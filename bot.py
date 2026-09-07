"""
Sristi - AI companion Telegram bot (MVP)

Setup:
  1. pip install -r requirements.txt
  2. Copy .env.example to .env and fill in your keys
  3. python bot.py

What this does:
  - Listens for Telegram messages
  - Tracks each user's conversation history in a local SQLite database
  - Periodically summarizes old messages into a short memory summary
    (so the persona "remembers" things without resending the whole history)
  - Calls Google's Gemini API with the Sristi persona prompt + recent history
  - Sends the reply back to the user
"""

import os
import re
import random
import sqlite3
import logging
import asyncio
from datetime import datetime, timezone, timedelta, time as dt_time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from persona import build_system_prompt

load_dotenv()

os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "")

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma-4-31b-it").strip()
DB_PATH = os.environ.get("DB_PATH", "sristi.db").strip()

RECENT_MESSAGES_LIMIT = 12       # how many raw messages to keep in the prompt every time
SUMMARIZE_AFTER_MESSAGES = 30    # once a user's raw history passes this, summarize the old part

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sristi-bot")

client = None


def get_openai_client() -> OpenAI:
    global client
    if client is None:
        if not GEMINI_API_KEY or GEMINI_API_KEY in ("your_gemini_api_key_here", "your_openai_api_key_here", "your_openrouter_api_key_here"):
            raise ValueError("GEMINI_API_KEY is not configured. Please set it in your .env file.")
        client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return client


# ---------- Database ----------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            first_seen TEXT NOT NULL,
            memory_summary TEXT DEFAULT '',
            engagement_score REAL DEFAULT 0.5,
            proactive_sent_today INTEGER DEFAULT 0,
            proactive_reset_date TEXT DEFAULT '',
            last_proactive_at TIMESTAMP,
            consecutive_unanswered INTEGER DEFAULT 0,
            last_message_at TIMESTAMP,
            is_blocked INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    # Check and add columns if upgrading an existing database
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    columns_to_add = [
        ("engagement_score", "REAL DEFAULT 0.5"),
        ("proactive_sent_today", "INTEGER DEFAULT 0"),
        ("proactive_reset_date", "TEXT DEFAULT ''"),
        ("last_proactive_at", "TIMESTAMP"),
        ("consecutive_unanswered", "INTEGER DEFAULT 0"),
        ("last_message_at", "TIMESTAMP"),
        ("is_blocked", "INTEGER DEFAULT 0"),
    ]
    for col_name, col_type in columns_to_add:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def get_or_create_user(user_id: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        now = datetime.now(timezone.utc).isoformat()
        today_nepal = datetime.now(ZoneInfo("Asia/Kathmandu")).strftime("%Y-%m-%d")
        conn.execute(
            """INSERT INTO users (
                user_id, first_seen, memory_summary, engagement_score,
                proactive_sent_today, proactive_reset_date, last_proactive_at,
                consecutive_unanswered, last_message_at, is_blocked
            ) VALUES (?, ?, '', 0.5, 0, ?, NULL, 0, NULL, 0)""",
            (user_id, now, today_nepal),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    user_dict = dict(row)
    conn.close()
    return user_dict


def split_into_messages(reply: str) -> list[str]:
    """Splits a reply into separate message parts on the [||] delimiter,
    stripping whitespace and filtering out any empty parts."""
    parts = [p.strip() for p in reply.split("[||]")]
    return [p for p in parts if p]


def compute_reply_delay(incoming_text: str, reply_text: str) -> float:
    """Returns a delay in seconds meant to simulate realistic human reading
    and typing time."""
    # Reading time: a moment to read what the user sent, scales with length
    reading_time = min(len(incoming_text) * random.uniform(0.015, 0.03), 2.5)

    # Typing time: scales with the length of the reply being sent
    typing_time = len(reply_text) * random.uniform(0.035, 0.06)

    # Small base delay so even short replies don't feel instant
    base_delay = random.uniform(0.6, 1.2)

    total = base_delay + reading_time + typing_time

    # Cap it so users never wait uncomfortably long, floor so it's never
    # instant either
    return max(0.8, min(total, 6.0))


def save_message(user_id: int, role: str, content: str, timestamp: str | None = None):
    conn = sqlite3.connect(DB_PATH)
    ts = timestamp if timestamp is not None else datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, role, content, ts),
    )
    conn.commit()
    conn.close()


def get_recent_messages(user_id: int, limit: int = RECENT_MESSAGES_LIMIT):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def count_messages(user_id: int) -> int:
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,)).fetchone()[0]
    conn.close()
    return count


def update_memory_summary(user_id: int, new_summary: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET memory_summary = ? WHERE user_id = ?", (new_summary, user_id))
    conn.commit()
    conn.close()


def trim_old_messages(user_id: int, keep_last: int = RECENT_MESSAGES_LIMIT):
    """Delete everything except the most recent `keep_last` messages for this user."""
    conn = sqlite3.connect(DB_PATH)
    ids_to_keep = conn.execute(
        "SELECT id FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, keep_last),
    ).fetchall()
    if ids_to_keep:
        keep_ids = tuple(i[0] for i in ids_to_keep)
        placeholders = ",".join("?" * len(keep_ids))
        conn.execute(
            f"DELETE FROM messages WHERE user_id = ? AND id NOT IN ({placeholders})",
            (user_id, *keep_ids),
        )
    conn.commit()
    conn.close()


# ---------- Helpers ----------

def get_nepal_time_string() -> str:
    now = datetime.now(ZoneInfo("Asia/Kathmandu"))
    return now.strftime("%A, %I:%M %p")  # e.g. "Thursday, 11:42 PM"


def get_days_known(first_seen_iso: str) -> int:
    first_seen = datetime.fromisoformat(first_seen_iso)
    delta = datetime.now(timezone.utc) - first_seen
    return delta.days


def strip_thinking_tags(reply: str) -> str:
    """Strip out anything between <thought> and </thought> tags (including the tags themselves)."""
    if not reply:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", reply, flags=re.DOTALL)
    return cleaned.strip()


def call_openai_with_retry(messages: list, max_retries: int = 3, initial_delay: float = 1.0) -> str:
    """Call OpenAI / Gemini API with retry logic and backoff, returning cleaned response text."""
    ai_client = get_openai_client()
    last_err: Exception = Exception("LLM API call failed after retries")
    for attempt in range(max_retries):
        try:
            response = ai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                extra_body={
                    "extra_body": {
                        "google": {
                            "thinking_config": {
                                "include_thoughts": False,
                            }
                        }
                    }
                },
            )
            content = response.choices[0].message.content
            cleaned = strip_thinking_tags(content) if content else ""
            return cleaned
        except Exception as e:
            last_err = e
            logger.warning(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(initial_delay * (2 ** attempt))
    raise last_err


from decimal import Decimal, ROUND_HALF_UP


def get_daily_budget(engagement_score: float) -> int:
    """Return daily proactive message budget (5 to 10) based on engagement score."""
    if engagement_score < 0.4:
        return 5
    if engagement_score >= 0.8:
        return 10
    budget = Decimal("5") + (Decimal(str(round(engagement_score, 4))) - Decimal("0.4")) * Decimal("12.5")
    return int(budget.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def is_within_active_hours(dt: datetime | None = None) -> bool:
    """Check if current time is between 8am and 10pm Nepal time."""
    if dt is None:
        dt = datetime.now(ZoneInfo("Asia/Kathmandu"))
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Asia/Kathmandu"))
    else:
        dt = dt.astimezone(ZoneInfo("Asia/Kathmandu"))
    return 8 <= dt.hour < 22


def should_send_proactive_message(user: dict, now: datetime | None = None) -> bool:
    """Determine if a proactive message should be sent to the user."""
    # User is not blocked
    if bool(user.get("is_blocked", 0)):
        return False

    # days_known for this user is 5 or more
    if get_days_known(user["first_seen"]) < 5:
        return False

    # is_within_active_hours() is True
    if not is_within_active_hours(now):
        return False

    # consecutive_unanswered is less than 2
    if user.get("consecutive_unanswered", 0) >= 2:
        return False

    # Nepal current date for daily counter reset
    nepal_tz = ZoneInfo("Asia/Kathmandu")
    current_nepal_dt = now.astimezone(nepal_tz) if (now and now.tzinfo) else (
        now.replace(tzinfo=nepal_tz) if now else datetime.now(nepal_tz)
    )
    today_nepal = current_nepal_dt.strftime("%Y-%m-%d")

    # Reset proactive_sent_today to 0 first if proactive_reset_date isn't today
    if user.get("proactive_reset_date") != today_nepal:
        user["proactive_sent_today"] = 0
        user["proactive_reset_date"] = today_nepal
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE users SET proactive_sent_today = 0, proactive_reset_date = ? WHERE user_id = ?",
            (today_nepal, user["user_id"]),
        )
        conn.commit()
        conn.close()

    # proactive_sent_today < get_daily_budget(engagement_score)
    score = user.get("engagement_score", 0.5)
    daily_budget = get_daily_budget(score)
    if user.get("proactive_sent_today", 0) >= daily_budget:
        return False

    # At least (14 hours / daily_budget) has passed since last_proactive_at, with ±20% jitter
    last_proactive_at = user.get("last_proactive_at")
    if last_proactive_at:
        if isinstance(last_proactive_at, str):
            last_dt = datetime.fromisoformat(last_proactive_at)
        else:
            last_dt = last_proactive_at
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        current_utc = now if now is not None else datetime.now(timezone.utc)
        if current_utc.tzinfo is None:
            current_utc = current_utc.replace(tzinfo=timezone.utc)

        elapsed_seconds = (current_utc - last_dt).total_seconds()
        base_interval_seconds = (14.0 / daily_budget) * 3600.0
        jitter_factor = random.uniform(0.8, 1.2)
        required_interval = base_interval_seconds * jitter_factor

        if elapsed_seconds < required_interval:
            return False

    return True


def generate_proactive_message(user: dict, nepal_now: datetime | None = None) -> list[str]:
    """Generate an unprompted proactive outreach message from Sristi as a list of message parts."""
    if nepal_now is None:
        nepal_now = datetime.now(ZoneInfo("Asia/Kathmandu"))
    elif nepal_now.tzinfo is None:
        nepal_now = nepal_now.replace(tzinfo=ZoneInfo("Asia/Kathmandu"))
    else:
        nepal_now = nepal_now.astimezone(ZoneInfo("Asia/Kathmandu"))

    valid_types = ["random thought"]
    nepal_hour = nepal_now.hour

    if 8 <= nepal_hour < 12:
        valid_types.append("morning greeting")
    if 18 <= nepal_hour < 22:
        valid_types.append("evening check-in")
    if user.get("memory_summary") and user["memory_summary"].strip():
        valid_types.append("follow up on something unresolved")

    outreach_type = random.choice(valid_types)
    days_known = get_days_known(user["first_seen"])
    current_time = get_nepal_time_string()

    system_prompt = build_system_prompt(
        current_time=current_time,
        memory_summary=user.get("memory_summary", ""),
        days_known=days_known,
    )

    history = get_recent_messages(user["user_id"])
    instruction = (
        f"Write a short, natural message reaching out to this person first, in your usual texting style. "
        f"The reason: {outreach_type}. Keep it brief, like a real unprompted text."
    )

    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "system", "content": instruction}]
    )

    try:
        reply = call_openai_with_retry(messages)
        return split_into_messages(reply) if reply else []
    except Exception:
        logger.exception(f"Error generating proactive message for user {user.get('user_id')}")
        return []


def update_user_incoming_engagement(user_id: int):
    """Update engagement score and consecutive_unanswered upon receiving any incoming user message."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        return

    user = dict(row)
    last_proactive_at = user.get("last_proactive_at")
    last_message_at = user.get("last_message_at")
    score = user.get("engagement_score", 0.5)

    is_reply_to_proactive = False
    if last_proactive_at:
        if not last_message_at:
            is_reply_to_proactive = True
        else:
            dt_proactive = datetime.fromisoformat(last_proactive_at)
            dt_message = datetime.fromisoformat(last_message_at)
            if dt_proactive > dt_message:
                is_reply_to_proactive = True

    now_utc = datetime.now(timezone.utc).isoformat()
    if is_reply_to_proactive:
        new_score = min(1.0, score * 0.9 + 1.0 * 0.1)
        conn.execute(
            """UPDATE users
               SET consecutive_unanswered = 0,
                   engagement_score = ?,
                   last_message_at = ?
               WHERE user_id = ?""",
            (new_score, now_utc, user_id),
        )
    else:
        conn.execute(
            "UPDATE users SET last_message_at = ? WHERE user_id = ?",
            (now_utc, user_id),
        )
    conn.commit()
    conn.close()


def check_daily_unanswered_proactive(now_dt: datetime | None = None):
    """Check for users whose last_proactive_at was > 12h ago with no reply since, increment consecutive_unanswered and decrease engagement_score."""
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    elif now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    users = conn.execute("SELECT * FROM users WHERE last_proactive_at IS NOT NULL").fetchall()

    for user_row in users:
        user = dict(user_row)
        last_proactive_at = user["last_proactive_at"]
        last_message_at = user["last_message_at"]

        dt_proactive = datetime.fromisoformat(last_proactive_at)
        if dt_proactive.tzinfo is None:
            dt_proactive = dt_proactive.replace(tzinfo=timezone.utc)

        elapsed_seconds = (now_dt - dt_proactive).total_seconds()
        if elapsed_seconds <= 12 * 3600:
            continue

        had_reply = False
        if last_message_at:
            dt_message = datetime.fromisoformat(last_message_at)
            if dt_message.tzinfo is None:
                dt_message = dt_message.replace(tzinfo=timezone.utc)
            if dt_message >= dt_proactive:
                had_reply = True

        if not had_reply:
            if last_message_at:
                unanswered_count = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE user_id = ? AND role = 'assistant' AND timestamp > ?",
                    (user["user_id"], last_message_at),
                ).fetchone()[0]
            else:
                unanswered_count = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE user_id = ? AND role = 'assistant'",
                    (user["user_id"],),
                ).fetchone()[0]

            if unanswered_count == 0 or user["consecutive_unanswered"] < unanswered_count:
                new_consecutive = user["consecutive_unanswered"] + 1
                new_score = max(0.0, user["engagement_score"] * 0.9 + 0.0 * 0.1)
                conn.execute(
                    """UPDATE users
                       SET consecutive_unanswered = ?,
                           engagement_score = ?
                       WHERE user_id = ?""",
                    (new_consecutive, new_score, user["user_id"]),
                )
                logger.info(
                    f"User {user['user_id']} penalized for unanswered proactive message: "
                    f"consecutive_unanswered={new_consecutive}, engagement_score={new_score:.3f}"
                )

    conn.commit()
    conn.close()


def summarize_old_messages(user_id: int, existing_summary: str):
    """Ask the model to fold older raw messages into a short updated memory summary."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id ASC",
        (user_id,),
    ).fetchall()
    conn.close()

    # Keep the most recent RECENT_MESSAGES_LIMIT raw, summarize everything before that
    to_summarize = rows[:-RECENT_MESSAGES_LIMIT] if len(rows) > RECENT_MESSAGES_LIMIT else []
    if not to_summarize:
        return

    transcript = "\n".join(f"{role}: {content}" for role, content in to_summarize)

    prompt = (
        "Summarize the important, memorable facts and emotional moments from this conversation "
        "in 3-5 short bullet points. Focus on things a close friend would actually remember "
        "(feelings, events, plans, preferences) - skip small talk. "
        "Combine with the existing summary below if relevant, keep it concise.\n\n"
        f"Existing summary:\n{existing_summary or '(none yet)'}\n\n"
        f"New conversation to fold in:\n{transcript}"
    )

    try:
        new_summary = call_openai_with_retry([{"role": "user", "content": prompt}])
        if new_summary:
            update_memory_summary(user_id, new_summary)
            trim_old_messages(user_id, keep_last=RECENT_MESSAGES_LIMIT)
            logger.info(f"Updated memory summary for user {user_id}")
    except Exception:
        logger.exception("Failed to summarize old messages for user %s", user_id)


# ---------- Core reply logic ----------

def generate_reply(user_id: int, user_message: str) -> list[str]:
    try:
        update_user_incoming_engagement(user_id)
    except Exception:
        logger.exception("Error updating incoming engagement for user %s", user_id)

    user = get_or_create_user(user_id)
    days_known = get_days_known(user["first_seen"])
    current_time = get_nepal_time_string()

    system_prompt = build_system_prompt(
        current_time=current_time,
        memory_summary=user["memory_summary"],
        days_known=days_known,
    )

    history = get_recent_messages(user_id)
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_message}
    ]

    try:
        content = call_openai_with_retry(messages)
        reply = content if content else "hmm kehi issue vayo jasto cha, feri bhana na"
    except Exception:
        logger.exception("Error generating reply")
        reply = "hmm kehi issue vayo jasto cha, feri bhana na"

    parts = split_into_messages(reply)
    if not parts:
        parts = ["hmm kehi issue vayo jasto cha, feri bhana na"]

    save_message(user_id, "user", user_message)

    base_time = datetime.now(timezone.utc)
    for i, part in enumerate(parts):
        ts = (base_time + timedelta(seconds=i * 2)).isoformat()
        save_message(user_id, "assistant", part, timestamp=ts)

    if count_messages(user_id) > SUMMARIZE_AFTER_MESSAGES:
        summarize_old_messages(user_id, user["memory_summary"])

    return parts


# ---------- Telegram handlers & jobs ----------

async def send_split_message_sequence(
    bot, chat_id: int, parts: list[str],
    reply_to_message=None, incoming_text: str = "",
):
    """Send a sequence of message parts with typing action and realistic delays.

    For the first part, the delay simulates reading the incoming message and
    typing the reply. For follow-up parts (double-texting), the delay simulates
    re-reading the previous part and typing the next one.
    """
    for i, part in enumerate(parts):
        # Determine what text the delay should be based on
        context_text = incoming_text if i == 0 else parts[i - 1]

        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            logger.warning("Failed to send chat action typing")

        delay = compute_reply_delay(context_text, part)
        await asyncio.sleep(delay)

        if reply_to_message:
            await reply_to_message.reply_text(part)
        else:
            await bot.send_message(chat_id=chat_id, text=part)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    get_or_create_user(update.effective_user.id)
    await update.message.reply_text("namaste!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    try:
        reply_parts = generate_reply(user_id, user_text)
    except Exception:
        logger.exception("Error generating reply")
        reply_parts = ["ali network issue vayo, feri try garnus is"]

    await send_split_message_sequence(
        bot=context.bot,
        chat_id=update.effective_chat.id,
        parts=reply_parts,
        reply_to_message=update.message,
        incoming_text=user_text,
    )


async def proactive_messaging_job(context: ContextTypes.DEFAULT_TYPE):
    """Job running every 45 minutes to send proactive messages to eligible users."""
    logger.info("Checking users for proactive messages...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM users WHERE is_blocked = 0").fetchall()
    users = [dict(r) for r in rows]
    conn.close()

    for user in users:
        try:
            if get_days_known(user["first_seen"]) < 5:
                continue
            if should_send_proactive_message(user):
                parts = generate_proactive_message(user)
                if not parts:
                    continue
                user_id = user["user_id"]
                await send_split_message_sequence(
                    bot=context.bot,
                    chat_id=user_id,
                    parts=parts,
                )
                now_utc = datetime.now(timezone.utc)
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    """UPDATE users
                       SET proactive_sent_today = proactive_sent_today + 1,
                           last_proactive_at = ?
                       WHERE user_id = ?""",
                    (now_utc.isoformat(), user_id),
                )
                conn.commit()
                conn.close()

                for i, part in enumerate(parts):
                    ts = (now_utc + timedelta(seconds=i * 2)).isoformat()
                    save_message(user_id, "assistant", part, timestamp=ts)

                logger.info(f"Sent proactive message to user {user_id} ({len(parts)} part(s))")
        except Exception:
            logger.exception(f"Error in proactive message for user {user.get('user_id')}")


async def daily_engagement_audit_job(context: ContextTypes.DEFAULT_TYPE):
    """Daily audit around midnight Nepal time for unanswered proactive messages."""
    logger.info("Running daily unanswered proactive audit...")
    try:
        check_daily_unanswered_proactive()
    except Exception:
        logger.exception("Error during daily proactive audit")


def check_configuration():
    issues = []
    if not GEMINI_API_KEY or GEMINI_API_KEY in ("your_gemini_api_key_here", "your_openai_api_key_here", "your_openrouter_api_key_here"):
        issues.append("- GEMINI_API_KEY is missing or unconfigured in .env (get one from https://aistudio.google.com)")
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
        issues.append("- TELEGRAM_BOT_TOKEN is missing or unconfigured in .env")
    return issues


def cli_chat():
    """Terminal chat mode to test Sristi directly without needing Telegram."""
    init_db()
    print("\n=======================================================")
    print("  Sristi CLI Chat (Terminal Mode)")
    print("  Type your message and press Enter. Type 'exit' to quit.")
    print("=======================================================")
    if not GEMINI_API_KEY or GEMINI_API_KEY in ("your_gemini_api_key_here", "your_openai_api_key_here"):
        print("[!] Note: GEMINI_API_KEY is not set in .env yet.")
        print("    Please set your key in .env to talk with Sristi.\n")
        return
    user_id = 12345678  # Dedicated CLI test user ID
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Bye!")
                break
            reply_parts = generate_reply(user_id, user_input)
            for part in reply_parts:
                print(f"Sristi: {part}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")


def main():
    init_db()
    issues = check_configuration()
    if issues:
        print("\n" + "=" * 60)
        print(" CONFIGURATION REQUIRED BEFORE RUNNING TELEGRAM BOT:")
        print("=" * 60)
        for issue in issues:
            print(f" {issue}")
        print("\n Steps to configure:")
        print(" 1. Open the '.env' file in this folder.")
        print(" 2. Set your GEMINI_API_KEY (from https://aistudio.google.com).")
        print(" 3. Set your TELEGRAM_BOT_TOKEN (from @BotFather on Telegram).")
        print(" 4. Run 'python bot.py' again.")
        print("=" * 60 + "\n")
        return

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    pid = os.getpid()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Bot starting... [Timestamp: {start_time}] [PID: {pid}]")
    print(f"\n=======================================================")
    print(f"  Bot starting... [Time: {start_time}] [PID: {pid}]")
    print(f"  Press Ctrl+C in this terminal to stop the bot cleanly.")
    print(f"=======================================================\n")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if app.job_queue:
        # Job 1: Proactive messaging check every 45 minutes
        app.job_queue.run_repeating(proactive_messaging_job, interval=45 * 60, first=60)
        # Job 2: Daily audit around midnight Nepal time
        app.job_queue.run_daily(
            daily_engagement_audit_job,
            time=dt_time(hour=0, minute=0, tzinfo=ZoneInfo("Asia/Kathmandu")),
        )
        logger.info("Proactive messaging jobs successfully registered on JobQueue.")
    else:
        logger.warning("JobQueue not initialized. Proactive jobs will not run.")

    logger.info(f"Sristi bot is running and listening for Telegram messages (PID: {pid})...")
    app.run_polling(drop_pending_updates=True)



if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        cli_chat()
    else:
        main()
