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
import sqlite3
import logging
import asyncio
from datetime import datetime, timezone
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
            memory_summary TEXT DEFAULT ''
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
    conn.commit()
    conn.close()


def get_or_create_user(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT user_id, first_seen, memory_summary FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT INTO users (user_id, first_seen, memory_summary) VALUES (?, ?, '')", (user_id, now))
        conn.commit()
        row = (user_id, now, "")
    conn.close()
    return {"user_id": row[0], "first_seen": row[1], "memory_summary": row[2]}


def save_message(user_id: int, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, role, content, datetime.now(timezone.utc).isoformat()),
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

    ai_client = get_openai_client()
    response = ai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
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
    raw_summary = response.choices[0].message.content or ""
    new_summary = strip_thinking_tags(raw_summary)
    if new_summary:
        update_memory_summary(user_id, new_summary)
        trim_old_messages(user_id, keep_last=RECENT_MESSAGES_LIMIT)
        logger.info(f"Updated memory summary for user {user_id}")


# ---------- Core reply logic ----------

def generate_reply(user_id: int, user_message: str) -> str:
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

    ai_client = get_openai_client()
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
    cleaned_reply = strip_thinking_tags(content) if content else ""
    reply = cleaned_reply if cleaned_reply else "hmm kehi issue vayo jasto cha, feri bhana na"

    save_message(user_id, "user", user_message)
    save_message(user_id, "assistant", reply)

    if count_messages(user_id) > SUMMARIZE_AFTER_MESSAGES:
        summarize_old_messages(user_id, user["memory_summary"])

    return reply


# ---------- Telegram handlers ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    get_or_create_user(update.effective_user.id)
    await update.message.reply_text("namaste!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    try:
        reply = generate_reply(user_id, user_text)
    except Exception:
        logger.exception("Error generating reply")
        reply = "ali network issue vayo, feri try garnus is"
    await update.message.reply_text(reply)


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
            reply = generate_reply(user_id, user_input)
            print(f"Sristi: {reply}")
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
    logger.info(f"Sristi bot is running and listening for Telegram messages (PID: {pid})...")
    app.run_polling(drop_pending_updates=True)



if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        cli_chat()
    else:
        main()
