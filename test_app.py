"""
Automated tests for Sristi Telegram bot and Persona logic.
Tests DB, prompt generation, message trimming, and mock reply generation.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import bot
from persona import build_system_prompt


class TestSristiBot(unittest.TestCase):
    def setUp(self):
        # Use an in-memory or temp db for testing
        self.test_db = "test_sristi.db"
        bot.DB_PATH = self.test_db
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        bot.init_db()

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_database_init_and_user_creation(self):
        user = bot.get_or_create_user(1001)
        self.assertEqual(user["user_id"], 1001)
        self.assertEqual(user["memory_summary"], "")
        self.assertTrue(len(user["first_seen"]) > 0)

        # Calling again should retrieve the same user
        same_user = bot.get_or_create_user(1001)
        self.assertEqual(same_user["first_seen"], user["first_seen"])

    def test_message_saving_and_retrieval(self):
        user_id = 2002
        bot.get_or_create_user(user_id)

        bot.save_message(user_id, "user", "hi")
        bot.save_message(user_id, "assistant", "namaste")

        self.assertEqual(bot.count_messages(user_id), 2)
        recent = bot.get_recent_messages(user_id, limit=5)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["role"], "user")
        self.assertEqual(recent[0]["content"], "hi")
        self.assertEqual(recent[1]["role"], "assistant")
        self.assertEqual(recent[1]["content"], "namaste")

    def test_trim_old_messages(self):
        user_id = 3003
        bot.get_or_create_user(user_id)
        for i in range(15):
            bot.save_message(user_id, "user", f"msg {i}")

        self.assertEqual(bot.count_messages(user_id), 15)
        bot.trim_old_messages(user_id, keep_last=5)
        self.assertEqual(bot.count_messages(user_id), 5)

        recent = bot.get_recent_messages(user_id, limit=10)
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[0]["content"], "msg 10")
        self.assertEqual(recent[-1]["content"], "msg 14")

    def test_nepal_time_string(self):
        time_str = bot.get_nepal_time_string()
        self.assertTrue(any(day in time_str for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]))
        self.assertTrue("AM" in time_str or "PM" in time_str)

    def test_persona_prompt_generation(self):
        prompt = build_system_prompt("Friday, 11:30 AM", "Likes cold coffee", 3)
        self.assertIn("Friday, 11:30 AM", prompt)
        self.assertIn("Likes cold coffee", prompt)
        self.assertIn("Days since you first talked: 3", prompt)
        self.assertIn("Sristi, a 22-year-old girl from Kathmandu", prompt)

    @patch("bot.get_openai_client")
    def test_generate_reply_flow(self, mock_get_client):
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "k gardai xau?"
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_client.return_value = mock_client

        user_id = 4004
        reply = bot.generate_reply(user_id, "hello")
        self.assertEqual(reply, ["k gardai xau?"])

        # Check that extra_headers is not passed
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertNotIn("extra_headers", kwargs)

        # Check DB messages saved
        messages = bot.get_recent_messages(user_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "hello")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "k gardai xau?")

    @patch("bot.get_openai_client")
    def test_generate_reply_strips_thoughts(self, mock_get_client):
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "<thought>\nUser said hi.\nI should respond in character.\n</thought>\n\nNamaste! Kasto xa?"
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_client.return_value = mock_client

        user_id = 5005
        reply = bot.generate_reply(user_id, "hi")
        self.assertEqual(reply, ["Namaste! Kasto xa?"])

        # Check that the cleaned reply without thoughts was saved in DB
        messages = bot.get_recent_messages(user_id)
        self.assertEqual(messages[1]["content"], "Namaste! Kasto xa?")

    def test_strip_thinking_tags(self):
        self.assertEqual(bot.strip_thinking_tags("<thought>internal thoughts</thought>clean reply"), "clean reply")
        self.assertEqual(bot.strip_thinking_tags("<thought>\nmulti\nline\n</thought>\n  clean  "), "clean")
        self.assertEqual(bot.strip_thinking_tags("no thought tags"), "no thought tags")
        self.assertEqual(bot.strip_thinking_tags("<thought>only thoughts</thought>"), "")
        self.assertEqual(bot.strip_thinking_tags(""), "")

    def test_gemini_client_configuration(self):
        saved_key = bot.GEMINI_API_KEY
        saved_client = bot.client
        try:
            bot.GEMINI_API_KEY = "test-gemini-key"
            bot.client = None
            client = bot.get_openai_client()
            self.assertEqual(str(client.base_url), "https://generativelanguage.googleapis.com/v1beta/openai/")
            self.assertEqual(bot.MODEL_NAME, "gemma-4-31b-it")
        finally:
            bot.GEMINI_API_KEY = saved_key
            bot.client = saved_client

    def test_get_daily_budget(self):
        # Below 0.4 -> 5
        self.assertEqual(bot.get_daily_budget(0.0), 5)
        self.assertEqual(bot.get_daily_budget(0.2), 5)
        self.assertEqual(bot.get_daily_budget(0.39), 5)
        # Above 0.8 -> 10
        self.assertEqual(bot.get_daily_budget(0.8), 10)
        self.assertEqual(bot.get_daily_budget(0.95), 10)
        self.assertEqual(bot.get_daily_budget(1.0), 10)
        # Scaled between 0.4 and 0.8
        self.assertEqual(bot.get_daily_budget(0.4), 5)
        self.assertEqual(bot.get_daily_budget(0.6), 8)
        self.assertEqual(bot.get_daily_budget(0.7), 9)

    def test_is_within_active_hours(self):
        from zoneinfo import ZoneInfo
        from datetime import datetime
        tz = ZoneInfo("Asia/Kathmandu")
        self.assertTrue(bot.is_within_active_hours(datetime(2026, 9, 7, 8, 0, tzinfo=tz)))
        self.assertTrue(bot.is_within_active_hours(datetime(2026, 9, 7, 14, 30, tzinfo=tz)))
        self.assertTrue(bot.is_within_active_hours(datetime(2026, 9, 7, 21, 59, tzinfo=tz)))
        self.assertFalse(bot.is_within_active_hours(datetime(2026, 9, 7, 22, 0, tzinfo=tz)))
        self.assertFalse(bot.is_within_active_hours(datetime(2026, 9, 7, 7, 59, tzinfo=tz)))
        self.assertFalse(bot.is_within_active_hours(datetime(2026, 9, 7, 1, 0, tzinfo=tz)))

    def test_should_send_proactive_message(self):
        from zoneinfo import ZoneInfo
        from datetime import datetime, timezone, timedelta
        tz = ZoneInfo("Asia/Kathmandu")
        now_nepal = datetime(2026, 9, 7, 12, 0, tzinfo=tz)

        # 1. Less than 5 days known -> False
        user_new = {
            "user_id": 9001,
            "first_seen": (now_nepal - timedelta(days=3)).astimezone(timezone.utc).isoformat(),
            "engagement_score": 0.5,
            "proactive_sent_today": 0,
            "proactive_reset_date": "2026-09-07",
            "consecutive_unanswered": 0,
            "is_blocked": 0,
            "last_proactive_at": None,
        }
        self.assertFalse(bot.should_send_proactive_message(user_new, now=now_nepal))

        # 2. Blocked user -> False
        user_blocked = dict(user_new)
        user_blocked["first_seen"] = (now_nepal - timedelta(days=10)).astimezone(timezone.utc).isoformat()
        user_blocked["is_blocked"] = 1
        self.assertFalse(bot.should_send_proactive_message(user_blocked, now=now_nepal))

        # 3. Inactive hours -> False
        night_time = datetime(2026, 9, 7, 23, 0, tzinfo=tz)
        user_ok = dict(user_blocked)
        user_ok["is_blocked"] = 0
        self.assertFalse(bot.should_send_proactive_message(user_ok, now=night_time))

        # 4. consecutive_unanswered >= 2 -> False
        user_unanswered = dict(user_ok)
        user_unanswered["consecutive_unanswered"] = 2
        self.assertFalse(bot.should_send_proactive_message(user_unanswered, now=now_nepal))

        # 5. Over budget today -> False
        user_budget_max = dict(user_ok)
        user_budget_max["engagement_score"] = 0.5  # budget is 6
        user_budget_max["proactive_sent_today"] = 6
        self.assertFalse(bot.should_send_proactive_message(user_budget_max, now=now_nepal))

        # 6. Reset date different -> resets today to 0 and checks budget
        user_reset = bot.get_or_create_user(9002)
        import sqlite3
        conn = sqlite3.connect(self.test_db)
        old_seen = (now_nepal - timedelta(days=10)).astimezone(timezone.utc).isoformat()
        conn.execute(
            """UPDATE users 
               SET first_seen = ?, proactive_sent_today = 6, proactive_reset_date = '2026-09-06'
               WHERE user_id = 9002""",
            (old_seen,),
        )
        conn.commit()
        conn.close()
        user_refreshed = bot.get_or_create_user(9002)
        self.assertTrue(bot.should_send_proactive_message(user_refreshed, now=now_nepal))
        # Verify proactive_sent_today was reset in DB
        user_after = bot.get_or_create_user(9002)
        self.assertEqual(user_after["proactive_sent_today"], 0)
        self.assertEqual(user_after["proactive_reset_date"], "2026-09-07")

        # 7. Interval since last_proactive_at
        user_recent = dict(user_ok)
        user_recent["engagement_score"] = 0.5  # budget 6 -> interval = 14/6 = 2.33h = ~8400s
        # 10 minutes ago should fail
        user_recent["last_proactive_at"] = (now_nepal - timedelta(minutes=10)).astimezone(timezone.utc).isoformat()
        self.assertFalse(bot.should_send_proactive_message(user_recent, now=now_nepal))
        # 5 hours ago (> 2.33h * 1.2 = 2.8h) should succeed
        user_recent["last_proactive_at"] = (now_nepal - timedelta(hours=5)).astimezone(timezone.utc).isoformat()
        self.assertTrue(bot.should_send_proactive_message(user_recent, now=now_nepal))

    @patch("bot.call_openai_with_retry")
    def test_generate_proactive_message_outreach_types(self, mock_call_llm):
        from zoneinfo import ZoneInfo
        from datetime import datetime, timezone, timedelta
        tz = ZoneInfo("Asia/Kathmandu")
        mock_call_llm.return_value = "k gardai xau?"

        user = {
            "user_id": 8001,
            "first_seen": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            "memory_summary": "",
        }

        # Morning test (9 AM) -> outreach types can include morning greeting or random thought
        morning_dt = datetime(2026, 9, 7, 9, 0, tzinfo=tz)
        msg = bot.generate_proactive_message(user, nepal_now=morning_dt)
        self.assertEqual(msg, ["k gardai xau?"])
        prompt_arg = mock_call_llm.call_args[0][0][-1]["content"]
        self.assertTrue("morning greeting" in prompt_arg or "random thought" in prompt_arg)

        # Afternoon test (2 PM) without memory summary -> only random thought
        afternoon_dt = datetime(2026, 9, 7, 14, 0, tzinfo=tz)
        bot.generate_proactive_message(user, nepal_now=afternoon_dt)
        prompt_arg = mock_call_llm.call_args[0][0][-1]["content"]
        self.assertIn("random thought", prompt_arg)

        # Evening test (8 PM) with memory summary -> can be evening check-in, follow up, or random thought
        user_with_memory = dict(user)
        user_with_memory["memory_summary"] = "Likes tea and momos"
        evening_dt = datetime(2026, 9, 7, 20, 0, tzinfo=tz)
        bot.generate_proactive_message(user_with_memory, nepal_now=evening_dt)
        prompt_arg = mock_call_llm.call_args[0][0][-1]["content"]
        self.assertTrue(
            "evening check-in" in prompt_arg or 
            "follow up on something unresolved" in prompt_arg or 
            "random thought" in prompt_arg
        )

    @patch("bot.get_openai_client")
    def test_engagement_tracking_on_user_reply(self, mock_get_client):
        from datetime import datetime, timezone, timedelta
        import sqlite3
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "reply back"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_client.return_value = mock_client

        user_id = 7001
        bot.get_or_create_user(user_id)

        # Set user with a pending proactive message sent 30 mins ago
        now = datetime.now(timezone.utc)
        proactive_time = (now - timedelta(minutes=30)).isoformat()
        old_msg_time = (now - timedelta(hours=2)).isoformat()

        conn = sqlite3.connect(self.test_db)
        conn.execute(
            """UPDATE users 
               SET last_proactive_at = ?,
                   last_message_at = ?,
                   consecutive_unanswered = 1,
                   engagement_score = 0.5
               WHERE user_id = ?""",
            (proactive_time, old_msg_time, user_id),
        )
        conn.commit()
        conn.close()

        # User sends a reply
        bot.generate_reply(user_id, "hey sristi")

        updated_user = bot.get_or_create_user(user_id)
        # consecutive_unanswered must reset to 0
        self.assertEqual(updated_user["consecutive_unanswered"], 0)
        # engagement_score should increase: 0.5 * 0.9 + 1.0 * 0.1 = 0.55
        self.assertAlmostEqual(updated_user["engagement_score"], 0.55, places=3)
        # last_message_at should be updated
        self.assertTrue(updated_user["last_message_at"] > proactive_time)

    def test_check_daily_unanswered_proactive(self):
        from datetime import datetime, timezone, timedelta
        import sqlite3
        user_id_1 = 6001  # Unanswered > 12h
        user_id_2 = 6002  # Replied
        user_id_3 = 6003  # Recent (<12h)

        bot.get_or_create_user(user_id_1)
        bot.get_or_create_user(user_id_2)
        bot.get_or_create_user(user_id_3)

        now = datetime.now(timezone.utc)
        proactive_14h_ago = (now - timedelta(hours=14)).isoformat()
        proactive_2h_ago = (now - timedelta(hours=2)).isoformat()
        reply_13h_ago = (now - timedelta(hours=13)).isoformat()

        conn = sqlite3.connect(self.test_db)
        # User 1: proactive 14h ago, no reply
        conn.execute(
            """UPDATE users 
               SET last_proactive_at = ?,
                   last_message_at = NULL,
                   consecutive_unanswered = 0,
                   engagement_score = 0.5
               WHERE user_id = ?""",
            (proactive_14h_ago, user_id_1),
        )
        # User 2: proactive 14h ago, but replied 13h ago
        conn.execute(
            """UPDATE users 
               SET last_proactive_at = ?,
                   last_message_at = ?,
                   consecutive_unanswered = 0,
                   engagement_score = 0.5
               WHERE user_id = ?""",
            (proactive_14h_ago, reply_13h_ago, user_id_2),
        )
        # User 3: proactive only 2h ago
        conn.execute(
            """UPDATE users 
               SET last_proactive_at = ?,
                   last_message_at = NULL,
                   consecutive_unanswered = 0,
                   engagement_score = 0.5
               WHERE user_id = ?""",
            (proactive_2h_ago, user_id_3),
        )
        conn.commit()
        conn.close()

        # Run the audit
        bot.check_daily_unanswered_proactive(now_dt=now)

        u1 = bot.get_or_create_user(user_id_1)
        self.assertEqual(u1["consecutive_unanswered"], 1)
        # engagement_score: 0.5 * 0.9 + 0.0 * 0.1 = 0.45
        self.assertAlmostEqual(u1["engagement_score"], 0.45, places=3)

        u2 = bot.get_or_create_user(user_id_2)
        self.assertEqual(u2["consecutive_unanswered"], 0)
        self.assertEqual(u2["engagement_score"], 0.5)

        u3 = bot.get_or_create_user(user_id_3)
        self.assertEqual(u3["consecutive_unanswered"], 0)
        self.assertEqual(u3["engagement_score"], 0.5)

    def test_database_migration_existing_schema(self):
        import sqlite3
        migration_db = "test_migration.db"
        bot.DB_PATH = migration_db
        if os.path.exists(migration_db):
            os.remove(migration_db)

        # Create old schema table
        conn = sqlite3.connect(migration_db)
        conn.execute("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                first_seen TEXT NOT NULL,
                memory_summary TEXT DEFAULT ''
            )
        """)
        conn.execute("INSERT INTO users (user_id, first_seen) VALUES (111, '2026-09-01T00:00:00+00:00')")
        conn.commit()
        conn.close()

        # Run init_db to test migration
        bot.init_db()

        # Verify columns exist
        u = bot.get_or_create_user(111)
        self.assertEqual(u["user_id"], 111)
        self.assertEqual(u["engagement_score"], 0.5)
        self.assertEqual(u["consecutive_unanswered"], 0)
        self.assertEqual(u["is_blocked"], 0)

        # Cleanup
        if os.path.exists(migration_db):
            os.remove(migration_db)
        bot.DB_PATH = self.test_db

    @patch("bot.generate_proactive_message")
    @patch("bot.should_send_proactive_message")
    def test_proactive_messaging_job_flow(self, mock_should_send, mock_generate_msg):
        import asyncio
        from unittest.mock import AsyncMock
        from datetime import datetime, timezone, timedelta

        mock_should_send.return_value = True
        mock_generate_msg.return_value = ["k gardai xau sathi?"]

        user_id = 9999
        bot.get_or_create_user(user_id)
        # Set days_known >= 5
        conn = bot.sqlite3.connect(self.test_db)
        old_seen = (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()
        conn.execute("UPDATE users SET first_seen = ?, proactive_sent_today = 0 WHERE user_id = ?", (old_seen, user_id))
        conn.commit()
        conn.close()

        mock_context = MagicMock()
        mock_context.bot.send_message = AsyncMock()

        asyncio.run(bot.proactive_messaging_job(mock_context))

        mock_context.bot.send_message.assert_awaited_once_with(chat_id=user_id, text="k gardai xau sathi?")

        # Check DB update
        updated_user = bot.get_or_create_user(user_id)
        self.assertEqual(updated_user["proactive_sent_today"], 1)
        self.assertIsNotNone(updated_user["last_proactive_at"])

        # Check message saved to messages table with role 'assistant'
        recent = bot.get_recent_messages(user_id)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["role"], "assistant")
        self.assertEqual(recent[0]["content"], "k gardai xau sathi?")

    def test_split_into_messages(self):
        # Single message without delimiter
        self.assertEqual(bot.split_into_messages("single line text"), ["single line text"])
        
        # Double message with delimiter
        text = "haha hoina yar\n[||]\nbuba le call gardai hunuhuncha"
        self.assertEqual(
            bot.split_into_messages(text),
            ["haha hoina yar", "buba le call gardai hunuhuncha"]
        )

        # Trims whitespace and filters empty parts
        text_with_empty = "  [||]  first thought  [||]   [||] second thought  [||]  "
        self.assertEqual(
            bot.split_into_messages(text_with_empty),
            ["first thought", "second thought"]
        )

        # Empty string
        self.assertEqual(bot.split_into_messages("   \n  "), [])

    @patch("bot.get_openai_client")
    def test_generate_reply_double_texting(self, mock_get_client):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "haha sachikai? [||] maile ta thyakkai tehi socheko thie"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_client.return_value = mock_client

        user_id = 1212
        bot.get_or_create_user(user_id)

        parts = bot.generate_reply(user_id, "kura milena")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "haha sachikai?")
        self.assertEqual(parts[1], "maile ta thyakkai tehi socheko thie")

        # Verify DB messages saved: 1 user message + 2 assistant messages
        conn = bot.sqlite3.connect(self.test_db)
        rows = conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE user_id = ? ORDER BY id ASC",
            (user_id,),
        ).fetchall()
        conn.close()

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][0], "user")
        self.assertEqual(rows[0][1], "kura milena")
        self.assertEqual(rows[1][0], "assistant")
        self.assertEqual(rows[1][1], "haha sachikai?")
        self.assertEqual(rows[2][0], "assistant")
        self.assertEqual(rows[2][1], "maile ta thyakkai tehi socheko thie")

        # The second assistant message timestamp should be after the first assistant message timestamp
        self.assertTrue(rows[2][2] > rows[1][2])

    @patch("bot.compute_reply_delay", return_value=1.5)
    @patch("asyncio.sleep")
    def test_handle_message_multi_part_typing_delay(self, mock_sleep, mock_delay):
        import asyncio
        from unittest.mock import AsyncMock

        mock_update = MagicMock()
        mock_update.effective_user.id = 1313
        mock_update.effective_chat.id = 1313
        mock_update.message.text = "hello"
        mock_update.message.reply_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.bot.send_chat_action = AsyncMock()

        with patch("bot.generate_reply", return_value=["part one", "part two - follow up thought"]):
            asyncio.run(bot.handle_message(mock_update, mock_context))

        # Typing action should fire for BOTH parts (first + follow-up)
        self.assertEqual(mock_context.bot.send_chat_action.await_count, 2)
        # Sleep called for both parts with compute_reply_delay value
        self.assertEqual(mock_sleep.await_count, 2)
        for call in mock_sleep.call_args_list:
            self.assertEqual(call[0][0], 1.5)

        # Both parts replied
        self.assertEqual(mock_update.message.reply_text.await_count, 2)
        mock_update.message.reply_text.assert_any_await("part one")
        mock_update.message.reply_text.assert_any_await("part two - follow up thought")

    @patch("bot.compute_reply_delay", return_value=1.0)
    @patch("asyncio.sleep")
    @patch("bot.generate_proactive_message")
    @patch("bot.should_send_proactive_message")
    def test_proactive_messaging_job_multi_part(self, mock_should_send, mock_generate_msg, mock_sleep, mock_delay):
        import asyncio
        from unittest.mock import AsyncMock
        from datetime import datetime, timezone, timedelta

        mock_should_send.return_value = True
        mock_generate_msg.return_value = ["good morning!", "k cha khabar?"]

        user_id = 1414
        bot.get_or_create_user(user_id)
        conn = bot.sqlite3.connect(self.test_db)
        old_seen = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        conn.execute("UPDATE users SET first_seen = ?, proactive_sent_today = 0 WHERE user_id = ?", (old_seen, user_id))
        conn.commit()
        conn.close()

        mock_context = MagicMock()
        mock_context.bot.send_message = AsyncMock()
        mock_context.bot.send_chat_action = AsyncMock()

        asyncio.run(bot.proactive_messaging_job(mock_context))

        # Two messages sent
        self.assertEqual(mock_context.bot.send_message.await_count, 2)
        mock_context.bot.send_message.assert_any_await(chat_id=user_id, text="good morning!")
        mock_context.bot.send_message.assert_any_await(chat_id=user_id, text="k cha khabar?")

        # Typing action sent for both parts
        self.assertEqual(mock_context.bot.send_chat_action.await_count, 2)

        # Sleep called for both parts
        self.assertEqual(mock_sleep.await_count, 2)

        # Two messages saved in messages table
        recent = bot.get_recent_messages(user_id)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["content"], "good morning!")
        self.assertEqual(recent[1]["content"], "k cha khabar?")

    def test_compute_reply_delay_bounds(self):
        """Verify compute_reply_delay always returns a value between 0.8 and 6.0."""
        for _ in range(100):
            delay = bot.compute_reply_delay("short", "hi")
            self.assertGreaterEqual(delay, 0.8)
            self.assertLessEqual(delay, 6.0)

        for _ in range(100):
            delay = bot.compute_reply_delay("a" * 500, "b" * 500)
            self.assertGreaterEqual(delay, 0.8)
            self.assertLessEqual(delay, 6.0)

        # Empty inputs should still produce a valid delay
        delay = bot.compute_reply_delay("", "")
        self.assertGreaterEqual(delay, 0.8)
        self.assertLessEqual(delay, 6.0)


if __name__ == "__main__":
    unittest.main()
