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
        self.assertEqual(reply, "k gardai xau?")

        # Check DB messages saved
        messages = bot.get_recent_messages(user_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "hello")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "k gardai xau?")

    def test_gemini_client_configuration(self):
        saved_key = bot.OPENAI_API_KEY
        saved_client = bot.client
        try:
            bot.OPENAI_API_KEY = "test-gemini-key"
            bot.client = None
            client = bot.get_openai_client()
            self.assertEqual(str(client.base_url), "https://generativelanguage.googleapis.com/v1beta/openai/")
            self.assertEqual(bot.MODEL_NAME, "gemini-3.6-flash")
        finally:
            bot.OPENAI_API_KEY = saved_key
            bot.client = saved_client


if __name__ == "__main__":
    unittest.main()
