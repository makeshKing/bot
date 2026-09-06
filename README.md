# Sristi - AI companion Telegram bot (MVP)

## What's in here
- `persona.py` - Sristi's personality and style, as a prompt template
- `bot.py` - the bot itself: Telegram handling, memory, OpenAI calls
- `requirements.txt` - Python dependencies
- `.env.example` - copy this to `.env` and fill in your real keys

## Setup

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Get a Google AI Studio API key**
   - Sign up at [aistudio.google.com](https://aistudio.google.com) and create an API key. This calls Gemma 4 (`gemma-4-31b-it`) directly through Google's own infrastructure rather than through OpenRouter.

3. **Create a Telegram bot**
   - Message @BotFather on Telegram
   - Send `/newbot`, follow the prompts, copy the token it gives you

4. **Set up your environment**
   ```
   cp .env.example .env
   ```
   Then open `.env` and paste in your real `GEMINI_API_KEY` and `TELEGRAM_BOT_TOKEN`.

5. **Run it**
   ```
   python bot.py
   ```
   Open Telegram, find your bot, send `/start`, and start chatting.

   *(Optional)* Test the bot directly in your terminal without Telegram:
   ```
   python bot.py --cli
   ```

## How the memory works
- Every message is saved to a local SQLite file (`sristi.db`).
- The last 12 messages are always sent to the model as context.
- Once a user's history passes 30 messages, the older messages get
  summarized into a short "memory summary" by the model itself, and
  the raw old messages are deleted to keep things efficient. This is
  what lets Sristi "remember" things from days ago without you paying
  to resend the entire chat history every single time.

## Things to change before showing this to real users
- **Persona**: edit `persona.py` to adjust her personality, backstory, or style.
- **Model**: Gemma 4 is served via Google's OpenAI-compatible endpoint using the model ID `gemma-4-31b-it`. If you wish to use other Gemini models (e.g., `gemini-2.5-flash`), you can swap `MODEL_NAME` in `.env`.
- **Safety**: this MVP has no content moderation layer. Before any public
  launch, add a moderation check (either OpenAI's moderation endpoint or a
  custom filter) on both user input and model output.
- **Hosting**: this runs on your machine via polling. To keep it running
  24/7, deploy it to a small server (Railway, Render, or a basic VPS) instead.
- **Database**: SQLite is fine for testing with a handful of users. If you
  get real traction, move to Postgres so multiple server instances can share
  the same data.
