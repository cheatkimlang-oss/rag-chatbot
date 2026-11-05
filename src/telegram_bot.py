"""Run the RUPP student assistant as a Telegram bot."""
import argparse
import asyncio
from getpass import getpass
import logging
import os
import re
import sys
from time import monotonic

import portalocker

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update, __version__
from telegram.constants import ChatAction, ChatType
from telegram.error import Conflict, InvalidToken, TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from answer import DEFAULT_MODEL, EXAMPLE_QUESTIONS, build_answer
from curriculum import load_curriculum
from hybrid import load_hybrid
from retrieve import COLLECTIONS, PROJECT_DIR, load_passages
from student_followups import resolve_followup


POLLING_LOCK = PROJECT_DIR / "index" / "telegram-polling.lock"


def question_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(question, callback_data=f"question:{index}")]
        for index, question in enumerate(EXAMPLE_QUESTIONS)
    ])


def message_parts(text, limit=1800):
    """Stay below Telegram's 4096-character limit, including UTF-16 pairs."""
    while len(text) > limit:
        end = text.rfind("\n", 0, limit + 1)
        if end <= 0:
            end = text.rfind(" ", 0, limit + 1)
        if end <= 0:
            end = limit
        yield text[:end]
        text = text[end:].lstrip("\n ")
    if text:
        yield text


def conversation_state(update, context):
    user = update.effective_user
    if user is None or user.is_bot or context.user_data is None:
        return {}
    chat = update.effective_chat
    if chat.type == ChatType.PRIVATE:
        return context.user_data
    # Keep a student's group follow-ups separate from their private chat.
    return context.user_data.setdefault("groups", {}).setdefault(chat.id, {})


class StudentBot:
    def __init__(self, rows, snapshot, search, model):
        self.rows = rows
        self.snapshot = snapshot
        self.search = search
        self.model = model
        self.last_conflict_warning = float("-inf")

    async def show_questions(self, update, context):
        if update.effective_chat.type == ChatType.PRIVATE:
            instructions = "Tap a question below or type your own. Use /examples to see the suggestions again."
        else:
            username = context.bot.username
            instructions = (
                f"Tap a question below or type /ask@{username} followed by your question. "
                f"Use /examples@{username} to see these suggestions again."
            )
        await update.effective_message.reply_text(
            f"Welcome to the RUPP student assistant. {instructions}",
            reply_markup=question_keyboard(),
        )

    async def answer(self, message, question, state):
        question = question.strip()
        if not question or len(question) > 500:
            await message.reply_text("Please type a question of up to 500 characters.")
            return
        reply = resolve_followup(question, self.snapshot, state)
        if reply is None:
            await message.reply_chat_action(ChatAction.TYPING)
            reply = await asyncio.to_thread(
                build_answer, question, self.rows, self.model, self.snapshot,
                official=True, search=self.search,
            )
        for part in message_parts(reply):
            await message.reply_text(part, disable_web_page_preview=True)

    async def on_message(self, update, context):
        message = update.effective_message
        await self.answer(message, message.text, conversation_state(update, context))

    async def on_ask(self, update, context):
        message = update.effective_message
        question = " ".join(context.args)
        if not question:
            command = "/ask" if update.effective_chat.type == ChatType.PRIVATE else f"/ask@{context.bot.username}"
            await message.reply_text(f"Add your question after the command. For example:\n{command} How many years for DSE?")
            return
        await self.answer(message, question, conversation_state(update, context))

    async def on_button(self, update, context):
        callback = update.callback_query
        message = callback.message
        if not isinstance(message, Message) or message.chat.type not in (ChatType.PRIVATE, ChatType.GROUP, ChatType.SUPERGROUP):
            await callback.answer()
            return
        index = int(callback.data.split(":", 1)[1])
        if index >= len(EXAMPLE_QUESTIONS):
            await callback.answer("Use /examples for the current list.", show_alert=True)
            return
        await callback.answer()
        await self.answer(message, EXAMPLE_QUESTIONS[index], conversation_state(update, context))

    def application(self, token):
        app = Application.builder().token(token).post_init(announce_connection).build()
        private = filters.ChatType.PRIVATE
        student_chats = private | filters.ChatType.GROUPS
        app.add_handler(CommandHandler(["start", "help", "examples"], self.show_questions, filters=student_chats))
        app.add_handler(CommandHandler("ask", self.on_ask, filters=student_chats))
        app.add_handler(CallbackQueryHandler(self.on_button, pattern=r"^question:[0-9]$"))
        app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, self.on_message))
        app.add_error_handler(self.on_error)
        return app

    async def on_error(self, update, context):
        # Request exceptions can contain the bot token in their URL.
        if isinstance(context.error, Conflict):
            now = monotonic()
            if now - self.last_conflict_warning >= 30:
                logging.warning(
                    "Telegram polling conflict. Keep just one running copy of this bot, including on other devices or hosts."
                )
                self.last_conflict_warning = now
            return
        logging.warning("Telegram update failed (%s).", type(context.error).__name__)


def valid_token_shape(token):
    return re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token) is not None


def configure_logging(token):
    class RedactingFormatter(logging.Formatter):
        def format(self, record):
            return super().format(record).replace(token, "[redacted]")

    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter("%(levelname)s: %(message)s"))
    handler.addFilter(
        lambda record: not (record.exc_info and isinstance(record.exc_info[1], InvalidToken))
    )
    logging.basicConfig(handlers=[handler], level=logging.WARNING, force=True)


async def announce_connection(application):
    print(f"Connected as @{application.bot.username}. Send /start in Telegram.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check local data without contacting Telegram")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        rows = load_passages(COLLECTIONS["official"])
        snapshot = load_curriculum()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if snapshot is None:
        parser.error("Refresh the official curriculum before starting Telegram chat.")

    search = load_hybrid(rows)
    if args.check:
        print(
            f"python-telegram-bot {__version__} | {len(rows)} official passages | "
            f"{len(EXAMPLE_QUESTIONS)} buttons | {'hybrid' if search else 'keyword'} search ready."
        )
        return

    bot = StudentBot(rows, snapshot, search, args.model)
    lock = portalocker.Lock(POLLING_LOCK, mode="a+b", timeout=0, fail_when_locked=True)
    try:
        lock.acquire()
    except portalocker.exceptions.AlreadyLocked:
        parser.exit(1, "Another copy of the Telegram bot is already open. Keep using its PowerShell window.\n")
    except (OSError, portalocker.exceptions.LockException):
        parser.exit(1, "Could not lock the Telegram bot's local index folder. Check that the folder is writable.\n")
    try:
        run_bot(bot, parser)
    finally:
        lock.release()


def run_bot(bot, parser):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if token and not valid_token_shape(token):
        parser.error("TELEGRAM_BOT_TOKEN must contain exactly one BotFather token.")
    if not token and sys.stdin.isatty():
        try:
            while not token:
                pasted = getpass("Paste your BotFather token (hidden): ").strip()
                if valid_token_shape(pasted):
                    token = pasted
                else:
                    print("Paste only the token line, once. Then press Enter.")
        except (KeyboardInterrupt, EOFError):
            print("\nToken entry canceled. Run the command again when your token is ready.")
            return
    if not token:
        parser.error("Set TELEGRAM_BOT_TOKEN or run this in an interactive terminal to enter it.")

    configure_logging(token)
    print("Connecting to Telegram...", flush=True)
    try:
        bot.application(token).run_polling(allowed_updates=[Update.MESSAGE, Update.CALLBACK_QUERY])
    except InvalidToken:
        parser.exit(1, "Telegram rejected the token. Run again and paste the newest token from BotFather.\n")
    except TelegramError as exc:
        parser.exit(1, f"Could not start Telegram polling ({type(exc).__name__}). Check the connection.\n")


if __name__ == "__main__":
    main()
