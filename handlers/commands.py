from telegram import Update
from telegram.ext import ContextTypes


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
	"""Handle /start command — greet the user."""
	if not update.message:
		return

	engine = context.application.bot_data["config"].engine
	await update.message.reply_text(
		f"Send me a voice message and I will transcribe it.\n"
		f"Engine: {engine}"
	)

