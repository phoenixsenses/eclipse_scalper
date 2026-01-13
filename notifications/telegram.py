from telegram import Bot
import asyncio
import time

class Notifier:
    def __init__(self, token, chat_id):
        self.bot = Bot(token) if token else None
        self.chat_id = chat_id
        self.last_alert = 0.0

    async def speak(self, text: str, priority: str = 'critical'):
        if not self.bot or not self.chat_id:
            return
        now = time.time()
        if priority == 'normal' and now - self.last_alert < 15:
            return
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"<pre>{text}</pre>",
                parse_mode='HTML'
            )
            if priority == 'critical':
                self.last_alert = now
        except Exception as e:
            print(f"Telegram failed: {e}")