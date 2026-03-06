from telegram import Bot
import time

from notifications.x_twitter import XTweetPublisher


class Notifier:
    def __init__(self, token, chat_id):
        self.bot = Bot(token) if token else None
        self.chat_id = chat_id
        self.last_alert = 0.0
        self._x = XTweetPublisher()

    async def speak(self, text: str, priority: str = 'critical', silent: bool = False) -> bool:
        if not self.bot or not self.chat_id:
            return False
        now = time.time()
        if priority == 'normal' and now - self.last_alert < 15:
            return False
        sent_ok = False
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"<pre>{text}</pre>",
                parse_mode='HTML',
                disable_notification=bool(silent),
            )
            if priority == 'critical':
                self.last_alert = now
            sent_ok = True
        except Exception as e:
            print(f"Telegram failed: {e}")

        # X/Twitter - optional, fire-and-forget, errors never propagate
        try:
            self._x.publish(text)
        except Exception as exc:
            print(f"X publisher error: {exc}")
        return sent_ok
