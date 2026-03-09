from html import escape as _html_escape
import logging

from telegram import Bot
import time

_log = logging.getLogger("notifications.telegram")

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
            safe_text = _html_escape(str(text or ""), quote=False)
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"<pre>{safe_text}</pre>",
                parse_mode='HTML',
                disable_notification=bool(silent),
            )
            if priority == 'critical':
                self.last_alert = now
            sent_ok = True
        except Exception as e:
            _log.error("Telegram failed: %s", e)

        # X/Twitter - optional, fire-and-forget, errors never propagate
        try:
            self._x.publish(text)
        except Exception as exc:
            _log.error("X publisher error: %s", exc)
        return sent_ok
