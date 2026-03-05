"""X (Twitter) publisher for Eclipse Scalper notifications.

Disabled by default (X_TWITTER_ENABLED=0).
Set X_TWITTER_ENABLED=1 and all four OAuth 1.0a credentials to enable.

Never raises — all errors are caught and logged internally.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(value: str) -> bool:
    """Parse env-style boolean strings: 1/true/True/yes/YES → True."""
    return value.strip().lower() in _TRUTHY


class XTweetPublisher:
    """Publishes short notifications to X (Twitter) via tweepy v2 Client.

    Usage::

        pub = XTweetPublisher()
        if pub.can_publish():
            pub.publish("Signal fired: BTCUSDT long @ 65000")

    All credential handling is via environment variables — see can_publish().
    """

    MAX_CHARS: int = 280
    _ELLIPSIS: str = "…"  # single Unicode char = 1 char towards limit

    def __init__(self) -> None:
        self._enabled: bool = _is_truthy(os.getenv("X_TWITTER_ENABLED", "0"))
        self._consumer_key: Optional[str] = os.getenv("X_CONSUMER_KEY") or None
        self._consumer_secret: Optional[str] = os.getenv("X_CONSUMER_SECRET") or None
        self._access_token: Optional[str] = os.getenv("X_ACCESS_TOKEN") or None
        self._access_token_secret: Optional[str] = os.getenv("X_ACCESS_TOKEN_SECRET") or None
        self._cooldown_sec: float = float(os.getenv("X_TWITTER_COOLDOWN_SEC", "30"))
        self._last_posted: float = 0.0
        self._client = None  # lazy-initialised on first publish

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def can_publish(self) -> bool:
        """Return True only if enabled AND all four credentials are present."""
        if not self._enabled:
            return False
        return all([
            self._consumer_key,
            self._consumer_secret,
            self._access_token,
            self._access_token_secret,
        ])

    def publish(self, text: str) -> bool:
        """Post text as a tweet.

        Returns:
            True  — tweet sent successfully.
            False — disabled, missing creds, cooldown active, or any error.

        Never raises.
        """
        if not self.can_publish():
            return False

        now = time.monotonic()
        remaining = self._cooldown_sec - (now - self._last_posted)
        if remaining > 0:
            logger.debug("X publisher: cooldown active (%.1fs remaining)", remaining)
            return False

        tweet_text = self._truncate(text)
        try:
            client = self._get_client()
            client.create_tweet(text=tweet_text)
            self._last_posted = time.monotonic()
            logger.info("X publisher: tweet posted (%d chars)", len(tweet_text))
            return True
        except Exception as exc:
            logger.warning("X publisher: failed to post tweet: %s", exc)
            return False

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _get_client(self):
        """Lazy-init tweepy Client (import deferred so tweepy is optional)."""
        if self._client is None:
            import tweepy  # noqa: PLC0415
            self._client = tweepy.Client(
                consumer_key=self._consumer_key,
                consumer_secret=self._consumer_secret,
                access_token=self._access_token,
                access_token_secret=self._access_token_secret,
            )
        return self._client

    def _truncate(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self.MAX_CHARS:
            return text
        return text[: self.MAX_CHARS - len(self._ELLIPSIS)] + self._ELLIPSIS
