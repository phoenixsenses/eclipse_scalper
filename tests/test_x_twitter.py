"""Tests for notifications/x_twitter.py

Run with: pytest tests/test_x_twitter.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────
# Helper — build a publisher with controlled env
# ─────────────────────────────────────────────

def _make_publisher(
    enabled: str = "1",
    consumer_key: str = "fake_ck",
    consumer_secret: str = "fake_cs",
    access_token: str = "fake_at",
    access_token_secret: str = "fake_ats",
    cooldown: str = "0",
):
    """Instantiate XTweetPublisher with patched env vars."""
    env = {
        "X_TWITTER_ENABLED": enabled,
        "X_CONSUMER_KEY": consumer_key,
        "X_CONSUMER_SECRET": consumer_secret,
        "X_ACCESS_TOKEN": access_token,
        "X_ACCESS_TOKEN_SECRET": access_token_secret,
        "X_TWITTER_COOLDOWN_SEC": cooldown,
    }
    with patch.dict("os.environ", env):
        from notifications.x_twitter import XTweetPublisher
        return XTweetPublisher()


# ─────────────────────────────────────────────
# a) Disabled → publish returns False
# ─────────────────────────────────────────────

def test_disabled_can_publish_false():
    pub = _make_publisher(enabled="0")
    assert pub.can_publish() is False


def test_disabled_publish_returns_false():
    pub = _make_publisher(enabled="0")
    assert pub.publish("hello") is False


def test_disabled_does_not_call_tweepy():
    pub = _make_publisher(enabled="0")
    mock_client = MagicMock()
    pub._client = mock_client
    pub.publish("hello")
    mock_client.create_tweet.assert_not_called()


# ─────────────────────────────────────────────
# b) Enabled but missing credentials → False
# ─────────────────────────────────────────────

def test_missing_consumer_key_returns_false():
    pub = _make_publisher(consumer_key="")
    assert pub.can_publish() is False
    assert pub.publish("hello") is False


def test_missing_consumer_secret_returns_false():
    pub = _make_publisher(consumer_secret="")
    assert pub.can_publish() is False
    assert pub.publish("hello") is False


def test_missing_access_token_returns_false():
    pub = _make_publisher(access_token="")
    assert pub.can_publish() is False
    assert pub.publish("hello") is False


def test_missing_access_token_secret_returns_false():
    pub = _make_publisher(access_token_secret="")
    assert pub.can_publish() is False
    assert pub.publish("hello") is False


# ─────────────────────────────────────────────
# c) Enabled with valid creds → True + create_tweet called
# ─────────────────────────────────────────────

def test_enabled_with_creds_returns_true():
    pub = _make_publisher(cooldown="0")
    pub._client = MagicMock()
    assert pub.publish("signal fired") is True


def test_enabled_calls_create_tweet_with_text():
    pub = _make_publisher(cooldown="0")
    mock_client = MagicMock()
    pub._client = mock_client
    pub.publish("BTCUSDT long signal @ 65000")
    mock_client.create_tweet.assert_called_once_with(text="BTCUSDT long signal @ 65000")


def test_can_publish_true_when_all_creds_present():
    pub = _make_publisher()
    assert pub.can_publish() is True


# ─────────────────────────────────────────────
# d) >280 chars → truncated to exactly 280
# ─────────────────────────────────────────────

def test_long_text_truncated_to_280():
    pub = _make_publisher(cooldown="0")
    mock_client = MagicMock()
    pub._client = mock_client

    long_text = "X" * 300
    pub.publish(long_text)

    sent = mock_client.create_tweet.call_args.kwargs["text"]
    assert len(sent) == 280


def test_long_text_ends_with_ellipsis():
    pub = _make_publisher(cooldown="0")
    mock_client = MagicMock()
    pub._client = mock_client

    pub.publish("A" * 300)
    sent = mock_client.create_tweet.call_args.kwargs["text"]
    assert sent.endswith("…")


def test_exact_280_chars_not_truncated():
    pub = _make_publisher(cooldown="0")
    mock_client = MagicMock()
    pub._client = mock_client

    text = "B" * 280
    pub.publish(text)
    sent = mock_client.create_tweet.call_args.kwargs["text"]
    assert sent == text
    assert not sent.endswith("…")


def test_short_text_unchanged():
    pub = _make_publisher(cooldown="0")
    mock_client = MagicMock()
    pub._client = mock_client

    pub.publish("short signal")
    sent = mock_client.create_tweet.call_args.kwargs["text"]
    assert sent == "short signal"


# ─────────────────────────────────────────────
# e) Exceptions from tweepy → swallowed, no crash
# ─────────────────────────────────────────────

def test_tweepy_runtime_error_returns_false():
    pub = _make_publisher(cooldown="0")
    mock_client = MagicMock()
    mock_client.create_tweet.side_effect = RuntimeError("network error")
    pub._client = mock_client

    result = pub.publish("some signal")
    assert result is False  # swallowed — trading loop safe


def test_tweepy_exception_no_crash():
    pub = _make_publisher(cooldown="0")
    mock_client = MagicMock()
    mock_client.create_tweet.side_effect = Exception("unexpected")
    pub._client = mock_client

    # Must not raise
    try:
        pub.publish("test")
    except Exception:
        pytest.fail("publish() raised — trading loop would have crashed")


def test_tweepy_exception_does_not_update_last_posted():
    pub = _make_publisher(cooldown="0")
    mock_client = MagicMock()
    mock_client.create_tweet.side_effect = RuntimeError("fail")
    pub._client = mock_client

    before = pub._last_posted
    pub.publish("test")
    # _last_posted must not be updated on failure
    assert pub._last_posted == before


# ─────────────────────────────────────────────
# Cooldown guard
# ─────────────────────────────────────────────

def test_cooldown_blocks_second_call():
    pub = _make_publisher(cooldown="60")
    mock_client = MagicMock()
    pub._client = mock_client

    first = pub.publish("first tweet")
    second = pub.publish("second tweet")

    assert first is True
    assert second is False
    assert mock_client.create_tweet.call_count == 1


def test_zero_cooldown_allows_immediate_repeat():
    pub = _make_publisher(cooldown="0")
    mock_client = MagicMock()
    pub._client = mock_client

    assert pub.publish("tweet 1") is True
    assert pub.publish("tweet 2") is True
    assert mock_client.create_tweet.call_count == 2


# ─────────────────────────────────────────────
# _is_truthy helper
# ─────────────────────────────────────────────

def test_truthy_strings_accepted():
    from notifications.x_twitter import _is_truthy
    for val in ("1", "true", "True", "TRUE", "yes", "YES", "on", "ON"):
        assert _is_truthy(val) is True, f"Expected {val!r} to be truthy"


def test_falsy_strings_rejected():
    from notifications.x_twitter import _is_truthy
    for val in ("0", "false", "False", "no", "NO", "", "off", "OFF", "disabled"):
        assert _is_truthy(val) is False, f"Expected {val!r} to be falsy"
