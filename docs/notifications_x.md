# X (Twitter) Notifications — Setup Guide

Eclipse Scalper can optionally post a tweet whenever a critical signal or event is emitted.
This feature is **disabled by default** and requires an X Developer account.

---

## 1. Create an X Developer App

1. Go to [developer.twitter.com/en/portal/dashboard](https://developer.twitter.com/en/portal/dashboard)
2. Click **"+ Add App"** (or create a new Project → App)
3. Choose a name (e.g. `eclipse-scalper-alerts`)
4. Select **"App Permissions"** → set to **Read and Write**
   - ⚠️ Must be **Read + Write**, not Read-only — posting requires write access

---

## 2. Generate OAuth 1.0a Keys

In your App dashboard → **"Keys and Tokens"**:

| Key | Where to find |
|-----|--------------|
| `Consumer Key` | "Consumer Keys" section → **API Key** |
| `Consumer Secret` | "Consumer Keys" section → **API Key Secret** |
| `Access Token` | "Authentication Tokens" section → **Access Token** |
| `Access Token Secret` | "Authentication Tokens" section → **Access Token Secret** |

> Make sure the Access Token is generated **after** you set Read+Write permissions.
> If you generated the token before changing permissions, regenerate it.

---

## 3. Set Environment Variables

Add these to your `.env` file (already in `.gitignore`):

```env
# X (Twitter) Notifications
X_TWITTER_ENABLED=1
X_CONSUMER_KEY=your_consumer_key_here
X_CONSUMER_SECRET=your_consumer_secret_here
X_ACCESS_TOKEN=your_access_token_here
X_ACCESS_TOKEN_SECRET=your_access_token_secret_here

# Optional: minimum seconds between tweets (default: 30)
X_TWITTER_COOLDOWN_SEC=30
```

Set `X_TWITTER_ENABLED=0` (or omit entirely) to disable without removing credentials.

---

## 4. Install tweepy

```bash
pip install "tweepy>=4.14.0"
```

---

## 5. How It Works

- `XTweetPublisher` is created inside `Notifier.__init__()` in `notifications/telegram.py`
- Every `Notifier.speak()` call also fires `XTweetPublisher.publish(text)` after Telegram
- If X fails for any reason (network, rate limit, wrong creds), the error is **logged and swallowed** — the trading loop is never affected
- Text longer than 280 characters is **truncated with `…`** automatically
- A **cooldown guard** (default 30 s) prevents tweet spam if many signals fire in quick succession

---

## 6. Sample Tweet Format

Tweets are the raw `text` passed to `Notifier.speak()`. Example output:

```
⚡ BTCUSDT LONG signal fired
Score: 0.73 | Regime: trending_up | Conf: 0.81
NPA: +0.000142 | Fill rate: 68%
```

Keep messages under 280 chars for best results; longer messages are auto-truncated.

---

## 7. Truthy Values for `X_TWITTER_ENABLED`

Any of these values enable posting: `1`, `true`, `True`, `yes`, `YES`, `on`

Any of these disable it: `0`, `false`, `no`, `off` (or missing entirely)

---

## ⚠️ Security Warnings

- **Never commit API keys** to git. The `.env` file is listed in `.gitignore` — keep it that way.
- **Never paste keys into code, logs, or chat messages.**
- If a key is accidentally exposed, immediately **Regenerate** it on the X developer portal.
- Rotate keys periodically as good practice.
- The dashboard Settings page masks all values containing `KEY`, `SECRET`, or `TOKEN`.

---

## 8. Testing Without Live Tweets

```python
# Run unit tests — no network calls, tweepy is mocked
pytest tests/test_x_twitter.py -v
```

To manually verify credentials without posting publicly, use the
[X API v2 Sandbox](https://developer.twitter.com/en/portal/products/free) if available on your plan,
or temporarily set `X_TWITTER_COOLDOWN_SEC=9999` to block real posts during testing.
