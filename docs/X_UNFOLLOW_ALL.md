# X Unfollow All

Local Playwright tool for unfollowing accounts from your X Following list.

Install once:

```powershell
python -m pip install playwright
python -m playwright install chromium
```

Dry run first:

```powershell
python tools\x_unfollow_all.py --username YOUR_USERNAME --dry-run --max 5
```

Run for real:

```powershell
python tools\x_unfollow_all.py --username YOUR_USERNAME
```

Faster, with the typed confirmation skipped:

```powershell
python tools\x_unfollow_all.py --username YOUR_USERNAME --yes
```

Use your normal logged-in Chrome profile:

```powershell
python tools\x_unfollow_all.py --username YOUR_USERNAME --yes --normal-chrome-profile
```

If your logged-in Chrome profile is not `Default`, pass the Chrome profile folder name:

```powershell
python tools\x_unfollow_all.py --username YOUR_USERNAME --yes --normal-chrome-profile --chrome-profile-directory "Profile 1"
```

Notes:

- Use your X username without `@`.
- A separate browser profile opens and keeps its session under `runtime/x_unfollow_browser`.
- Your normal Chrome login is not reused. If you are not logged in inside the opened browser, log in manually there, then press Enter in the terminal.
- Add `--system-chrome` if you want the opened browser to be installed Chrome instead of bundled Playwright Chromium. It still uses the separate `runtime/x_unfollow_browser` profile.
- Add `--normal-chrome-profile` to use your existing Chrome login. Close every Chrome window first, otherwise Chrome may reject the profile because it is already in use.
- The tool writes a CSV log to `reports/x_unfollow_log.csv`.
- Stop anytime with `Ctrl+C`.
- X can rate-limit or restrict accounts for aggressive bulk actions. The default delay is 2-5 seconds between unfollows; increase it with `--delay-min` and `--delay-max` if needed.
