#!/usr/bin/env python
"""Unfollow everyone from an X account through a local browser session.

This uses Playwright against the regular X web UI. You log in manually in the
opened browser; the script then walks your Following page and clicks
Following -> Unfollow with conservative delays.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from datetime import datetime, timezone
from os import environ
from pathlib import Path


FOLLOWING_TEXTS = (
    "Following",
    "Takip ediliyor",
)
UNFOLLOW_TEXTS = (
    "Unfollow",
    "Takibi b\u0131rak",
    "Takipten \u00e7\u0131k",
)
PROFILE_LINK_TESTID = "AppTabBar_Profile_Link"
LOGIN_HINTS = (
    "Log in",
    "Sign in",
    "Giris yap",
    "Giri\u015f yap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open X in a browser and unfollow accounts from your Following list."
    )
    parser.add_argument(
        "--username",
        help="Your X username without @. If omitted, the script tries to detect it after login.",
    )
    parser.add_argument(
        "--profile-dir",
        default="runtime/x_unfollow_browser",
        help="Persistent browser profile directory. Default: runtime/x_unfollow_browser",
    )
    parser.add_argument(
        "--log",
        default="reports/x_unfollow_log.csv",
        help="CSV log path. Default: reports/x_unfollow_log.csv",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Maximum accounts to unfollow. 0 means no explicit cap.",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=2.0,
        help="Minimum delay between actions in seconds. Default: 2.0",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=5.0,
        help="Maximum delay between actions in seconds. Default: 5.0",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless. Not recommended because manual login is usually needed.",
    )
    parser.add_argument(
        "--system-chrome",
        action="store_true",
        help="Use installed Chrome instead of bundled Playwright Chromium.",
    )
    parser.add_argument(
        "--normal-chrome-profile",
        action="store_true",
        help="Use your normal Chrome user data directory. Close all Chrome windows first.",
    )
    parser.add_argument(
        "--chrome-profile-directory",
        default="Default",
        help="Chrome profile directory to use with --normal-chrome-profile. Default: Default",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the final typed confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Open and scan, but do not click unfollow buttons.",
    )
    parser.add_argument(
        "--cdp-url",
        help="Attach to an already-running Chrome remote debugging URL, e.g. http://127.0.0.1:9222",
    )
    parser.add_argument(
        "--login-wait-seconds",
        type=int,
        default=300,
        help="How long to wait for manual login when stdin is not interactive. Default: 300",
    )
    return parser.parse_args()


def die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def import_playwright():
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError:
        die(
            "Playwright kurulu degil.\n"
            "Kurulum:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chromium"
        )
    return sync_playwright, PlaywrightTimeoutError


def normalize_username(username: str | None) -> str | None:
    if not username:
        return None
    username = username.strip().lstrip("@")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,15}", username):
        die(f"Gecersiz X username: {username!r}")
    return username


def wait_for_manual_login(page, wait_seconds: int) -> None:
    print("X aciliyor. Gerekirse tarayicida manuel giris yap.")
    force_goto(page, "https://x.com/home")
    deadline = time.monotonic() + max(1, wait_seconds)
    while True:
        if is_logged_in(page):
            print("Login dogrulandi.")
            return
        print("Bu Playwright profili normal Chrome'dan ayridir; X'e acilan pencerede giris yap.")
        try:
            input("Giris tamamlaninca Enter'a bas...")
        except EOFError:
            if time.monotonic() >= deadline:
                die("Login bekleme suresi doldu. Acilan tarayicida X'e giris yapilamadi.")
            page.wait_for_timeout(5_000)
        force_goto(page, "https://x.com/home")


def force_goto(page, url: str) -> None:
    print(f"Sayfa aciliyor: {url}")
    page.bring_to_front()
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(2_000)
    print(f"Aktif URL: {page.url}")


def is_logged_in(page) -> bool:
    profile_selectors = [
        f'a[data-testid="{PROFILE_LINK_TESTID}"]',
        'a[aria-label="Profile"]',
        'a[aria-label="Profil"]',
        'a[href="/compose/post"]',
        'a[data-testid="SideNav_NewTweet_Button"]',
    ]
    for selector in profile_selectors:
        try:
            if page.locator(selector).first.is_visible(timeout=2_000):
                return True
        except Exception:
            pass
    for text in LOGIN_HINTS:
        try:
            if page.locator(f'text="{text}"').first.is_visible(timeout=500):
                return False
        except Exception:
            pass
    return False


def detect_username(page) -> str:
    selectors = [
        f'a[data-testid="{PROFILE_LINK_TESTID}"]',
        'a[aria-label="Profile"]',
        'a[aria-label="Profil"]',
    ]
    for selector in selectors:
        try:
            href = page.locator(selector).first.get_attribute("href", timeout=5_000)
        except Exception:
            href = None
        if href:
            username = href.rstrip("/").split("/")[-1]
            username = normalize_username(username)
            if username:
                print(f"Kullanici adi algilandi: @{username}")
                return username
    die("Username otomatik algilanamadi. Komutu --username SENIN_KULLANICIN ile calistir.")


def text_button_selector(texts: tuple[str, ...]) -> str:
    parts = []
    for text in texts:
        escaped = text.replace('"', '\\"')
        parts.append(f'button:has-text("{escaped}")')
    return ", ".join(parts)


def following_button_selector() -> str:
    text_selector = text_button_selector(FOLLOWING_TEXTS)
    return (
        'button[data-testid$="-unfollow"], '
        f"{text_selector}, "
        'button[aria-label^="Following @"], '
        'button[aria-label*="Takip ediliyor"]'
    )


def confirm_button_selector() -> str:
    text_selector = text_button_selector(UNFOLLOW_TEXTS)
    return f'button[data-testid="confirmationSheetConfirm"], {text_selector}'


def best_effort_handle_from_button(button) -> str:
    try:
        article = button.locator("xpath=ancestor::article[1]")
        links = article.locator('a[href^="/"][role="link"]')
        for index in range(min(links.count(), 8)):
            href = links.nth(index).get_attribute("href") or ""
            candidate = href.strip("/").split("/")[0]
            if re.fullmatch(r"[A-Za-z0-9_]{1,15}", candidate):
                return f"@{candidate}"
    except Exception:
        pass
    try:
        label = button.get_attribute("aria-label") or ""
        match = re.search(r"@([A-Za-z0-9_]{1,15})", label)
        if match:
            return f"@{match.group(1)}"
    except Exception:
        pass
    return ""


def append_log(log_path: Path, handle: str, status: str, detail: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(["ts_utc", "handle", "status", "detail"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), handle, status, detail])


def confirm_destructive_action(args: argparse.Namespace) -> None:
    if args.dry_run or args.yes:
        return
    print("Bu islem takip ettigin hesaplari geri alinmaz sekilde takipten cikarir.")
    print("Devam etmek icin aynen sunu yaz: UNFOLLOW ALL")
    if input("> ").strip() != "UNFOLLOW ALL":
        die("Iptal edildi.", code=2)


def sleep_between(args: argparse.Namespace) -> None:
    low = max(0.0, args.delay_min)
    high = max(low, args.delay_max)
    time.sleep(random.uniform(low, high))


def default_chrome_user_data_dir() -> Path:
    local_app_data = environ.get("LOCALAPPDATA")
    if not local_app_data:
        die("LOCALAPPDATA bulunamadi; --profile-dir ile profil dizini ver.")
    return Path(local_app_data) / "Google" / "Chrome" / "User Data"


def click_unfollow(page, button, args: argparse.Namespace, timeout_error) -> tuple[bool, str]:
    handle = best_effort_handle_from_button(button)
    if args.dry_run:
        return True, handle

    button.scroll_into_view_if_needed(timeout=10_000)
    button.click(timeout=10_000)
    confirm = page.locator(confirm_button_selector()).last
    try:
        confirm.wait_for(state="visible", timeout=7_000)
        confirm.click(timeout=10_000)
        return True, handle
    except timeout_error:
        page.keyboard.press("Escape")
        return False, handle


def run() -> int:
    args = parse_args()
    args.username = normalize_username(args.username)
    if args.delay_max < args.delay_min:
        die("--delay-max, --delay-min degerinden kucuk olamaz.")

    confirm_destructive_action(args)
    sync_playwright, timeout_error = import_playwright()
    profile_dir = default_chrome_user_data_dir() if args.normal_chrome_profile else Path(args.profile_dir)
    log_path = Path(args.log)

    with sync_playwright() as p:
        if args.cdp_url:
            browser = p.chromium.connect_over_cdp(args.cdp_url)
            context = browser.contexts[0] if browser.contexts else browser.new_context()
        else:
            launch_kwargs = {
                "user_data_dir": str(profile_dir),
                "headless": args.headless,
                "viewport": {"width": 1280, "height": 900},
                "slow_mo": 50,
            }
            if args.system_chrome or args.normal_chrome_profile:
                launch_kwargs["channel"] = "chrome"
            if args.system_chrome or args.normal_chrome_profile:
                launch_kwargs["args"] = [
                    f"--profile-directory={args.chrome_profile_directory}",
                ]
            context = p.chromium.launch_persistent_context(**launch_kwargs)
            browser = context

        page = context.new_page()
        page.bring_to_front()

        try:
            wait_for_manual_login(page, args.login_wait_seconds)
            username = args.username or detect_username(page)
            force_goto(page, f"https://x.com/{username}/following")
            page.wait_for_timeout(5_000)
            if not is_logged_in(page):
                die("Following sayfasinda login algilanamadi. Acilan tarayicida X'e giris yapip tekrar calistir.")

            unfollowed = 0
            failed = 0
            idle_rounds = 0
            seen_scroll_y = -1

            print("Following listesi taraniyor. Durdurmak icin Ctrl+C.")
            while True:
                if args.max and unfollowed >= args.max:
                    print(f"Limit doldu: {unfollowed}/{args.max}")
                    break

                buttons = page.locator(following_button_selector())
                count = buttons.count()
                if count > 0:
                    idle_rounds = 0
                    ok, handle = click_unfollow(page, buttons.first, args, timeout_error)
                    if ok:
                        unfollowed += 1
                        status = "dry_run_found" if args.dry_run else "unfollowed"
                        append_log(log_path, handle, status)
                        print(f"{unfollowed}: {status} {handle}".rstrip())
                    else:
                        failed += 1
                        append_log(log_path, handle, "failed", "confirm button not found")
                        print(f"Hata: onay butonu bulunamadi {handle}".rstrip())
                    sleep_between(args)
                    continue

                page.mouse.wheel(0, 1800)
                page.wait_for_timeout(2_000)
                scroll_y = page.evaluate("window.scrollY")
                if scroll_y == seen_scroll_y:
                    idle_rounds += 1
                else:
                    idle_rounds = 0
                    seen_scroll_y = scroll_y

                if idle_rounds >= 4:
                    print("Yeni Following butonu bulunamadi; islem tamam veya sayfa limite geldi.")
                    break

            print(f"Bitti. Basarili: {unfollowed}, hata: {failed}, log: {log_path}")
        finally:
            browser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
