/*
Rate-limited mode. Paste into DevTools Console on:
  https://x.com/phoenixsenses/following

Stop anytime:
  window.__xUnfollowStop = true
*/
(async () => {
  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const randomDelay = (min, max) => delay(min + Math.random() * (max - min));
  const log = (...args) => console.log("[x-unfollow-safe]", ...args);

  window.__xUnfollowStop = false;
  window.__xUnfollowSaw429 = false;

  if (!window.__xUnfollowFetchPatched) {
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const response = await originalFetch(...args);
      const url = String(args[0]?.url || args[0] || "");
      if (response.status === 429 && url.includes("/friendships/destroy")) {
        window.__xUnfollowSaw429 = true;
        log("429 rate-limit detected; backing off");
      }
      return response;
    };
    window.__xUnfollowFetchPatched = true;
  }

  const followingTexts = new Set(["Following", "Takip ediliyor"]);
  const confirmTexts = new Set(["Unfollow", "Takibi b\u0131rak", "Takipten \u00e7\u0131k"]);

  function visible(el) {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
  }

  function text(el) {
    return (el.innerText || el.textContent || "").trim();
  }

  function findFollowingButton() {
    return [...document.querySelectorAll('button, [role="button"]')]
      .filter(visible)
      .find((button) => {
        const testId = button.getAttribute("data-testid") || "";
        const aria = button.getAttribute("aria-label") || "";
        const label = text(button);
        return (
          testId.endsWith("-unfollow") ||
          followingTexts.has(label) ||
          aria.startsWith("Following @") ||
          aria.includes("Takip ediliyor")
        );
      });
  }

  function findConfirmButton() {
    const explicit = document.querySelector('button[data-testid="confirmationSheetConfirm"]');
    if (explicit && visible(explicit)) return explicit;
    return [...document.querySelectorAll('button, [role="button"]')]
      .filter(visible)
      .find((button) => confirmTexts.has(text(button)));
  }

  function handleFromButton(button) {
    const article = button.closest("article");
    if (!article) return "";
    for (const link of article.querySelectorAll('a[href^="/"]')) {
      const candidate = link.getAttribute("href").split("/").filter(Boolean)[0] || "";
      if (/^[A-Za-z0-9_]{1,15}$/.test(candidate)) return "@" + candidate;
    }
    return "";
  }

  let unfollowed = 0;
  let misses = 0;
  let backoffMs = 0;

  log("started; stop with: window.__xUnfollowStop = true");

  while (!window.__xUnfollowStop) {
    if (window.__xUnfollowSaw429) {
      window.__xUnfollowSaw429 = false;
      backoffMs = backoffMs ? Math.min(backoffMs * 2, 30 * 60 * 1000) : 10 * 60 * 1000;
      log("waiting after 429", Math.round(backoffMs / 1000), "seconds");
      await delay(backoffMs);
      continue;
    }

    const button = findFollowingButton();
    if (!button) {
      window.scrollBy(0, Math.floor(window.innerHeight * 0.85));
      await delay(1800);
      misses += 1;
      if (misses >= 10) break;
      continue;
    }

    misses = 0;
    const handle = handleFromButton(button);
    button.scrollIntoView({ block: "center" });
    await randomDelay(700, 1600);
    button.click();
    await randomDelay(900, 1800);

    const confirm = findConfirmButton();
    if (!confirm) {
      log("confirm not found", handle);
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      await randomDelay(3000, 6000);
      continue;
    }

    confirm.click();
    unfollowed += 1;
    backoffMs = 0;
    log("unfollowed", unfollowed, handle);
    await randomDelay(4000, 8000);
  }

  log("finished", { unfollowed, stopped: Boolean(window.__xUnfollowStop) });
})();
