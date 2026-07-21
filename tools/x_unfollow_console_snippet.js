/*
Paste this into DevTools Console on:
  https://x.com/phoenixsenses/following

Stop anytime by running:
  window.__xUnfollowStop = true
*/
(async () => {
  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const randomDelay = (min, max) => delay(min + Math.random() * (max - min));
  const log = (...args) => console.log("[x-unfollow]", ...args);

  window.__xUnfollowStop = false;

  const followingTexts = new Set(["Following", "Takip ediliyor"]);
  const confirmTexts = new Set(["Unfollow", "Takibi b\u0131rak", "Takipten \u00e7\u0131k"]);

  function visible(el) {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
  }

  function buttonText(button) {
    return (button.innerText || button.textContent || "").trim();
  }

  function findFollowingButton() {
    const buttons = [...document.querySelectorAll('button, [role="button"]')].filter(visible);
    return buttons.find((button) => {
      const testId = button.getAttribute("data-testid") || "";
      const aria = button.getAttribute("aria-label") || "";
      const text = buttonText(button);
      return (
        testId.endsWith("-unfollow") ||
        followingTexts.has(text) ||
        aria.startsWith("Following @") ||
        aria.includes("Takip ediliyor")
      );
    });
  }

  function findConfirmButton() {
    const explicit = document.querySelector('button[data-testid="confirmationSheetConfirm"]');
    if (explicit && visible(explicit)) return explicit;
    const buttons = [...document.querySelectorAll('button, [role="button"]')].filter(visible);
    return buttons.find((button) => confirmTexts.has(buttonText(button)));
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

  log("started; stop with: window.__xUnfollowStop = true");

  while (!window.__xUnfollowStop) {
    const button = findFollowingButton();
    if (!button) {
      window.scrollBy(0, Math.floor(window.innerHeight * 0.85));
      await delay(1600);
      misses += 1;
      if (misses >= 8) break;
      continue;
    }

    misses = 0;
    const handle = handleFromButton(button);
    button.scrollIntoView({ block: "center" });
    await randomDelay(500, 1200);
    button.click();
    await randomDelay(700, 1400);

    const confirm = findConfirmButton();
    if (!confirm) {
      log("confirm button not found", handle);
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
      await randomDelay(1500, 2500);
      continue;
    }

    confirm.click();
    unfollowed += 1;
    log("unfollowed", unfollowed, handle);
    await randomDelay(2500, 5500);
  }

  log("finished", { unfollowed, stopped: Boolean(window.__xUnfollowStop) });
})();
