/*
Fast mode. Paste into DevTools Console on:
  https://x.com/phoenixsenses/following

Stop anytime:
  window.__xUnfollowStop = true
*/
(async () => {
  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const log = (...args) => console.log("[x-unfollow-fast]", ...args);

  window.__xUnfollowStop = false;

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

  function findFollowingButtons() {
    return [...document.querySelectorAll('button, [role="button"]')]
      .filter(visible)
      .filter((button) => {
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
  log("started; stop with: window.__xUnfollowStop = true");

  while (!window.__xUnfollowStop) {
    const buttons = findFollowingButtons();

    if (!buttons.length) {
      window.scrollBy(0, Math.floor(window.innerHeight * 0.95));
      await delay(450);
      misses += 1;
      if (misses >= 12) break;
      continue;
    }

    misses = 0;
    for (const button of buttons.slice(0, 4)) {
      if (window.__xUnfollowStop) break;
      if (!document.contains(button) || !visible(button)) continue;

      const handle = handleFromButton(button);
      button.scrollIntoView({ block: "center" });
      await delay(120);
      button.click();
      await delay(220);

      const confirm = findConfirmButton();
      if (!confirm) {
        log("confirm not found", handle);
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
        await delay(350);
        continue;
      }

      confirm.click();
      unfollowed += 1;
      log("unfollowed", unfollowed, handle);
      await delay(650);
    }

    window.scrollBy(0, Math.floor(window.innerHeight * 0.65));
    await delay(350);
  }

  log("finished", { unfollowed, stopped: Boolean(window.__xUnfollowStop) });
})();
