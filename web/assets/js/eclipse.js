/* ============================================================
   ECLIPSE — shared page behaviour
   Lights the rail as sections enter view, marks the current nav
   item, and drives the mobile menu. No dependencies.
   ============================================================ */
(function () {
  "use strict";

  /* ---- mark the current page in the header nav --------------- */
  var here = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".hdr-nav a").forEach(function (a) {
    var target = a.getAttribute("href");
    if (!target) return;
    if (target === here || (here === "research-e-der.html" && target === "research.html")) {
      a.classList.add("is-here");
      a.setAttribute("aria-current", "page");
    }
  });

  /* ---- mobile menu ------------------------------------------- */
  var burger = document.querySelector(".hdr-burger");
  var nav = document.querySelector(".hdr-nav");
  if (burger && nav) {
    burger.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      burger.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  /* ---- light the rail ---------------------------------------- */
  var taps = document.querySelectorAll(".tap, .rv");
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduced || !("IntersectionObserver" in window)) {
    taps.forEach(function (el) { el.classList.add("is-lit"); });
    return;
  }

  var seen = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      e.target.classList.add("is-lit");
      seen.unobserve(e.target);
    });
  }, { rootMargin: "0px 0px -14% 0px", threshold: 0.06 });

  taps.forEach(function (el) { seen.observe(el); });
})();
