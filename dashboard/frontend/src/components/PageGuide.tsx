import React from "react";
import { useEffect, useMemo, useState } from "react";

interface GuideItem {
  icon: string;
  titleTr: string;
  titleEn: string;
  descTr: string;
  descEn: string;
}

interface PageGuideProps {
  icon: string;
  titleTr: string;
  titleEn: string;
  subtitleTr: string;
  subtitleEn: string;
  items: GuideItem[];
}

export default function PageGuide(props: PageGuideProps) {
  const { icon, titleTr, titleEn, subtitleTr, subtitleEn, items } = props;
  const storageKey = useMemo(() => `eclipse.guide.${titleEn.toLowerCase().replace(/\s+/g, "_")}`, [titleEn]);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw === "0") setOpen(false);
    } catch {
      // noop
    }
  }, [storageKey]);

  function toggleGuide() {
    setOpen((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(storageKey, next ? "1" : "0");
      } catch {
        // noop
      }
      return next;
    });
  }

  return (
    <section className="page-guide">
      <div className="page-guide-title" style={{ justifyContent: "space-between" }}>
        <span>{titleTr} / {titleEn}</span>
        <button className="guide-toggle" onClick={toggleGuide}>
          {open ? "Hide guide" : "Show guide"}
        </button>
      </div>
      {open && (
        <>
          <div className="page-guide-sub">
            {subtitleTr}
            <br />
            {subtitleEn}
          </div>
          <div className="signpost-grid">
            {items.map((item, idx) => (
              <div className="signpost-card" key={`${item.titleEn}_${idx}`}>
                <div className="signpost-title">
                  {idx + 1}. {item.titleTr} / {item.titleEn}
                </div>
                <div className="signpost-desc">
                  {item.descTr}
                  <br />
                  {item.descEn}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
