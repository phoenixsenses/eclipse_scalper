# -*- coding: utf-8 -*-
"""LANE MIND SELF-TEST -- known-answer acceptance for `--who` and the corpus reader.

WHY THIS EXISTS.  Five defects were found in these tools on 2026-08-27 and every single one was
found by a lane USING them or asking a question, never by me re-reading my own code.  Twice I
published an absence that came from a probe I wrote myself against a pattern I chose myself.  So
before telling the lanes the tools are fixed, the tools are tested against answers taken from the
corpus itself rather than from my expectations.

THE CONSTRUCTION.  Truth is sampled FROM the shelf, not authored:

  PHRASES ACROSS A LINE BREAK  take real spans where a two-word phrase straddles a newline in the
                               raw file.  A human reading the page sees "word1 word2".  `--who`
                               must find every one.  This is defect 1 (D-E22): a space compiled to
                               EXACTLY one space.
  HYPHEN-BROKEN WORDS          take real spans where the typesetter split a word.  A human sees
                               one word.  `--who` must find it.  This is defect 3 (D-E23), which
                               hid behind CRLF.
  LIGATURE WORDS               words carrying `ffi`/`fi`/`fl` as single glyphs.  Defect 0, the one
                               `corpus_text_v1` was built for.
  NUL-BYTE SOURCES             the 3 files a raw reader skips entirely as "binary".
  NEGATIVE CONTROLS            strings that CANNOT be on the shelf.  A repair that makes
                               everything match is worse than the defect; recall without this is
                               not evidence.
  ESTATE HALF                  `--who` must still return the one section it was built to find
                               (frailty -> 437) and must still show CT-016 as CLOSED by CT-016-R.

Usage:  python tools/lane_mind_selftest_v1.py
Exit code is 0 only if every case passes.
"""
from __future__ import annotations

import glob
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tools.corpus_text_v1 as C          # noqa: E402
import tools.lane_mind_v1 as L            # noqa: E402

SEED = 20260827
N_PHRASE = 40
N_HYPHEN = 30
N_LIGATURE = 15

STOP = set("""the a an of in on to for and or is are was were be been being that this these those
with by from as at it its their his her our your not no if then than so such which who whom whose
we they he she you i but also can may might must shall should would could have has had do does
did one two both each other some any all more most less least when where while into over under
after before between during about""".split())
WORDY = re.compile(r"^[a-z][a-z]{2,}$")


def raw_bodies():
    """The shelf as a human sees it: NUL-safe, ligature-normalised, but NOT de-hyphenated.

    De-hyphenation is the thing under test, so the truth sample must be drawn before it.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(C.TEXT_DIR, "*.txt"))):
        t = open(f, "rb").read().decode("utf-8", "replace")
        for k, v in C.LIGATURES.items():
            t = t.replace(k, v)
        for k, v in C.DASHES.items():
            t = t.replace(k, v)
        out[os.path.basename(f)] = t
    return out


def sample_linebreak_phrases(raw, n, rng):
    """Real two-word phrases that straddle a newline.  A human types them with a space."""
    rx = re.compile(r"([A-Za-z]{3,})[ \t]*\r?\n[ \t]*([A-Za-z]{3,})")
    cand = set()
    for body in raw.values():
        for m in rx.finditer(body):
            w1, w2 = m.group(1).lower(), m.group(2).lower()
            if WORDY.match(w1) and WORDY.match(w2) and w1 not in STOP and w2 not in STOP:
                cand.add("%s %s" % (w1, w2))
    cand = sorted(cand)
    return rng.sample(cand, min(n, len(cand)))


def sample_hyphen_words(raw, n, rng):
    """Real words the typesetter split.  A human reads one word."""
    rx = re.compile(r"([A-Za-z]{3,})-[ \t]*\r?\n[ \t]*([a-z]{2,})")
    vocab = set()
    for body in raw.values():
        vocab |= {w.lower() for w in re.findall(r"[A-Za-z]{4,}", body)}
    cand = set()
    for body in raw.values():
        for m in rx.finditer(body):
            joined = (m.group(1) + m.group(2)).lower()
            if joined in vocab and len(joined) >= 6:
                cand.add(joined)
    cand = sorted(cand)
    return rng.sample(cand, min(n, len(cand)))


def sample_ligature_words(raw, n, rng):
    """Words that carry a ligature glyph in the source bytes."""
    lig = "".join(C.LIGATURES.keys())
    cand = set()
    for f in sorted(glob.glob(os.path.join(C.TEXT_DIR, "*.txt"))):
        t = open(f, "rb").read().decode("utf-8", "replace")
        for m in re.finditer(r"[A-Za-z%s]{6,}" % re.escape(lig), t):
            w = m.group(0)
            if any(ch in w for ch in lig):
                norm = w
                for k, v in C.LIGATURES.items():
                    norm = norm.replace(k, v)
                if norm.isalpha() and len(norm) >= 6:
                    cand.add(norm.lower())
    cand = sorted(cand)
    return rng.sample(cand, min(n, len(cand)))


def case(label, ok, detail=""):
    print("  %-5s %-46s %s" % ("PASS" if ok else "FAIL", label, detail))
    return ok


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rng = random.Random(SEED)
    raw = raw_bodies()
    failures = []

    print("LANE MIND SELF-TEST   truth sampled from the shelf, not authored\n")

    # ---------------------------------------------------------------- corpus half
    print("PHRASES THAT STRADDLE A LINE BREAK   (defect 1: a space meant exactly one space)")
    ph = sample_linebreak_phrases(raw, N_PHRASE, rng)
    miss = [p for p in ph if L.who_corpus([p])["total"] == 0]
    if not case("%d real line-break phrases, all findable" % len(ph), not miss,
                "missed: %s" % (miss[:5] if miss else "none")):
        failures.append("linebreak_phrases")

    print("\nWORDS THE TYPESETTER SPLIT   (defect 3: never folded, hid behind CRLF)")
    hy = sample_hyphen_words(raw, N_HYPHEN, rng)
    missh = [w for w in hy if L.who_corpus([w])["total"] == 0]
    if not case("%d real hyphen-broken words, all findable" % len(hy), not missh,
                "missed: %s" % (missh[:5] if missh else "none")):
        failures.append("hyphen_words")

    print("\nLIGATURE WORDS   (defect 0: the reason corpus_text_v1 exists)")
    lg = sample_ligature_words(raw, N_LIGATURE, rng)
    missl = [w for w in lg if L.who_corpus([w])["total"] == 0]
    if not case("%d real ligature words, all findable" % len(lg), not missl,
                "missed: %s" % (missl[:5] if missl else "none")):
        failures.append("ligature_words")

    print("\nNUL-BYTE SOURCES   (a raw reader calls these binary and skips them)")
    nul = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(C.TEXT_DIR, "*.txt")))
           if b"\x00" in open(f, "rb").read()]
    loaded = set(C.bodies())
    stems = {os.path.splitext(n)[0] for n in nul}
    seen = {s for s in stems if any(s in k or k in s for k in loaded)}
    if not case("%d NUL files, all loaded" % len(nul), len(seen) == len(stems),
                "loaded: %d/%d" % (len(seen), len(stems))):
        failures.append("nul_files")

    print("\nNEGATIVE CONTROLS   (a repair that matches everything is worse than the defect)")
    junk = ["zqxjkvw", "qqzzxx frobnicator", "hapax legomenon zzq",
            "unobtainium liquidity", "flibbertigibbet hazard", "xyzzy plugh",
            "nonexistentterm estimator", "quuxbaz spread"]
    fired = [j for j in junk if L.who_corpus([j])["total"] > 0]
    if not case("%d impossible strings, all return zero" % len(junk), not fired,
                "fired: %s" % (fired if fired else "none")):
        failures.append("negative_controls")

    print("\nEMBEDDED-MATCH REPORTING   (defect 2: a substring matched the NEGATION)")
    r_phrase = L.who_corpus(["overlapping returns"])
    econ = r_phrase["per_source"].get("ECONOPHYS_ODM", {})
    r_stem = L.who_corpus(["identifiab"])
    stem_emb = sum(v.get("embedded_hits", 0) for v in r_stem["per_source"].values())
    stem_tot = r_stem["total"]
    ok = econ.get("embedded_hits", 0) == econ.get("hits", -1) and stem_emb == stem_tot
    if not case("phrase flags its negation, stem not filtered", ok,
                "phrase %s/%s embedded, stem %d/%d embedded and still returned"
                % (econ.get("embedded_hits"), econ.get("hits"), stem_emb, stem_tot)):
        failures.append("embedded_reporting")

    # ---------------------------------------------------------------- estate half
    print("\nESTATE HALF   (the cases --who was built for)")
    bl, sec = L.blocks(), L.sections()

    # the one section --who was built to find.  its first version searched TITLES ONLY and missed
    # it, because 437 carries the word in its BODY and closes in prose rather than a verdict block.
    fr = str(L.who(["frailty"], bl, sec))
    if not case("frailty still returns section 437", "437" in fr,
                "" if "437" in fr else "the case the tool exists for"):
        failures.append("estate_437")

    # a resolution row must CLOSE its parent, or a lane is sent to reopen settled work.
    ct = L.contradictions()
    txt = str(ct)
    open_ids = [c.get("id") for c in ct if isinstance(c, dict) and c.get("open")]         if isinstance(ct, list) else []
    if not case("CT-016 is not listed as open", "CT-016" not in [str(i) for i in open_ids],
                "open: %s" % (open_ids[:8] if open_ids else "n/a")):
        failures.append("estate_ct016")

    # an empty estate result is a CLAIM, not a default: a term the record certainly contains
    # must come back non-empty, or the estate half is silently dead.
    liv = str(L.who(["censoring"], bl, sec))
    if not case("a term the record certainly carries returns hits", len(liv) > 200,
                "%d chars" % len(liv)):
        failures.append("estate_liveness")

    print("\n" + "=" * 78)
    if failures:
        print("SELF-TEST FAILED: %s" % ", ".join(failures))
        return 1
    print("SELF-TEST PASSED -- every case answered from the shelf, negative controls silent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
