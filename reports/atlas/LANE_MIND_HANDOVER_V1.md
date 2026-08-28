# LANE MIND — HANDOVER

**For whoever takes over `tools/lane_mind_v1.py` and its corpus reader.**
Written 2026-08-28 by lane D, which built them. Everything below is measured; where it is an
estimate it says so.

---

## 1 · What this is, in one paragraph

Four Claude sessions ("lanes" A, B, C, D) work the same repository in parallel and cannot see each
other. They coordinate through **one append-only file**, `reports/atlas/_SHARED_LOG.md` (9,725
lines). `lane_mind_v1.py` is the **recall layer** over that record plus a 13-book corpus on disk.
It **writes nothing** — every command prints to stdout and creates no file. Delete it and nothing
is lost.

```
tools/lane_mind_v1.py            1,080 lines   the recall layer, 7 commands
tools/corpus_text_v1.py            171 lines   the ONLY correct reader for data/literature_v2/text/*.txt
tools/lane_mind_selftest_v1.py     273 lines   13 known-answer cases, exit 0 only if all pass
reports/atlas/_SHARED_LOG.md     9,725 lines   THE RECORD.  Append-only.  Never edited.
reports/atlas/LANE_MIND_PROTOCOL_V1.md         the one-page rule sheet
reports/atlas/LANE_PROMPT.txt                  what the operator sends a lane each round
```

Commands: `--brief LANE` · `--who TERM...` · `--inbox LANE` · `--promises LANE` · `--owed` ·
`--ct` · `--check`, with `--json`, `--full`, `--no-corpus`.

---

## 2 · Invariants. Break any of these and the tool is worse than nothing

1. **The record is sacred and append-only.** Never edit an earlier block. A correction is a NEW
   block that names what it withdraws. The original stays malformed forever — that is by design.
2. **The tool writes nothing.** Derived data goes to stdout; rules go to files. Every stale surface
   in this estate on 2026-08-27 was a derived file on disk.
3. **`--json` is the contract, the human text is not.** Another lane's gate parsed the human
   `--check` line, and a wording change broke it. Their gate fail-closed, which is why nothing was
   lost. Change the prose freely; keep the JSON keys.
4. **Never edit another lane's file.** A defect found there is a FINDING reported to that lane.
5. **No `git push`.** `origin` is PUBLIC and the diff is 1,645 files / 1.55M insertions including
   `SYSTEM_STATE.md`. Commit freely; push never.
6. **Section numbers collide across lanes by design.** Identity is the stable ID (`D-E42`), never
   the `§` number. No renumbering, ever.

---

## 3 · Run this before and after every change

```
python tools/lane_mind_selftest_v1.py     # 13 known-answer cases, exit 0 only if all pass
python tools/lane_mind_v1.py --check      # record invariants; 2 OPEN and 2 superseded is correct
```

Plus one assertion that lives outside the suite and must be re-checked after **any** change to
`who()`:

```
--who "frailty" (as lane D) MUST return section 437 among INDEPENDENT_PRIOR.
```

§437 predates lane D entirely and is the duplication that motivated the whole tool. Two separate
defects were caught by that single case alone. **If it stops holding, the change is wrong.**

---

## 4 · The defect register. Read this before writing code

Eleven defects were found in these tools in two days. **Not one was found by a unit test.** They
were found by another lane using the tool, by an outside question, or by reading a number next to
an estimate. Four of them are the *same shape* recommitted.

| # | defect | how it presented | how it was caught |
|---|---|---|---|
| 1 | plain `grep` on the corpus | false zero on `identifiab` | memory of a prior incident |
| 2 | `--who` searched titles only | missed §437, the one case it existed for | running it on that case |
| 3 | phrase matched with `re.escape` | a space meant EXACTLY one space; PDFs break lines. **6.0% of phrase hits invisible** | control phrases known to be on the shelf |
| 4 | substring match | `overlapping returns` matched `NON-overlapping returns` | reading the snippet |
| 5 | hyphen fold never implemented | docstring claimed it; **7,566 breaks, 2,205 real words** | a sample of "phrases" that were broken words |
| 6 | CRLF | a probe for `-\n` returns EXACTLY ZERO on CRLF text | the number was implausible |
| 7 | block parser | header swallowed body under `re.S`; **103 of 120 blocks parsed, 7 messages to lane D never delivered** | another lane's report |
| 8 | `SECT` matched one header shape | **54.8% of SYSTEM_STATE invisible** | another lane measured it |
| 9 | `rank()` broke ties by index | passed `rho(x,x)=+1` and `rho(x,-x)=-1`; returned **0.52 against a constant** | two table rows contradicting each other |
| 10 | `--who` read 3 fields of 9 | **50.8% of the log invisible**, including all four `to X` lines | asking "does it find everything" |
| 11 | bare `except` + wrong constant | every query returned ZERO register rows **and reported success** | zero on every query is a symptom |

**The three rules that fall out of this list:**

- **A bare `except` around an estate read is never acceptable.** Defect 11 was committed *inside*
  the fix for another instance of the same shape. A missing file must produce a visible row.
- **A known-answer test that passes is not coverage.** Defect 9 passed both obvious tests and still
  fabricated structure on ties. Defect 3's fix passed its own suite while defect 10 sat open.
- **Print the diagnostic inside the table** — n, sd, count, at-risk. Two defects were caught by the
  number beside the estimate and none by the estimate itself.

---

## 5 · What `--who` means now, and why it is not a hit count

`--who > 0` **is not prior work**. Measured on lane D's own distinctive terms, **six of eight were
100% the searcher's own blocks**; lane C measured 67–91% on theirs. Every estate hit is therefore
classified:

| class | meaning | evidence? |
|---|---|---|
| `SELF` | this lane's own writing | **no** |
| `INDEPENDENT_PRIOR` | another writer, **before** this lane first raised the term | **yes** |
| `ECHO_RISK` | another writer, **after** | **cannot tell** |
| `CORPUS` | the 13-book shelf | separate leg, never mixed in |

The classes are not arbitrary. `ECHO_RISK` is Hernán & Robins' **conditioning on a common effect**:
the causes are (A) the topic is worth studying and (E) this lane raised it; the common effect is
(Y) another lane wrote about it. H&R 8.6 states that conditioning on a common effect *always*
induces association in at least one stratum while the other can stay clean — which is exactly why
**a zero here is strong and a non-zero is weak**. H&R 8.5 says adjusting requires positivity for
the selection, and **positivity fails**: a term only one lane uses has no chance of being written
by another. So the class is **reported, never adjusted away**.

The cut is the first line at which the asking lane mentions the term, computed **per file** —
SYSTEM_STATE and the shared log have no common line axis, and comparing them on one was defect
number twelve, caught by the §437 assertion. The approximation errs in one direction only: it can
over-count `ECHO_RISK`, never under-count it.

**Pass the lane letter** (`--brief D --who "..."`) or hits come back unclassified.

---

## 6 · Known gaps, with sizes, none of them fixed

These are measured and open. They are not bugs to be surprised by later.

- **Polysemy.** The discriminating threshold (500) counts *frequency*, not *meaning*. `correction`
  returns 77 hits across 9 sources and every one is "finite-sample correction", not record
  amendment. A detector that guessed meaning would be a fabricated field, so this is stated rather
  than patched.
- **Source regime.** Nothing in the corpus output says which market a passage is about. Bouchaud's
  "100 days or more" is an equities statement and reads identically to a crypto one. The honest
  form is to publish the caveat with the citation.
- **`LANE_CHARTERS_V1.md`** (13,115 bytes) is still not searched by `--who`.
- **Two permanently open `--check` rows** — `B (18 sections)` and `LANE D OPENED` — legitimately
  non-ID headers. They will never close, and that is correct.
- **`--promises` is a heuristic.** Content-word overlap scores writing style as much as
  follow-through. It is null-calibrated per lane and the rate is **withheld** unless |z| > 2 AND at
  least 5 kept; lane A currently fails that gate and its rate is not shown. The flagged instances
  are always listed, because the instance is the product and the count is not.

---

## 7 · Working rules for changes

1. Read the mail first: `--brief <LANE> --ct`. A silently skipped message is an unclosed debt.
2. **Verify an incoming correction** — the correction itself can be wrong. One reported the right
   defect and named the wrong file.
3. Before appending a block to the record, **validate it on a COPY** with `--check`. Append-only
   means a malformed block costs two blocks; that has happened twice.
4. After any change touching `who()`, re-run the §437 assertion by hand.
5. Every claim about a number gets the number re-derived at least once. Re-deriving published
   figures surfaced defects nothing else would have.

---

```verdict
HANDOVER_V1_ISSUED_BY_LANE_D
THE_RECORD_IS_SACRED_THE_RECALL_IS_DISPOSABLE_AND_WRITES_NOTHING
JSON_IS_THE_CONTRACT_THE_HUMAN_TEXT_IS_NOT
SECTION_437_MUST_STAY_INDEPENDENT_PRIOR_FOR_FRAILTY_AFTER_EVERY_CHANGE_TO_WHO
ELEVEN_DEFECTS_IN_TWO_DAYS_NONE_FOUND_BY_A_UNIT_TEST
FOUR_OF_THEM_WERE_THE_SAME_SHAPE_RECOMMITTED
A_BARE_EXCEPT_AROUND_AN_ESTATE_READ_IS_NEVER_ACCEPTABLE
A_HIT_COUNT_IS_NOT_PRIOR_WORK_PASS_THE_LANE_LETTER
KNOWN_GAPS_POLYSEMY_SOURCE_REGIME_LANE_CHARTERS_STATED_NOT_HIDDEN
NO_PUSH_ORIGIN_IS_PUBLIC
```
