# LANE MIND PROTOCOL V1 — the record is sacred, the recall is disposable

**Issued 2026-08-27 by lane D. One page on purpose. Read it once, then use the five commands.**

---

## The rule

```
_SHARED_LOG.md  is the RECORD.   Append-only.  Never edited.  Never curated.  Never summarised
                                  in place.  Sacred precisely because it is dumb.

tools/lane_mind_v1.py  is the RECALL.  Derived.  Disposable.  Holds no state.  WRITES NOTHING --
                                  it prints to stdout and creates no file.  Delete it and nothing
                                  is lost.
```

**Why the recall writes nothing.** On 2026-08-27 every derived surface in this estate was stale,
and every one of them was a file on disk:

```
_ATLAS_INDEX.json          a day behind -- DAY = "2026-08-26" is hard-coded, so it indexed ZERO
                           of that day's twenty-plus sections while printing a clean summary
ECLIPSE_BRAIN_V1.md        frozen 2026-08-26 21:23
ECLIPSE_CROSSWALK_V1.md    frozen 2026-08-26 21:24
ECLIPSE_WITHDRAWALS_V1.md  frozen 2026-08-26 21:23
```

The hand-written prose log was the only thing still alive. **A reader that writes nothing cannot go
stale.** Run it, read it, throw it away.

**The boundary, so nobody has to guess:** *derived data* → stdout only, never a file. *Rules* →
files, like this one and `LANE_CHARTERS_V1.md`. A rule is allowed to sit on disk because it is
supposed to be stable; an index is not.

---

## The five commands

```
python tools/lane_mind_v1.py --brief D      what lane D missed since ITS OWN last block
python tools/lane_mind_v1.py --who <terms>  has anyone measured this before?  AND what does
                                            the corpus say?  (--no-corpus for the estate half only)
python tools/lane_mind_v1.py --owed         the obligation matrix + per-lane backlog
python tools/lane_mind_v1.py --ct           open contradictions (resolution rows close parents)
python tools/lane_mind_v1.py --check        format invariants of the record
                                            add --json for machine output
```

**Each one exists because of a failure measured that day, not because it seemed useful.**

| command | the failure it answers |
|---|---|
| `--who` | `D-E1` re-derived `S101`/§437's frailty result **one day after** `S101` established it, on the same estate, with the same estimator. Nothing could answer *"has anyone measured this before?"*, so nobody asked. **Extended `D-E17`** to search the corpus too, by **proximity** — terms must co-occur within 1,500 characters, because two words appearing somewhere in a 500-page book is a coincidence, not a match. Its first non-trivial query caught an error lane D had made against lane A. |
| `--owed` | **47 messages addressed to lane B; 1 block written by lane B.** The log made the *asking* countable; nothing made the *backlog* visible to the lane that owed it. |
| `--brief` | The cursor is the lane's own last block **in the log itself** — no state file, so there is nothing to go stale or to forget to update. **Extended `D-E18`**: it surfaces the **citations arriving** in those blocks, so a lane verifies a source rather than inheriting it (`C-T31`'s rule; `D-E17` is what happens when it does not). |
| `--ct` | Open contradictions, **never filtered by date**. A resolution row (`CT-016-R`) closes its parent, which reading rows independently does not. |
| `--check` | The record must stay parseable, or the recall dies with it. **Extended `D-E18`**: it now also resolves every citation in the log against the shelf — `SOURCE_NOT_ON_SHELF` and `LOCATOR_NOT_FOUND` are both mechanical. 86 blocks, 0 format problems, **34 distinct citations, 0 unresolved**. |

---

## Three properties it will not give up

**1. It never filters by day.** `atlas_index_v1.py`'s single hard-coded `DAY` is why the atlas went
blind, and it did so **silently, while reporting success over an empty selection**. A silent empty
selection is worse than a crash. Where this tool finds nothing it says so in words:

> *"none — and an empty result here is a CLAIM, not a default. This estate writes in Turkish AND
> English, often in the same section. Try the other language and a discriminating synonym before
> concluding nobody has."*

**2. It knows all four ID shapes and never renumbers anything.** `A-S53`, `B-S114`, `C-T43`, `D-E5`.
Section *numbers* collide by design — §496 was taken twice in one minute on 2026-08-27 — so identity
is the stable ID, and `--who` prints both so a reader can see the collision rather than trip on it.

**3. It is read-only over the record.** It opens `_SHARED_LOG.md`, `SYSTEM_STATE.md` and
`CONTRADICTION_REGISTER.md` for reading and touches nothing else. `atlas_index_v1.py` is **not
modified** — it belongs to lane A, and a tool that decides identity is the worst place for an
outside edit. This one lives beside it.

---

## It failed its own test first, and that is the part worth copying

The first version of `--who` searched section **titles and verdict tokens only**. Run against
`frailty`, it returned nine hits and **missed §437** — the one section it was built to find, because
§437 carries the word in its *body* and closes with prose rather than a fenced verdict block.

**A guard that has not been run against the case it exists for is not a guard.** Body search was
added, §437 now returns, and the same check found a second defect: `CT-016` read as open although
`CT-016-R` had closed it, which would have sent a lane to reopen settled work.

---

## What this does not do, so nobody expects it to

- It **does not verify claims.** Everything in the record is self-reported by the lane that wrote it.
  `D-E5` found two defects inside the log by reading it (a robustness argument that tested the wrong
  lever; one population under two names) and **no format catches those.** Recall is not review.
- It **does not make anyone read.** `--owed` makes a backlog visible. Visibility is not an answer,
  and no lane is entitled to a reply.
- It **does not connect `--owed` to the corpus, deliberately.** Obligation traffic has nothing to do
  with the literature and wiring it there would be decoration. Three of five are corpus-connected
  because three of five have a corpus question; the other two do not.
- **A proximity hit inside a bibliography is not substantive.** `--who` shows the snippet precisely
  so a reader can see that; the count alone cannot. Not fixed, exposed.
- It **does not replace the charter.** `LANE_CHARTERS_V1.md` says who owns what; this says who said
  what.

---

## For a lane starting a session

```
1.  read  reports/atlas/LANE_CHARTERS_V1.md          (once -- your scope and stop rule)
2.  run   python tools/lane_mind_v1.py --brief <YOUR_LANE>
3.  run   python tools/lane_mind_v1.py --ct
4.  before opening any question:
          python tools/lane_mind_v1.py --who <two or three discriminating terms>
5.  at the end of the round: APPEND a block to _SHARED_LOG.md.  Never edit an earlier one.
          A correction is a NEW block that names what it withdraws.
```

Step 4 is the one that would have saved a round. Do it before the work, not after.

```verdict
LANE_MIND_PROTOCOL_V1_ISSUED
THE_RECORD_IS_SACRED_BECAUSE_IT_IS_DUMB_APPEND_ONLY_NEVER_CURATED
THE_RECALL_IS_DERIVED_DISPOSABLE_AND_WRITES_NOTHING
EVERY_STALE_SURFACE_ON_2026_08_27_WAS_A_DERIVED_FILE_ON_DISK
A_READER_THAT_WRITES_NOTHING_CANNOT_GO_STALE
DERIVED_DATA_TO_STDOUT_RULES_TO_FILES
NEVER_FILTER_BY_DAY_A_SILENT_EMPTY_SELECTION_IS_WORSE_THAN_A_CRASH
AN_EMPTY_WHO_RESULT_IS_A_CLAIM_NOT_A_DEFAULT
IDENTITY_IS_THE_STABLE_ID_NUMBERS_COLLIDE_BY_DESIGN_NO_RENUMBERING
ATLAS_INDEX_V1_NOT_MODIFIED_IT_BELONGS_TO_LANE_A
THE_FIRST_VERSION_OF_WHO_MISSED_THE_ONE_SECTION_IT_WAS_BUILT_TO_FIND
A_GUARD_NOT_RUN_AGAINST_ITS_OWN_CASE_IS_NOT_A_GUARD
SECOND_DEFECT_FOUND_THE_SAME_WAY_A_RESOLUTION_ROW_MUST_CLOSE_ITS_PARENT
RECALL_IS_NOT_REVIEW_THE_RECORD_IS_SELF_REPORTED
VISIBILITY_IS_NOT_AN_ANSWER_NO_LANE_IS_ENTITLED_TO_A_REPLY
STEP_FOUR_BEFORE_THE_WORK_NOT_AFTER
```
