# CORPUS AUDIT PROMPT V1 — the reusable instruction, and why each clause is there

**Issued 2026-08-27 by lane D, from the operator's own prompt plus the failures that day exposed.**
The operator's schema was already right; what was missing is everything that makes it executable by
a lane that does **not** have that day's context.

---

## Where the original breaks

| clause | what happens without a fence |
|---|---|
| *"tüm nesneleri… her iddia"* | **Unbounded quantifier.** Measured that day: 914 corpus questions, 329 verdict tokens, 51 log blocks, ~500 sections. "All" is not executable — it yields a truncated answer presented as complete. |
| *"uygula"* | **No scope fence.** The audit finds defects in *other lanes'* artifacts. Charter rule 5: a lane may contradict another, never silently overwrite. Unfenced, "apply" means trespass. |
| *"verdict üret"* | **No vocabulary** ⟹ ad-hoc tokens the index cannot group. |
| *"külliyatla uyumu"* | **Four different things.** Predicted / contradicted / declared unidentifiable / **silent**. The fourth is the most valuable — it is how `A-S62`'s carry branch was found to be the one thing the shelf cannot advise on. |
| — | **No "has this been answered?" gate.** `D-E1` re-derived `S101`/§437 one day later on the same estate. |
| — | **No stop rule**, where every lane charter has one. |
| — | **No treatment of self-reported numbers.** The log is self-reported; two defects were found *inside* it that no format catches. |

---

## The prompt

```
STEP 0 — BEFORE OPENING ANYTHING
  python tools/lane_mind_v1.py --brief <YOUR_LANE>
  python tools/lane_mind_v1.py --ct
  For every item you are about to open:
      python tools/lane_mind_v1.py --who <2-3 discriminating terms>
  If it has been measured, you INHERIT it with a citation or you RE-MEASURE it and say why.
  You do not re-derive it silently.  An empty --who result is a CLAIM: this estate writes in
  Turkish and English, so try both before concluding nobody has.

STEP 1 — EXTRACT MECHANICALLY, SELECT EXPLICITLY
  Extract the FULL population by machine and save it, so the selection can be audited:
      corpus objects and demands   -> from data/literature_v2/text via tools/corpus_text_v1.py
                                      (NUL-safe + ligature-normalised; a plain grep misses up to
                                      100% of hits on fi/fl terms and skips 3 of 13 files entirely)
      shared-log claims            -> every block's verdict / stands / withdraws
  Then STATE THE SELECTION RULE and the number left unselected.  Extraction is mechanical;
  selection is judgement, and it must be labelled as judgement.

STEP 2 — ONE ROW PER ITEM, SEVEN FIELDS, NO PROSE
  source            book + section/passage, or block stable ID + log line
  object_or_action  the thing the corpus GENERATES, or the action it DEMANDS
  estate_counterpart  the artifact path or stable ID that already holds it -- or NONE
  gap               what is missing, contradictory, or merely unstated
  test_or_fix       the smallest check that would settle it
  owner             which lane owns the fix (never assume it is you)
  verdict           from the CLOSED LIST below

  VERDICT VOCABULARY (closed -- add one only by declaring it in the same document)
      TEXTBOOK_PREDICTED            the corpus already says it
      TEXTBOOK_CONTRADICTED         the corpus says otherwise
      CORPUS_SAYS_NOT_IDENTIFIABLE  the object cannot be identified, by theorem
      CORPUS_ADDS_A_MISSING_CONDITION   right but incomplete
      CORPUS_REGIME_MISMATCH        the terms exist, this regime is outside them
      BEYOND_THE_SHELF              machine-checked absence -- name the terms used
      ALREADY_ANSWERED_ON_RECORD    with the stable ID
      OPEN_ANSWERABLE / OPEN_BLOCKED    and say what blocks it

STEP 3 — NUMBERS YOU LEAN ON
  Any number your conclusion DEPENDS on is recomputed, or it is marked `SELF_REPORTED` -- the
    record is self-reported and recall is not review.
  Any null you test against is CALIBRATED before you read the test -- 2 of 6 needed it in D-E4
    and both changed the answer; one moved 28 orders of magnitude in p and flipped a verdict.
  Any absence claim names the reader and the discriminating terms it used.
  Any threshold your result depends on is DECLARED and its sweep is PUBLISHED -- one unjustified
    notional floor moved a central duration by 4.0x.
  Samples are named BY ARTIFACT PATH, never in prose -- two published sections used one name for
    two populations differing by a single threshold and 6.35x in median size.
  NEVER pool across scales; standardise within unit first -- a raw pooled CV came out larger than
    every individual symbol's.
  NEVER trust a mean over disagreeing units -- a mean of -0.23 hid +0.86 / +0.65 / -2.19.

STEP 4 — APPLY, FENCED
  Apply ONLY inside your own scope, and ONLY additively:
      your own artifacts        -- yes
      a NEW file beside another lane's tool  -- yes
      another lane's file       -- NO.  It becomes a finding addressed to that lane.
      guardrailed surfaces      -- NO, ever.
  Prioritise by: (a) it unblocks another lane, (b) it is cheap and settles a live contradiction,
  (c) it prevents a repeat of a failure already on record.  Say what you did NOT do and why.

STEP 5 — CLOSE
  Append ONE block to reports/atlas/_SHARED_LOG.md.  Never edit an earlier block; a correction is
    a NEW block that names what it withdraws.  Write every `to X` line, even when it is `-`.
  Add a SYSTEM_STATE section closing with a fenced ```verdict block -- without it your work is
    invisible to every index this estate has.
  Contradictions go to CONTRADICTION_REGISTER.md, withdrawals to the atlas.
  Verify the record still parses:   python tools/lane_mind_v1.py --check

STOP RULE
  Stop when the ledger is issued and the fenced applications are done.
  Do NOT widen the extraction to make the ledger look complete -- publish the unselected count
  instead.  If an item needs data you do not have, that is a row with OPEN_BLOCKED, not a detour.
```

---

## What deliberately is *not* in it

- **No target row count.** A quota makes a lane pad the ledger. The unselected count does the same
  job honestly.
- **No requirement that the audit find something.** But note lane B's charter precedent: *"an audit
  that produces no findings… means the audit was not adversarial."* Both can be true — say which.
- **No instruction to fix what you find.** That is Step 4's fence, and it is the clause most likely
  to be dropped by someone in a hurry. It is also the one that protects the estate.

```verdict
CORPUS_AUDIT_PROMPT_V1_ISSUED
THE_OPERATORS_SEVEN_FIELD_SCHEMA_WAS_ALREADY_THE_VALUABLE_PART
UNBOUNDED_QUANTIFIERS_REPLACED_BY_MECHANICAL_EXTRACTION_PLUS_A_DECLARED_SELECTION_RULE
SELECTION_IS_JUDGEMENT_AND_MUST_BE_LABELLED_AS_JUDGEMENT
VERDICT_VOCABULARY_CLOSED_SO_THE_INDEX_CAN_GROUP_IT
BEYOND_THE_SHELF_IS_A_VERDICT_NOT_AN_OMISSION
APPLY_IS_FENCED_ADDITIVE_ONLY_AND_NEVER_ANOTHER_LANES_FILE
STEP_ZERO_WHO_BEFORE_OPENING_ANYTHING
NUMBERS_YOU_LEAN_ON_ARE_RECOMPUTED_OR_MARKED_SELF_REPORTED
NULLS_ARE_CALIBRATED_BEFORE_THE_TEST_IS_READ
THRESHOLDS_DECLARED_AND_THEIR_SWEEP_PUBLISHED
SAMPLES_NAMED_BY_ARTIFACT_PATH_NEVER_IN_PROSE
NO_POOLING_ACROSS_SCALES_STANDARDISE_WITHIN_UNIT_FIRST
NO_MEAN_OVER_DISAGREEING_UNITS
CLOSE_STEP_ENDS_WITH_A_RECORD_PARSE_CHECK
STOP_RULE_PRESENT_PUBLISH_THE_UNSELECTED_COUNT_INSTEAD_OF_WIDENING
```
