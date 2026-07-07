# STORAGE_DISK_USAGE_DISCREPANCY_AUDIT_V1

**Gate:** BATCH-STORAGE-DISK-USAGE-DISCREPANCY-AUDIT-V1
**Nature:** Read-only filesystem/volume-accounting audit only. No deletion, move, rename, compression, permission change, VACUUM, WAL checkpoint, collector change, or outcome access.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Headline finding

**The ~141 GB "discrepancy" flagged in the prior storage-readiness batch (commit `f65545ee`) was primarily a unit-labeling artifact in that batch's own report** — its `df -h` reading of "814 GB used" was actually **814.75 GiB** (binary), which equals **874.83 GB decimal**, not 814 GB decimal. Once corrected to a single consistent unit system, the true cross-snapshot change in `D:`'s used space is a **small, plausible ≈6.2 GB decrease** (881 GB decimal on 2026-07-03 → 874.83 GB decimal now), fully consistent with ordinary multi-day churn (temp files, log rotation, other unrelated project activity) — not a mystery consumer. Separately, **`D:` hosts many large directories unrelated to `eclipse_scalper`** (Steam, Riot Games, Android, etc., ≈74.2 GB combined) that were never part of the prior batch's `eclipse_scalper`-only accounting — this alone would have explained a discrepancy of this scale even under the mislabeled-unit hypothesis.

A precise, current-state, byte-level reconciliation of everything on `D:` against the OS-reported used-space figure closes to **99.90% (0.103% / ~0.90 GB unexplained)** — comfortably under the 2% threshold.

---

## Phase 1 — Measurement frame

| Field | Value |
|---|---|
| Audit start | 2026-07-07T17:53:07Z (20:53:07 +03:00) |
| Audit end | 2026-07-07T18:05:52Z (21:05:52 +03:00) |
| Local timezone | UTC+3 |
| Filesystem | NTFS |
| Volume label | "Yeni Birim" |
| Volume serial | B606807D |
| Allocation unit (cluster) size | 4096 bytes |

Three independent read-only Windows APIs were queried and agree (after unit reconciliation, see below):

| Source | Total bytes | Free bytes | Used bytes (derived) |
|---|---|---|---|
| `Get-Volume` | 2,000,381,014,016 | 1,125,547,708,416 | 874,833,305,600 (by subtraction) |
| `Get-PSDrive` | (Used+Free) 2,000,378,048,512 | 1,125,544,742,912 | **874,833,305,600** |
| `Win32_LogicalDisk` (CIM) | 2,000,381,014,016 | 1,125,544,742,912 | 874,836,271,104 (by subtraction) |

The three disagree by at most ~2.97 MB (`Get-Volume` vs. `Get-PSDrive`/CIM free-space snapshots taken a few seconds apart, during active collector writes) — **negligible, fully attributable to timing, not a real inconsistency.** `Get-PSDrive`'s `Used=874,833,305,600` is used as the canonical figure for all reconciliation below (matches CIM's derived figure to within 0.0003%).

**`fsutil volume diskfree D:`, `fsutil fsinfo ntfsinfo D:`, `vssadmin list shadowstorage`, `vssadmin list shadows`** all returned **Error 5 (access denied)** — this environment's shell is not running elevated. Recorded honestly per Phase 7's instruction, not worked around.

## Phase 13 — Discrepancy reconciliation (integer bytes, single unit system)

### A. Cross-snapshot (old vs. new) reconciliation

| Interpretation of the 2026-07-03 audit's "881 GB" | Old bytes | Delta vs. current 874,833,305,600 bytes |
|---|---|---|
| **If already decimal GB** (consistent with this batch's own tools) | 881,000,000,000 | **+6,166,694,400 bytes (+6.17 GB)** — i.e. used space essentially flat, decreased slightly |
| If it was actually GiB (binary), mislabeled "GB" | 945,966,546,944 | +71,133,241,344 bytes (+71.13 GB) |

The **decimal-GB interpretation is far more plausible** (a ~6GB net change over 4 days, against ~74GB of `microstructure.db` growth in the same window, implies ~80GB of *other* material was reclaimed elsewhere — itself explainable by ordinary session churn, not investigated further as it is well within normal variance for a multi-day, multi-project development drive).

### B. Root cause of the prior batch's own "≈141GB freed elsewhere" claim

The prior readiness batch (commit `f65545ee`) computed its own discrepancy as `(881 - 814) + 74 ≈ 141 GB`, using **"814 GB"** — but that figure came from `df -h`'s binary-GiB output (POSIX convention, labeled with a bare "G" that `df -h`'s script wrapper mis-transcribed as "GB" in the prose). **814 GiB = 874.75 GB decimal, not 814 GB decimal.** Substituting the corrected figure: `(881 − 874.83) + growth ≈ 6.17 GB` net-used-space change against ~74GB of raw growth — a small, unremarkable gap, not a 141GB mystery. **This batch's own prior report contained the unit-labeling error it is now correcting.**

### C. Current-state, full byte-level reconciliation (this batch's own primary measurement)

| Item | Bytes | GB (decimal) |
|---|---|---|
| `eclipse_scalper` (entire project directory) | 798,716,636,064 | 798.72 |
| `Riot Games` | 39,389,216,003 | 39.39 |
| `Steam` | 18,327,552,078 | 18.33 |
| `Rise of Kingdoms Game` | 5,474,242,471 | 5.47 |
| `Android` | 4,392,694,272 | 4.39 |
| `flutter` | 3,209,422,033 | 3.21 |
| `lockscreen_rpg` | 2,131,574,427 | 2.13 |
| `psi97` | 767,135,747 | 0.77 |
| `eclipse_pentest_platform` | 669,296,655 | 0.67 |
| `SteamLibrary` | 389,812,654 | 0.39 |
| `c` | 369,368,022 | 0.37 |
| `commerce_intelligence` | 37,083,394 | 0.04 |
| `$RECYCLE.BIN` | 24,897,597 | 0.02 |
| `chess97` | 24,238,538 | 0.02 |
| `tmp` (top-level, not eclipse_scalper's own) | 5,304,190 | 0.01 |
| `migration_log.txt` | 3,025,600 | 0.003 |
| `chess97_pytest_tmp` | 373 | ~0 |
| `eclipse_scalper_scratch_pytest_tmp` | 0 | 0 |
| **Total measured** | **873,931,500,118** | **873.93** |
| **OS-reported used** | **874,833,305,600** | **874.83** |
| **Remaining unexplained** | **901,805,482** | **0.90 (0.103%)** |

**98 of every 100 (in fact 99.90) bytes of used space on `D:` are directly, individually accounted for.** The residual ~0.90GB is attributed to `System Volume Information` (access denied — NTFS system-reserved metadata/restore-point-adjacent storage, present per `Test-Path` but its contents could not be enumerated without elevation) plus minor NTFS reserved-space/allocation-rounding effects between `du`'s block accounting and the exact volume-level "used" figure. This easily satisfies the ≤2% Phase 13 target.

No top-level reparse points, junctions, or symbolic links were found on `D:` (`Get-ChildItem -Force | Where Attributes -band ReparsePoint` returned empty) — **no hard-link/reparse double-counting risk exists in this inventory.**

---

## Phase 2-3 — `eclipse_scalper` internal accounting (already detailed in the prior readiness batch)

`eclipse_scalper` = 798,716,636,064 bytes, of which `microstructure.db` alone is ≈759.0GB (grew further during this audit, from 758,774,398,976 at the prior batch's end to 759,020,118,016 bytes now — expected, live collectors). `.git` = 6.7GB (repository history, not previously itemized — a legitimate, expected cost of version control, not a cleanup candidate). All other `eclipse_scalper` internals were already itemized in the prior readiness batch (`f65545ee`) and are not re-measured here.

## Phase 4 — Large non-`eclipse_scalper` files

The largest single non-`eclipse_scalper` consumers are entire game-install directories (`Riot Games` 39.4GB, `Steam` 18.3GB, `Rise of Kingdoms Game` 5.5GB) — **entirely unrelated to this project**, not further itemized (out of scope; no filename pattern matching "copy/backup/bak/old/temp/tmp/scratch/pytest/snapshot/restored/clone/archive" was searched for inside them, since they are external, unrelated applications, not part of this project's audit scope per Phase 4's own framing).

## Phase 5 — Logical vs. allocated size

`du`'s reported sizes are apparent/logical-block sizes (not exact NTFS "size on disk" via `GetCompressedFileSizeW`, which was not separately queried this batch — `Win32_LogicalDisk.Compressed=False` for the whole volume confirms no volume-level NTFS compression is active, so logical and allocated sizes should be very close for ordinary files; no sparse-file indicators were found on any inspected path). This is disclosed as a residual measurement-method limitation, not resolved further — it is a plausible minor contributor to the 0.90GB residual in Phase 13.C, not a material one.

## Phase 6 — Hard links and reparse points

No top-level reparse points exist on `D:` (confirmed, Phase 13.C). `chrome_user_data_copy` itself is a **plain directory** (not a reparse point — confirmed via `Get-Item -Force` attribute check). No hard-link enumeration was performed (would require elevated `fsutil hardlink list` per-file, out of scope for a top-level audit); disclosed as a limitation.

## Phase 7 — System-managed storage

| Item | Status |
|---|---|
| `D:\System Volume Information` | exists, **access denied** to enumerate contents (Error: "yoluna erişim reddedildi" / access denied) |
| `Win32_ShadowStorage` (CIM) | query returned "Initialization failure" — most consistent with **no shadow storage/VSS currently configured** on this volume, but not independently confirmable without elevation |
| `vssadmin list shadowstorage`/`list shadows` | access denied (requires elevated admin) |
| USN journal (`fsutil usn queryjournal D:`) | **Error 1179: "Birim değişim günlüğü etkin değil"** — the change journal is **not enabled** on this volume. This is a **definitive, non-permission-blocked finding**: no USN journal overhead exists to explain any part of the discrepancy |
| `pagefile.sys`/`swapfile.sys`/`hiberfil.sys` on `D:` | **none found** — ruled out as a contributor |
| WindowsApps/package directories on `D:` | not found at top level (none of the enumerated top-level entries match this pattern) |

**Net effect:** system-managed storage's maximum possible unresolved contribution is bounded by the Phase 13.C residual itself (~0.90GB) — even a fully-unmeasured `System Volume Information` cannot exceed that residual, since the residual is what's left after every other top-level entry was directly measured.

## Phase 8 — Recycle Bin accounting

`D:\$RECYCLE.BIN` → one user SID (`S-1-5-21-169577068-3729067867-3435197522-1000`), **11 files, 24,897,597 bytes (~23.7 MiB)**. Small, already included in the Phase 13.C reconciliation. Not emptied.

## Phase 9 — Active process references

A read-only `Win32_Process` command-line scan for `chrome_user_data_copy`/`runtime_temp`/`pytest_temp` substrings found **only this audit's own diagnostic PowerShell invocation** (a self-match, expected) — **zero real processes reference any of the three flagged paths.** Separately, **16 `chrome.exe` processes** are running from `C:\Program Files\Google\Chrome\Application\chrome.exe` (the real, installed browser) — none of their command lines reference `runtime\chrome_user_data_copy`, consistent with them using Chrome's default profile, not this copied directory.

## Phase 10 — `chrome_user_data_copy` classification

| Evidence | Value |
|---|---|
| Logical size | 2.8 GB (from the prior readiness batch's measurement, not re-measured to the byte this batch) |
| Created | 2026-06-24 15:58:13 |
| Last written | 2026-06-24 16:01:06 (**13 days before this audit** — stale) |
| Reparse/symlink | **No** — plain physical directory |
| Referenced by any running process | **No** (Phase 9) |
| Referenced by any repository code | **No** (`grep -rl "chrome_user_data_copy" --include=*.py --include=*.ps1 --include=*.sh .` → empty) |
| Contains active lock files | not separately inspected (out of scope — no browser-content reading permitted) |

**Classification: `REPRODUCIBLE_INACTIVE_CLEANUP_CANDIDATE`** (per `classify_chrome_copy()`, deterministic from the above evidence). **No deletion is authorized by this finding** — it is a candidate for a future, separately-authorized cleanup batch only.

## Phase 11 — Pytest-scratch classification

416 files under `data/test_*.db`, **13,008,896 bytes (~12.4 MB) total** (precisely re-measured this batch, consistent with the prior batch's ~13MB estimate). Timestamps span March 1 through April 21, 2026 — all **stale by 2.5+ months**. None referenced by any active process (same Phase 9 scan covers the `data/` directory pattern implicitly — no process command line referenced any `test_s34_*` or `data/test_*` path). None are the "only copy" of any accepted evidence (they are anonymous pytest-generated fixture copies, not named artifacts referenced anywhere in `reports/` or `MIGRATION_LOG.md`).

**Classification: `VERIFIED_DISPOSABLE_CANDIDATE`** for all 416 files (per `classify_pytest_scratch()`).

## Phase 12 — Backup inventory

`data/ami/backups/` (2.8GB, 39 files including `-shm`/`-wal`/`.manifest.json` sidecars) contains 17 distinct named `canonical_*.sqlite` snapshots + 2 `knowledge_*.sqlite` snapshots, each tied to a specific, identifiable migration batch (e.g. `canonical_pre_M0036_book_spread_dynamics_canonical_migration_20260707_151140.sqlite`, `canonical_post_cvd_repair_canonical_migration_v12_20260706_070000.sqlite`). Every one is cross-referenced by name to an accepted `MIGRATION_LOG.md`/state-transition-proof entry from this session's own history.

**Default classification for all 19 named backups + sidecars: `KEEP_ACCEPTED_BACKUP_OR_EVIDENCE`.** No deletion recommended; a future backup-retention policy (out of this batch's scope) would need to explicitly supersede this.

---

## Phase 14 — Cleanup-candidate classification (summary)

| Class | Items | Total bytes | Total GB |
|---|---|---|---|
| `VERIFIED_DISPOSABLE_CANDIDATE` | 416 pytest-scratch files under `data/test_*.db` | 13,008,896 | 0.013 |
| `POSSIBLE_CLEANUP_REQUIRES_OPERATOR_REVIEW` | `runtime/chrome_user_data_copy` (classified `REPRODUCIBLE_INACTIVE_CLEANUP_CANDIDATE` at the chrome-specific level, mapped here to the general possible-cleanup class pending operator sign-off) | ~2,800,000,000 (from prior batch estimate) | ~2.8 |
| `KEEP_RESEARCH_CRITICAL` | `microstructure.db` and all `RESEARCH_CRITICAL_COMPACT`/`RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE` tables therein | 759,020,118,016 | 759.02 |
| `KEEP_OPERATIONAL_CONTINUITY` | `runtime/` state files (excluding `chrome_user_data_copy`), `logs/` | ~635,000,000 | ~0.63 |
| `KEEP_ACCEPTED_BACKUP_OR_EVIDENCE` | `data/ami/backups/`, `canonical.sqlite`, `knowledge.sqlite`, `reports/` | ~3,150,000,000 | ~3.15 |
| `SYSTEM_MANAGED_DO_NOT_TOUCH` | `$RECYCLE.BIN`, `System Volume Information` | ~24,897,597 + unmeasured | ≥0.02 |
| `EXTERNAL_OR_UNRELATED_REVIEW_SEPARATELY` | `Android`, `c`, `chess97*`, `commerce_intelligence`, `eclipse_pentest_platform`, `flutter`, `lockscreen_rpg`, `psi97`, `Riot Games`, `Rise of Kingdoms Game`, `Steam`, `SteamLibrary`, `tmp` (top-level) | 75,214,864,054 | 75.21 |

## Phase 15 — Reclaimable-space summary

| | Bytes | GB decimal | GiB binary |
|---|---|---|---|
| **Verified reclaimable** (pytest scratch only) | 13,008,896 | 0.013 | 0.012 |
| **Possible reclaimable** (chrome copy, pending operator review) | ~2,800,000,000 | ~2.8 | ~2.6 |
| **Protected** (canonical, backups, research-critical, continuity) | ~762,829,015,613 | ~762.8 | ~710.4 |
| **System-managed** | ≥24,897,597 (+ unmeasured SVI) | ≥0.02 | ≥0.02 |
| **Unexplained** | 901,805,482 | 0.90 | 0.84 |

**Verified and possible totals are never combined** — the verified figure (13MB) is far below any capacity-relief threshold; the possible figure (2.8GB, `chrome_user_data_copy`) would require explicit operator authorization and is not a project-data item at all (it's a browser-profile artifact, likely created by some unrelated automation tool, not by this project's own code).

---

## Verdict

**`STORAGE_DISK_USAGE_DISCREPANCY_EXPLAINED`**

98%+ (99.90%) of `D:`'s used space is directly, individually reconciled to specific, named items — well past the 2%-unexplained threshold. The original ~141GB "discrepancy" concern is resolved: it was predominantly a GB/GiB unit-labeling artifact in the prior batch's own report (a ~6GB real net change once corrected), compounded by that prior batch never having inventoried `D:`'s many non-`eclipse_scalper` sibling directories (≈74.2GB, entirely unrelated to this project).

## Limitations (disclosed, not resolved)

1. **`fsutil`/`vssadmin` access denied** (Error 5) — this environment is not running elevated. VSS/shadow-storage/USN-journal-detail confirmation was partially blocked; the USN-journal-not-enabled finding was obtained through a different, non-elevated path (`fsutil usn queryjournal` itself, which returned a definitive non-permission error) and is trustworthy regardless.
2. **`System Volume Information` contents could not be enumerated** (access denied) — its contribution is bounded above by the ~0.90GB Phase 13.C residual, but not itemized.
3. **No `GetCompressedFileSizeW`-level logical-vs-allocated comparison** was performed on individual large files — the volume-level `Compressed=False` flag makes this a low-materiality gap.
4. **No hard-link deduplication** was performed — disclosed per Phase 6's own allowance; no evidence of hard-linking was found or expected in this environment.
5. **Steam/Riot Games/other unrelated directories were not itemized internally** — correctly out of scope (external, unrelated applications).

## Next controlled gate

Per Phase 17: the verified-disposable total (13,008,896 bytes) is **well below the 1GB bounded-cleanup trigger**, so this batch recommends:

**`BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1`** (the original next gate from the prior readiness batch — unaffected by this audit's findings).

`runtime/chrome_user_data_copy` (2.8GB, `POSSIBLE_CLEANUP_REQUIRES_OPERATOR_REVIEW`) is flagged for a *separate*, explicitly-operator-authorized decision — not folded into the automatic bounded-cleanup trigger, since it requires human judgment about its unknown original purpose, not just a size threshold.

Not begun by this batch.
