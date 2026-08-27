# LANE A · PREREG V1 — ADDENDUM A
### the effective-bet estimator of §6 is under-specified, and one reading of it is degenerate

**Binds to** `LANE_A_PREREG_V1.md`, body sha256 `6bac365a88a4782ca86716eaed8a193b425fa4bd20aa141ca3a55dc707ace754`,
frozen `2026-08-27T03:22:40Z`.

**The source is not modified.** Precedent: `IMMUTABLE_ERRATA_LEDGER_V1` (§393) — six entries, zero
sources touched. This file is hashed on the same terms and the evaluator verifies both.

**No outcome has been read.** Everything here is a design quantity: correlations of raw returns, and
the same quantities under a pure-noise null. The estimand `f` remains unmeasured.

---

## A1 · What §6 said, and what it left open

§6 specified `effective bets = trace(C)/lambda_max`, with a `1/n` noise floor and a rule to take the
lower estimate when the two are within `1.5×`. §8b then measured `3.27` and built the feasibility
table on it.

**Two things were left open and both turn out to matter:**

1. §6 did not say whether `C` is the **raw** correlation matrix or one **denoised and detoned**.
   López de Prado (MLAM §3638) is explicit — *"in a nonexperimental setting, the researcher should
   denoise and detone the correlation matrix"* — and MLAM §2.6 gives the reason this test in
   particular needs it: *"Detoning is the principal components analysis analogue to computing
   beta-adjusted (or market-adjusted) returns in regression analysis."* **A cross-sectional test's
   signs are cross-sectionally balanced, so the market mode is hedged away and should not consume a
   bet.** §8b's matrix was raw, and its top mode carries 30.5 % of the variance.
2. §6 gave the `1/n` floor as the null but never **measured** it. It is not `1/n`.

## A2 · The matrix is thinner than §8b reported

```
symbols  N = 186        return observations  T = 151        T/N = 0.812
```

`T = 151`, not the ~179 the 180-day window suggests: the correlation is built on the **intersection**
of days across all 186 symbols (§6 requires it — a correlation on unequal supports is not a
correlation), and that intersection is 152 days.

**MLAM §2.2 states the Marcenko–Pastur theorem for `1 < T/N`. At `T/N = 0.812` the sample
correlation matrix is singular by construction: rank ≤ 151 of 186, and 35 eigenvalues are exactly
zero.** The regime §8b measured in is outside the theorem's stated domain.

```
MP noise band          [0.012, 4.452]        lambda_+ = (1 + sqrt(N/T))^2
top 8 eigenvalues      56.80  4.71  4.01  3.88  3.66  3.22  3.17  2.86
eigenvalues > lambda_+ 2 of 186
```

**Two.** 186 crypto perpetuals over 151 common days yield **two** directions distinguishable from
noise.

## A3 · Five estimators, and the null that separates them

The decisive check — the estate's own standing rule, *test any incremental-fit statistic on pure
noise first* — run on iid normal returns at the identical `(N, T)` through the identical pipeline:

```
estimator                    REAL      NOISE     real/noise
raw   trace/lambda_max       3.27      43.57         0.08x
raw   exp(entropy)          41.21      99.95         0.41x
dn+dt trace/lambda_max      29.39     177.46         0.17x
dn+dt exp(entropy)         178.58     184.99         0.97x   <- DEGENERATE
MP signal count              2.00       0.00           inf   <- zero under the null
```

**`dn+dt exp(entropy)` is degenerate and is struck out.** It cannot distinguish this market from
white noise. The mechanism is plain once seen: LdP's residual-eigenvalue denoising replaces every
noise eigenvalue with their common average, which makes the spectrum maximally flat *on purpose*,
and spectral entropy measures flatness. The statistic reports its own construction.

**And §8b's `3.27` is not an absolute count.** Its null is `43.57`. The number is meaningful only as
a ratio — the market is `13×` more concentrated than noise — and §8b published it as though it were
a bet count. **Every non-MP estimator here has a large noise floor and the real matrix scores BELOW
it**, which is the right direction (a real market is more correlated than noise) but means the
absolute value is set by `N` and `T`, not by market structure.

## A4 · What this does to the verdict

```
effective bets basis          value    S_pooled   years to t=2   f for a 1-year verdict
MP signal count                2.00      0.0172         13,550                  10.79%
§8b raw trace/lambda_max       3.27      0.0220          8,276                   9.54%
dn+dt trace/lambda_max        29.39      0.0659            922                   5.51%
raw exp(entropy)              41.21      0.0780            657                   5.06%
--- struck out ---
dn+dt exp(entropy)           178.58      0.1624            151                   3.51%
```

**§8b's headline figure of 9.54 % is not robust.** Across the four surviving estimators the required
capture spans **5.06 % – 10.79 %**, a factor of 2.1, and 9.54 % is the second-most pessimistic of
them. **That number is withdrawn as a point estimate and replaced by the range.**

**The verdict is robust.** Every surviving estimator requires a pooled capture of at least **5.06 %**.
The estate's measured single-leg captures are **1–2 %** (`A-S14`) and the best dark-family cell ever
measured is **2.09 %** (`A-S43`). The most favourable defensible treatment still demands **2.4× more
capture than anything ever measured here**, and the least favourable demands `5×`.

**`S0` (DO NOT START) stands, and its trigger is restated on the range:** the test becomes runnable
when some other route demonstrates a pooled capture at or above **5.06 %** — the most permissive
surviving estimator's requirement, chosen deliberately so that the gate cannot be accused of being
set by the pessimistic end.

## A5 · The result underneath, which is not a number

Four estimators of the same quantity disagree by `13×`, one is degenerate, and the one clean under
the null (`MP signal count`) measures *detectability in this sample* rather than *bets available to
a strategy*. They differ because **the effective-bet count of a test depends on the correlation of
its SIGNED returns, and the sign vector is exactly what this prereg refuses to name.**

```
EFFECTIVE_BETS_NOT_POINT_IDENTIFIED_PRE_OUTCOME
```

This is the same shape as §388's `rho is not point-identifiable pre-outcome`, reached from an
unrelated direction. It is a limit on the design, not a defect in the data, and it is why A4 reports
a range and gates on its permissive end.

## A6 · What §6 now requires

Superseding the estimator clause of §6 (and only that clause; §6's steps 1, 2, 4 and 5 stand):

```
3'  effective BETS is reported as a RANGE over the four surviving estimators, each with
    its measured null on iid returns at the same (N, T).  An estimator whose real/noise
    ratio exceeds 0.90 is DEGENERATE and is excluded from the range.
    No single value is treated as the count.  N_eff is computed at BOTH ends and the
    verdict must hold at both, or it is not a verdict.
```

---

## Freeze block

```
STATUS                 FROZEN
binds to               LANE_A_PREREG_V1.md body sha256 6bac365a88a4782c...
strikes out            dn+dt exp(entropy) as DEGENERATE (real/noise 0.97)
withdraws              §8b's 9.54% as a point estimate; replaced by 5.06%-10.79%
leaves standing        S0 DO NOT START; the infeasibility verdict; the whole of §1-§7
driver                 tools/s47_effective_bets.py
data                   reports/research/h2_response_shape_v1/S47_EFFECTIVE_BETS_V1.json
sha256 of this file    6d94b802bab30d5b4482c8b21016adf6c32c2e7e6ea74079e44d3c711ea59ac1
frozen at              2026-08-27T03:52:25Z
```

**The hashed subject is this file's text UP TO the line `## Freeze block`** — everything above,
nothing below.
