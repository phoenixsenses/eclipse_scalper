# LANE A · PREREGISTRATION V1 — the pooled cross-sectional capture test

**Status: `FROZEN`.** The hash and timestamp are in the freeze block at the foot of this file. **No outcome has been read, and none may be
read except through `tools/lane_a_evaluator_v1.py`, whose interlocks verify this hash.**

Written 2026-08-27 by lane A under `LANE_CHARTERS_V1.md`. The charter's failure condition is
explicit: *"a prereg with a free parameter left in it."* §3 enumerates every parameter and how it is
removed.

---

## 1 · What this tests, and what it does not

§463 established that this estate cannot settle whether a capture of any useful scale exists across a
cross-section of symbols: 201 days of panel gives `t = 0.65`. §469 established the constraint is
**days, not symbols**, and §470 that no download changes it — the archive depth is what it is.

Forward time will accrue whether or not anyone is ready for it. **Being ready is the whole task.**

```
ESTIMAND     the pooled single-leg capture  f = E[s·r] / E[|r|]
             measured across a mechanically defined universe, each symbol at its own
             derived horizon, on data that did not exist when this file was frozen.

H0           f = 0
H1           f > 0        (one-sided; a reliably negative f is a separate finding, see §9)

NOT TESTED   whether any particular rule works.  This measures whether the SIGN CARRIES
             at all, cross-sectionally, on fresh data.  A specific rule is a separate
             preregistration with its own multiplicity.
```

---

## 2 · A correction this preregistration is built on

Every economic figure in §460–§473 used `f = 3.35 %`, carried as *"the demonstrated capture."*
Traced to source (`SYSTEM_STATE.md` L39494, study `A-S14`):

```
ULAŞILABİLİR YAKALAMA (= |rho1|)   p50 3.35%   p90 9.43%   p99 18.40%
```

It is **the median absolute lag-1 autocorrelation of HEDGED HOURLY PAIR spreads**, over 19,110 pairs,
**two legs, cost 20 bps** — and `A-S14`'s own verdict on it was **8.8× short**. It is a distribution,
not a point, and it belongs to a different regime from the single-leg frontier that inherited it.

**`A-S14`'s same table gives the single-leg figure:**

```
route                       achieved    required    short
directional single leg        1-2%       13-19%     ~10x
pairs / stat arb              3.35%       29.6%      8.8x
```

**§460's frontier is single-leg at c = 10 bps. It used the pairs number — favourable on both counts.**

What that costs, compounding with §467's measured `k = 0.6966`:

```
                     published   k-corrected   at f = 1.5%   at f = 1.0%
BASUSDT  S_max          0.3742        0.2852        0.0572        0.0254
BTCUSDT  S_max          0.0454        0.0346        0.0069        0.0031
BTCUSDT  h*           31.6 days     41.5 days     207 days      466 days
BTCUSDT  h_be          7.9 days     10.4 days      51.7 days     116 days
trades for t = 2         29,380        29,380       146,544       329,726
```

**The frontier as published was optimistic by roughly 5× in Sharpe and 6.5× in horizon**, on top of
§467's 24 %. Recorded here rather than quietly absorbed; §460–§473's *structure* is unaffected —
the closed form, the invariants and the universal curve are all `f`-independent in form.

---

## 3 · Every parameter, and how it is removed

| parameter | status | how |
|---|---|---|
| **f, the capture** | **NOT PREREGISTERED** | It is the estimand. The test measures it. `f_design` below enters only through power. |
| **horizon h** | **DERIVED** | `h* = [2c/(k·f_design·σ_d)]²` per symbol, from that symbol's own measured `σ_d` and `c` (§460). Not chosen. |
| **f_design** | **FROZEN at 1.0 %** | §4 — derived from §465's asymmetry, not picked. |
| **k = E\|r\|/σ** | **MEASURED** | 0.6966 (§467), re-measured on the freeze-date panel and written into the freeze block. |
| **cost c** | **MEASURED** | per symbol: `spread + fee`. Spread = one tick (§452, 12 of 15 symbols). Fee from `OD-033` if answered, else `BINANCE_BASE` — and the freeze block records which. |
| **universe** | **RULE** | §5. Mechanical, outcome-free, evaluated at the freeze date. |
| **N_eff** | **MEASURED** (§8b: 3.27, not the 8 first assumed) | §6. The exact procedure, because three sections got this wrong (`A-S35`, `A-S41`, `A-S43`). |
| **the bar** | **DERIVED** | §7. One preregistered hypothesis ⇒ K = 1 ⇒ no False-Strategy correction; the FST-corrected bar is reported as a sensitivity and both must pass. |
| **when to stop** | **FROZEN** | §8. |

---

## 4 · Why `f_design = 1.0 %`, derived and not chosen

`h*` depends on `f`, and `f` is what is being measured. The circularity is broken by noting that
**`h*` affects power only, never validity**: a test at a suboptimal horizon is still a valid test of
`f ≠ 0`, merely a weaker one.

So `f_design` is chosen to make the *error* safe, and §465 says exactly which direction that is.
The universal curve `S/S_max = v(2−v)`, `v = √(h*/h)`, is symmetric in `v` and therefore violently
asymmetric in `h`:

```
lose 25% of the Sharpe:   h = 4·h*  (too long)   or   h = 0.444·h*  (too short)
lose 50%:                 h = 11.6·h*            or   h = 0.343·h*
below h_be = h*/4:        NEGATIVE, and diverging as h -> 0
```

**Being too short is catastrophic; being too long is merely wasteful.** `h* ∝ 1/f²`, so
**underestimating `f` makes `h*` too long — the safe side.**

`f_design = 1.0 %` is the **low end** of `A-S14`'s single-leg range (1–2 %). If the true `f` is 2 %,
`h*` is 4× too long and the curve costs 25 % of the attainable Sharpe. If `f_design` had been set at
the high end and the truth were 1 %, `h` would be `0.25·h*` — **at the break-even horizon, where the
Sharpe is zero.**

```
THE DESIGN PARAMETER IS SET LOW ON PURPOSE.  THE PENALTY FOR BEING LOW IS 25%.
THE PENALTY FOR BEING HIGH IS EVERYTHING.
```

---

## 5 · Universe rule — mechanical, outcome-free, evaluated at the freeze date

A symbol enters if and only if **all** of the following hold on the 180 calendar days ending at the
freeze date. Every quantity is a cost or a coverage quantity; **no return enters the filter.**

```
U1  coverage      >= 150 of the 180 days carry >= 1200 of 1440 one-minute bars
U2  liquidity     median daily quote volume >= L  (L derived in U2a)
U3  measurable    the tick is recoverable: >= 1 non-zero one-minute close change
U4  not pegged    median sigma_daily >= 50 bps      (excludes stablecoins; A-S28 measured
                  USDCUSDT at 1.2 bps, required capture 837%)
U5  priced        the symbol carries price data for the whole window in ONE store
                  (A-S36: the panel has two disjoint ingests; a symbol split across
                   them is excluded, not stitched)
```

**U2a — the liquidity floor, derived.** `A-S30` measured Kyle's λ per symbol and `A-S30` §5 gives
the trade size at which impact equals the fee. The floor is set so that **the intended position is
at most 1 % of the impact-equals-fee size**, i.e. impact ≤ 1 % of cost:

```
L  =  the smallest median daily notional for which  lambda(h*) * Q_intended <= 0.01 * c
Q_intended is written into the freeze block by the operator.  If it is not supplied,
L defaults to $5,000,000/day and the freeze block records that the default was used.
```

**No top-N. No ranking. No selection.** `A-S28`'s screen ranked symbols on `c/σ_d`; that ranking is
**not** used here. Ranking and then testing the top is the multiplicity error `CLAUDE.md` names:
*"sembol sembol bakıp en iyisini seçmek §200'ün suçunu tekrarlar."*

---

## 6 · N_eff — specified exactly, because it was got wrong three times

`A-S35`, `A-S41` and `A-S43` each mistook a row count for an `N`. `A-S43`'s headline looked like a
90 bps edge against a 10 bps cost and was `t = 0.41`. The procedure is therefore frozen:

```
1  a TRADE is one (symbol, non-overlapping window of length h*_symbol).
2  independent TIME UNITS = the number of non-overlapping windows in the evaluation
   span, computed per symbol and then taken as the MAXIMUM across symbols
   (symbols share the calendar; they do not add time).
3  effective BETS = trace(C) / lambda_max of the correlation matrix of symbol returns
   at the evaluation horizon, computed on the EVALUATION data.
   Reported alongside its noise floor 1/n and alongside rho_bar; if the eigenvalue
   share is within 1.5x of 1/n, effective bets is reported as ">= n/1.5, uninformative"
   and the LOWER of the two estimates is used.   (A-S36 caught exactly this.)
4  N_eff = (independent time units) x (effective bets).
5  SE = sd(per-trade signed return) / sqrt(N_eff).
```

**Any figure quoted without `N_eff` alongside it is a reporting defect, not a result.**

---

## 7 · Decision rule

```
PRIMARY     t = f_hat / SE(f_hat),  one-sided, computed with the N_eff of §6.
            PASS requires  t >= 2.0
SENSITIVITY the False-Strategy-corrected bar (López de Prado MLAM §8.5):
              E[max SR]/sd(SR) ~ (1-g)*Z^-1[1-1/K] + g*Z^-1[1-1/(K e)]
            with K = 1 this is 0 and adds nothing; K = 1 is asserted by this file
            existing and naming ONE hypothesis.  If more than one variant is ever run
            on the same window, K is the count of variants and BOTH bars must clear.
BOTH        must pass.  Either failing is a FAIL.
```

**One-sided is deliberate.** `H1: f > 0`. A significantly negative pooled `f` is not a pass with the
sign flipped — `A-S13` and §309 both record that inverting a result whose direction was chosen after
seeing it is not a hypothesis. A negative result is recorded as §9 specifies.

---

## 8 · Stop rule — frozen, with artifact-backed triggers

Modelled on §397's frozen stop rule.

```
S0  see section 8b -- DO NOT START; the span to a verdict is 8,276 years at f_design.
S1  ACCRUE until N_eff >= N_required, where
      N_required = (2.0 / (k * f_design / 2))^2 = (2.0 / (0.6966*0.010/2))^2 = 329,726
    computed at f_design and re-derivable from this file alone.
S2  EVALUATE ONCE.  One look.  The evaluator writes its verdict and the verdict stands.
S3  ABORT if the universe drops below 20 symbols meeting §5 at evaluation time.
S4  ABORT if any symbol's realised coverage in the evaluation span falls below U1's
    threshold -- the symbol is dropped, and if that takes the count below S3, abort.
S5  ABORT if OD-033 is answered and the resulting c differs from the frozen c by more
    than 2x.  A 2x cost change moves h* by 4x and the design horizon is then wrong by
    more than the 25% budget of section 4.  Re-freeze, do not re-interpret.
S6  NO EXTENSION.  If t falls short, the answer is "not established at N_required",
    not "accrue more".  Accruing until significance is the error §200 documents.
```

---

## 8b · Feasibility, measured — and the reason this prereg does not schedule itself

The evaluator's `--dry` mode was run against the 610-symbol panel at the lawful cutoff. Everything
below is measured, not assumed; `effective bets` in particular was a **guess of 8 in the first draft
of the evaluator and the measurement replaced it.**

```
UNIVERSE at 2026-08-21        admitted 186 of 610   (409 fail U1 coverage, 15 fail U2)
DERIVED HORIZONS              shortest RAVEUSDT   h*   1.06 d
                              median   SPXUSDT    h*  30.00 d
                              longest  BTCUSDC    h* 188.76 d
EFFECTIVE BETS                3.27 of 186     rho_bar +0.2318   lambda_max 56.80
```

**186 crypto perpetuals carry 3.27 independent bets.** That is the load-bearing number. The
cross-section is not 186 wide; it is three wide, because everything moves together (§336's common
market clock, measured here from a different direction and agreeing).

Feasibility follows, with `S_annual = √(365/h*)·k·f/2` and `t = S_annual·√years`:

```
                h* (d)     S_annual     years to t = 2
RAVEUSDT          1.06       0.0645              961
SPXUSDT          30.00       0.0121           27,100
POOLED           30.00       0.0220            8,276
```

**The inversion is the decision-relevant figure, and it does not depend on `f_design`** — `h* ∝ 1/f²`
makes `S_annual ∝ f²`, so time-to-verdict scales as `f⁻⁴`, and inverting is `f_design`-free algebra:

```
verdict in 1 year   requires pooled capture  f >= 9.54 %
verdict in 2 years                           f >= 8.02 %
verdict in 5 years                           f >= 6.38 %

every single-leg capture this estate has measured   1-2 %   (A-S14)
best dark-family cell measured                      2.09 %  (A-S43)
```

**A pooled cross-sectional capture test is not runnable on this estate at any capture this estate
has ever measured — short by roughly a factor of five in `f`, which is four orders of magnitude in
time.** §463 reached the same wall by observation (201 days gives `t = 0.65`); this reaches it by
construction, from the frontier's own closed form, and says by how much.

**This does not weaken the prereg; it is what the prereg is for.** A preregistration whose power
calculation is done honestly *before* the outcome is exactly the instrument that stops a decade of
accrual from being spent on an undetectable effect. `S6` already forbade accruing to significance.
`8b` says the accrual would not get there.

**One clause is added to the stop rule as a result:**

```
S0  DO NOT START.  This prereg is frozen in the RUNNABLE-BUT-NOT-RUN state.  It is
    executed if and only if a pooled capture at or above 6.38% is demonstrated by some
    OTHER route first -- at which point the years-to-verdict falls inside a human span
    and S1..S6 take over unchanged.  Absent that, freezing it is the whole deliverable:
    the machine exists, the parameters are dead, and no one has to re-derive them.
```

---

## 9 · What a failure means, and what it does not

```
PASS               f > 0 established cross-sectionally on fresh data at N_required.
                   This is NOT a strategy.  It licenses ONE thing: a rule-level
                   preregistration, with its own multiplicity budget.
FAIL (t < 2.0)     "not established at this N."  It does NOT establish f = 0.
                   The MDE is f_design = 1.0%; anything smaller was never in scope.
NEGATIVE (t <= -2) recorded as a finding in its own right and NOT traded inverted.
                   §463 measured what happens when an effect is large enough to look
                   tradeable in either direction: it is 19-64x larger than the frontier
                   permits an alpha to be, and is therefore mechanics.  Any negative
                   result here is checked against the frontier's ceiling FIRST.
```

---

## 10 · What would falsify this design itself

Stated so a reader can attack the design rather than the result:

- **If capture varies with horizon** (`A-S40` measured `p ≈ −0.5` for order flow), then a single
  `f` at a single `h*` is a point on a curve, not a constant. This design is valid at its own
  horizon and says nothing about others. **`p` is unmeasured for every family except order flow.**
- **If the universe rule admits a symbol whose price series is stitched across ingests**, U5 fails
  silently. `A-S36` found exactly that defect after five sections had used the panel.
- **If `k` differs materially at the evaluation horizon** — `A-S39` measured `k` rising with horizon
  as `h^0.0339` — the frozen `k` is wrong by that factor and `N_required` with it.
- **If the effective-bet count is materially above the 3.27 measured in §8b**, `N_required` is
  reached sooner and the stop rule is conservative -- the safe direction. The first draft of the
  evaluator ASSUMED 8; the measurement more than halved it, and the assumption was optimistic.

---

## 11 · Freeze block

```
STATUS                 FROZEN
f_design               0.010                       (§4, derived)
k                      0.6966                      (§467; re-measure at freeze)
N_required             329,726                     (§8, from f_design and k)
bar                    t >= 2.0 one-sided AND the K=1 FST bar
cost basis             BINANCE_BASE unless OD-033 answered   <- record which at freeze
Q_intended             NOT SUPPLIED -> L defaults to $5,000,000/day
evaluation window      NOT ALLOCATED.  This file does not claim one.
                       Window allocation is an operator decision (cf. OD-031), and the
                       forward-lane reservations in §400/§408 are not touched here.
sha256 of this file    6bac365a88a4782ca86716eaed8a193b425fa4bd20aa141ca3a55dc707ace754
frozen at              2026-08-27T03:22:40Z
```

**The hashed subject is this file's text UP TO the line `## 11 · Freeze block`** --
everything above, nothing below. `sha256(body)` reproduces the value above exactly.

**Until `STATUS` reads `FROZEN` and the hash is filled, no outcome may be read and nothing in this
file is binding.**
