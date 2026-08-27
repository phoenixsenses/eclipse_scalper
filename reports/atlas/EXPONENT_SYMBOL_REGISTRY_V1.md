# EXPONENT SYMBOL REGISTRY V1 — keyed on the object, never the letter

`SEVEN_LETTERS_CARRY_FIFTEEN_OBJECTS_FIVE_ARE_OVERLOADED` · built 2026-08-27T03:24:08Z · lane C, `C-T25`

> three rounds found the same failure -- p carries three objects, zeta two, gamma two -- and the atlas index cannot see any of it, because both objects emit tokens containing the same letter. A token index keys on STRINGS; a collision is a fact about MEANINGS.

**7 letters carry 15 distinct objects; 5 letters carry more than one.**

**The case for this file.** CT-016 pitted A's exponential fill curve against C's power law and stayed open a day. C-T22 closed it by showing the abscissae differ: A's delta is DEPTH in bps (Cartea Eq. 8.1), C's x is QUEUE POSITION. That was a symbol collision, and it cost two lanes a day and a register entry.

**And this lane is implicated.** section 478, written by this lane, used `zeta` for the Hill tail exponent of returns -- Bouchaud's own usage for that quantity, and a THIRD object under a letter this lane had already found carrying two.

---

## The registry

| letter | object | definition | conditions on | measured | owner | shares its letter with |
|---|---|---|---|---|---|---|
| `zeta` | **ZETA_WINDOW_IMBALANCE** | outer-region exponent of R against |dV| over windows of T trades | net imbalance of ALL participants in a window | 0.416 / 0.439 / 0.495 | A-S30 | ZETA_SINGLE_ORDER_SIZE, ZETA_RETURN_TAIL |
| `zeta` | **ZETA_SINGLE_ORDER_SIZE** | R(v,1) = A (v/V_best)^zeta <s>, Bouchaud Eq. 11.7 | ONE market order's size | 0.166 / 0.230 / 0.262 at 600 s; 0.63-0.72 at lag-1 and mechanical | C-T20 | ZETA_WINDOW_IMBALANCE, ZETA_RETURN_TAIL |
| `zeta` | **ZETA_RETURN_TAIL** | tail exponent of the unconditional return distribution, P(|r|>x) ~ x^-zeta | nothing -- an unconditional distributional exponent | Hill 2.33-3.83 across k on 60-minute moves; Bouchaud reports ~3 universally | section 478 | ZETA_WINDOW_IMBALANCE, ZETA_SINGLE_ORDER_SIZE |
| `gamma` | **GAMMA_METAORDER_IMPACT** | concavity of price response in metaorder size, I ~ Q^gamma | a metaorder, requiring child-to-parent identity | NOT IDENTIFIABLE on anonymised data (Bouchaud 12.2) | C-T24 | GAMMA_LMF_SIGN_MEMORY |
| `gamma` | **GAMMA_LMF_SIGN_MEMORY** | decay of the order-sign autocorrelation, C(l) ~ l^-gamma_LMF | trade signs only -- no identity needed | 0.7746 / 0.7892 / 0.2092 (SOL an aggregation artefact) | C-T24 | GAMMA_METAORDER_IMPACT |
| `delta` | **DELTA_CASCADE_IMPACT** | dP = k Q^delta on cascade episodes | a whole cascade episode, simultaneous aggregate | 0.684 / 0.666 / 0.696 | C-T20 | DELTA_QUOTE_DEPTH |
| `delta` | **DELTA_QUOTE_DEPTH** | depth in bps from the mid, the abscissa of Cartea Eq. (8.1) exp(-kappa*delta) | a price level, not a quantity | A-S45 fitted kappa ~ 0.0097/bps over an hour | A-S45 | DELTA_CASCADE_IMPACT |
| `kappa` | **KAPPA_RESPONSE_T_EXPONENT** | R(dV,T) = R(1) T^kappa F(dV/(V_D T^chi)) -- prefactor exponent in the collapsed scaling form | a scaled imbalance argument held fixed | not reported alone; enters as kappa-chi = 0.25-0.30 | A-S30, C-T21 | KAPPA_UNCONDITIONAL_RESPONSE, KAPPA_FILL_DECAY_RATE |
| `kappa` | **KAPPA_UNCONDITIONAL_RESPONSE** | d log R / d log T with no scaling collapse | nothing held fixed | 0.6507 / 0.5782 / 0.5209 | C-T23 | KAPPA_RESPONSE_T_EXPONENT, KAPPA_FILL_DECAY_RATE |
| `kappa` | **KAPPA_FILL_DECAY_RATE** | rate constant in Cartea Eq. (8.1), P(fill) = exp(-kappa*delta) | depth in bps | 0.0097/bps (A-S45); 0.00956 re-derived (C-T22) | A-S45 | KAPPA_RESPONSE_T_EXPONENT, KAPPA_UNCONDITIONAL_RESPONSE |
| `chi` | **CHI_VOLUME_SCALE** | exponent of the volume normaliser inside F(dV/(V_D T^chi)) | aggregation window length | 0.6498 / 0.6817 / 0.5902 as sd(dV) ~ T^chi | C-T23 | — |
| `p` | **P_PREDICTOR_CAPTURE_DECAY** | f(h) = R(h)/E|r|(h) ~ h^p for a real predictor | a forecast formed before the horizon | -0.409 / -0.495 / -0.508 | A-S40 | P_CONTEMPORANEOUS_FLOW_RATIO, P_LAGGED_FLOW_RATIO |
| `p` | **P_CONTEMPORANEOUS_FLOW_RATIO** | same formula, flow and return measured over the SAME window | nothing lagged | -0.026 / -0.014 / +0.000 | C-T23 | P_PREDICTOR_CAPTURE_DECAY, P_LAGGED_FLOW_RATIO |
| `p` | **P_LAGGED_FLOW_RATIO** | same formula, signal from a prior window, response measured forward | a lag | +0.215/+0.009 at h<=16; -0.721/-0.785 at h>=256 | C-T23 | P_PREDICTOR_CAPTURE_DECAY, P_CONTEMPORANEOUS_FLOW_RATIO |
| `alpha` | **ALPHA_METAORDER_SIZE_TAIL** | Pareto tail exponent of the metaorder size distribution | metaorder sizes | 1.775 / 1.789 as UPPER BOUNDS via LMF; book: equities ~1.5, Bitcoin ~1.10 | C-T24 | — |

### Notes that matter

- **ZETA_WINDOW_IMBALANCE** — A itself holds ZETA_IS_NOT_DELTA
- **ZETA_SINGLE_ORDER_SIZE** — book range 0-0.3
- **ZETA_RETURN_TAIL** — Bouchaud's own usage; the lane writing this registry used the letter for a third object
- **GAMMA_METAORDER_IMPACT** — C-T20's 0.373/0.369 was indirect via Eq. 16.16 and withdrawn by C-T21; its bias sign is known (aggregate substitution underestimates)
- **GAMMA_LMF_SIGN_MEMORY** — alpha_metaorder_size = gamma_LMF + 1 under the LMF model
- **DELTA_CASCADE_IMPACT** — A holds DELTA_IS_ASSUMED_NOT_MEASURED and DELTA_IS_NOT_MEASURABLE_ON_PUBLIC_DATA for its own delta
- **DELTA_QUOTE_DEPTH** — THIS COLLISION COST A DAY: CT-016 pitted A's exponential against C's power law until C-T22 showed A's abscissa is DEPTH and C's is QUEUE POSITION
- **KAPPA_RESPONSE_T_EXPONENT** — confirmed three times as the DIFFERENCE kappa-chi
- **KAPPA_UNCONDITIONAL_RESPONSE** — NOT the collapsed-scaling kappa; this is why C-T23's kappa-chi differs from A-S30's and C-T21's
- **KAPPA_FILL_DECAY_RATE** — a RATE with units 1/bps, not a dimensionless exponent
- **CHI_VOLUME_SCALE** — p - (kappa-chi) = chi - alpha_E|r| exactly
- **P_PREDICTOR_CAPTURE_DECAY** — A-S40 itself: a single fitted p ~ -0.5 is the AVERAGE OF A TRANSITION
- **P_CONTEMPORANEOUS_FLOW_RATIO** — shares A's formula and is not A's object
- **P_LAGGED_FLOW_RATIO** — reproduces A-S40's transition on independent windows
- **ALPHA_METAORDER_SIZE_TAIL** — SOL's 1.209 is an aggregation artefact and must not be compared with the book

---

## Read-only audit of lane A's frozen preregistration

Artifact `reports/atlas/LANE_A_PREREG_V1.md`, status **FROZEN**, frozen 2026-08-27T03:21:20Z. **Read-only:** charter rule 5 -- a lane may contradict another, never silently overwrite.

### Verdict: `NO_MISUSE_OF_AN_EXPONENT_IN_LANE_A_PREREG`

- the exponent appears in section 10, the FALSIFICATION section, not in the machinery
- it is attributed to A-S40
- it is scoped by family -- 'for order flow'
- the design's claim is explicitly restricted to its own horizon

**C-T23's warning was already satisfied before it was written.**

### Two things the frozen text cannot know

**p is not a constant within order flow either**

- measured: C-T23 on independent windows: +0.215/+0.009 at h<=16, -0.721/-0.785 at h>=256
- effect: the clause's DIRECTION is right; its MAGNITUDE is unbounded, because near the transition the local exponent moves fast. A design at one h* is unaffected; a reader inferring a sensitivity from 'p ~ -0.5' is not.

**'p is unmeasured for every family except order flow' is superseded**

- measured: C-T23 measured p on two further constructions: the contemporaneous flow-response ratio (~0) and the lagged one
- effect: more is measured than the frozen text knew, and what is measured shows the single-number reading is the one thing it should not be

> neither item is a defect in the design; both are facts a future reader of a frozen file would otherwise take from it and carry away wrong

*Hash check:* a self-referential hash cannot match its own file once written -- the declared digest covers the file as it stood before the line was filled in. Recorded, not treated as a defect.

## What is NOT claimed

- That the registry is complete. It covers the exponent family this lane has measured or read; other families will have their own collisions.
- That any measurement changes. Nothing is re-measured here; entries cite the sections that own them.
- That lane A's prereg needs editing. It is frozen and it passes the audit.

```verdict
SEVEN_LETTERS_CARRY_FIFTEEN_OBJECTS_FIVE_ARE_OVERLOADED
NO_MISUSE_OF_AN_EXPONENT_IN_LANE_A_PREREG
P_IS_NOT_A_CONSTANT_WITHIN_ORDER_FLOW_EITHER
P_UNMEASURED_EXCEPT_ORDER_FLOW_IS_SUPERSEDED
SYMBOL_COLLISION_IS_INVISIBLE_TO_A_TOKEN_INDEX
ZETA_KAPPA_AND_P_EACH_CARRY_THREE_OBJECTS
CT_016_WAS_A_SYMBOL_COLLISION_AND_IT_COST_A_DAY
THIS_LANE_IS_IMPLICATED_ZETA_HAS_THREE_OBJECTS
```
