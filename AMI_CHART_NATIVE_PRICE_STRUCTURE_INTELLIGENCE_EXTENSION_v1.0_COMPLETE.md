# AMI Chart-Native Price Structure Intelligence Extension

## Candle, Swing, Sweep, Breakout, Compression and Human-Observation Research Specification

**Document ID:** `AMI-CHART-STRUCTURE-EXT-0001`  
**Version:** `1.0.0`  
**Status:** `PROPOSED CANONICAL EXTENSION / RESEARCH-ONLY / AGENT HANDOFF`  
**Target parent:** `AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.3_COMPLETE`  
**Proposed question range:** `Q867–Q1058`  
**Date:** `2026-07-03`  
**Default mode:** `ORDERLESS / FORWARD-OBSERVATION / NO LIVE MUTATION`

---

# 0. Purpose

The existing AMI architecture already covers:

- LONG and SHORT as connected structural phases;
- pre-event and post-event research;
- event timing;
- signal aging;
- scalp, intraday and swing separation;
- failed-fade LONG;
- early, T0, delayed and late SHORT;
- multi-timeframe conflict;
- execution feasibility;
- evidence independence;
- forward validation;
- epistemic governance.

The missing layer is not a lack of market questions.

The missing layer is a sufficiently explicit and operational representation of what an experienced human sees directly on a chart:

```text
candle quality
wick rejection
close location
push count
momentum geometry
swing grammar
sweep anatomy
breakout acceptance
retest quality
compression shape
channel behavior
relative strength
session opening structure
setup cancellation
visual pattern evolution
```

This extension converts those visual observations into:

```text
timestamped chart observations
→ measurable features
→ deterministic pattern states
→ orderless observer routes
→ forward outcomes
→ research questions
→ preregistered experiments
→ governed knowledge
```

The objective is not to teach AMI named chart patterns as folklore.

The objective is to make visual price structure measurable, falsifiable, forward-safe and compatible with AMI’s existing evidence hierarchy.

---

# 1. Core Principle

A human statement such as:

> “The move looked exhausted before the event.”

is not yet a feature.

AMI must translate it into observable components:

```text
third push produced less displacement
upper-wick ratio increased
close location deteriorated
overlap increased
time-to-new-high increased
relative strength weakened
breakout acceptance failed
```

The governing transformation is:

```text
VISUAL IMPRESSION
→ ATOMIC OBSERVATIONS
→ NUMERIC FEATURES
→ STRUCTURE STATE
→ HYPOTHESIS
→ FROZEN OBSERVER
→ FORWARD EVIDENCE
```

No visual claim becomes alpha merely because it appears obvious in hindsight.

---

# 2. Safety and Governance Boundaries

This extension must not:

```text
place orders
change live routes
change shadow routes
change leverage
change sizing
change stop or take-profit behavior
change .env
grant live permission
auto-promote a discovered pattern
retroactively label a pattern as forward evidence
use a future event to select a pre-event setup
```

All outputs must remain:

```text
RESEARCH_ONLY
ORDERLESS
FORWARD_OBSERVATION
NO_ORDER_EFFECT
```

Every feature, pattern and observer must include:

```yaml
feature_version:
pattern_version:
observer_version:
schema_version:
activation_timestamp:
code_commit:
source_hash:
known_at_ts:
available_at_ts:
```

---

# 3. Chart-Native Research Architecture

```text
RAW PRICE / VOLUME / BOOK / FLOW
                ↓
        CANDLE NORMALIZER
                ↓
          SWING EXTRACTOR
                ↓
          LEVEL REGISTRY
                ↓
     CHART FEATURE FACTORY
                ↓
      PATTERN STATE ENGINE
                ↓
       SETUP LIFECYCLE ENGINE
                ↓
         OBSERVER ROUTES
                ↓
        POSITION PATH LEDGER
                ↓
        FORWARD OUTCOMES
                ↓
 QUESTION / EXPERIMENT / KNOWLEDGE
```

---

# 4. Canonical Objects

## 4.1 Candle Object

```yaml
candle_id:
symbol:
venue:
timeframe:
open_ts:
close_ts:
open:
high:
low:
close:
volume:
trade_count:
taker_buy_volume:
taker_sell_volume:
is_closed:
partial_status:
data_quality:
source_hash:
```

## 4.2 Swing Object

```yaml
swing_id:
symbol:
timeframe:
swing_type: HIGH | LOW
pivot_ts:
pivot_price:
confirmation_ts:
confirmation_method:
prominence_bps:
duration_bars:
left_strength:
right_strength:
known_at_ts:
```

`pivot_ts` and `known_at_ts` must be separate.  
A swing high is not knowable at its exact peak until the confirmation rule is satisfied.

## 4.3 Level Object

```yaml
level_id:
level_type:
price:
origin_ts:
known_at_ts:
timeframe:
touch_count:
rejection_count:
acceptance_count:
last_touch_ts:
strength_score:
source_type:
```

Possible `level_type` values:

```text
PREVIOUS_DAY_HIGH
PREVIOUS_DAY_LOW
PREVIOUS_WEEK_HIGH
PREVIOUS_WEEK_LOW
SESSION_HIGH
SESSION_LOW
OPENING_RANGE_HIGH
OPENING_RANGE_LOW
SWING_HIGH
SWING_LOW
EQUAL_HIGH_ZONE
EQUAL_LOW_ZONE
BREAKOUT_LEVEL
VWAP
ANCHORED_VWAP
VOLUME_PROFILE_HVN
VOLUME_PROFILE_LVN
CHANNEL_BOUNDARY
TRENDLINE
CUSTOM_OPERATOR_LEVEL
```

## 4.4 Chart Pattern Object

```yaml
pattern_id:
pattern_family:
pattern_subtype:
symbol:
timeframe:
start_ts:
detected_ts:
known_at_ts:
end_ts:
direction_hypothesis:
confidence_descriptive:
component_features:
invalidation_rule:
status:
pattern_version:
record_type:
```

## 4.5 Setup Object

```yaml
setup_id:
pattern_id:
structural_cycle_id:
direction:
horizon_class:
setup_state:
detected_ts:
armed_ts:
triggered_ts:
cancelled_ts:
expired_ts:
completed_ts:
entry_reference:
invalidation_reference:
cancellation_reason:
observer_version:
```

---

# 5. Setup Lifecycle State Machine

```text
DISCOVERED
→ FORMING
→ ARMED
→ TRIGGERED
→ ACTIVE
→ COMPLETED
```

Alternative branches:

```text
FORMING → CANCELLED_BEFORE_ENTRY
ARMED → STALE
ARMED → INVALIDATED
TRIGGERED → FAILED
TRIGGERED → EXPIRED
```

Canonical states:

```text
DISCOVERED
FORMING
ARMED
TRIGGERED
ACTIVE
COMPLETED
CANCELLED_BEFORE_ENTRY
INVALIDATED
STALE
EXPIRED
FAILED
UNKNOWN
```

A setup that disappears before entry is valuable evidence and must not be deleted.

---

# 6. Candle and Close-Location Morphology

## 6.1 Atomic candle features

For each closed candle:

```text
range = high - low
body = abs(close - open)
upper_wick = high - max(open, close)
lower_wick = min(open, close) - low
```

Normalized:

```text
body_ratio = body / range
upper_wick_ratio = upper_wick / range
lower_wick_ratio = lower_wick / range
close_location_value = (close - low) / range
open_location_value = (open - low) / range
directional_body = (close - open) / range
```

Additional features:

```text
range_zscore
volume_zscore
trade_count_zscore
taker_imbalance
close_vs_vwap
close_vs_level
range_vs_ATR
gap_from_previous_close
```

## 6.2 Multi-candle morphology

```text
consecutive_upper_wick_count
consecutive_lower_wick_count
body_compression_rate
range_compression_rate
close_location_trend
wick_asymmetry_trend
follow_through_ratio
expansion_retracement_ratio
inside_bar_count
outside_bar_count
overlap_ratio
```

## 6.3 Close quality labels

```text
CLOSE_NEAR_HIGH
CLOSE_NEAR_LOW
MID_RANGE_CLOSE
REJECTION_CLOSE
ACCEPTANCE_CLOSE
INDECISION_CLOSE
FOLLOW_THROUGH_CONFIRMED
FOLLOW_THROUGH_FAILED
```

Thresholds must be versioned and selected on training data only.

---

# 7. Multi-Push and Momentum Geometry

A price move must be represented as a sequence of pushes, not a single return.

## 7.1 Push Object

```yaml
push_id:
direction:
start_swing_id:
end_swing_id:
start_ts:
end_ts:
displacement_bps:
duration_seconds:
bars:
path_length_bps:
efficiency_ratio:
volume:
liquidation_notional:
average_speed:
peak_speed:
acceleration:
pullback_after_bps:
known_at_ts:
```

## 7.2 Push efficiency

```text
efficiency_ratio =
absolute displacement /
sum of absolute bar-to-bar movement
```

High efficiency:

```text
directional, low overlap
```

Low efficiency:

```text
choppy, high overlap
```

## 7.3 Momentum decay

For push `n` versus push `n-1`:

```text
displacement_decay
speed_decay
volume_efficiency_decay
impact_per_liquidation_decay
time_to_new_extreme_increase
overlap_increase
wick_rejection_increase
```

## 7.4 Push sequence labels

```text
FIRST_PUSH
SECOND_PUSH
THIRD_PUSH
EXTENDED_PUSH
PARABOLIC_TERMINAL_PUSH
DECAYING_PUSH
FAILED_PUSH
REACCELERATING_PUSH
```

---

# 8. Swing Grammar

Individual higher highs and lower lows are insufficient. AMI must represent sequences.

## 8.1 Canonical swing tokens

```text
HH
HL
LH
LL
EH  equal high
EL  equal low
SH  swept high
SL  swept low
RH  reclaimed high
RL  reclaimed low
```

## 8.2 Example grammar patterns

```text
HH → shallow HL → strong HH
HH → deep HL → weak HH
HH → EH → sweep → close below
LL → failed LL → reclaim
LH → compression → breakdown
HL → compression → breakout
LL → shallow LH → new LL
LL → deep LH → failed LL
```

## 8.3 Grammar attributes

```text
retracement_depth
swing_duration
swing_displacement
progress_per_bar
swing_overlap
distance_to_major_level
relative_volume
relative_strength
state_age
```

## 8.4 Pattern status

A swing grammar pattern must be identified using only confirmed swings known at that time.

---

# 9. Liquidity Sweep Anatomy

A sweep is not defined only as “price crossed a high or low.”

## 9.1 Sweep components

```text
liquidity_zone
penetration_bps
wick_only_or_body
time_outside_level
close_back_inside
volume_at_sweep
taker_imbalance_at_sweep
liquidation_at_sweep
reclaim_speed
first_pullback_depth
second_retest_depth
post-sweep displacement
```

## 9.2 Sweep types

```text
WICK_SWEEP
BODY_BREAK_AND_REJECT
SLOW_AUCTION_ABOVE_THEN_FAIL
FAST_STOP_RUN
DOUBLE_SWEEP
SESSION_SWEEP
PREVIOUS_DAY_LEVEL_SWEEP
EQUAL_HIGH_SWEEP
EQUAL_LOW_SWEEP
FAILED_SWEEP
TRUE_BREAKOUT_AFTER_SWEEP
```

## 9.3 Sweep versus breakout discriminator

Candidate features:

```text
time_above_level
close_count_above_level
volume_acceptance
retest_hold
range_expansion_after_break
book_refill_on_retest
relative_strength_after_break
return_below_level_speed
```

---

# 10. Breakout and Retest Quality

## 10.1 Breakout features

```text
breakout_displacement_bps
breakout_range_vs_ATR
close_distance_beyond_level
number_of_closes_beyond_level
time_beyond_level
volume_expansion
trade_count_expansion
flow_confirmation
OI_confirmation
BTC_confirmation
```

## 10.2 Retest features

```text
retest_number
retest_delay_seconds
retest_depth_bps
retest_depth_fraction
time_at_level
volume_contraction
counterflow_intensity
wick_rejection
close_location
post-retest_progress
```

## 10.3 Acceptance labels

```text
NO_ACCEPTANCE
WEAK_ACCEPTANCE
TEMPORARY_ACCEPTANCE
STRONG_ACCEPTANCE
ACCEPTANCE_THEN_FAILURE
```

## 10.4 Breakout states

```text
BREAKOUT_FORMING
BREAKOUT_CONFIRMED
BREAKOUT_ACCEPTED
BREAKOUT_RETESTING
BREAKOUT_FAILED
BREAKOUT_EXHAUSTED
```

LONG and SHORT breakout families must be tested separately.

---

# 11. Compression Taxonomy

Compression must be decomposed into shape and internal quality.

## 11.1 Compression shapes

```text
HORIZONTAL_RANGE_COMPRESSION
ASCENDING_BASE_COMPRESSION
DESCENDING_CEILING_COMPRESSION
SYMMETRIC_COIL
TREND_FLAG
VOLATILITY_CONTRACTION
WEDGE_COMPRESSION
IRREGULAR_NOISY_COMPRESSION
FAKE_COMPRESSION
```

## 11.2 Compression metrics

```text
range_width_decay
ATR_decay
realized_volatility_decay
high_slope
low_slope
apex_distance
touch_count
false_break_count
volume_decay
trade_count_decay
book_instability
flow_disagreement
```

## 11.3 Compression quality

```text
CLEAN
ORDERLY
ASYMMETRIC
NOISY
UNSTABLE
FAKE
```

## 11.4 Release metrics

```text
release_direction
release_speed
release_efficiency
volume_expansion
acceptance
retest_quality
false_release
```

---

# 12. Trendline and Channel Behavior

Trendlines must be algorithmic and versioned, not drawn differently after the outcome.

## 12.1 Trendline object

```yaml
trendline_id:
anchor_1:
anchor_2:
construction_method:
known_at_ts:
slope:
touch_count:
violation_count:
validity_score:
```

## 12.2 Channel object

```yaml
channel_id:
midline:
upper_boundary:
lower_boundary:
slope:
width:
construction_method:
known_at_ts:
```

## 12.3 Channel features

```text
upper_touch_count
lower_touch_count
midline_hold_count
midline_reject_count
overshoot_bps
time_outside_channel
reentry_speed
slope_acceleration
slope_deceleration
channel_width_change
```

## 12.4 Channel events

```text
BOUNDARY_TOUCH
BOUNDARY_REJECTION
BOUNDARY_ACCEPTANCE
OVERSHOOT_AND_REENTRY
MIDLINE_HOLD
MIDLINE_FAILURE
CHANNEL_BREAK
CHANNEL_BREAK_RETEST
```

---

# 13. Relative-Strength Chart Intelligence

BTC synchronization alone is insufficient. AMI must represent relative performance.

## 13.1 Relative-strength pairs

```text
ETH vs BTC
SOL vs BTC
asset vs sector basket
asset vs total crypto market
asset spot vs perpetual
venue A vs venue B
```

## 13.2 Features

```text
relative_return_1m
relative_return_5m
relative_return_15m
relative_return_1h
relative_swing_state
relative_breakout_state
relative_reclaim_lead_seconds
relative_drawdown
beta_adjusted_residual
correlation_break
```

## 13.3 Visual relative-strength states

```text
LEADING_UP
LEADING_RECOVERY
HOLDING_WHILE_BTC_FALLS
FAILING_TO_CONFIRM_BTC_HIGH
UNDERPERFORMING
RELATIVE_BREAKDOWN
DECOUPLED
UNKNOWN
```

## 13.4 Lead-lag questions

The system must distinguish:

```text
asset leads BTC
BTC leads asset
simultaneous
no stable lead
```

---

# 14. Session and Opening Structure

Clock time must be connected to session-native chart structures.

## 14.1 Session objects

```text
ASIA
LONDON
US_PREOPEN
US_OPEN
US_MIDDAY
US_CLOSE
WEEKEND
FUNDING_WINDOW
DAILY_ROLLOVER
```

## 14.2 Opening structures

```text
ASIA_RANGE
ASIA_RANGE_SWEEP
LONDON_BREAKOUT
LONDON_FAILED_BREAKOUT
US_OPEN_REVERSAL
US_OPEN_CONTINUATION
OPENING_RANGE_BREAKOUT
OPENING_RANGE_FAILURE
PREVIOUS_SESSION_RECLAIM
SESSION_HANDOFF_FAILURE
```

## 14.3 Opening range features

```text
opening_range_minutes
opening_range_width
break_direction
break_speed
acceptance_time
retest_depth
volume_expansion
relative_strength
previous_session_context
```

---

# 15. Unconditional SHORT Genesis

The existing unconditional LONG-genesis logic must be mirrored with an independently designed SHORT-genesis programme.

This is not created by inverting LONG thresholds.

## 15.1 Candidate SHORT genesis families

```text
distribution before liquidation
failed high
weak reclaim
range breakdown
relative-strength failure
third-push exhaustion
session sweep failure
breakout acceptance failure
channel overshoot and reentry
compression release down
OI expansion without price acceptance
crowded funding plus failed continuation
```

## 15.2 Required controls

```text
all timestamps
matched non-event timestamps
event-never-arrived timestamps
same-session controls
same-regime controls
opposite-direction controls
```

---

# 16. Setup Cancellation Before Entry

A setup may fail before a trade exists.

## 16.1 Cancellation reasons

```text
EXPECTED_LH_NOT_FORMED
EXPECTED_HL_NOT_FORMED
NEW_EXTREME_INVALIDATED
RECLAIM_FAILED
ACCEPTANCE_OPPOSITE_DIRECTION
PULLBACK_TOO_DEEP
CONFIRMATION_TOO_LATE
SIGNAL_STALE
VOLUME_DIED
FLOW_REVERSED
RELATIVE_STRENGTH_FLIPPED
REGIME_CHANGED
DATA_BECAME_STALE
```

## 16.2 Cancellation metrics

```text
time_from_detection_to_cancel
maximum hypothetical favorable excursion before cancel
maximum hypothetical adverse excursion before cancel
event_arrival_after_cancel
post-cancel direction
avoided_loss
missed_move
```

## 16.3 Why cancellations matter

Without cancellation records, research keeps only patterns that survived long enough to trigger. That creates survivorship bias.

---

# 17. Chart Observation → Measurable Feature Registry

This is the bridge between human chart intuition and AMI research.

## 17.1 Chart Observation record

```yaml
chart_observation_id:
observer_id:
created_at:
symbol:
venue:
timeframe:
observation_ts:
screenshot_capture_ts:
direction_hypothesis:
expected_horizon:
visible_pattern:
free_text_reason:
invalidation:
confidence_self_reported:
screenshot_hash:
chart_settings:
known_future_hidden:
linked_event_id:
linked_cycle_id:
status:
```

## 17.2 Measurable translation

```yaml
translation_id:
chart_observation_id:
atomic_claim:
candidate_feature:
formula:
required_data:
availability_time:
expected_sign:
alternative_explanation:
negative_control:
implementation_status:
```

## 17.3 Observation statuses

```text
UNREVIEWED
TRANSLATION_REQUIRED
FEATURE_MAPPED
DATA_BLOCKED
OBSERVER_READY
PREREG_READY
FALSIFIED
SUPPORTED_DESCRIPTIVE
ARCHIVED
```

## 17.4 Human annotation rule

Human chart observations may generate hypotheses.

They do not count as independent evidence unless:

- timestamped before the outcome;
- future candles are hidden;
- screenshot and chart settings are hashed;
- expected horizon and invalidation are recorded;
- the observer is evaluated on future observations.

---

# 18. Additional Chart-Native Families

## 18.1 Volume and participation coupling

```text
high volume + no progress
low volume + strong progress
volume climax
trade-count climax
large-trade concentration
participation breadth
```

## 18.2 Auction inefficiency and repair

Candidate objects:

```text
gap
fast single-print region
low-volume node traversal
fair-value-gap-like imbalance
unfinished auction proxy
revisit and fill behavior
```

These must be defined quantitatively and not imported as discretionary terminology.

## 18.3 Distance and extension

```text
distance_from_VWAP
distance_from_anchored_VWAP
distance_from_ATR_band
distance_from_channel
distance_from_volume_node
distance_from_session_open
```

## 18.4 Pattern conflict

Example:

```text
5m sweep SHORT
1h accepted breakout LONG
4h mature uptrend
```

The system must permit:

```text
NO_TRADE
SCALP_ONLY
WAIT_FOR_CONFIRMATION
```

---

# 19. Feature Dictionary

At minimum, implement the following feature families.

## Candle

```text
body_ratio
upper_wick_ratio
lower_wick_ratio
close_location_value
directional_body
range_zscore
volume_zscore
follow_through_ratio
expansion_retracement_ratio
overlap_ratio
```

## Push

```text
push_number
push_displacement_bps
push_duration
push_speed
push_acceleration
push_efficiency
displacement_decay
speed_decay
impact_decay
overlap_increase
```

## Swing

```text
swing_token
retracement_depth
swing_progress
time_to_new_extreme
swing_duration
swing_overlap
equal_high_distance
equal_low_distance
```

## Sweep

```text
penetration_bps
time_outside_level
close_back_inside
reclaim_speed
sweep_volume_zscore
post_sweep_displacement
first_retest_depth
second_retest_depth
```

## Breakout/retest

```text
breakout_close_distance
closes_beyond_level
time_beyond_level
volume_expansion
retest_delay
retest_depth
retest_volume_contraction
acceptance_score
```

## Compression

```text
range_width_decay
ATR_decay
high_slope
low_slope
apex_distance
touch_count
false_break_count
compression_quality
```

## Channel

```text
channel_slope
channel_width
touch_count
overshoot_bps
reentry_speed
midline_behavior
```

## Relative strength

```text
beta_adjusted_residual
relative_return
relative_reclaim_lead
relative_swing_divergence
correlation_break
```

## Session

```text
session_range_position
opening_range_position
previous_session_level_distance
session_sweep_state
session_handoff_state
```

---

# 20. Observer Families

All observer families are hypothetical.

## LONG

```text
CANDLE_REJECTION_LONG
FAILED_BREAKDOWN_LONG
SWEEP_RECLAIM_LONG
ASCENDING_COMPRESSION_LONG
BREAKOUT_RETEST_LONG
CHANNEL_REENTRY_LONG
RELATIVE_STRENGTH_LONG
SESSION_SWEEP_LONG
THIRD_PUSH_EXHAUSTION_LONG
```

## SHORT

```text
CANDLE_REJECTION_SHORT
FAILED_BREAKOUT_SHORT
SWEEP_REJECTION_SHORT
DESCENDING_COMPRESSION_SHORT
BREAKDOWN_RETEST_SHORT
CHANNEL_REENTRY_SHORT
RELATIVE_WEAKNESS_SHORT
SESSION_SWEEP_SHORT
THIRD_PUSH_EXHAUSTION_SHORT
UNCONDITIONAL_STRUCTURE_SHORT
```

## Cancellation observers

```text
SETUP_CANCELLED_BEFORE_ENTRY
SETUP_STALE_BEFORE_ENTRY
CONFIRMATION_TOO_LATE
```

## Management observers

```text
FIRST_RETEST_EXIT
ACCEPTANCE_FAILURE_EXIT
CHANNEL_REENTRY_EXIT
RELATIVE_STRENGTH_FLIP_EXIT
SWEEP_FAILURE_EXIT
```

---

# 21. Outcome Horizons

Each observer must be evaluated at:

```text
30s
1m
3m
5m
10m
15m
30m
45m
60m
90m
2h
4h
6h
12h
24h
```

Required path outputs:

```text
MFE
MAE
time_to_MFE
time_to_MAE
time_to_first_positive
time_underwater
new_extreme_probability
reclaim_probability
giveback
best_observed_horizon
```

Scalp, intraday and swing variants must remain separate route families.

---

# 22. Matched Controls and Hindsight Protection

## 22.1 Candidate generation

Patterns must be generated across all eligible timestamps, not only before known events.

## 22.2 Controls

```text
same symbol, same hour, no event
same regime, random timestamp
time-shifted pattern
opposite direction
pattern component removed
matched volatility
matched session
```

## 22.3 Human observation controls

For human-submitted chart observations:

- chart must hide future candles;
- submission time must be recorded;
- invalidation must be specified;
- no editing after horizon completion;
- screenshots after the outcome are `POST_HOC_EXPLANATION`, not forward evidence.

## 22.4 Discovery versus validation

```text
DISCOVERY_FEATURE_VERSION
VALIDATION_FEATURE_VERSION
```

Material threshold changes require a new version and new forward N.

---

# 23. Proposed New Canonical Question Families

The following questions extend the existing Q001–Q866 system.

They are proposed as `Q867–Q1058`.

---

## Q867–Q878 — Candle and Close Morphology

**Q867.** Does upper-wick ratio in the final three closed candles improve early SHORT selection?  
**Q868.** Does lower-wick ratio in the final three closed candles improve early LONG selection?  
**Q869.** Does close location within the candle range predict next-horizon continuation independently of candle direction?  
**Q870.** Does a large expansion candle followed by weak follow-through predict failed breakout?  
**Q871.** Does repeated rejection at the same level outperform a single extreme wick?  
**Q872.** Does body compression with rising wick size predict exhaustion?  
**Q873.** Does expansion retracement fraction distinguish healthy pullback from reversal?  
**Q874.** Does a close beyond a structural level matter more than intrabar penetration?  
**Q875.** Do two consecutive acceptance closes outperform one close for breakout confirmation?  
**Q876.** Does candle morphology retain value after controlling for ATR and session?  
**Q877.** Are candle morphology effects directionally asymmetric between LONG and SHORT?  
**Q878.** Which candle features remain stable in forward data across regimes?

---

## Q879–Q890 — Multi-Push and Momentum Geometry

**Q879.** Does third-push displacement decay predict SHORT before a BUY-side event?  
**Q880.** Does third-push sell-side decay predict LONG before a SELL-side reversal?  
**Q881.** Does increasing time-to-new-high predict momentum exhaustion?  
**Q882.** Does increasing overlap across pushes predict failed continuation?  
**Q883.** Does liquidation notional rising while displacement falls identify terminal pressure?  
**Q884.** Does a parabolic final push differ from an orderly third push?  
**Q885.** Is push efficiency more predictive than raw return?  
**Q886.** Does speed decay provide earlier warning than slope decay?  
**Q887.** Does reacceleration after decay invalidate an exhaustion observer?  
**Q888.** Which push count is most informative after controlling for move age?  
**Q889.** Are push-decay signals scalp-only or useful at swing horizons?  
**Q890.** Do multi-push features survive matched non-event controls?

---

## Q891–Q902 — Swing Grammar

**Q891.** Does `HH → deep HL → weak HH` predict SHORT better than generic lower momentum?  
**Q892.** Does `LL → failed LL → reclaim` predict LONG before liquidation reversal?  
**Q893.** Does an equal-high sweep followed by close below create a distinct SHORT route?  
**Q894.** Does an equal-low sweep followed by reclaim create a distinct LONG route?  
**Q895.** Does retracement depth determine whether a higher low is healthy or fragile?  
**Q896.** Does swing duration add information beyond retracement depth?  
**Q897.** Does reduced progress per swing identify mature trend state?  
**Q898.** Does a failed new extreme outperform simple divergence labels?  
**Q899.** Which swing grammar patterns remain valid across 5m, 15m and 1h?  
**Q900.** Are LONG and SHORT grammar families structurally asymmetric?  
**Q901.** Does grammar quality improve event-independent setup selection?  
**Q902.** Can swing grammar reduce false early entries without excessive delay?

---

## Q903–Q914 — Liquidity Sweep Anatomy

**Q903.** Does wick-only penetration differ from body acceptance beyond a level?  
**Q904.** Does reclaim speed after a sweep predict reversal magnitude?  
**Q905.** Does time spent outside the swept level distinguish sweep from breakout?  
**Q906.** Does sweep volume without acceptance predict reversal?  
**Q907.** Does liquidation at the sweep improve or degrade reversal expectancy?  
**Q908.** Is the first pullback after a sweep better than immediate entry?  
**Q909.** Is the second retest safer but economically too late?  
**Q910.** Do equal-high and previous-day-high sweeps behave differently?  
**Q911.** Do session sweeps behave differently from structural swing sweeps?  
**Q912.** Does book refill after a sweep identify genuine absorption?  
**Q913.** What earliest feature separates a true breakout after sweep from a failed sweep?  
**Q914.** Which sweep definitions survive untouched forward observation?

---

## Q915–Q926 — Breakout and Retest Quality

**Q915.** Does close distance beyond the breakout level predict continuation?  
**Q916.** Does time beyond the level matter more than breakout candle size?  
**Q917.** Does decreasing retest volume improve breakout continuation?  
**Q918.** Does retest depth have a nonlinear optimal range?  
**Q919.** Is the first retest superior to the second retest after fees and missed movement?  
**Q920.** Does acceptance require multiple closes or one strong close?  
**Q921.** Does breakout range expansion predict sustainable continuation or terminal exhaustion?  
**Q922.** Does failed acceptance create a better opposite-direction entry than the original breakout?  
**Q923.** Does OI expansion confirm or weaken breakout quality?  
**Q924.** Does relative strength improve breakout validation?  
**Q925.** Are breakout and breakdown mechanics asymmetric?  
**Q926.** Which retest-quality features are knowable early enough to remain economical?

---

## Q927–Q938 — Compression Taxonomy

**Q927.** Do horizontal, ascending-base and descending-ceiling compressions have different direction priors?  
**Q928.** Does symmetric coil shape contain directional information without flow confirmation?  
**Q929.** Does volatility contraction plus volume contraction improve release quality?  
**Q930.** Does compression touch count improve or degrade breakout expectancy?  
**Q931.** Do repeated false breaks strengthen or weaken the final release?  
**Q932.** Can fake compression be identified from unstable book and flow behavior?  
**Q933.** Does apex proximity affect breakout timing and failure probability?  
**Q934.** Does compression duration determine scalp versus swing behavior?  
**Q935.** Does trend-context compression differ from range-context compression?  
**Q936.** Does release efficiency matter more than release direction at T0?  
**Q937.** Can compression cancellation be identified before a false release?  
**Q938.** Which compression families remain stable across sessions and regimes?

---

## Q939–Q950 — Trendline and Channel Behavior

**Q939.** Does repeated upper-channel contact predict exhaustion?  
**Q940.** Does channel slope acceleration predict terminal movement?  
**Q941.** Does channel slope deceleration predict transition?  
**Q942.** Does overshoot-and-reentry create a robust reversal observer?  
**Q943.** Does midline hold support continuation?  
**Q944.** Does midline rejection improve opposite-direction timing?  
**Q945.** Does a channel break require retest confirmation?  
**Q946.** Does channel width expansion indicate healthy trend or instability?  
**Q947.** Are algorithmic channel results stable across construction methods?  
**Q948.** Does a trendline break add information beyond swing grammar?  
**Q949.** Does channel location improve exit timing after MFE?  
**Q950.** Which channel features remain valid after multiple-testing correction?

---

## Q951–Q962 — Relative Strength and Lead-Lag

**Q951.** Does ETH failing to confirm a BTC high predict ETH SHORT?  
**Q952.** Does ETH holding while BTC falls predict ETH LONG?  
**Q953.** Does asset reclaim before BTC improve LONG timing?  
**Q954.** Does BTC reclaim before the asset improve waiting versus immediate entry?  
**Q955.** Does ETH/BTC structure lead ETH/USDT direction?  
**Q956.** Does beta-adjusted residual outperform raw relative return?  
**Q957.** Does correlation breakdown indicate regime change or temporary divergence?  
**Q958.** Does relative-strength leadership persist across session boundaries?  
**Q959.** Does relative weakness improve failed-breakout SHORT selection?  
**Q960.** Does relative strength improve failed-breakdown LONG selection?  
**Q961.** Is relative strength more useful for entry, hold or exit?  
**Q962.** Which lead-lag relationship survives forward observation across venues?

---

## Q963–Q974 — Session and Opening Structures

**Q963.** Does Asia-range sweep predict London reversal or continuation?  
**Q964.** Does London breakout acceptance predict US-session continuation?  
**Q965.** Does US-open reversal differ after an overnight trend versus overnight range?  
**Q966.** Does previous-session high reclaim create LONG continuation?  
**Q967.** Does previous-session low reclaim create SHORT failure and LONG transition?  
**Q968.** Does opening-range breakout quality outperform fixed clock-time entries?  
**Q969.** Does session handoff create predictable failed continuation?  
**Q970.** Are sweep outcomes different at session open versus mid-session?  
**Q971.** Does session range position improve event interpretation?  
**Q972.** Does weekend session structure require separate thresholds?  
**Q973.** Does funding-window proximity alter opening structure outcomes?  
**Q974.** Which session-native structures remain robust after regime controls?

---

## Q975–Q986 — Unconditional SHORT Genesis

**Q975.** Can distribution be detected before any liquidation event?  
**Q976.** Does failed high plus weak reclaim create a valid event-independent SHORT observer?  
**Q977.** Does third-push exhaustion create SHORT without a future event condition?  
**Q978.** Does relative-strength failure create early SHORT before market-wide weakness?  
**Q979.** Does descending compression create SHORT genesis before breakdown?  
**Q980.** Does channel overshoot-and-reentry create SHORT genesis?  
**Q981.** Does OI expansion without price acceptance improve SHORT selection?  
**Q982.** Does crowded funding matter only when price progress fails?  
**Q983.** Does session sweep failure create independent SHORT alpha?  
**Q984.** Which unconditional SHORT candidates outperform matched no-event controls?  
**Q985.** Are unconditional SHORT routes scalp, intraday or swing structures?  
**Q986.** Which candidates deserve new frozen forward buckets?

---

## Q987–Q998 — Setup Cancellation and Staleness

**Q987.** Does cancellation before entry avoid material losses?  
**Q988.** How often does cancellation also miss a valid move?  
**Q989.** Which cancellation reason has the highest avoided-loss value?  
**Q990.** Does confirmation delay cause economic staleness before statistical invalidation?  
**Q991.** Does a new extreme always invalidate a reversal setup?  
**Q992.** Does pullback depth provide a better cancellation rule than elapsed time?  
**Q993.** Does volume decay cancel breakout setups?  
**Q994.** Does flow reversal cancel sweep-reversal setups?  
**Q995.** Does relative-strength flip cancel otherwise valid setups?  
**Q996.** Does regime change require immediate cancellation or observer downgrade?  
**Q997.** Can stale setups be revived by a new structural cycle?  
**Q998.** Which cancellation policies survive forward validation without excessive opportunity loss?

---

## Q999–Q1010 — Human Chart Observation Registry

**Q999.** Which recurring human chart observations can be translated into measurable features?  
**Q1000.** Which human observations cannot be reproduced from existing data?  
**Q1001.** Does timestamped human intuition outperform matched mechanical baselines?  
**Q1002.** Which observers are systematically overconfident?  
**Q1003.** Which visual patterns show high inter-observer agreement?  
**Q1004.** Does agreement predict forward survival?  
**Q1005.** Which human terms map to multiple incompatible numeric definitions?  
**Q1006.** Can screenshot annotations reveal missing feature families?  
**Q1007.** Does hiding future candles materially reduce apparent pattern quality?  
**Q1008.** Are post-hoc visual explanations distinguishable from real-time observations?  
**Q1009.** Which human observations are useful only for hypothesis generation?  
**Q1010.** Can AMI learn a stable translation dictionary from visual language to features?

---

## Q1011–Q1022 — Volume, Participation and Progress

**Q1011.** Does high volume with low price progress predict absorption or exhaustion?  
**Q1012.** Does low volume with high progress predict liquidity vacuum continuation?  
**Q1013.** Does trade-count expansion add information beyond volume?  
**Q1014.** Does large-trade concentration predict continuation or terminal activity?  
**Q1015.** Does participation breadth distinguish healthy breakout from isolated print?  
**Q1016.** Does volume climax require close failure to become reversal information?  
**Q1017.** Does price progress per unit volume decay across pushes?  
**Q1018.** Does liquidation-adjusted price impact improve terminal-push detection?  
**Q1019.** Does volume contraction during retest improve continuation?  
**Q1020.** Does flow disagreement predict failed acceptance?  
**Q1021.** Which participation features are robust across venues?  
**Q1022.** Are participation features most useful for entry, hold or exit?

---

## Q1023–Q1034 — Auction Inefficiency and Repair

**Q1023.** Do fast low-volume traversals predict later revisit?  
**Q1024.** Does revisit probability depend on trend strength?  
**Q1025.** Does partial repair support continuation while full repair predicts reversal?  
**Q1026.** Do imbalance zones improve pullback entry timing?  
**Q1027.** Does gap age reduce predictive value?  
**Q1028.** Does session boundary alter repair probability?  
**Q1029.** Does volume-profile location improve imbalance interpretation?  
**Q1030.** Does book depth validate or reject auction-inefficiency signals?  
**Q1031.** Are inefficiency effects asymmetric between upward and downward moves?  
**Q1032.** Do repeated unrepaired zones increase trend persistence?  
**Q1033.** Can repair completion improve exit timing?  
**Q1034.** Which quantitative definitions survive negative controls?

---

## Q1035–Q1046 — Multi-Timeframe Visual Nesting

**Q1035.** When is a 5m SHORT merely a pullback inside a 4h LONG?  
**Q1036.** When does a 5m sweep become a 1h reversal?  
**Q1037.** Does higher-timeframe acceptance override lower-timeframe rejection?  
**Q1038.** Does lower-timeframe momentum decay provide early warning of higher-timeframe transition?  
**Q1039.** Which timeframe should define the structural cycle?  
**Q1040.** How should conflicting candle morphology be fused across timeframes?  
**Q1041.** Does daily structural location determine scalp hold limits?  
**Q1042.** Does weekly context alter breakout versus sweep interpretation?  
**Q1043.** Can timeframe conflict identify `SCALP_ONLY` rather than `NO_TRADE`?  
**Q1044.** Does timeframe alignment improve entry or merely reduce frequency?  
**Q1045.** Which visual features are timeframe-invariant?  
**Q1046.** Which features require independent thresholds per timeframe?

---

## Q1047–Q1058 — Validation, Execution and Evidence Independence

**Q1047.** Do chart-native observers survive executable fill and fee modeling?  
**Q1048.** Does confirmation latency remove the apparent edge?  
**Q1049.** Are multiple pattern labels from one structural cycle independent evidence?  
**Q1050.** How much researcher exposure occurred before each pattern was frozen?  
**Q1051.** Which chart features are redundant with existing flow or state features?  
**Q1052.** Which pattern families survive top-day and top-cycle removal?  
**Q1053.** Which families survive chronological and untouched holdout?  
**Q1054.** Which families survive matched non-event controls?  
**Q1055.** Which pattern observers are regime-limited?  
**Q1056.** Which pattern observers become OOD under new volatility conditions?  
**Q1057.** Which discovered patterns deserve forward observer activation?  
**Q1058.** Which chart-native claims remain descriptive and must not become operational?

---

# 24. Question Registry Requirements

Every new question must carry:

```yaml
qid:
family:
question:
required_data:
required_features:
minimum_raw_n:
minimum_independent_cycle_n:
minimum_days:
required_regimes:
required_sessions:
required_forward_duration:
current_status:
blocked_reason:
next_valid_test:
permission_ceiling:
```

Allowed statuses:

```text
QUESTION_TEXT_CANONICAL
BLOCKED_BY_DATA
BLOCKED_BY_FEATURE
BLOCKED_BY_SAMPLE
BLOCKED_BY_REGIME
FORWARD_ACCUMULATING
READY_FOR_PREREG
PREREGISTERED
ANSWERED
FALSIFIED
```

---

# 25. Database Additions

Recommended tables:

```text
ami_chart_observations
ami_chart_observation_translations
ami_chart_candles
ami_chart_swings
ami_chart_levels
ami_chart_pushes
ami_chart_patterns
ami_chart_setups
ami_chart_setup_transitions
ami_chart_observer_entries
ami_chart_outcomes
ami_chart_question_progress
ami_chart_feature_dictionary
```

## 25.1 `ami_chart_observations`

```sql
CREATE TABLE IF NOT EXISTS ami_chart_observations (
    chart_observation_id TEXT PRIMARY KEY,
    observer_id TEXT,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    venue TEXT,
    timeframe TEXT NOT NULL,
    observation_ts TEXT NOT NULL,
    screenshot_capture_ts TEXT,
    direction_hypothesis TEXT,
    expected_horizon TEXT,
    visible_pattern TEXT,
    free_text_reason TEXT,
    invalidation TEXT,
    confidence_self_reported REAL,
    screenshot_hash TEXT,
    chart_settings_json TEXT,
    known_future_hidden INTEGER NOT NULL,
    linked_event_id TEXT,
    linked_cycle_id TEXT,
    status TEXT NOT NULL,
    source_hash TEXT NOT NULL
);
```

## 25.2 `ami_chart_setups`

```sql
CREATE TABLE IF NOT EXISTS ami_chart_setups (
    setup_id TEXT PRIMARY KEY,
    pattern_id TEXT NOT NULL,
    structural_cycle_id TEXT,
    direction TEXT NOT NULL,
    horizon_class TEXT NOT NULL,
    setup_state TEXT NOT NULL,
    detected_ts TEXT NOT NULL,
    armed_ts TEXT,
    triggered_ts TEXT,
    cancelled_ts TEXT,
    expired_ts TEXT,
    completed_ts TEXT,
    entry_reference REAL,
    invalidation_reference REAL,
    cancellation_reason TEXT,
    observer_version TEXT NOT NULL,
    source_hash TEXT NOT NULL
);
```

---

# 26. API Additions

```http
GET  /api/v1/chart/patterns
GET  /api/v1/chart/setups
GET  /api/v1/chart/setups/{setup_id}
GET  /api/v1/chart/features/{symbol}
GET  /api/v1/chart/questions
GET  /api/v1/chart/observations
POST /api/v1/chart/observations
POST /api/v1/chart/observations/{id}/translate
GET  /api/v1/chart/relative-strength
GET  /api/v1/chart/session-structures
```

All write endpoints are research metadata only.  
No endpoint may connect to order execution.

---

# 27. Dashboard Pages

## Page A — Chart Structure Laboratory

Show:

```text
current swing grammar
push count
momentum decay
major levels
compression type
breakout/retest state
channel state
relative strength
session structure
```

## Page B — Setup Lifecycle

Columns:

```text
setup
detected
armed
triggered
cancelled
stale
outcome pending
completed
```

## Page C — Candle Morphology Matrix

Rows:

```text
pattern family
```

Columns:

```text
5m
15m
30m
45m
1h
2h
4h
```

Metrics:

```text
independent N
median
WR
MFE
MAE
recovery
```

## Page D — Sweep and Breakout Laboratory

Compare:

```text
wick sweep
body rejection
temporary acceptance
true breakout
failed breakout
first retest
second retest
```

## Page E — Compression Explorer

Show compression shape, quality, release and failure.

## Page F — Relative Strength

Display asset/BTC structure, lead-lag and divergence.

## Page G — Session Structure

Display Asia, London, US-open and handoff patterns.

## Page H — Human Observation Inbox

Workflow:

```text
submitted
translation required
feature mapped
data blocked
observer ready
prereg ready
```

## Page I — Feature Translation Dictionary

Human phrase:

```text
“third push looked weak”
```

maps to:

```text
push_number = 3
displacement_decay < threshold
speed_decay < threshold
overlap_increase > threshold
```

## Page J — Setup Cancellation

Show:

```text
cancel reason
avoided loss
missed move
time to cancel
post-cancel direction
```

---

# 28. Test Suite

## Safety and timing

1. Chart observer cannot call order APIs.  
2. Partial candle cannot be treated as closed candle.  
3. Swing pivot cannot be used before `known_at_ts`.  
4. Future event cannot select a pre-event chart pattern.  
5. Screenshot captured after outcome cannot count as forward observation.  
6. Edited observation after outcome creates a new version.  
7. Missing chart data is not zero.  
8. Existing live/shadow route diff remains zero.  

## Feature correctness

9. Wick and body ratios handle zero-range candles safely.  
10. Close-location calculation matches reference examples.  
11. Push efficiency matches path calculation.  
12. Push count is stable under restart.  
13. Swing grammar uses confirmed swings only.  
14. Sweep penetration uses the correct level version.  
15. Breakout acceptance does not use future closes.  
16. Retest depth is direction-correct.  
17. Compression slopes use only known points.  
18. Channel anchors are immutable after freeze.  
19. Relative-strength residual uses frozen beta methodology.  
20. Session boundaries use UTC correctly.  

## Lifecycle and controls

21. Cancelled setup remains in the dataset.  
22. Stale setup is not silently deleted.  
23. Event-never-arrived early signals are retained.  
24. Same structural cycle does not inflate independent N.  
25. Multiple pattern labels remain separate raw rows but one cycle.  
26. Historical replay does not increase forward N.  
27. Human observation is not treated as alpha evidence by default.  
28. Matched non-event controls are generated without future information.  
29. Opposite-direction control uses the same timestamp.  
30. Pattern component ablation is reproducible.  

## Statistical integrity

31. Discovery thresholds are not reused as untouched validation thresholds.  
32. Multiple-testing family ID is recorded.  
33. Researcher-exposure count is recorded before freeze.  
34. Top-day removal is computed correctly.  
35. Top-cycle removal is computed correctly.  
36. Regime-specific metrics are not pooled silently.  
37. Raw N and independent N are shown together.  
38. Pending outcomes are not counted as losses or zero.  
39. Censored paths are labeled.  
40. Executable fill metrics remain separate from mark-price metrics.  

## Dashboard and API

41. Dashboard setup counts match DB.  
42. Question progress matches registry.  
43. Observation screenshot hash is preserved.  
44. API returns explicit null for unavailable features.  
45. API exposes pattern and feature versions.  
46. Chart pages show sample-quality banner.  
47. Setup cancellation page includes missed-move risk.  
48. Human observation page distinguishes pre-outcome and post-hoc.  
49. Aggregate rebuild equals incremental aggregate.  
50. Rollback disables chart services without touching trading services.  

---

# 29. Implementation Phases

## Phase 0 — Audit

```text
inspect existing candles, states, swings and chart code
identify duplicate feature implementations
identify available screenshot/chart metadata
map integration points
record protected files
```

## Phase 1 — Core feature foundation

```text
candle normalizer
swing extractor
level registry
push extractor
feature dictionary
known-at contract
```

## Phase 2 — Pattern engines

```text
candle morphology
swing grammar
sweep anatomy
breakout/retest
compression
channel
relative strength
session structures
```

## Phase 3 — Setup lifecycle

```text
forming
armed
triggered
cancelled
stale
expired
completed
```

## Phase 4 — Human observation bridge

```text
observation form
screenshot provenance
translation workflow
feature mapping
```

## Phase 5 — Observer and outcome layer

```text
orderless LONG/SHORT observers
matched controls
path outcomes
raw N / independent N
```

## Phase 6 — Dashboard and reports

```text
chart structure lab
setup lifecycle
sweep/breakout lab
relative strength
session structures
observation inbox
question progress
```

## Phase 7 — Preregistration candidates

Only after descriptive and control analysis:

```text
shortlist candidate observers
freeze exact definitions
set activation timestamp
start N from zero
```

---

# 30. Definition of Done

This extension is complete only when:

```text
candle features are versioned
swings have known-at timestamps
push geometry is measurable
swing grammar is stored
sweeps and breakouts are distinct
compression shapes are distinct
channel construction is deterministic
relative strength is explicit
session structures are explicit
unconditional SHORT genesis exists
setup cancellation is recorded
human chart observations are timestamped
visual language maps to features
matched controls exist
raw N and independent N are separate
Q867–Q1058 are registered
dashboard pages work
tests pass
live/shadow behavior is unchanged
no operational permission is granted
```

---

# 31. Deliverables

```text
Chart-native feature code
Candle normalizer
Swing extractor
Level registry
Push geometry engine
Pattern state engines
Setup lifecycle engine
Human observation registry
Screenshot provenance support
Observer routes
Matched-control generator
DB migrations
API contracts
Dashboard pages
Question registry extension Q867–Q1058
Data dictionary
Test suite
Benchmark
Migration notes
Rollback plan
CHANGELOG
Decision Record
SYSTEM_STATE update
Untouched live components list
```

---

# 32. Coding Agent Instruction

```text
Read the parent AMI whitepaper, artifact reconstruction protocol and forward
observatory/timing dashboard specification before implementing this extension.

Treat this document as a proposed chart-native research extension.

Do not modify any existing closed experiment verdict. Do not change live
executor, shadow order behavior, risk, sizing, leverage, .env or route logic.

First audit existing candle, swing, structure, chart and screenshot code.
Reuse canonical components where scientifically equivalent. Do not create a
second conflicting feature implementation without recording the conflict.

Implement the foundation in this order:

1. candle normalization and partial-candle safety;
2. confirmed swing extraction with pivot_ts and known_at_ts separation;
3. level registry;
4. push and momentum geometry;
5. chart feature dictionary;
6. pattern engines;
7. setup lifecycle including cancellation and staleness;
8. human chart observation registry;
9. orderless observer routes and matched controls;
10. API, dashboard and question progress;
11. tests, benchmark and rollback.

Do not train a prediction model. Do not optimize thresholds on forward data.
Do not activate hundreds of buckets automatically.

After descriptive historical exploration and controls, produce a shortlist of
candidate observers. Each material candidate must receive a frozen definition,
version, activation timestamp and forward N starting from zero.

All chart observations submitted after outcomes must be labeled POST_HOC and
must not count as forward evidence.

At delivery provide changed files, protected untouched files, git diff proof,
migration output, example candle/swing/push/pattern/setup rows, observation-to-
feature translation examples, API samples, dashboard screenshots, test results,
benchmark, rollback instructions and known limitations.
```

---

# 33. Final Scientific Rule

Chart intelligence is valuable only when the system can distinguish:

```text
what looked obvious afterward
from
what was measurable and knowable beforehand
```

The extension must therefore optimize for:

```text
clarity
timestamp integrity
measurability
negative controls
evidence independence
forward survival
```

not for the number of visually attractive patterns discovered.

```text
A CHART PATTERN IS NOT A CLAIM.
A CLAIM IS NOT AN EDGE.
AN EDGE IS NOT A PERMISSION.
```
