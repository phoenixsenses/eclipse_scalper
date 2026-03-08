# Event Merged Banner Policy Handoff

## Purpose

`event_merged_banner_policy` turns the effective watchboard into a single
operator-facing top banner when multiple fresh high-priority lanes are active
at the same time.

Its job is to answer one runtime question:

- should the dashboard show one merged banner instead of separate top banners

## Runtime Input

Preferred input:

- `tools.event_merged_banner_policy`
- example artifact:
  - `reports/EVENT_MERGED_BANNER_POLICY_REAL.json`

Upstream context already applied before this payload:

- suppression policy
- persistence policy
- effective lane ranking

## What Runtime Should Render

- `summary.banner_mode`
- `summary.focus_lanes`
- `banner.headline`
- `banner.recommended_action`
- `banner.reasons`
- `focus_rows`

This is intended for:

- top dashboard header
- overview banner
- operator status strip

## Runtime Rules

- if `banner_mode = merged`, show a single combined header message
- keep lane-level cards below the banner; this payload does not replace lane tables
- `focus_rows` order should match operator emphasis order
- if a focus row is already `degrade`, keep it visually secondary inside the merged context
- do not auto-wire this payload into execution logic

## Current Real Example

Latest real merged banner currently shows:

- `banner_mode = merged`
- `focus_lanes = [return_shock, book_proxy_pressure, volatility_burst]`
- `top_lane = return_shock`
- `top_action = escalate_monitoring`

This means runtime should present a single high-priority banner for the current
multi-lane fresh event cluster instead of making the operator scan multiple
competing headers.
