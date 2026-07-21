# AMI Drift Monitor (research-only)

> 2026-07-03 07:34 UTC

```json
{
 "status": "UNUSABLE",
 "per_feature": {
  "ret5m": {
   "psi": 0.043,
   "miss_delta": 0.0,
   "status": "STABLE"
  },
  "rv30m": {
   "psi": 0.155,
   "miss_delta": 0.0,
   "status": "WARNING"
  },
  "ofi10m": {
   "psi": 0.004,
   "miss_delta": 0.257,
   "status": "STABLE"
  },
  "stress10m": {
   "psi": 0.547,
   "miss_delta": 0.0,
   "status": "SHIFTED"
  },
  "buyliq10m": {
   "psi": 0.362,
   "miss_delta": 0.0,
   "status": "SHIFTED"
  },
  "fund_vel_1h": {
   "psi": 0.06,
   "miss_delta": 0.0,
   "status": "STABLE"
  },
  "spread5m": {
   "psi": 7.381,
   "miss_delta": 0.189,
   "status": "SHIFTED"
  },
  "trades10m": {
   "psi": 0.89,
   "miss_delta": 0.0,
   "status": "SHIFTED"
  },
  "ret1h": {
   "psi": 0.043,
   "miss_delta": 0.0,
   "status": "STABLE"
  }
 },
 "latent_occupancy_drift_tv": 0.506,
 "transition_matrix_drift": 0.205,
 "recommendations": [
  "applicability_restrict(latent knowledge)",
  "shadow_permission_suspend_suggest",
  "retest_request",
  "data_quality_investigation"
 ],
 "authority_note": "monitor izin degistiremez; oneriler Epistemic Governor'a gider"
}
```

*Monitor izin degistiremez; oneriler governor'a gider. Runner: `python -m ami.latent.drift_monitor`*