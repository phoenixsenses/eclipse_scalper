# S34 Sub-Minute Lead-Lag (precursor's last refuge: the second scale)

Generated: `2026-06-28T22:45:44.699651+00:00`  |  7.0d window, 5s returns, samples=120960

Peak at a positive lag (sec) => lead leads follow at that horizon (HFT precursor). Peak at 0 => contemporaneous.

| Pair (lead->follow) | peak lag (s) | peak corr | corr@-5s | corr@0 | corr@+5s | corr@+10s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC->ETH | 0 | 0.841 | 0.172 | 0.841 | 0.131 | 0.054 |
| BTC->SOL | 0 | 0.764 | 0.156 | 0.764 | 0.139 | 0.043 |
| ETH->SOL | 0 | 0.788 | 0.12 | 0.788 | 0.14 | 0.047 |
| ETH->BTC | 0 | 0.841 | 0.131 | 0.841 | 0.172 | 0.07 |
