# S34 SELL Bear-Day Filter Research

Generated: `2026-06-27T08:33:21.601625+00:00`

Tests whether adding `max_day_trend_bps=0.0` (bearish days only) to SELL rules
produces the same improvement that `min_day_trend_bps=0.0` gives BUY rules.

| Rule | Filter | Sigs | NF% | N | Median | Top3R | WR | H1 | H2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_SELL_500K | RAW | 222 | 45% | 122 | -15.6 | -479.0 | 37% | -26.5 | -6.7 |
| ETH_SELL_500K | BEAR<=0 | 222 | 45% | 94 | -15.6 | -90.3 | 37% | -24.8 | -5.6 |
| ETH_SELL_500K | STRONGBEAR<-1% | 222 | 45% | 62 | -20.1 | +81.4 | 40% | -26.5 | +44.4 |
| ETH_SELL_1M | RAW | 114 | 32% | 77 | -13.5 | +41.6 | 31% | -15.4 | -10.2 |
| ETH_SELL_1M | BEAR<=0 | 114 | 32% | 67 | -12.9 | +155.1 | 31% | -16.3 | -10.2 |
| ETH_SELL_1M | STRONGBEAR<-1% | 114 | 32% | 52 | -14.9 | -19.2 | 29% | -18.4 | -12.7 |
| SOL_SELL_200K | RAW | 58 | 21% | 46 | -11.5 | +180.9 | 39% | -14.9 | +1.8 |
| SOL_SELL_200K | BEAR<=0 | 58 | 21% | 39 | -9.4 | +273.3 | 44% | -12.1 | +2.0 |
| SOL_SELL_200K | STRONGBEAR<-1% | 58 | 21% | 33 | -10.8 | +213.1 | 42% | -12.4 | +2.2 |
| SOL_SELL_100K | RAW | 107 | 19% | 86 | -37.0 | -850.6 | 29% | -17.5 | -38.3 |
| SOL_SELL_100K | BEAR<=0 | 107 | 19% | 66 | -20.3 | -402.3 | 33% | -12.1 | -38.2 |
| SOL_SELL_100K | STRONGBEAR<-1% | 107 | 19% | 43 | -12.8 | -243.8 | 35% | -12.8 | -23.9 |