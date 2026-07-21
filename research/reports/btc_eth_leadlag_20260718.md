# BTC/ETH Forced-Order Lead-Lag Measurement

Generated: 2026-07-18T14:15:05+00:00  
Database: `D:\eclipse_scalper\data\microstructure.db` (opened SQLite `mode=ro`)  
Gap source: `D:\eclipse_scalper\SYSTEM_STATE.md`

## Scope

Research measurement only. This report defines no signal, trading rule, parameter optimization, or holdout evaluation. At every occurrence below, **positive lag means BTC leads ETH**, computed as `corr(BTC[t], ETH[t + lag])`.

## Step 0: Schema Discovery

### `liq_heatmap`

```sql
CREATE TABLE liq_heatmap (symbol TEXT NOT NULL, price_bucket REAL NOT NULL, side TEXT NOT NULL, event_count INTEGER NOT NULL, notional_sum REAL NOT NULL, PRIMARY KEY(symbol, price_bucket, side))
```

| cid | name | type | notnull | default | pk |
|---|---|---|---|---|---|
| 0 | symbol | TEXT | 1 | None | 1 |
| 1 | price_bucket | REAL | 1 | None | 2 |
| 2 | side | TEXT | 1 | None | 3 |
| 3 | event_count | INTEGER | 1 | None | 0 |
| 4 | notional_sum | REAL | 1 | None | 0 |
### `liquidations`

```sql
CREATE TABLE liquidations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional REAL NOT NULL,
    trade_time_ms INTEGER NOT NULL
)
```

| cid | name | type | notnull | default | pk |
|---|---|---|---|---|---|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | ts_ms | INTEGER | 1 | None | 0 |
| 2 | symbol | TEXT | 1 | None | 0 |
| 3 | side | TEXT | 1 | None | 0 |
| 4 | price | REAL | 1 | None | 0 |
| 5 | quantity | REAL | 1 | None | 0 |
| 6 | notional | REAL | 1 | None | 0 |
| 7 | trade_time_ms | INTEGER | 1 | None | 0 |

The event table is `liquidations`. Symbols present: 752 distinct values; `BTCUSDT` (117,312 rows) and `ETHUSDT` (125,343 rows) are confirmed. The timestamp used is `ts_ms`, Unix epoch milliseconds in UTC; its magnitude and UTC conversion span 2026-02-15T14:30:18.195Z to 2026-07-18T14:14:35.529Z. `trade_time_ms` is also Unix epoch milliseconds; `ts_ms` is the collector/event timestamp used by the required binning rule.

### Symbol Values Present

| symbol | event_count |
|---|---|
| 0GUSDT | 1056 |
| 1000000BOBUSDT | 398 |
| 1000000MOGUSDT | 187 |
| 1000BONKUSDC | 264 |
| 1000BONKUSDT | 1904 |
| 1000CATUSDT | 154 |
| 1000CHEEMSUSDT | 186 |
| 1000FLOKIUSDT | 933 |
| 1000LUNCUSDT | 1997 |
| 1000PEPEUSDC | 577 |
| 1000PEPEUSDT | 7198 |
| 1000RATSUSDT | 1387 |
| 1000SATSUSDT | 815 |
| 1000SHIBUSDC | 136 |
| 1000SHIBUSDT | 2155 |
| 1000XECUSDT | 5138 |
| 1INCHUSDT | 252 |
| 1MBABYDOGEUSDT | 799 |
| 2ZUSDT | 515 |
| 4USDT | 1750 |
| AAOIUSDT | 916 |
| AAPLUSDT | 289 |
| AAVEUSDC | 289 |
| AAVEUSDT | 3664 |
| AAVEUSD_PERP | 27 |
| ACEUSDT | 774 |
| ACHUSDT | 343 |
| ACTUSDT | 1966 |
| ACUUSDT | 147 |
| ACXUSDT | 144 |
| ADAUSDC | 399 |
| ADAUSDT | 7090 |
| ADAUSD_PERP | 129 |
| ADBEUSDT | 99 |
| AERGOUSDT | 1736 |
| AEROUSDT | 1114 |
| AEVOUSDT | 351 |
| AGLDUSDT | 6509 |
| AGTUSDT | 2789 |
| AIAUSDT | 1231 |
| AIGENSYNUSDT | 3075 |
| AINUSDT | 1380 |
| AIOTUSDT | 2179 |
| AIOUSDT | 6072 |
| AIXBTUSDT | 467 |
| AKEUSDT | 6727 |
| AKTUSDT | 568 |
| ALABUSDT | 134 |
| ALCHUSDT | 412 |
| ALGOUSDT | 1162 |
| ALICEUSDT | 3779 |
| ALLOUSDT | 13836 |
| ALLUSDT | 115 |
| ALPINEUSDT | 180 |
| ALTUSDT | 652 |
| AMATUSDT | 235 |
| AMDUSDT | 429 |
| AMZNUSDT | 129 |
| ANIMEUSDT | 359 |
| ANKRUSDT | 382 |
| ANTHROPICUSDT | 183 |
| APEUSDT | 1099 |
| API3USDT | 477 |
| APPUSDT | 64 |
| APRUSDT | 1056 |
| APTUSDT | 1707 |
| ARBUSDC | 121 |
| ARBUSDT | 2002 |
| ARCUSDT | 649 |
| ARIAUSDT | 746 |
| ARKMUSDT | 557 |
| ARKUSDT | 554 |
| ARMUSDT | 390 |
| ARPAUSDT | 3083 |
| ARUSDT | 692 |
| ARXUSDT | 2732 |
| ASMLUSDT | 171 |
| ASRUSDT | 343 |
| ASTERUSDT | 1983 |
| ASTRUSDT | 519 |
| ASTSUSDT | 476 |
| ATHUSDT | 245 |
| ATOMUSDT | 1371 |
| ATUSDT | 1111 |
| AUCTIONUSDT | 205 |
| AUSDT | 410 |
| AVAAIUSDT | 1095 |
| AVAUSDT | 145 |
| AVAXUSDC | 253 |
| AVAXUSDT | 3263 |
| AVAXUSD_PERP | 18 |
| AVGOUSDT | 284 |
| AVNTUSDT | 623 |
| AWEUSDT | 429 |
| AXLUSDT | 454 |
| AXSUSDT | 1676 |
| AXTIUSDT | 713 |
| AZTECUSDT | 205 |
| B2USDT | 434 |
| BABAUSDT | 270 |
| BABYUSDT | 2444 |
| BANANAS31USDT | 1855 |
| BANANAUSDT | 346 |
| BANDUSDT | 306 |
| BANKUSDT | 7338 |
| BANUSDT | 510 |
| BARDUSDT | 326 |
| BASEDUSDT | 4475 |
| BASUSDT | 4502 |
| BATUSDT | 445 |
| BBUSDT | 406 |
| BBXUSDT | 206 |
| BCHUSDC | 258 |
| BCHUSDT | 3345 |
| BCHUSD_PERP | 27 |
| BEAMXUSDT | 244 |
| BEATUSDT | 18562 |
| BELUSDT | 2863 |
| BERAUSDT | 967 |
| BEUSDT | 343 |
| BICOUSDT | 11047 |
| BIGTIMEUSDT | 294 |
| BILLUSDT | 4894 |
| BIOUSDC | 265 |
| BIOUSDT | 1842 |
| BIRBUSDT | 4599 |
| BLESSUSDT | 10174 |
| BLUAIUSDT | 1422 |
| BLURUSDT | 503 |
| BMNRUSDT | 175 |
| BMTUSDT | 134 |
| BNBUSDC | 1051 |
| BNBUSDT | 7205 |
| BNBUSD_260925 | 22 |
| BNBUSD_261225 | 7 |
| BNBUSD_PERP | 123 |
| BNCUSDT | 76 |
| BNTUSDT | 84 |
| BOMEUSDC | 189 |
| BOMEUSDT | 781 |
| BOTUSDT | 105 |
| BRETTUSDT | 563 |
| BREVUSDT | 1681 |
| BRKBUSDT | 31 |
| BROCCOLI714USDT | 338 |
| BROCCOLIF3BUSDT | 668 |
| BRUSDT | 2372 |
| BSBUSDT | 7812 |
| BSPUSDT | 100 |
| BSVUSDT | 327 |
| BTCDOMUSDT | 112 |
| BTCU | 56 |
| BTCUSD1 | 137 |
| BTCUSDC | 4900 |
| BTCUSDT | 117312 |
| BTCUSDT_260626 | 474 |
| BTCUSDT_260925 | 445 |
| BTCUSDT_261225 | 136 |
| BTCUSD_260925 | 16 |
| BTCUSD_261225 | 6 |
| BTCUSD_PERP | 499 |
| BTRUSDT | 564 |
| BTWUSDT | 13770 |
| BULLAUSDT | 1207 |
| BUSDT | 4918 |
| BXUSDT | 87 |
| BZUSDT | 1852 |
| C98USDT | 553 |
| CAKEUSDT | 465 |
| CAPUSDT | 2082 |
| CARVUSDT | 797 |
| CATIUSDT | 583 |
| CATUSDT | 25 |
| CBRSUSDT | 428 |
| CCUSDT | 409 |
| CELOUSDT | 956 |
| CELRUSDT | 220 |
| CETUSUSDT | 160 |
| CFGUSDT | 570 |
| CFXUSDT | 553 |
| CGPTUSDT | 380 |
| CHILLGUYUSDT | 241 |
| CHIPUSDT | 2819 |
| CHRUSDT | 534 |
| CHZUSDT | 2798 |
| CIENUSDT | 90 |
| CKBUSDT | 262 |
| CLANKERUSDT | 366 |
| CLOUSDT | 5755 |
| CLUSDT | 4486 |
| COAIUSDT | 2876 |
| COHRUSDT | 207 |
| COINUSDT | 146 |
| COLLECTUSDT | 1336 |
| COMPUSDT | 580 |
| COOKIEUSDT | 675 |
| COPPERUSDT | 195 |
| COSTUSDT | 39 |
| COSUSDT | 183 |
| COTIUSDT | 309 |
| COWUSDT | 188 |
| CRCLUSDT | 996 |
| CRDOUSDT | 196 |
| CRMUSDT | 67 |
| CROSSUSDT | 353 |
| CRVUSDC | 92 |
| CRVUSDT | 1310 |
| CRWDUSDT | 225 |
| CRWVUSDT | 217 |
| CSCOUSDT | 46 |
| CTKUSDT | 126 |
| CTRUSDT | 779 |
| CTSIUSDT | 429 |
| CUSDT | 1064 |
| CVCUSDT | 168 |
| CVXUSDT | 163 |
| CYBERUSDT | 260 |
| CYSUSDT | 508 |
| DASHUSDT | 2540 |
| DATAIPUSDC | 53 |
| DATAIPUSDT | 227 |
| DEEPUSDT | 295 |
| DELLUSDT | 187 |
| DEXEUSDT | 7625 |
| DIAUSDT | 121 |
| DISUSDT | 33 |
| DKNGUSDT | 68 |
| DODOXUSDT | 4254 |
| DOGEUSDC | 806 |
| DOGEUSDT | 7801 |
| DOGEUSD_PERP | 73 |
| DOGSUSDT | 1777 |
| DOLOUSDT | 305 |
| DOODUSDT | 405 |
| DOTUSDT | 2398 |
| DOTUSD_PERP | 15 |
| DRAMUSDT | 1350 |
| DRIFTUSDT | 1026 |
| DUSDT | 319 |
| DUSKUSDT | 594 |
| DYDXUSDT | 3868 |
| DYMUSDT | 657 |
| EBAYUSDT | 74 |
| EDENUSDT | 2195 |
| EDGEUSDT | 2555 |
| EDUUSDT | 766 |
| EGLDUSDT | 882 |
| EIGENUSDT | 2213 |
| ELSAUSDT | 299 |
| ENAUSDC | 364 |
| ENAUSDT | 5090 |
| ENJUSDT | 1214 |
| ENSOUSDT | 626 |
| ENSUSDT | 499 |
| EPICUSDT | 4700 |
| ERAUSDT | 229 |
| ESPORTSUSDT | 19804 |
| ESPUSDT | 583 |
| ETCUSDT | 1242 |
| ETCUSD_PERP | 5 |
| ETHBTC | 80 |
| ETHFIUSDC | 186 |
| ETHFIUSDT | 1416 |
| ETHU | 44 |
| ETHUSD1 | 83 |
| ETHUSDC | 5318 |
| ETHUSDT | 125343 |
| ETHUSDT_260626 | 377 |
| ETHUSDT_260925 | 389 |
| ETHUSDT_261225 | 126 |
| ETHUSD_260925 | 30 |
| ETHUSD_261225 | 11 |
| ETHUSD_PERP | 362 |
| ETHWUSDT | 240 |
| EULUSDT | 249 |
| EVAAUSDT | 10731 |
| EWJUSDT | 50 |
| EWTUSDT | 50 |
| EWYUSDT | 924 |
| EWZUSDT | 46 |
| FARTCOINUSDT | 2764 |
| FETUSDT | 2767 |
| FFUSDT | 706 |
| FHEUSDT | 818 |
| FIDAUSDT | 2888 |
| FIGHTUSDT | 945 |
| FILUSDC | 140 |
| FILUSDT | 2339 |
| FILUSD_PERP | 23 |
| FLEXUSDT | 30 |
| FLNCUSDT | 233 |
| FLOCKUSDT | 1085 |
| FLOWUSDT | 202 |
| FLUIDUSDT | 540 |
| FLUXUSDT | 145 |
| FOGOUSDT | 1429 |
| FOLKSUSDT | 5720 |
| FORMUSDT | 437 |
| FRAXUSDT | 269 |
| FUSDT | 227 |
| FWDIUSDT | 50 |
| GALAUSDT | 1561 |
| GASUSDT | 364 |
| GENIUSUSDT | 1759 |
| GEVUSDT | 51 |
| GIGGLEUSDT | 1387 |
| GLMUSDT | 324 |
| GLWUSDT | 852 |
| GMEUSDT | 40 |
| GMTUSDT | 706 |
| GMXUSDT | 188 |
| GOATUSDT | 307 |
| GOOGLUSDT | 385 |
| GPSUSDT | 794 |
| GRAMUSDT | 695 |
| GRASSUSDT | 2453 |
| GRIFFAINUSDT | 392 |
| GRTUSDT | 534 |
| GTCUSDT | 233 |
| GUAUSDT | 5529 |
| GUNUSDT | 1393 |
| GUSDT | 3292 |
| GWEIUSDT | 8494 |
| HAEDALUSDT | 296 |
| HANAUSDT | 314 |
| HBARUSDC | 88 |
| HBARUSDT | 1346 |
| HDUSDT | 39 |
| HEIUSDT | 5734 |
| HEMIUSDT | 666 |
| HFTUSDT | 452 |
| HIGHUSDT | 206 |
| HIMSUSDT | 164 |
| HIVEUSDT | 361 |
| HMSTRUSDT | 4800 |
| HOLOUSDT | 395 |
| HOMEUSDT | 12731 |
| HOODUSDT | 197 |
| HOTUSDT | 2227 |
| HPEUSDT | 84 |
| HUMAUSDT | 422 |
| HUSDT | 12890 |
| HYPERUSDT | 707 |
| HYPEUSDT | 14469 |
| HYUNDAIUSDT | 134 |
| IBMUSDT | 390 |
| ICNTUSDT | 1318 |
| ICPUSDT | 2112 |
| ICXUSDT | 266 |
| IDOLUSDT | 3779 |
| IDUSDT | 4779 |
| ILVUSDT | 245 |
| IMXUSDT | 426 |
| INITUSDT | 314 |
| INJUSDT | 2685 |
| INTCUSDT | 1232 |
| INTWUSDT | 130 |
| INUSDT | 5879 |
| INXUSDT | 846 |
| IOSTUSDT | 415 |
| IOTAUSDT | 1021 |
| IOTXUSDT | 315 |
| IOUSDT | 1337 |
| IPUSDC | 153 |
| IPUSDT | 910 |
| IRENUSDT | 204 |
| IRYSUSDT | 451 |
| IWMUSDT | 30 |
| JASMYUSDT | 645 |
| JCTUSDT | 6417 |
| JELLYJELLYUSDT | 1401 |
| JOEUSDT | 125 |
| JPMUSDT | 48 |
| JSTUSDT | 383 |
| JTOUSDT | 7140 |
| JUPUSDT | 1587 |
| KAIAUSDT | 235 |
| KAITOUSDC | 410 |
| KAITOUSDT | 3158 |
| KASUSDT | 472 |
| KATUSDT | 2079 |
| KAVAUSDT | 361 |
| KERNELUSDT | 402 |
| KGENUSDT | 1096 |
| KITEUSDT | 648 |
| KLACUSDT | 167 |
| KMNOUSDT | 493 |
| KNCUSDT | 254 |
| KOMAUSDT | 279 |
| KORUUSDT | 6263 |
| KSMUSDT | 385 |
| KSTRUSDT | 68 |
| LABUSDT | 18280 |
| LAUSDT | 817 |
| LAYERUSDT | 2227 |
| LDOUSDT | 1132 |
| LIGHTUSDT | 788 |
| LINEAUSDT | 386 |
| LINKUSDC | 241 |
| LINKUSDT | 2914 |
| LINKUSD_PERP | 20 |
| LISTAUSDT | 280 |
| LITEUSDT | 432 |
| LITUSDT | 3632 |
| LLYUSDT | 87 |
| LPTUSDT | 318 |
| LQTYUSDT | 202 |
| LRCXUSDT | 197 |
| LSKUSDT | 132 |
| LTCUSDC | 174 |
| LTCUSDT | 2536 |
| LTCUSD_PERP | 12 |
| LUMIAUSDT | 1160 |
| LUNA2USDT | 486 |
| LYNUSDT | 677 |
| MAGICUSDT | 918 |
| MAGMAUSDT | 7724 |
| MANAUSDT | 819 |
| MANTAUSDT | 3789 |
| MANTRAUSDT | 1288 |
| MASKUSDT | 375 |
| MAVIAUSDT | 309 |
| MAVUSDT | 182 |
| MBOXUSDT | 195 |
| MEGAUSDT | 2957 |
| MELANIAUSDT | 563 |
| MEMEUSDT | 1698 |
| MERLUSDT | 463 |
| METAUSDT | 204 |
| METISUSDT | 274 |
| METUSDT | 1744 |
| MEUSDT | 640 |
| MEWUSDT | 256 |
| MINAUSDT | 516 |
| MINIMAXUSDT | 48 |
| MIRAUSDT | 1391 |
| MITOUSDT | 1054 |
| MMTUSDT | 2990 |
| MOCAUSDT | 128 |
| MONUSDT | 507 |
| MOODENGUSDT | 533 |
| MORPHOUSDT | 807 |
| MOVEUSDT | 2490 |
| MOVRUSDT | 773 |
| MRVLUSDT | 2067 |
| MSFTUSDT | 365 |
| MSTRUSDT | 1018 |
| MTLUSDT | 191 |
| MUBARAKUSDT | 470 |
| MUSDT | 4624 |
| MUUSDT | 6717 |
| MUUUSDT | 219 |
| MVLLUSDT | 731 |
| MYXUSDT | 3416 |
| NAORISUSDT | 1226 |
| NATGASUSDT | 308 |
| NBISUSDT | 407 |
| NEARUSDC | 341 |
| NEARUSDT | 6736 |
| NEARUSD_PERP | 31 |
| NEIROUSDT | 1370 |
| NEOUSDC | 46 |
| NEOUSDT | 451 |
| NEWTUSDT | 188 |
| NFLXUSDT | 248 |
| NFPUSDT | 3993 |
| NIGHTUSDT | 678 |
| NILUSDT | 1117 |
| NMRUSDT | 278 |
| NOKUSDT | 624 |
| NOMUSDT | 1829 |
| NOTUSDT | 1430 |
| NOWUSDT | 109 |
| NVDAUSDT | 934 |
| NVOUSDT | 41 |
| NXPCUSDT | 490 |
| OGNUSDT | 897 |
| OGUSDT | 315 |
| ONDOUSDT | 4043 |
| ONDSUSDT | 190 |
| ONEUSDT | 518 |
| ONGUSDT | 593 |
| ONTUSDT | 522 |
| ONUSDT | 865 |
| OPENAIUSDT | 275 |
| OPENUSDT | 1175 |
| OPGUSDT | 4055 |
| OPNUSDT | 6093 |
| OPUSDT | 1787 |
| ORCAUSDT | 823 |
| ORCLUSDT | 500 |
| ORDERUSDT | 131 |
| ORDIUSDC | 388 |
| ORDIUSDT | 2862 |
| OUSDT | 1978 |
| PARTIUSDT | 1267 |
| PAXGUSDT | 1731 |
| PAYPUSDT | 104 |
| PENDLEUSDT | 988 |
| PENGUUSDC | 241 |
| PENGUUSDT | 2101 |
| PEOPLEUSDT | 573 |
| PHAROSUSDT | 515 |
| PHAUSDT | 625 |
| PIEVERSEUSDT | 1188 |
| PIPPINUSDT | 5201 |
| PIXELUSDT | 539 |
| PLAYUSDT | 10425 |
| PLTRUSDT | 225 |
| PLUMEUSDT | 579 |
| PNUTUSDC | 87 |
| PNUTUSDT | 513 |
| POLUSDT | 831 |
| POLYXUSDT | 181 |
| POPCATUSDT | 950 |
| PORTALUSDT | 1970 |
| POWERUSDT | 2442 |
| POWRUSDT | 1672 |
| PRLUSDT | 509 |
| PROMPTUSDT | 216 |
| PROMUSDT | 366 |
| PROVEUSDT | 432 |
| PTBUSDT | 1499 |
| PUMPBTCUSDT | 205 |
| PUMPUSDT | 3125 |
| PUNDIXUSDT | 1392 |
| PYTHUSDT | 1269 |
| QCOMUSDT | 299 |
| QNTUSDT | 226 |
| QNTXUSDT | 301 |
| QQQUSDT | 587 |
| QTUMUSDT | 256 |
| QUSDT | 475 |
| RAREUSDT | 870 |
| RAVEUSDT | 8852 |
| RAYSOLUSDT | 247 |
| RECALLUSDT | 654 |
| REDUSDT | 397 |
| RENDERUSDT | 1301 |
| RESOLVUSDT | 2177 |
| REUSDT | 25480 |
| REZUSDT | 390 |
| RIFUSDT | 11527 |
| RIVERUSDT | 3959 |
| RIVNUSDT | 100 |
| RKLBUSDT | 579 |
| RLCUSDT | 343 |
| ROBOUSDT | 1036 |
| RONINUSDT | 470 |
| ROSEUSDT | 802 |
| RPLUSDT | 2395 |
| RSRUSDT | 529 |
| RUNEUSDT | 779 |
| RVNUSDT | 332 |
| SAFEUSDT | 426 |
| SAGAUSDT | 1450 |
| SAHARAUSDT | 4000 |
| SAMSUNGUSDT | 1074 |
| SANDUSDT | 1144 |
| SANTOSUSDT | 310 |
| SAPIENUSDT | 321 |
| SCRTUSDT | 483 |
| SCRUSDT | 87 |
| SEIUSDT | 1371 |
| SENTUSDT | 2682 |
| SFPUSDT | 204 |
| SHELLUSDT | 149 |
| SIGNUSDT | 323 |
| SIRENUSDT | 21819 |
| SKHYNIXUSDT | 6014 |
| SKHYUSDT | 2463 |
| SKLUSDT | 3838 |
| SKRUSDT | 355 |
| SKYAIUSDT | 7325 |
| SKYUSDT | 413 |
| SLPUSDT | 656 |
| SLXUSDT | 11428 |
| SMCIUSDT | 165 |
| SNDKUSDT | 8424 |
| SNOWUSDT | 46 |
| SNXUSDT | 923 |
| SNXXUSDT | 1028 |
| SOLUSDC | 2894 |
| SOLUSDT | 24459 |
| SOLUSD_260925 | 15 |
| SOLUSD_261225 | 1 |
| SOLUSD_PERP | 150 |
| SOLVUSDT | 223 |
| SOMIUSDT | 371 |
| SONICUSDT | 85 |
| SONYUSDT | 105 |
| SOONUSDT | 536 |
| SOPHUSDT | 305 |
| SOXLUSDT | 7609 |
| SOXSUSDT | 448 |
| SPACEUSDT | 2741 |
| SPCXUSDT | 15728 |
| SPELLUSDT | 1001 |
| SPKUSDT | 668 |
| SPORTFUNUSDT | 1294 |
| SPXUSDT | 1391 |
| SPYUSDT | 160 |
| SQDUSDT | 1071 |
| SQQQUSDT | 174 |
| SSVUSDT | 214 |
| STABLEUSDT | 541 |
| STARUSDT | 345 |
| STBLUSDT | 380 |
| STEEMUSDT | 102 |
| STGUSDT | 11854 |
| STORJUSDT | 531 |
| STOUSDT | 1095 |
| STRCUSDT | 35 |
| STRKUSDT | 756 |
| STXUSDT | 370 |
| STXXUSDT | 344 |
| SUIUSDC | 393 |
| SUIUSDT | 5667 |
| SUIUSD_PERP | 22 |
| SUNUSDT | 123 |
| SUPERUSDT | 325 |
| SUSDT | 2309 |
| SUSHIUSDT | 739 |
| SWARMSUSDT | 960 |
| SXTUSDT | 2765 |
| SYNUSDT | 15578 |
| SYRUPUSDT | 624 |
| TACUSDT | 4846 |
| TAGUSDT | 2610 |
| TAIKOUSDT | 7123 |
| TAKEUSDT | 397 |
| TAOUSDT | 5534 |
| TAUSDT | 595 |
| TERUSDT | 40 |
| THETAUSDT | 858 |
| THEUSDT | 1514 |
| TIAUSDC | 170 |
| TIAUSDT | 1921 |
| TLMUSDT | 9116 |
| TNSRUSDT | 6718 |
| TONUSDT | 1793 |
| TOSHIUSDT | 541 |
| TOWNSUSDT | 314 |
| TQQQUSDT | 242 |
| TRADOORUSDT | 4281 |
| TRBUSDT | 1243 |
| TREEUSDT | 261 |
| TRIAUSDT | 3567 |
| TRUMPUSDC | 253 |
| TRUMPUSDT | 6022 |
| TRUSTUSDT | 312 |
| TRUTHUSDT | 1060 |
| TRXUSDT | 1195 |
| TRXUSD_PERP | 6 |
| TSLAUSDT | 620 |
| TSMUSDT | 400 |
| TSTUSDT | 912 |
| TTWOUSDT | 29 |
| TURBOUSDT | 940 |
| TURTLEUSDT | 138 |
| TUSDT | 5384 |
| TUTUSDT | 314 |
| TWTUSDT | 248 |
| TXNUSDT | 16 |
| TZAUSDT | 30 |
| UAIUSDT | 2697 |
| UBERUSDT | 61 |
| UBUSDT | 13570 |
| UMAUSDT | 221 |
| UNIUSDC | 243 |
| UNIUSDT | 3486 |
| UNIUSD_PERP | 28 |
| URNMUSDT | 68 |
| USARUSDT | 166 |
| USDCUSDT | 58 |
| USELESSUSDT | 1404 |
| USTCUSDT | 328 |
| USUALUSDT | 463 |
| USUSDT | 2851 |
| UVXYUSDT | 170 |
| VANAUSDT | 289 |
| VANRYUSDT | 5284 |
| VELODROMEUSDT | 613 |
| VELVETUSDT | 50594 |
| VETUSDT | 795 |
| VICUSDT | 933 |
| VIRTUALUSDT | 1781 |
| VRTUSDT | 17 |
| VTHOUSDT | 229 |
| VUSDT | 35 |
| VVVUSDT | 3144 |
| WALUSDT | 637 |
| WAXPUSDT | 191 |
| WCTUSDT | 417 |
| WDCUSDT | 682 |
| WENUSDT | 26 |
| WETUSDT | 320 |
| WIFUSDC | 169 |
| WIFUSDT | 2143 |
| WLDUSDC | 1391 |
| WLDUSDT | 16673 |
| WLFIUSDC | 139 |
| WLFIUSDT | 945 |
| WMTUSDT | 57 |
| WOOUSDT | 237 |
| WUSDT | 1102 |
| XAGUSDT | 6110 |
| XAIUSDT | 448 |
| XANUSDT | 1091 |
| XAUTUSDT | 1305 |
| XAUUSDT | 8732 |
| XBIUSDT | 16 |
| XLEUSDT | 63 |
| XLMUSDT | 6503 |
| XLMUSD_PERP | 27 |
| XMRUSDT | 1822 |
| XNYUSDT | 587 |
| XPDUSDT | 178 |
| XPINUSDT | 3051 |
| XPLUSDT | 6108 |
| XPTUSDT | 254 |
| XRPUSDC | 1203 |
| XRPUSDT | 13917 |
| XRPUSD_260925 | 25 |
| XRPUSD_261225 | 1 |
| XRPUSD_PERP | 155 |
| XTZUSDT | 693 |
| XVGUSDT | 325 |
| XVSUSDT | 139 |
| YBUSDT | 329 |
| YFIUSDT | 608 |
| YGGUSDT | 437 |
| ZAMAUSDT | 633 |
| ZBTUSDT | 3472 |
| ZECUSDC | 1706 |
| ZECUSDT | 21567 |
| ZENUSDT | 997 |
| ZEREBROUSDT | 1667 |
| ZESTUSDT | 831 |
| ZETAUSDT | 268 |
| ZHIPUUSDT | 88 |
| ZILUSDT | 447 |
| ZKCUSDT | 1129 |
| ZKPUSDT | 1628 |
| ZKUSDT | 583 |
| ZMUSDT | 70 |
| ZORAUSDT | 539 |
| ZROUSDT | 1469 |
| ZRXUSDT | 273 |
| 币安人生USDT | 3273 |
| 我踏马来了USDT | 865 |
| 龙虾USDT | 2481 |

### BTC/ETH Timestamp Coverage

| symbol | event_count | min_ts_ms | min_ts_utc | max_ts_ms | max_ts_utc |
|---|---|---|---|---|---|
| BTCUSDT | 117312 | 1771165875289 | 2026-02-15T14:31:15.289Z | 1784383904435 | 2026-07-18T14:11:44.435Z |
| ETHUSDT | 125343 | 1771165818195 | 2026-02-15T14:30:18.195Z | 1784384075529 | 2026-07-18T14:14:35.529Z |

`price`, `quantity`, and stored `notional` are all present. Across BTC/ETH, 0 rows differ from `price * quantity` beyond max($0.01, 1e-9 relative), so stored notional is used and does not need derivation. Side values are `BUY` and `SELL`. Binance forced-order `SELL` closes a long, while `BUY` closes a short; the signed convention is therefore **long-liquidation (`SELL`) notional positive and short-liquidation (`BUY`) notional negative**.

Coverage stop-gate result: BTC is active on 112/112 ETH-active UTC days (100.0%). The stop gate is temporal: below 90% indicates materially thinner collection; raw event-count differences alone can reflect different liquidation incidence. BTC total event count is 93.6% of ETH, but collection dates and outage days align; BTC coverage is not materially thinner.

### Full Daily Coverage Map (UTC, Before Gap Exclusion)

| day | BTCUSDT | ETHUSDT |
|---|---|---|
| 2026-02-15 | 402 | 644 |
| 2026-02-16 | 1108 | 1321 |
| 2026-02-17 | 1140 | 1317 |
| 2026-02-18 | 1133 | 1296 |
| 2026-02-19 | 1025 | 1201 |
| 2026-02-20 | 1019 | 1106 |
| 2026-02-21 | 398 | 443 |
| 2026-02-22 | 420 | 601 |
| 2026-02-23 | 1799 | 1924 |
| 2026-02-24 | 1570 | 1544 |
| 2026-02-25 | 1928 | 2091 |
| 2026-02-26 | 1415 | 1573 |
| 2026-02-27 | 1304 | 1602 |
| 2026-02-28 | 1916 | 2062 |
| 2026-03-01 | 1483 | 1924 |
| 2026-03-02 | 1959 | 2079 |
| 2026-03-03 | 1923 | 2111 |
| 2026-03-04 | 2436 | 2475 |
| 2026-03-05 | 1584 | 1836 |
| 2026-03-06 | 1482 | 1521 |
| 2026-03-07 | 675 | 688 |
| 2026-03-08 | 1146 | 1103 |
| 2026-03-09 | 1542 | 1617 |
| 2026-03-10 | 1822 | 1529 |
| 2026-03-11 | 1298 | 1258 |
| 2026-03-12 | 1096 | 1186 |
| 2026-03-13 | 1701 | 1779 |
| 2026-03-14 | 459 | 520 |
| 2026-03-15 | 741 | 919 |
| 2026-03-16 | 1713 | 2347 |
| 2026-03-17 | 1444 | 1659 |
| 2026-03-18 | 1585 | 1933 |
| 2026-03-19 | 1830 | 2229 |
| 2026-03-20 | 1205 | 1360 |
| 2026-03-21 | 525 | 569 |
| 2026-03-22 | 1268 | 1503 |
| 2026-03-23 | 1901 | 2119 |
| 2026-03-24 | 1130 | 1183 |
| 2026-03-25 | 943 | 978 |
| 2026-03-26 | 952 | 1050 |
| 2026-03-27 | 1386 | 1268 |
| 2026-03-28 | 628 | 619 |
| 2026-03-29 | 705 | 720 |
| 2026-03-30 | 1036 | 1193 |
| 2026-03-31 | 1351 | 1509 |
| 2026-04-01 | 1092 | 1318 |
| 2026-04-02 | 1264 | 1281 |
| 2026-04-03 | 104 | 114 |
| 2026-04-04 | 105 | 131 |
| 2026-04-05 | 373 | 407 |
| 2026-04-06 | 314 | 391 |
| 2026-04-07 | 737 | 822 |
| 2026-04-08 | 518 | 575 |
| 2026-04-09 | 573 | 625 |
| 2026-04-10 | 320 | 330 |
| 2026-04-11 | 390 | 508 |
| 2026-04-12 | 964 | 1089 |
| 2026-04-13 | 1355 | 1559 |
| 2026-04-14 | 1413 | 1563 |
| 2026-04-15 | 1025 | 1086 |
| 2026-04-16 | 1197 | 1223 |
| 2026-04-17 | 1568 | 1553 |
| 2026-04-18 | 1073 | 1044 |
| 2026-04-19 | 1178 | 1256 |
| 2026-04-20 | 924 | 1141 |
| 2026-04-21 | 1039 | 958 |
| 2026-04-22 | 1175 | 1147 |
| 2026-04-23 | 789 | 751 |
| 2026-04-24 | 271 | 280 |
| 2026-04-25 | 305 | 291 |
| 2026-04-26 | 448 | 411 |
| 2026-04-27 | 271 | 285 |
| 2026-04-28 | 0 | 0 |
| 2026-04-29 | 0 | 0 |
| 2026-04-30 | 0 | 0 |
| 2026-05-01 | 0 | 0 |
| 2026-05-02 | 0 | 0 |
| 2026-05-03 | 0 | 0 |
| 2026-05-04 | 0 | 0 |
| 2026-05-05 | 0 | 0 |
| 2026-05-06 | 0 | 0 |
| 2026-05-07 | 0 | 0 |
| 2026-05-08 | 0 | 0 |
| 2026-05-09 | 0 | 0 |
| 2026-05-10 | 0 | 0 |
| 2026-05-11 | 0 | 0 |
| 2026-05-12 | 0 | 0 |
| 2026-05-13 | 0 | 0 |
| 2026-05-14 | 0 | 0 |
| 2026-05-15 | 0 | 0 |
| 2026-05-16 | 0 | 0 |
| 2026-05-17 | 0 | 0 |
| 2026-05-18 | 0 | 0 |
| 2026-05-19 | 0 | 0 |
| 2026-05-20 | 0 | 0 |
| 2026-05-21 | 0 | 0 |
| 2026-05-22 | 0 | 0 |
| 2026-05-23 | 0 | 0 |
| 2026-05-24 | 0 | 0 |
| 2026-05-25 | 0 | 0 |
| 2026-05-26 | 0 | 0 |
| 2026-05-27 | 0 | 0 |
| 2026-05-28 | 0 | 0 |
| 2026-05-29 | 0 | 0 |
| 2026-05-30 | 0 | 0 |
| 2026-05-31 | 0 | 0 |
| 2026-06-01 | 0 | 0 |
| 2026-06-02 | 0 | 0 |
| 2026-06-03 | 0 | 0 |
| 2026-06-04 | 0 | 0 |
| 2026-06-05 | 0 | 0 |
| 2026-06-06 | 259 | 200 |
| 2026-06-07 | 1592 | 1560 |
| 2026-06-08 | 1376 | 1462 |
| 2026-06-09 | 1524 | 1472 |
| 2026-06-10 | 1609 | 1548 |
| 2026-06-11 | 1102 | 1071 |
| 2026-06-12 | 1039 | 953 |
| 2026-06-13 | 474 | 360 |
| 2026-06-14 | 908 | 749 |
| 2026-06-15 | 1130 | 1375 |
| 2026-06-16 | 960 | 1182 |
| 2026-06-17 | 1292 | 1390 |
| 2026-06-18 | 1313 | 1344 |
| 2026-06-19 | 715 | 789 |
| 2026-06-20 | 614 | 586 |
| 2026-06-21 | 527 | 509 |
| 2026-06-22 | 997 | 976 |
| 2026-06-23 | 1128 | 1138 |
| 2026-06-24 | 2051 | 1942 |
| 2026-06-25 | 1844 | 1723 |
| 2026-06-26 | 1763 | 1653 |
| 2026-06-27 | 625 | 648 |
| 2026-06-28 | 788 | 676 |
| 2026-06-29 | 1376 | 1244 |
| 2026-06-30 | 1160 | 985 |
| 2026-07-01 | 1540 | 1280 |
| 2026-07-02 | 1218 | 1300 |
| 2026-07-03 | 724 | 937 |
| 2026-07-04 | 533 | 724 |
| 2026-07-05 | 562 | 711 |
| 2026-07-06 | 363 | 386 |
| 2026-07-07 | 0 | 0 |
| 2026-07-08 | 0 | 0 |
| 2026-07-09 | 0 | 0 |
| 2026-07-10 | 359 | 361 |
| 2026-07-11 | 345 | 501 |
| 2026-07-12 | 491 | 611 |
| 2026-07-13 | 1044 | 991 |
| 2026-07-14 | 591 | 685 |
| 2026-07-15 | 858 | 1181 |
| 2026-07-16 | 933 | 1035 |
| 2026-07-17 | 1182 | 1333 |
| 2026-07-18 | 92 | 124 |

## Step 1: Series Construction and Gap Exclusion

Both series sum stored forced-order notional into `floor(ts_ms / 1000)` bins. Empty allowed bins are explicitly zero in the statistical population, not missing. Known liquidation collection gaps were parsed from `SYSTEM_STATE.md`; exact exclusion edges are the adjacent observed events surrounding each flagged gap.

| flag | last_event_before_utc | first_event_after_utc | removed_1s_bins |
|---|---|---|---|
| SYSTEM_STATE Apr 23 6.9h gap | 2026-04-23T16:25:18.201Z | 2026-04-23T23:17:21.574Z | 24722 |
| SYSTEM_STATE Apr 24 12.3h gap | 2026-04-24T05:56:19.147Z | 2026-04-24T18:13:39.399Z | 44239 |
| SYSTEM_STATE Apr 27 7.5h gap | 2026-04-27T04:56:11.216Z | 2026-04-27T12:26:26.447Z | 27014 |
| SYSTEM_STATE 40.1-day complete blackout | 2026-04-27T14:27:26.345Z | 2026-06-06T17:43:52.123Z | 3467785 |
| SYSTEM_STATE July 6-10 routed-endpoint outage | 2026-07-06T10:06:39.307Z | 2026-07-10T11:18:06.518Z | 349886 |

Total removed bins: 3,913,646. Remaining aligned bins: 9,304,612. Removed ranges are not bridged in wall-clock time; they are omitted from the measurement population.

## Step 2: Cross-Correlation

Lag header and equation: **`lag_sec_btc_leads_positive`; positive lag means BTC leads ETH; `r = corr(BTC[t], ETH[t + lag])`.** Pearson N is reported at every lag.

### raw_notional

| lag_sec_btc_leads_positive | pearson_r | N |
|---|---|---|
| -30 | 0.00502860 | 9304432 |
| -29 | 0.00570665 | 9304438 |
| -28 | 0.01691032 | 9304444 |
| -27 | 0.00876067 | 9304450 |
| -26 | 0.00464947 | 9304456 |
| -25 | 0.00620817 | 9304462 |
| -24 | 0.00461191 | 9304468 |
| -23 | 0.00811350 | 9304474 |
| -22 | 0.00999353 | 9304480 |
| -21 | 0.00644213 | 9304486 |
| -20 | 0.00689918 | 9304492 |
| -19 | 0.01045534 | 9304498 |
| -18 | 0.00795086 | 9304504 |
| -17 | 0.00993734 | 9304510 |
| -16 | 0.00870125 | 9304516 |
| -15 | 0.00754438 | 9304522 |
| -14 | 0.00954638 | 9304528 |
| -13 | 0.00957814 | 9304534 |
| -12 | 0.00847361 | 9304540 |
| -11 | 0.00996434 | 9304546 |
| -10 | 0.01019370 | 9304552 |
| -9 | 0.01271891 | 9304558 |
| -8 | 0.00765971 | 9304564 |
| -7 | 0.00933944 | 9304570 |
| -6 | 0.01709970 | 9304576 |
| -5 | 0.02121716 | 9304582 |
| -4 | 0.01780521 | 9304588 |
| -3 | 0.02372290 | 9304594 |
| -2 | 0.05132488 | 9304600 |
| -1 | 0.07401559 | 9304606 |
| 0 | 0.15913715 | 9304612 |
| 1 | 0.10208006 | 9304606 |
| 2 | 0.03633642 | 9304600 |
| 3 | 0.02940621 | 9304594 |
| 4 | 0.02260573 | 9304588 |
| 5 | 0.02671293 | 9304582 |
| 6 | 0.01473040 | 9304576 |
| 7 | 0.01056763 | 9304570 |
| 8 | 0.01567026 | 9304564 |
| 9 | 0.02229929 | 9304558 |
| 10 | 0.01173891 | 9304552 |
| 11 | 0.05401028 | 9304546 |
| 12 | 0.01067876 | 9304540 |
| 13 | 0.00985279 | 9304534 |
| 14 | 0.00539004 | 9304528 |
| 15 | 0.01165519 | 9304522 |
| 16 | 0.00961453 | 9304516 |
| 17 | 0.01924011 | 9304510 |
| 18 | 0.01324061 | 9304504 |
| 19 | 0.01448496 | 9304498 |
| 20 | 0.01202176 | 9304492 |
| 21 | 0.00687143 | 9304486 |
| 22 | 0.00791868 | 9304480 |
| 23 | 0.00705999 | 9304474 |
| 24 | 0.00394000 | 9304468 |
| 25 | 0.00733093 | 9304462 |
| 26 | 0.00591601 | 9304456 |
| 27 | 0.00940108 | 9304450 |
| 28 | 0.01680948 | 9304444 |
| 29 | 0.00809721 | 9304438 |
| 30 | 0.00576772 | 9304432 |
### log1p_notional

| lag_sec_btc_leads_positive | pearson_r | N |
|---|---|---|
| -30 | 0.05254998 | 9304432 |
| -29 | 0.05393801 | 9304438 |
| -28 | 0.05512539 | 9304444 |
| -27 | 0.05605906 | 9304450 |
| -26 | 0.05765371 | 9304456 |
| -25 | 0.05844507 | 9304462 |
| -24 | 0.06069678 | 9304468 |
| -23 | 0.06171974 | 9304474 |
| -22 | 0.06414025 | 9304480 |
| -21 | 0.06492823 | 9304486 |
| -20 | 0.06742050 | 9304492 |
| -19 | 0.06850746 | 9304498 |
| -18 | 0.07021579 | 9304504 |
| -17 | 0.07252771 | 9304510 |
| -16 | 0.07449084 | 9304516 |
| -15 | 0.07716755 | 9304522 |
| -14 | 0.07857903 | 9304528 |
| -13 | 0.08052602 | 9304534 |
| -12 | 0.08469025 | 9304540 |
| -11 | 0.09020663 | 9304546 |
| -10 | 0.09496736 | 9304552 |
| -9 | 0.09904859 | 9304558 |
| -8 | 0.10392283 | 9304564 |
| -7 | 0.11007314 | 9304570 |
| -6 | 0.11984198 | 9304576 |
| -5 | 0.12799102 | 9304582 |
| -4 | 0.14075196 | 9304588 |
| -3 | 0.16062896 | 9304594 |
| -2 | 0.19379434 | 9304600 |
| -1 | 0.22979605 | 9304606 |
| 0 | 0.39597000 | 9304612 |
| 1 | 0.23351719 | 9304606 |
| 2 | 0.19624816 | 9304600 |
| 3 | 0.15992631 | 9304594 |
| 4 | 0.13905462 | 9304588 |
| 5 | 0.12699275 | 9304582 |
| 6 | 0.11814930 | 9304576 |
| 7 | 0.10961591 | 9304570 |
| 8 | 0.10325953 | 9304564 |
| 9 | 0.09849532 | 9304558 |
| 10 | 0.09347803 | 9304552 |
| 11 | 0.08895903 | 9304546 |
| 12 | 0.08530653 | 9304540 |
| 13 | 0.08227129 | 9304534 |
| 14 | 0.07959462 | 9304528 |
| 15 | 0.07654647 | 9304522 |
| 16 | 0.07453828 | 9304516 |
| 17 | 0.07299997 | 9304510 |
| 18 | 0.07026951 | 9304504 |
| 19 | 0.06915217 | 9304498 |
| 20 | 0.06665465 | 9304492 |
| 21 | 0.06419558 | 9304486 |
| 22 | 0.06264386 | 9304480 |
| 23 | 0.06031469 | 9304474 |
| 24 | 0.05866615 | 9304468 |
| 25 | 0.05694540 | 9304462 |
| 26 | 0.05708266 | 9304456 |
| 27 | 0.05551724 | 9304450 |
| 28 | 0.05412914 | 9304444 |
| 29 | 0.05270122 | 9304438 |
| 30 | 0.05155502 | 9304432 |
### signed_notional

| lag_sec_btc_leads_positive | pearson_r | N |
|---|---|---|
| -30 | 0.00505615 | 9304432 |
| -29 | 0.00579025 | 9304438 |
| -28 | 0.01667394 | 9304444 |
| -27 | 0.00877913 | 9304450 |
| -26 | 0.00471275 | 9304456 |
| -25 | 0.00608043 | 9304462 |
| -24 | 0.00457109 | 9304468 |
| -23 | 0.00814669 | 9304474 |
| -22 | 0.01004356 | 9304480 |
| -21 | 0.00651516 | 9304486 |
| -20 | 0.00689496 | 9304492 |
| -19 | 0.01057176 | 9304498 |
| -18 | 0.00805115 | 9304504 |
| -17 | 0.01001601 | 9304510 |
| -16 | 0.00876050 | 9304516 |
| -15 | 0.00754416 | 9304522 |
| -14 | 0.00966204 | 9304528 |
| -13 | 0.00968197 | 9304534 |
| -12 | 0.00857045 | 9304540 |
| -11 | 0.01006544 | 9304546 |
| -10 | 0.01033115 | 9304552 |
| -9 | 0.01279152 | 9304558 |
| -8 | 0.00770782 | 9304564 |
| -7 | 0.00883791 | 9304570 |
| -6 | 0.01678113 | 9304576 |
| -5 | 0.02122993 | 9304582 |
| -4 | 0.01789913 | 9304588 |
| -3 | 0.02383161 | 9304594 |
| -2 | 0.05140695 | 9304600 |
| -1 | 0.07395613 | 9304606 |
| 0 | 0.15688006 | 9304612 |
| 1 | 0.10205787 | 9304606 |
| 2 | 0.03634174 | 9304600 |
| 3 | 0.02947657 | 9304594 |
| 4 | 0.02271546 | 9304588 |
| 5 | 0.02680757 | 9304582 |
| 6 | 0.01476257 | 9304576 |
| 7 | 0.01061658 | 9304570 |
| 8 | 0.01577107 | 9304564 |
| 9 | 0.02238828 | 9304558 |
| 10 | 0.01185998 | 9304552 |
| 11 | 0.05410073 | 9304546 |
| 12 | 0.01070716 | 9304540 |
| 13 | 0.00980095 | 9304534 |
| 14 | 0.00548781 | 9304528 |
| 15 | 0.01174496 | 9304522 |
| 16 | 0.00971921 | 9304516 |
| 17 | 0.01933920 | 9304510 |
| 18 | 0.01331287 | 9304504 |
| 19 | 0.01451978 | 9304498 |
| 20 | 0.01191667 | 9304492 |
| 21 | 0.00694393 | 9304486 |
| 22 | 0.00783227 | 9304480 |
| 23 | 0.00709658 | 9304474 |
| 24 | 0.00392184 | 9304468 |
| 25 | 0.00723653 | 9304462 |
| 26 | 0.00596897 | 9304456 |
| 27 | 0.00945822 | 9304450 |
| 28 | 0.01681614 | 9304444 |
| 29 | 0.00816520 | 9304438 |
| 30 | 0.00577594 | 9304432 |

## Step 3: Monthly Stability

Each non-overlapping UTC calendar month is recomputed after the same gap exclusions. `argmax_lag_sec_btc_leads_positive` maximizes signed Pearson r, not absolute r.

| month_utc | variant | usable_bins | argmax_lag_sec_btc_leads_positive | peak_r | N_at_peak |
|---|---|---|---|---|---|
| 2026-02 | raw_notional | 1157382 | 0 | 0.27847679 | 1157382 |
| 2026-02 | log1p_notional | 1157382 | 0 | 0.36666564 | 1157382 |
| 2026-02 | signed_notional | 1157382 | 0 | 0.27815153 | 1157382 |
| 2026-03 | raw_notional | 2678400 | 0 | 0.01920448 | 2678400 |
| 2026-03 | log1p_notional | 2678400 | 0 | 0.37722606 | 2678400 |
| 2026-03 | signed_notional | 2678400 | 0 | 0.01626056 | 2678400 |
| 2026-04 | raw_notional | 2202472 | 0 | 0.12497760 | 2202472 |
| 2026-04 | log1p_notional | 2202472 | 0 | 0.41483106 | 2202472 |
| 2026-04 | signed_notional | 2202472 | 0 | 0.12458867 | 2202472 |
| 2026-06 | raw_notional | 2096168 | 0 | 0.19893054 | 2096168 |
| 2026-06 | log1p_notional | 2096168 | 0 | 0.41920440 | 2096168 |
| 2026-06 | signed_notional | 2096168 | 0 | 0.19889450 | 2096168 |
| 2026-07 | raw_notional | 1170190 | 0 | 0.18630669 | 1170190 |
| 2026-07 | log1p_notional | 1170190 | 0 | 0.39451740 | 1170190 |
| 2026-07 | signed_notional | 1170190 | 0 | 0.17031684 | 1170190 |

### Full Monthly Correlograms

Every row retains the convention **positive lag means BTC leads ETH**, computed as `corr(BTC[t], ETH[t + lag])`.

| month_utc | variant | lag_sec_btc_leads_positive | pearson_r | N |
|---|---|---|---|---|
| 2026-02 | raw_notional | -30 | 0.00058597 | 1157352 |
| 2026-02 | raw_notional | -29 | 0.00056769 | 1157353 |
| 2026-02 | raw_notional | -28 | 0.00093890 | 1157354 |
| 2026-02 | raw_notional | -27 | 0.00075133 | 1157355 |
| 2026-02 | raw_notional | -26 | 0.00115198 | 1157356 |
| 2026-02 | raw_notional | -25 | 0.00182005 | 1157357 |
| 2026-02 | raw_notional | -24 | 0.00033436 | 1157358 |
| 2026-02 | raw_notional | -23 | 0.00080050 | 1157359 |
| 2026-02 | raw_notional | -22 | 0.00036621 | 1157360 |
| 2026-02 | raw_notional | -21 | 0.00051272 | 1157361 |
| 2026-02 | raw_notional | -20 | 0.00093061 | 1157362 |
| 2026-02 | raw_notional | -19 | 0.00046752 | 1157363 |
| 2026-02 | raw_notional | -18 | 0.00114664 | 1157364 |
| 2026-02 | raw_notional | -17 | 0.00052660 | 1157365 |
| 2026-02 | raw_notional | -16 | 0.00074931 | 1157366 |
| 2026-02 | raw_notional | -15 | 0.00084205 | 1157367 |
| 2026-02 | raw_notional | -14 | 0.00349231 | 1157368 |
| 2026-02 | raw_notional | -13 | 0.00082448 | 1157369 |
| 2026-02 | raw_notional | -12 | 0.00112839 | 1157370 |
| 2026-02 | raw_notional | -11 | 0.00247951 | 1157371 |
| 2026-02 | raw_notional | -10 | 0.00155064 | 1157372 |
| 2026-02 | raw_notional | -9 | 0.00103220 | 1157373 |
| 2026-02 | raw_notional | -8 | 0.00082877 | 1157374 |
| 2026-02 | raw_notional | -7 | 0.00397796 | 1157375 |
| 2026-02 | raw_notional | -6 | 0.00107432 | 1157376 |
| 2026-02 | raw_notional | -5 | 0.00135576 | 1157377 |
| 2026-02 | raw_notional | -4 | 0.01840872 | 1157378 |
| 2026-02 | raw_notional | -3 | 0.00119792 | 1157379 |
| 2026-02 | raw_notional | -2 | 0.00467012 | 1157380 |
| 2026-02 | raw_notional | -1 | 0.00538796 | 1157381 |
| 2026-02 | raw_notional | 0 | 0.27847679 | 1157382 |
| 2026-02 | raw_notional | 1 | 0.00318965 | 1157381 |
| 2026-02 | raw_notional | 2 | 0.00290153 | 1157380 |
| 2026-02 | raw_notional | 3 | 0.00145295 | 1157379 |
| 2026-02 | raw_notional | 4 | 0.00452029 | 1157378 |
| 2026-02 | raw_notional | 5 | 0.00191882 | 1157377 |
| 2026-02 | raw_notional | 6 | 0.00311890 | 1157376 |
| 2026-02 | raw_notional | 7 | 0.00169964 | 1157375 |
| 2026-02 | raw_notional | 8 | 0.00155305 | 1157374 |
| 2026-02 | raw_notional | 9 | 0.00140959 | 1157373 |
| 2026-02 | raw_notional | 10 | 0.00070867 | 1157372 |
| 2026-02 | raw_notional | 11 | 0.00142511 | 1157371 |
| 2026-02 | raw_notional | 12 | 0.00112431 | 1157370 |
| 2026-02 | raw_notional | 13 | 0.00068790 | 1157369 |
| 2026-02 | raw_notional | 14 | 0.00064152 | 1157368 |
| 2026-02 | raw_notional | 15 | 0.00836631 | 1157367 |
| 2026-02 | raw_notional | 16 | 0.00090481 | 1157366 |
| 2026-02 | raw_notional | 17 | 0.00078339 | 1157365 |
| 2026-02 | raw_notional | 18 | 0.01912504 | 1157364 |
| 2026-02 | raw_notional | 19 | 0.00054590 | 1157363 |
| 2026-02 | raw_notional | 20 | 0.00036174 | 1157362 |
| 2026-02 | raw_notional | 21 | 0.00042801 | 1157361 |
| 2026-02 | raw_notional | 22 | 0.00089292 | 1157360 |
| 2026-02 | raw_notional | 23 | 0.00027859 | 1157359 |
| 2026-02 | raw_notional | 24 | 0.00081034 | 1157358 |
| 2026-02 | raw_notional | 25 | 0.00023502 | 1157357 |
| 2026-02 | raw_notional | 26 | 0.00118577 | 1157356 |
| 2026-02 | raw_notional | 27 | 0.00025577 | 1157355 |
| 2026-02 | raw_notional | 28 | 0.00026987 | 1157354 |
| 2026-02 | raw_notional | 29 | 0.00011756 | 1157353 |
| 2026-02 | raw_notional | 30 | 0.00043099 | 1157352 |
| 2026-02 | log1p_notional | -30 | 0.05444116 | 1157352 |
| 2026-02 | log1p_notional | -29 | 0.05533692 | 1157353 |
| 2026-02 | log1p_notional | -28 | 0.05962509 | 1157354 |
| 2026-02 | log1p_notional | -27 | 0.05824408 | 1157355 |
| 2026-02 | log1p_notional | -26 | 0.06144774 | 1157356 |
| 2026-02 | log1p_notional | -25 | 0.06038713 | 1157357 |
| 2026-02 | log1p_notional | -24 | 0.06557411 | 1157358 |
| 2026-02 | log1p_notional | -23 | 0.06476484 | 1157359 |
| 2026-02 | log1p_notional | -22 | 0.06206181 | 1157360 |
| 2026-02 | log1p_notional | -21 | 0.06728728 | 1157361 |
| 2026-02 | log1p_notional | -20 | 0.06754950 | 1157362 |
| 2026-02 | log1p_notional | -19 | 0.06459395 | 1157363 |
| 2026-02 | log1p_notional | -18 | 0.06777154 | 1157364 |
| 2026-02 | log1p_notional | -17 | 0.07256051 | 1157365 |
| 2026-02 | log1p_notional | -16 | 0.07064183 | 1157366 |
| 2026-02 | log1p_notional | -15 | 0.07465453 | 1157367 |
| 2026-02 | log1p_notional | -14 | 0.07909341 | 1157368 |
| 2026-02 | log1p_notional | -13 | 0.07986242 | 1157369 |
| 2026-02 | log1p_notional | -12 | 0.08245512 | 1157370 |
| 2026-02 | log1p_notional | -11 | 0.08792725 | 1157371 |
| 2026-02 | log1p_notional | -10 | 0.09214905 | 1157372 |
| 2026-02 | log1p_notional | -9 | 0.09422993 | 1157373 |
| 2026-02 | log1p_notional | -8 | 0.09920669 | 1157374 |
| 2026-02 | log1p_notional | -7 | 0.10434169 | 1157375 |
| 2026-02 | log1p_notional | -6 | 0.11251596 | 1157376 |
| 2026-02 | log1p_notional | -5 | 0.12159381 | 1157377 |
| 2026-02 | log1p_notional | -4 | 0.13360164 | 1157378 |
| 2026-02 | log1p_notional | -3 | 0.14805099 | 1157379 |
| 2026-02 | log1p_notional | -2 | 0.17968453 | 1157380 |
| 2026-02 | log1p_notional | -1 | 0.20670275 | 1157381 |
| 2026-02 | log1p_notional | 0 | 0.36666564 | 1157382 |
| 2026-02 | log1p_notional | 1 | 0.21109461 | 1157381 |
| 2026-02 | log1p_notional | 2 | 0.18148984 | 1157380 |
| 2026-02 | log1p_notional | 3 | 0.14699703 | 1157379 |
| 2026-02 | log1p_notional | 4 | 0.12653916 | 1157378 |
| 2026-02 | log1p_notional | 5 | 0.11683836 | 1157377 |
| 2026-02 | log1p_notional | 6 | 0.11142734 | 1157376 |
| 2026-02 | log1p_notional | 7 | 0.10361831 | 1157375 |
| 2026-02 | log1p_notional | 8 | 0.09830310 | 1157374 |
| 2026-02 | log1p_notional | 9 | 0.09548691 | 1157373 |
| 2026-02 | log1p_notional | 10 | 0.09092553 | 1157372 |
| 2026-02 | log1p_notional | 11 | 0.08779475 | 1157371 |
| 2026-02 | log1p_notional | 12 | 0.08226540 | 1157370 |
| 2026-02 | log1p_notional | 13 | 0.07931941 | 1157369 |
| 2026-02 | log1p_notional | 14 | 0.07791714 | 1157368 |
| 2026-02 | log1p_notional | 15 | 0.07718472 | 1157367 |
| 2026-02 | log1p_notional | 16 | 0.07224423 | 1157366 |
| 2026-02 | log1p_notional | 17 | 0.07042030 | 1157365 |
| 2026-02 | log1p_notional | 18 | 0.06887706 | 1157364 |
| 2026-02 | log1p_notional | 19 | 0.06643357 | 1157363 |
| 2026-02 | log1p_notional | 20 | 0.06549270 | 1157362 |
| 2026-02 | log1p_notional | 21 | 0.06362048 | 1157361 |
| 2026-02 | log1p_notional | 22 | 0.06189855 | 1157360 |
| 2026-02 | log1p_notional | 23 | 0.05864189 | 1157359 |
| 2026-02 | log1p_notional | 24 | 0.05926534 | 1157358 |
| 2026-02 | log1p_notional | 25 | 0.05917508 | 1157357 |
| 2026-02 | log1p_notional | 26 | 0.05711252 | 1157356 |
| 2026-02 | log1p_notional | 27 | 0.05588059 | 1157355 |
| 2026-02 | log1p_notional | 28 | 0.05355304 | 1157354 |
| 2026-02 | log1p_notional | 29 | 0.05244958 | 1157353 |
| 2026-02 | log1p_notional | 30 | 0.05207388 | 1157352 |
| 2026-02 | signed_notional | -30 | 0.00063609 | 1157352 |
| 2026-02 | signed_notional | -29 | 0.00066974 | 1157353 |
| 2026-02 | signed_notional | -28 | 0.00101299 | 1157354 |
| 2026-02 | signed_notional | -27 | 0.00080731 | 1157355 |
| 2026-02 | signed_notional | -26 | 0.00126380 | 1157356 |
| 2026-02 | signed_notional | -25 | 0.00196663 | 1157357 |
| 2026-02 | signed_notional | -24 | 0.00030171 | 1157358 |
| 2026-02 | signed_notional | -23 | 0.00095341 | 1157359 |
| 2026-02 | signed_notional | -22 | 0.00045164 | 1157360 |
| 2026-02 | signed_notional | -21 | 0.00063242 | 1157361 |
| 2026-02 | signed_notional | -20 | 0.00095306 | 1157362 |
| 2026-02 | signed_notional | -19 | 0.00052090 | 1157363 |
| 2026-02 | signed_notional | -18 | 0.00130208 | 1157364 |
| 2026-02 | signed_notional | -17 | 0.00061080 | 1157365 |
| 2026-02 | signed_notional | -16 | 0.00090890 | 1157366 |
| 2026-02 | signed_notional | -15 | 0.00093997 | 1157367 |
| 2026-02 | signed_notional | -14 | 0.00364577 | 1157368 |
| 2026-02 | signed_notional | -13 | 0.00095602 | 1157369 |
| 2026-02 | signed_notional | -12 | 0.00124904 | 1157370 |
| 2026-02 | signed_notional | -11 | 0.00221037 | 1157371 |
| 2026-02 | signed_notional | -10 | 0.00170742 | 1157372 |
| 2026-02 | signed_notional | -9 | 0.00107507 | 1157373 |
| 2026-02 | signed_notional | -8 | 0.00099414 | 1157374 |
| 2026-02 | signed_notional | -7 | 0.00407118 | 1157375 |
| 2026-02 | signed_notional | -6 | 0.00123251 | 1157376 |
| 2026-02 | signed_notional | -5 | 0.00150189 | 1157377 |
| 2026-02 | signed_notional | -4 | 0.01856838 | 1157378 |
| 2026-02 | signed_notional | -3 | 0.00136169 | 1157379 |
| 2026-02 | signed_notional | -2 | 0.00482951 | 1157380 |
| 2026-02 | signed_notional | -1 | 0.00551243 | 1157381 |
| 2026-02 | signed_notional | 0 | 0.27815153 | 1157382 |
| 2026-02 | signed_notional | 1 | 0.00331017 | 1157381 |
| 2026-02 | signed_notional | 2 | 0.00306478 | 1157380 |
| 2026-02 | signed_notional | 3 | 0.00159870 | 1157379 |
| 2026-02 | signed_notional | 4 | 0.00468540 | 1157378 |
| 2026-02 | signed_notional | 5 | 0.00202875 | 1157377 |
| 2026-02 | signed_notional | 6 | 0.00325899 | 1157376 |
| 2026-02 | signed_notional | 7 | 0.00013379 | 1157375 |
| 2026-02 | signed_notional | 8 | 0.00167924 | 1157374 |
| 2026-02 | signed_notional | 9 | 0.00092464 | 1157373 |
| 2026-02 | signed_notional | 10 | 0.00086591 | 1157372 |
| 2026-02 | signed_notional | 11 | 0.00138895 | 1157371 |
| 2026-02 | signed_notional | 12 | 0.00125996 | 1157370 |
| 2026-02 | signed_notional | 13 | 0.00074168 | 1157369 |
| 2026-02 | signed_notional | 14 | 0.00078207 | 1157368 |
| 2026-02 | signed_notional | 15 | 0.00846842 | 1157367 |
| 2026-02 | signed_notional | 16 | 0.00099357 | 1157366 |
| 2026-02 | signed_notional | 17 | 0.00090314 | 1157365 |
| 2026-02 | signed_notional | 18 | 0.01926977 | 1157364 |
| 2026-02 | signed_notional | 19 | 0.00054555 | 1157363 |
| 2026-02 | signed_notional | 20 | 0.00052386 | 1157362 |
| 2026-02 | signed_notional | 21 | 0.00057577 | 1157361 |
| 2026-02 | signed_notional | 22 | 0.00098093 | 1157360 |
| 2026-02 | signed_notional | 23 | 0.00027328 | 1157359 |
| 2026-02 | signed_notional | 24 | 0.00094391 | 1157358 |
| 2026-02 | signed_notional | 25 | 0.00037538 | 1157357 |
| 2026-02 | signed_notional | 26 | 0.00130040 | 1157356 |
| 2026-02 | signed_notional | 27 | 0.00039167 | 1157355 |
| 2026-02 | signed_notional | 28 | 0.00033655 | 1157354 |
| 2026-02 | signed_notional | 29 | 0.00025059 | 1157353 |
| 2026-02 | signed_notional | 30 | 0.00056199 | 1157352 |
| 2026-03 | raw_notional | -30 | 0.00283035 | 2678370 |
| 2026-03 | raw_notional | -29 | 0.00333243 | 2678371 |
| 2026-03 | raw_notional | -28 | 0.00054076 | 2678372 |
| 2026-03 | raw_notional | -27 | 0.01146299 | 2678373 |
| 2026-03 | raw_notional | -26 | 0.00527425 | 2678374 |
| 2026-03 | raw_notional | -25 | 0.00077118 | 2678375 |
| 2026-03 | raw_notional | -24 | 0.00063323 | 2678376 |
| 2026-03 | raw_notional | -23 | 0.00152097 | 2678377 |
| 2026-03 | raw_notional | -22 | 0.00246733 | 2678378 |
| 2026-03 | raw_notional | -21 | 0.00241137 | 2678379 |
| 2026-03 | raw_notional | -20 | 0.00228167 | 2678380 |
| 2026-03 | raw_notional | -19 | 0.00144287 | 2678381 |
| 2026-03 | raw_notional | -18 | 0.00097865 | 2678382 |
| 2026-03 | raw_notional | -17 | 0.00076145 | 2678383 |
| 2026-03 | raw_notional | -16 | 0.00111108 | 2678384 |
| 2026-03 | raw_notional | -15 | 0.00192177 | 2678385 |
| 2026-03 | raw_notional | -14 | 0.00163123 | 2678386 |
| 2026-03 | raw_notional | -13 | 0.00176244 | 2678387 |
| 2026-03 | raw_notional | -12 | 0.00099099 | 2678388 |
| 2026-03 | raw_notional | -11 | 0.00196182 | 2678389 |
| 2026-03 | raw_notional | -10 | 0.00163553 | 2678390 |
| 2026-03 | raw_notional | -9 | 0.00181491 | 2678391 |
| 2026-03 | raw_notional | -8 | 0.00135061 | 2678392 |
| 2026-03 | raw_notional | -7 | 0.00160981 | 2678393 |
| 2026-03 | raw_notional | -6 | 0.00453424 | 2678394 |
| 2026-03 | raw_notional | -5 | 0.00190127 | 2678395 |
| 2026-03 | raw_notional | -4 | 0.00274121 | 2678396 |
| 2026-03 | raw_notional | -3 | 0.00331019 | 2678397 |
| 2026-03 | raw_notional | -2 | 0.00321095 | 2678398 |
| 2026-03 | raw_notional | -1 | 0.00207371 | 2678399 |
| 2026-03 | raw_notional | 0 | 0.01920448 | 2678400 |
| 2026-03 | raw_notional | 1 | 0.00953547 | 2678399 |
| 2026-03 | raw_notional | 2 | 0.00328016 | 2678398 |
| 2026-03 | raw_notional | 3 | 0.00269289 | 2678397 |
| 2026-03 | raw_notional | 4 | 0.00215225 | 2678396 |
| 2026-03 | raw_notional | 5 | 0.00246419 | 2678395 |
| 2026-03 | raw_notional | 6 | 0.00205232 | 2678394 |
| 2026-03 | raw_notional | 7 | 0.00151387 | 2678393 |
| 2026-03 | raw_notional | 8 | 0.00123878 | 2678392 |
| 2026-03 | raw_notional | 9 | 0.00232937 | 2678391 |
| 2026-03 | raw_notional | 10 | 0.00271678 | 2678390 |
| 2026-03 | raw_notional | 11 | 0.00189548 | 2678389 |
| 2026-03 | raw_notional | 12 | 0.00282347 | 2678388 |
| 2026-03 | raw_notional | 13 | 0.00188827 | 2678387 |
| 2026-03 | raw_notional | 14 | 0.00106918 | 2678386 |
| 2026-03 | raw_notional | 15 | 0.00182677 | 2678385 |
| 2026-03 | raw_notional | 16 | 0.00078535 | 2678384 |
| 2026-03 | raw_notional | 17 | 0.00054753 | 2678383 |
| 2026-03 | raw_notional | 18 | 0.00040582 | 2678382 |
| 2026-03 | raw_notional | 19 | 0.00079567 | 2678381 |
| 2026-03 | raw_notional | 20 | 0.00148290 | 2678380 |
| 2026-03 | raw_notional | 21 | 0.00074727 | 2678379 |
| 2026-03 | raw_notional | 22 | 0.00083988 | 2678378 |
| 2026-03 | raw_notional | 23 | 0.00130374 | 2678377 |
| 2026-03 | raw_notional | 24 | 0.00066497 | 2678376 |
| 2026-03 | raw_notional | 25 | 0.00046297 | 2678375 |
| 2026-03 | raw_notional | 26 | 0.00060280 | 2678374 |
| 2026-03 | raw_notional | 27 | 0.00038329 | 2678373 |
| 2026-03 | raw_notional | 28 | 0.00047108 | 2678372 |
| 2026-03 | raw_notional | 29 | 0.00062542 | 2678371 |
| 2026-03 | raw_notional | 30 | 0.00073665 | 2678370 |
| 2026-03 | log1p_notional | -30 | 0.04938138 | 2678370 |
| 2026-03 | log1p_notional | -29 | 0.04866186 | 2678371 |
| 2026-03 | log1p_notional | -28 | 0.05064118 | 2678372 |
| 2026-03 | log1p_notional | -27 | 0.05243003 | 2678373 |
| 2026-03 | log1p_notional | -26 | 0.05299246 | 2678374 |
| 2026-03 | log1p_notional | -25 | 0.05540266 | 2678375 |
| 2026-03 | log1p_notional | -24 | 0.05704698 | 2678376 |
| 2026-03 | log1p_notional | -23 | 0.05718329 | 2678377 |
| 2026-03 | log1p_notional | -22 | 0.06056525 | 2678378 |
| 2026-03 | log1p_notional | -21 | 0.05944702 | 2678379 |
| 2026-03 | log1p_notional | -20 | 0.06257672 | 2678380 |
| 2026-03 | log1p_notional | -19 | 0.06377622 | 2678381 |
| 2026-03 | log1p_notional | -18 | 0.06616063 | 2678382 |
| 2026-03 | log1p_notional | -17 | 0.06771688 | 2678383 |
| 2026-03 | log1p_notional | -16 | 0.07074806 | 2678384 |
| 2026-03 | log1p_notional | -15 | 0.07257447 | 2678385 |
| 2026-03 | log1p_notional | -14 | 0.07679935 | 2678386 |
| 2026-03 | log1p_notional | -13 | 0.07651105 | 2678387 |
| 2026-03 | log1p_notional | -12 | 0.08128867 | 2678388 |
| 2026-03 | log1p_notional | -11 | 0.08654933 | 2678389 |
| 2026-03 | log1p_notional | -10 | 0.09057116 | 2678390 |
| 2026-03 | log1p_notional | -9 | 0.09391775 | 2678391 |
| 2026-03 | log1p_notional | -8 | 0.09940331 | 2678392 |
| 2026-03 | log1p_notional | -7 | 0.10473822 | 2678393 |
| 2026-03 | log1p_notional | -6 | 0.11582981 | 2678394 |
| 2026-03 | log1p_notional | -5 | 0.12737922 | 2678395 |
| 2026-03 | log1p_notional | -4 | 0.13773858 | 2678396 |
| 2026-03 | log1p_notional | -3 | 0.15671559 | 2678397 |
| 2026-03 | log1p_notional | -2 | 0.18930929 | 2678398 |
| 2026-03 | log1p_notional | -1 | 0.21693322 | 2678399 |
| 2026-03 | log1p_notional | 0 | 0.37722606 | 2678400 |
| 2026-03 | log1p_notional | 1 | 0.22292398 | 2678399 |
| 2026-03 | log1p_notional | 2 | 0.18992340 | 2678398 |
| 2026-03 | log1p_notional | 3 | 0.15669325 | 2678397 |
| 2026-03 | log1p_notional | 4 | 0.13596210 | 2678396 |
| 2026-03 | log1p_notional | 5 | 0.12426803 | 2678395 |
| 2026-03 | log1p_notional | 6 | 0.11510884 | 2678394 |
| 2026-03 | log1p_notional | 7 | 0.10533859 | 2678393 |
| 2026-03 | log1p_notional | 8 | 0.10049832 | 2678392 |
| 2026-03 | log1p_notional | 9 | 0.09411877 | 2678391 |
| 2026-03 | log1p_notional | 10 | 0.08906744 | 2678390 |
| 2026-03 | log1p_notional | 11 | 0.08526319 | 2678389 |
| 2026-03 | log1p_notional | 12 | 0.08265272 | 2678388 |
| 2026-03 | log1p_notional | 13 | 0.07813518 | 2678387 |
| 2026-03 | log1p_notional | 14 | 0.07751898 | 2678386 |
| 2026-03 | log1p_notional | 15 | 0.07174700 | 2678385 |
| 2026-03 | log1p_notional | 16 | 0.07086691 | 2678384 |
| 2026-03 | log1p_notional | 17 | 0.06866844 | 2678383 |
| 2026-03 | log1p_notional | 18 | 0.06571891 | 2678382 |
| 2026-03 | log1p_notional | 19 | 0.06367906 | 2678381 |
| 2026-03 | log1p_notional | 20 | 0.06031018 | 2678380 |
| 2026-03 | log1p_notional | 21 | 0.05849142 | 2678379 |
| 2026-03 | log1p_notional | 22 | 0.05736154 | 2678378 |
| 2026-03 | log1p_notional | 23 | 0.05592100 | 2678377 |
| 2026-03 | log1p_notional | 24 | 0.05472987 | 2678376 |
| 2026-03 | log1p_notional | 25 | 0.05354196 | 2678375 |
| 2026-03 | log1p_notional | 26 | 0.05217836 | 2678374 |
| 2026-03 | log1p_notional | 27 | 0.05153237 | 2678373 |
| 2026-03 | log1p_notional | 28 | 0.05009873 | 2678372 |
| 2026-03 | log1p_notional | 29 | 0.05011058 | 2678371 |
| 2026-03 | log1p_notional | 30 | 0.04840121 | 2678370 |
| 2026-03 | signed_notional | -30 | 0.00295959 | 2678370 |
| 2026-03 | signed_notional | -29 | 0.00337992 | 2678371 |
| 2026-03 | signed_notional | -28 | 0.00066594 | 2678372 |
| 2026-03 | signed_notional | -27 | 0.01158017 | 2678373 |
| 2026-03 | signed_notional | -26 | 0.00534348 | 2678374 |
| 2026-03 | signed_notional | -25 | 0.00090115 | 2678375 |
| 2026-03 | signed_notional | -24 | 0.00071706 | 2678376 |
| 2026-03 | signed_notional | -23 | 0.00160975 | 2678377 |
| 2026-03 | signed_notional | -22 | 0.00255625 | 2678378 |
| 2026-03 | signed_notional | -21 | 0.00256574 | 2678379 |
| 2026-03 | signed_notional | -20 | 0.00236789 | 2678380 |
| 2026-03 | signed_notional | -19 | 0.00154468 | 2678381 |
| 2026-03 | signed_notional | -18 | 0.00111809 | 2678382 |
| 2026-03 | signed_notional | -17 | 0.00088345 | 2678383 |
| 2026-03 | signed_notional | -16 | 0.00125057 | 2678384 |
| 2026-03 | signed_notional | -15 | 0.00205836 | 2678385 |
| 2026-03 | signed_notional | -14 | 0.00177678 | 2678386 |
| 2026-03 | signed_notional | -13 | 0.00181877 | 2678387 |
| 2026-03 | signed_notional | -12 | 0.00113753 | 2678388 |
| 2026-03 | signed_notional | -11 | 0.00208141 | 2678389 |
| 2026-03 | signed_notional | -10 | 0.00177941 | 2678390 |
| 2026-03 | signed_notional | -9 | 0.00197418 | 2678391 |
| 2026-03 | signed_notional | -8 | 0.00148690 | 2678392 |
| 2026-03 | signed_notional | -7 | 0.00173431 | 2678393 |
| 2026-03 | signed_notional | -6 | 0.00467965 | 2678394 |
| 2026-03 | signed_notional | -5 | 0.00204020 | 2678395 |
| 2026-03 | signed_notional | -4 | 0.00286692 | 2678396 |
| 2026-03 | signed_notional | -3 | 0.00345611 | 2678397 |
| 2026-03 | signed_notional | -2 | 0.00332274 | 2678398 |
| 2026-03 | signed_notional | -1 | 0.00220468 | 2678399 |
| 2026-03 | signed_notional | 0 | 0.01626056 | 2678400 |
| 2026-03 | signed_notional | 1 | 0.00962468 | 2678399 |
| 2026-03 | signed_notional | 2 | 0.00342683 | 2678398 |
| 2026-03 | signed_notional | 3 | 0.00283291 | 2678397 |
| 2026-03 | signed_notional | 4 | 0.00230434 | 2678396 |
| 2026-03 | signed_notional | 5 | 0.00261389 | 2678395 |
| 2026-03 | signed_notional | 6 | 0.00218561 | 2678394 |
| 2026-03 | signed_notional | 7 | 0.00165041 | 2678393 |
| 2026-03 | signed_notional | 8 | 0.00138383 | 2678392 |
| 2026-03 | signed_notional | 9 | 0.00242885 | 2678391 |
| 2026-03 | signed_notional | 10 | 0.00286878 | 2678390 |
| 2026-03 | signed_notional | 11 | 0.00202285 | 2678389 |
| 2026-03 | signed_notional | 12 | 0.00297428 | 2678388 |
| 2026-03 | signed_notional | 13 | 0.00202920 | 2678387 |
| 2026-03 | signed_notional | 14 | 0.00120358 | 2678386 |
| 2026-03 | signed_notional | 15 | 0.00194945 | 2678385 |
| 2026-03 | signed_notional | 16 | 0.00091419 | 2678384 |
| 2026-03 | signed_notional | 17 | 0.00058702 | 2678383 |
| 2026-03 | signed_notional | 18 | 0.00042806 | 2678382 |
| 2026-03 | signed_notional | 19 | 0.00088781 | 2678381 |
| 2026-03 | signed_notional | 20 | -0.00011963 | 2678380 |
| 2026-03 | signed_notional | 21 | 0.00087163 | 2678379 |
| 2026-03 | signed_notional | 22 | 0.00095118 | 2678378 |
| 2026-03 | signed_notional | 23 | 0.00137053 | 2678377 |
| 2026-03 | signed_notional | 24 | 0.00076870 | 2678376 |
| 2026-03 | signed_notional | 25 | 0.00053391 | 2678375 |
| 2026-03 | signed_notional | 26 | 0.00067458 | 2678374 |
| 2026-03 | signed_notional | 27 | 0.00051642 | 2678373 |
| 2026-03 | signed_notional | 28 | 0.00057768 | 2678372 |
| 2026-03 | signed_notional | 29 | 0.00071357 | 2678371 |
| 2026-03 | signed_notional | 30 | 0.00084696 | 2678370 |
| 2026-04 | raw_notional | -30 | 0.00269756 | 2202352 |
| 2026-04 | raw_notional | -29 | 0.00893803 | 2202356 |
| 2026-04 | raw_notional | -28 | 0.00245840 | 2202360 |
| 2026-04 | raw_notional | -27 | 0.00334723 | 2202364 |
| 2026-04 | raw_notional | -26 | 0.00206119 | 2202368 |
| 2026-04 | raw_notional | -25 | 0.00204851 | 2202372 |
| 2026-04 | raw_notional | -24 | 0.00351253 | 2202376 |
| 2026-04 | raw_notional | -23 | 0.00378669 | 2202380 |
| 2026-04 | raw_notional | -22 | 0.00288911 | 2202384 |
| 2026-04 | raw_notional | -21 | 0.00308431 | 2202388 |
| 2026-04 | raw_notional | -20 | 0.00292300 | 2202392 |
| 2026-04 | raw_notional | -19 | 0.00529301 | 2202396 |
| 2026-04 | raw_notional | -18 | 0.00326203 | 2202400 |
| 2026-04 | raw_notional | -17 | 0.00773743 | 2202404 |
| 2026-04 | raw_notional | -16 | 0.00801325 | 2202408 |
| 2026-04 | raw_notional | -15 | 0.00390854 | 2202412 |
| 2026-04 | raw_notional | -14 | 0.00395403 | 2202416 |
| 2026-04 | raw_notional | -13 | 0.00574054 | 2202420 |
| 2026-04 | raw_notional | -12 | 0.00929913 | 2202424 |
| 2026-04 | raw_notional | -11 | 0.00577291 | 2202428 |
| 2026-04 | raw_notional | -10 | 0.00485907 | 2202432 |
| 2026-04 | raw_notional | -9 | 0.00665958 | 2202436 |
| 2026-04 | raw_notional | -8 | 0.00588690 | 2202440 |
| 2026-04 | raw_notional | -7 | 0.01191136 | 2202444 |
| 2026-04 | raw_notional | -6 | 0.01376067 | 2202448 |
| 2026-04 | raw_notional | -5 | 0.02034789 | 2202452 |
| 2026-04 | raw_notional | -4 | 0.01086879 | 2202456 |
| 2026-04 | raw_notional | -3 | 0.01580285 | 2202460 |
| 2026-04 | raw_notional | -2 | 0.01971003 | 2202464 |
| 2026-04 | raw_notional | -1 | 0.03105445 | 2202468 |
| 2026-04 | raw_notional | 0 | 0.12497760 | 2202472 |
| 2026-04 | raw_notional | 1 | 0.02728272 | 2202468 |
| 2026-04 | raw_notional | 2 | 0.02336211 | 2202464 |
| 2026-04 | raw_notional | 3 | 0.04860376 | 2202460 |
| 2026-04 | raw_notional | 4 | 0.01841570 | 2202456 |
| 2026-04 | raw_notional | 5 | 0.03746085 | 2202452 |
| 2026-04 | raw_notional | 6 | 0.00671584 | 2202448 |
| 2026-04 | raw_notional | 7 | 0.00723344 | 2202444 |
| 2026-04 | raw_notional | 8 | 0.01007567 | 2202440 |
| 2026-04 | raw_notional | 9 | 0.01119713 | 2202436 |
| 2026-04 | raw_notional | 10 | 0.00810435 | 2202432 |
| 2026-04 | raw_notional | 11 | 0.00875432 | 2202428 |
| 2026-04 | raw_notional | 12 | 0.00758803 | 2202424 |
| 2026-04 | raw_notional | 13 | 0.01394640 | 2202420 |
| 2026-04 | raw_notional | 14 | 0.00409979 | 2202416 |
| 2026-04 | raw_notional | 15 | 0.00947300 | 2202412 |
| 2026-04 | raw_notional | 16 | 0.00499587 | 2202408 |
| 2026-04 | raw_notional | 17 | 0.00257704 | 2202404 |
| 2026-04 | raw_notional | 18 | 0.00210734 | 2202400 |
| 2026-04 | raw_notional | 19 | 0.00456353 | 2202396 |
| 2026-04 | raw_notional | 20 | 0.00888773 | 2202392 |
| 2026-04 | raw_notional | 21 | 0.00563928 | 2202388 |
| 2026-04 | raw_notional | 22 | 0.00345396 | 2202384 |
| 2026-04 | raw_notional | 23 | 0.00288642 | 2202380 |
| 2026-04 | raw_notional | 24 | 0.00325175 | 2202376 |
| 2026-04 | raw_notional | 25 | 0.00279135 | 2202372 |
| 2026-04 | raw_notional | 26 | 0.00671292 | 2202368 |
| 2026-04 | raw_notional | 27 | 0.01260182 | 2202364 |
| 2026-04 | raw_notional | 28 | 0.02147009 | 2202360 |
| 2026-04 | raw_notional | 29 | 0.00713734 | 2202356 |
| 2026-04 | raw_notional | 30 | 0.00532932 | 2202352 |
| 2026-04 | log1p_notional | -30 | 0.05116584 | 2202352 |
| 2026-04 | log1p_notional | -29 | 0.05416529 | 2202356 |
| 2026-04 | log1p_notional | -28 | 0.05410839 | 2202360 |
| 2026-04 | log1p_notional | -27 | 0.05428585 | 2202364 |
| 2026-04 | log1p_notional | -26 | 0.05634134 | 2202368 |
| 2026-04 | log1p_notional | -25 | 0.05496179 | 2202372 |
| 2026-04 | log1p_notional | -24 | 0.05815923 | 2202376 |
| 2026-04 | log1p_notional | -23 | 0.06115288 | 2202380 |
| 2026-04 | log1p_notional | -22 | 0.06191073 | 2202384 |
| 2026-04 | log1p_notional | -21 | 0.06306653 | 2202388 |
| 2026-04 | log1p_notional | -20 | 0.06689809 | 2202392 |
| 2026-04 | log1p_notional | -19 | 0.06997434 | 2202396 |
| 2026-04 | log1p_notional | -18 | 0.06805971 | 2202400 |
| 2026-04 | log1p_notional | -17 | 0.07310901 | 2202404 |
| 2026-04 | log1p_notional | -16 | 0.07629252 | 2202408 |
| 2026-04 | log1p_notional | -15 | 0.07751760 | 2202412 |
| 2026-04 | log1p_notional | -14 | 0.07713879 | 2202416 |
| 2026-04 | log1p_notional | -13 | 0.08214620 | 2202420 |
| 2026-04 | log1p_notional | -12 | 0.08374994 | 2202424 |
| 2026-04 | log1p_notional | -11 | 0.08886246 | 2202428 |
| 2026-04 | log1p_notional | -10 | 0.09429209 | 2202432 |
| 2026-04 | log1p_notional | -9 | 0.09995807 | 2202436 |
| 2026-04 | log1p_notional | -8 | 0.10112893 | 2202440 |
| 2026-04 | log1p_notional | -7 | 0.10646392 | 2202444 |
| 2026-04 | log1p_notional | -6 | 0.11760366 | 2202448 |
| 2026-04 | log1p_notional | -5 | 0.12332981 | 2202452 |
| 2026-04 | log1p_notional | -4 | 0.14053487 | 2202456 |
| 2026-04 | log1p_notional | -3 | 0.15862948 | 2202460 |
| 2026-04 | log1p_notional | -2 | 0.19244352 | 2202464 |
| 2026-04 | log1p_notional | -1 | 0.23493967 | 2202468 |
| 2026-04 | log1p_notional | 0 | 0.41483106 | 2202472 |
| 2026-04 | log1p_notional | 1 | 0.23692227 | 2202468 |
| 2026-04 | log1p_notional | 2 | 0.20394521 | 2202464 |
| 2026-04 | log1p_notional | 3 | 0.16120623 | 2202460 |
| 2026-04 | log1p_notional | 4 | 0.13950753 | 2202456 |
| 2026-04 | log1p_notional | 5 | 0.12651913 | 2202452 |
| 2026-04 | log1p_notional | 6 | 0.11601526 | 2202448 |
| 2026-04 | log1p_notional | 7 | 0.10378953 | 2202444 |
| 2026-04 | log1p_notional | 8 | 0.09851167 | 2202440 |
| 2026-04 | log1p_notional | 9 | 0.09615074 | 2202436 |
| 2026-04 | log1p_notional | 10 | 0.09064274 | 2202432 |
| 2026-04 | log1p_notional | 11 | 0.08761564 | 2202428 |
| 2026-04 | log1p_notional | 12 | 0.08375310 | 2202424 |
| 2026-04 | log1p_notional | 13 | 0.08190525 | 2202420 |
| 2026-04 | log1p_notional | 14 | 0.07658794 | 2202416 |
| 2026-04 | log1p_notional | 15 | 0.07295054 | 2202412 |
| 2026-04 | log1p_notional | 16 | 0.07089864 | 2202408 |
| 2026-04 | log1p_notional | 17 | 0.07039741 | 2202404 |
| 2026-04 | log1p_notional | 18 | 0.07166699 | 2202400 |
| 2026-04 | log1p_notional | 19 | 0.06911631 | 2202396 |
| 2026-04 | log1p_notional | 20 | 0.06765453 | 2202392 |
| 2026-04 | log1p_notional | 21 | 0.06680683 | 2202388 |
| 2026-04 | log1p_notional | 22 | 0.06316607 | 2202384 |
| 2026-04 | log1p_notional | 23 | 0.06170470 | 2202380 |
| 2026-04 | log1p_notional | 24 | 0.05725793 | 2202376 |
| 2026-04 | log1p_notional | 25 | 0.05616399 | 2202372 |
| 2026-04 | log1p_notional | 26 | 0.05902431 | 2202368 |
| 2026-04 | log1p_notional | 27 | 0.05539788 | 2202364 |
| 2026-04 | log1p_notional | 28 | 0.05674365 | 2202360 |
| 2026-04 | log1p_notional | 29 | 0.05209176 | 2202356 |
| 2026-04 | log1p_notional | 30 | 0.04987229 | 2202352 |
| 2026-04 | signed_notional | -30 | 0.00276154 | 2202352 |
| 2026-04 | signed_notional | -29 | 0.00903567 | 2202356 |
| 2026-04 | signed_notional | -28 | 0.00251575 | 2202360 |
| 2026-04 | signed_notional | -27 | 0.00320331 | 2202364 |
| 2026-04 | signed_notional | -26 | 0.00203543 | 2202368 |
| 2026-04 | signed_notional | -25 | 0.00206465 | 2202372 |
| 2026-04 | signed_notional | -24 | 0.00339696 | 2202376 |
| 2026-04 | signed_notional | -23 | 0.00386081 | 2202380 |
| 2026-04 | signed_notional | -22 | 0.00295879 | 2202384 |
| 2026-04 | signed_notional | -21 | 0.00304528 | 2202388 |
| 2026-04 | signed_notional | -20 | 0.00283386 | 2202392 |
| 2026-04 | signed_notional | -19 | 0.00540893 | 2202396 |
| 2026-04 | signed_notional | -18 | 0.00335095 | 2202400 |
| 2026-04 | signed_notional | -17 | 0.00777091 | 2202404 |
| 2026-04 | signed_notional | -16 | 0.00803628 | 2202408 |
| 2026-04 | signed_notional | -15 | 0.00402023 | 2202412 |
| 2026-04 | signed_notional | -14 | 0.00406972 | 2202416 |
| 2026-04 | signed_notional | -13 | 0.00581023 | 2202420 |
| 2026-04 | signed_notional | -12 | 0.00927107 | 2202424 |
| 2026-04 | signed_notional | -11 | 0.00583377 | 2202428 |
| 2026-04 | signed_notional | -10 | 0.00496616 | 2202432 |
| 2026-04 | signed_notional | -9 | 0.00671802 | 2202436 |
| 2026-04 | signed_notional | -8 | 0.00594035 | 2202440 |
| 2026-04 | signed_notional | -7 | 0.01192160 | 2202444 |
| 2026-04 | signed_notional | -6 | 0.01386313 | 2202448 |
| 2026-04 | signed_notional | -5 | 0.02017797 | 2202452 |
| 2026-04 | signed_notional | -4 | 0.01086200 | 2202456 |
| 2026-04 | signed_notional | -3 | 0.01582839 | 2202460 |
| 2026-04 | signed_notional | -2 | 0.01978795 | 2202464 |
| 2026-04 | signed_notional | -1 | 0.03105565 | 2202468 |
| 2026-04 | signed_notional | 0 | 0.12458867 | 2202472 |
| 2026-04 | signed_notional | 1 | 0.02737822 | 2202468 |
| 2026-04 | signed_notional | 2 | 0.02346628 | 2202464 |
| 2026-04 | signed_notional | 3 | 0.04869530 | 2202460 |
| 2026-04 | signed_notional | 4 | 0.01851678 | 2202456 |
| 2026-04 | signed_notional | 5 | 0.03756563 | 2202452 |
| 2026-04 | signed_notional | 6 | 0.00651507 | 2202448 |
| 2026-04 | signed_notional | 7 | 0.00732465 | 2202444 |
| 2026-04 | signed_notional | 8 | 0.01018442 | 2202440 |
| 2026-04 | signed_notional | 9 | 0.01129702 | 2202436 |
| 2026-04 | signed_notional | 10 | 0.00821350 | 2202432 |
| 2026-04 | signed_notional | 11 | 0.00881098 | 2202428 |
| 2026-04 | signed_notional | 12 | 0.00765616 | 2202424 |
| 2026-04 | signed_notional | 13 | 0.01364743 | 2202420 |
| 2026-04 | signed_notional | 14 | 0.00421024 | 2202416 |
| 2026-04 | signed_notional | 15 | 0.00958358 | 2202412 |
| 2026-04 | signed_notional | 16 | 0.00510699 | 2202408 |
| 2026-04 | signed_notional | 17 | 0.00267185 | 2202404 |
| 2026-04 | signed_notional | 18 | 0.00215588 | 2202400 |
| 2026-04 | signed_notional | 19 | 0.00466319 | 2202396 |
| 2026-04 | signed_notional | 20 | 0.00899709 | 2202392 |
| 2026-04 | signed_notional | 21 | 0.00572298 | 2202388 |
| 2026-04 | signed_notional | 22 | 0.00354889 | 2202384 |
| 2026-04 | signed_notional | 23 | 0.00299532 | 2202380 |
| 2026-04 | signed_notional | 24 | 0.00335081 | 2202376 |
| 2026-04 | signed_notional | 25 | 0.00289502 | 2202372 |
| 2026-04 | signed_notional | 26 | 0.00681146 | 2202368 |
| 2026-04 | signed_notional | 27 | 0.01270612 | 2202364 |
| 2026-04 | signed_notional | 28 | 0.02148899 | 2202360 |
| 2026-04 | signed_notional | 29 | 0.00724369 | 2202356 |
| 2026-04 | signed_notional | 30 | 0.00540533 | 2202352 |
| 2026-06 | raw_notional | -30 | 0.00796924 | 2096138 |
| 2026-06 | raw_notional | -29 | 0.00635221 | 2096139 |
| 2026-06 | raw_notional | -28 | 0.03272876 | 2096140 |
| 2026-06 | raw_notional | -27 | 0.01313252 | 2096141 |
| 2026-06 | raw_notional | -26 | 0.00654989 | 2096142 |
| 2026-06 | raw_notional | -25 | 0.01041564 | 2096143 |
| 2026-06 | raw_notional | -24 | 0.00695186 | 2096144 |
| 2026-06 | raw_notional | -23 | 0.01381879 | 2096145 |
| 2026-06 | raw_notional | -22 | 0.01768160 | 2096146 |
| 2026-06 | raw_notional | -21 | 0.01075840 | 2096147 |
| 2026-06 | raw_notional | -20 | 0.01166132 | 2096148 |
| 2026-06 | raw_notional | -19 | 0.01768269 | 2096149 |
| 2026-06 | raw_notional | -18 | 0.01361713 | 2096150 |
| 2026-06 | raw_notional | -17 | 0.01226533 | 2096151 |
| 2026-06 | raw_notional | -16 | 0.01216706 | 2096152 |
| 2026-06 | raw_notional | -15 | 0.01141227 | 2096153 |
| 2026-06 | raw_notional | -14 | 0.01420931 | 2096154 |
| 2026-06 | raw_notional | -13 | 0.01386851 | 2096155 |
| 2026-06 | raw_notional | -12 | 0.01100898 | 2096156 |
| 2026-06 | raw_notional | -11 | 0.01353910 | 2096157 |
| 2026-06 | raw_notional | -10 | 0.01673109 | 2096158 |
| 2026-06 | raw_notional | -9 | 0.02065519 | 2096159 |
| 2026-06 | raw_notional | -8 | 0.01022901 | 2096160 |
| 2026-06 | raw_notional | -7 | 0.00950723 | 2096161 |
| 2026-06 | raw_notional | -6 | 0.01468664 | 2096162 |
| 2026-06 | raw_notional | -5 | 0.02662370 | 2096163 |
| 2026-06 | raw_notional | -4 | 0.02448509 | 2096164 |
| 2026-06 | raw_notional | -3 | 0.02805436 | 2096165 |
| 2026-06 | raw_notional | -2 | 0.08606944 | 2096166 |
| 2026-06 | raw_notional | -1 | 0.12215725 | 2096167 |
| 2026-06 | raw_notional | 0 | 0.19893054 | 2096168 |
| 2026-06 | raw_notional | 1 | 0.16029699 | 2096167 |
| 2026-06 | raw_notional | 2 | 0.05172225 | 2096166 |
| 2026-06 | raw_notional | 3 | 0.02662296 | 2096165 |
| 2026-06 | raw_notional | 4 | 0.03011425 | 2096164 |
| 2026-06 | raw_notional | 5 | 0.01405330 | 2096163 |
| 2026-06 | raw_notional | 6 | 0.01847981 | 2096162 |
| 2026-06 | raw_notional | 7 | 0.01254921 | 2096161 |
| 2026-06 | raw_notional | 8 | 0.02456409 | 2096160 |
| 2026-06 | raw_notional | 9 | 0.03840083 | 2096159 |
| 2026-06 | raw_notional | 10 | 0.01755796 | 2096158 |
| 2026-06 | raw_notional | 11 | 0.10372636 | 2096157 |
| 2026-06 | raw_notional | 12 | 0.01537597 | 2096156 |
| 2026-06 | raw_notional | 13 | 0.01137778 | 2096155 |
| 2026-06 | raw_notional | 14 | 0.00758953 | 2096154 |
| 2026-06 | raw_notional | 15 | 0.01262434 | 2096153 |
| 2026-06 | raw_notional | 16 | 0.01358063 | 2096152 |
| 2026-06 | raw_notional | 17 | 0.03038233 | 2096151 |
| 2026-06 | raw_notional | 18 | 0.02105950 | 2096150 |
| 2026-06 | raw_notional | 19 | 0.02475896 | 2096149 |
| 2026-06 | raw_notional | 20 | 0.01732610 | 2096148 |
| 2026-06 | raw_notional | 21 | 0.00893795 | 2096147 |
| 2026-06 | raw_notional | 22 | 0.01280519 | 2096146 |
| 2026-06 | raw_notional | 23 | 0.01116610 | 2096145 |
| 2026-06 | raw_notional | 24 | 0.00530653 | 2096144 |
| 2026-06 | raw_notional | 25 | 0.01271490 | 2096143 |
| 2026-06 | raw_notional | 26 | 0.00796144 | 2096142 |
| 2026-06 | raw_notional | 27 | 0.01189060 | 2096141 |
| 2026-06 | raw_notional | 28 | 0.02273933 | 2096140 |
| 2026-06 | raw_notional | 29 | 0.01255176 | 2096139 |
| 2026-06 | raw_notional | 30 | 0.00850395 | 2096138 |
| 2026-06 | log1p_notional | -30 | 0.05842211 | 2096138 |
| 2026-06 | log1p_notional | -29 | 0.06293489 | 2096139 |
| 2026-06 | log1p_notional | -28 | 0.06165651 | 2096140 |
| 2026-06 | log1p_notional | -27 | 0.06362258 | 2096141 |
| 2026-06 | log1p_notional | -26 | 0.06526186 | 2096142 |
| 2026-06 | log1p_notional | -25 | 0.06678881 | 2096143 |
| 2026-06 | log1p_notional | -24 | 0.06800408 | 2096144 |
| 2026-06 | log1p_notional | -23 | 0.06980294 | 2096145 |
| 2026-06 | log1p_notional | -22 | 0.07422222 | 2096146 |
| 2026-06 | log1p_notional | -21 | 0.07379260 | 2096147 |
| 2026-06 | log1p_notional | -20 | 0.07633864 | 2096148 |
| 2026-06 | log1p_notional | -19 | 0.07790624 | 2096149 |
| 2026-06 | log1p_notional | -18 | 0.07865104 | 2096150 |
| 2026-06 | log1p_notional | -17 | 0.08042437 | 2096151 |
| 2026-06 | log1p_notional | -16 | 0.08165800 | 2096152 |
| 2026-06 | log1p_notional | -15 | 0.08591320 | 2096153 |
| 2026-06 | log1p_notional | -14 | 0.08512157 | 2096154 |
| 2026-06 | log1p_notional | -13 | 0.08812017 | 2096155 |
| 2026-06 | log1p_notional | -12 | 0.09378937 | 2096156 |
| 2026-06 | log1p_notional | -11 | 0.09938171 | 2096157 |
| 2026-06 | log1p_notional | -10 | 0.10486485 | 2096158 |
| 2026-06 | log1p_notional | -9 | 0.10995331 | 2096159 |
| 2026-06 | log1p_notional | -8 | 0.11451600 | 2096160 |
| 2026-06 | log1p_notional | -7 | 0.12323628 | 2096161 |
| 2026-06 | log1p_notional | -6 | 0.13153103 | 2096162 |
| 2026-06 | log1p_notional | -5 | 0.13640182 | 2096163 |
| 2026-06 | log1p_notional | -4 | 0.15044898 | 2096164 |
| 2026-06 | log1p_notional | -3 | 0.17313196 | 2096165 |
| 2026-06 | log1p_notional | -2 | 0.20941443 | 2096166 |
| 2026-06 | log1p_notional | -1 | 0.25186712 | 2096167 |
| 2026-06 | log1p_notional | 0 | 0.41920440 | 2096168 |
| 2026-06 | log1p_notional | 1 | 0.25385336 | 2096167 |
| 2026-06 | log1p_notional | 2 | 0.21033583 | 2096166 |
| 2026-06 | log1p_notional | 3 | 0.17265742 | 2096165 |
| 2026-06 | log1p_notional | 4 | 0.15188418 | 2096164 |
| 2026-06 | log1p_notional | 5 | 0.13859716 | 2096163 |
| 2026-06 | log1p_notional | 6 | 0.12967991 | 2096162 |
| 2026-06 | log1p_notional | 7 | 0.12428406 | 2096161 |
| 2026-06 | log1p_notional | 8 | 0.11492110 | 2096160 |
| 2026-06 | log1p_notional | 9 | 0.10916064 | 2096159 |
| 2026-06 | log1p_notional | 10 | 0.10399780 | 2096158 |
| 2026-06 | log1p_notional | 11 | 0.09725464 | 2096157 |
| 2026-06 | log1p_notional | 12 | 0.09307932 | 2096156 |
| 2026-06 | log1p_notional | 13 | 0.09141852 | 2096155 |
| 2026-06 | log1p_notional | 14 | 0.08834925 | 2096154 |
| 2026-06 | log1p_notional | 15 | 0.08536423 | 2096153 |
| 2026-06 | log1p_notional | 16 | 0.08474276 | 2096152 |
| 2026-06 | log1p_notional | 17 | 0.08445784 | 2096151 |
| 2026-06 | log1p_notional | 18 | 0.07723150 | 2096150 |
| 2026-06 | log1p_notional | 19 | 0.07970586 | 2096149 |
| 2026-06 | log1p_notional | 20 | 0.07610458 | 2096148 |
| 2026-06 | log1p_notional | 21 | 0.07270701 | 2096147 |
| 2026-06 | log1p_notional | 22 | 0.07206832 | 2096146 |
| 2026-06 | log1p_notional | 23 | 0.06849194 | 2096145 |
| 2026-06 | log1p_notional | 24 | 0.06677049 | 2096144 |
| 2026-06 | log1p_notional | 25 | 0.06481798 | 2096143 |
| 2026-06 | log1p_notional | 26 | 0.06489213 | 2096142 |
| 2026-06 | log1p_notional | 27 | 0.06276423 | 2096141 |
| 2026-06 | log1p_notional | 28 | 0.06031754 | 2096140 |
| 2026-06 | log1p_notional | 29 | 0.06056243 | 2096139 |
| 2026-06 | log1p_notional | 30 | 0.05989247 | 2096138 |
| 2026-06 | signed_notional | -30 | 0.00812305 | 2096138 |
| 2026-06 | signed_notional | -29 | 0.00649125 | 2096139 |
| 2026-06 | signed_notional | -28 | 0.03225702 | 2096140 |
| 2026-06 | signed_notional | -27 | 0.01324911 | 2096141 |
| 2026-06 | signed_notional | -26 | 0.00672852 | 2096142 |
| 2026-06 | signed_notional | -25 | 0.01018313 | 2096143 |
| 2026-06 | signed_notional | -24 | 0.00693903 | 2096144 |
| 2026-06 | signed_notional | -23 | 0.01385644 | 2096145 |
| 2026-06 | signed_notional | -22 | 0.01773901 | 2096146 |
| 2026-06 | signed_notional | -21 | 0.01091227 | 2096147 |
| 2026-06 | signed_notional | -20 | 0.01169879 | 2096148 |
| 2026-06 | signed_notional | -19 | 0.01785514 | 2096149 |
| 2026-06 | signed_notional | -18 | 0.01376246 | 2096150 |
| 2026-06 | signed_notional | -17 | 0.01240791 | 2096151 |
| 2026-06 | signed_notional | -16 | 0.01233066 | 2096152 |
| 2026-06 | signed_notional | -15 | 0.01139684 | 2096153 |
| 2026-06 | signed_notional | -14 | 0.01436969 | 2096154 |
| 2026-06 | signed_notional | -13 | 0.01405951 | 2096155 |
| 2026-06 | signed_notional | -12 | 0.01119701 | 2096156 |
| 2026-06 | signed_notional | -11 | 0.01372562 | 2096157 |
| 2026-06 | signed_notional | -10 | 0.01693707 | 2096158 |
| 2026-06 | signed_notional | -9 | 0.02077872 | 2096159 |
| 2026-06 | signed_notional | -8 | 0.01038186 | 2096160 |
| 2026-06 | signed_notional | -7 | 0.00970613 | 2096161 |
| 2026-06 | signed_notional | -6 | 0.01479817 | 2096162 |
| 2026-06 | signed_notional | -5 | 0.02676919 | 2096163 |
| 2026-06 | signed_notional | -4 | 0.02467212 | 2096164 |
| 2026-06 | signed_notional | -3 | 0.02824934 | 2096165 |
| 2026-06 | signed_notional | -2 | 0.08622380 | 2096166 |
| 2026-06 | signed_notional | -1 | 0.12227251 | 2096167 |
| 2026-06 | signed_notional | 0 | 0.19889450 | 2096168 |
| 2026-06 | signed_notional | 1 | 0.16036465 | 2096167 |
| 2026-06 | signed_notional | 2 | 0.05187697 | 2096166 |
| 2026-06 | signed_notional | 3 | 0.02673897 | 2096165 |
| 2026-06 | signed_notional | 4 | 0.03029523 | 2096164 |
| 2026-06 | signed_notional | 5 | 0.01420924 | 2096163 |
| 2026-06 | signed_notional | 6 | 0.01863519 | 2096162 |
| 2026-06 | signed_notional | 7 | 0.01273180 | 2096161 |
| 2026-06 | signed_notional | 8 | 0.02471903 | 2096160 |
| 2026-06 | signed_notional | 9 | 0.03856165 | 2096159 |
| 2026-06 | signed_notional | 10 | 0.01773996 | 2096158 |
| 2026-06 | signed_notional | 11 | 0.10386999 | 2096157 |
| 2026-06 | signed_notional | 12 | 0.01538038 | 2096156 |
| 2026-06 | signed_notional | 13 | 0.01140128 | 2096155 |
| 2026-06 | signed_notional | 14 | 0.00771999 | 2096154 |
| 2026-06 | signed_notional | 15 | 0.01274755 | 2096153 |
| 2026-06 | signed_notional | 16 | 0.01372735 | 2096152 |
| 2026-06 | signed_notional | 17 | 0.03053972 | 2096151 |
| 2026-06 | signed_notional | 18 | 0.02120338 | 2096150 |
| 2026-06 | signed_notional | 19 | 0.02488497 | 2096149 |
| 2026-06 | signed_notional | 20 | 0.01731668 | 2096148 |
| 2026-06 | signed_notional | 21 | 0.00903971 | 2096147 |
| 2026-06 | signed_notional | 22 | 0.01257967 | 2096146 |
| 2026-06 | signed_notional | 23 | 0.01125331 | 2096145 |
| 2026-06 | signed_notional | 24 | 0.00520713 | 2096144 |
| 2026-06 | signed_notional | 25 | 0.01246524 | 2096143 |
| 2026-06 | signed_notional | 26 | 0.00803117 | 2096142 |
| 2026-06 | signed_notional | 27 | 0.01193640 | 2096141 |
| 2026-06 | signed_notional | 28 | 0.02272993 | 2096140 |
| 2026-06 | signed_notional | 29 | 0.01262672 | 2096139 |
| 2026-06 | signed_notional | 30 | 0.00847199 | 2096138 |
| 2026-07 | raw_notional | -30 | 0.00191337 | 1170130 |
| 2026-07 | raw_notional | -29 | 0.00179622 | 1170132 |
| 2026-07 | raw_notional | -28 | 0.00124547 | 1170134 |
| 2026-07 | raw_notional | -27 | 0.00543856 | 1170136 |
| 2026-07 | raw_notional | -26 | 0.00421251 | 1170138 |
| 2026-07 | raw_notional | -25 | 0.00390254 | 1170140 |
| 2026-07 | raw_notional | -24 | 0.00240817 | 1170142 |
| 2026-07 | raw_notional | -23 | 0.00227805 | 1170144 |
| 2026-07 | raw_notional | -22 | 0.00341911 | 1170146 |
| 2026-07 | raw_notional | -21 | 0.00178680 | 1170148 |
| 2026-07 | raw_notional | -20 | 0.00218147 | 1170150 |
| 2026-07 | raw_notional | -19 | 0.00318695 | 1170152 |
| 2026-07 | raw_notional | -18 | 0.00300444 | 1170154 |
| 2026-07 | raw_notional | -17 | 0.01681074 | 1170156 |
| 2026-07 | raw_notional | -16 | 0.00603423 | 1170158 |
| 2026-07 | raw_notional | -15 | 0.00693987 | 1170160 |
| 2026-07 | raw_notional | -14 | 0.01159632 | 1170162 |
| 2026-07 | raw_notional | -13 | 0.01040067 | 1170164 |
| 2026-07 | raw_notional | -12 | 0.00636400 | 1170166 |
| 2026-07 | raw_notional | -11 | 0.01438504 | 1170168 |
| 2026-07 | raw_notional | -10 | 0.00536467 | 1170170 |
| 2026-07 | raw_notional | -9 | 0.00688375 | 1170172 |
| 2026-07 | raw_notional | -8 | 0.00927403 | 1170174 |
| 2026-07 | raw_notional | -7 | 0.01360963 | 1170176 |
| 2026-07 | raw_notional | -6 | 0.05276973 | 1170178 |
| 2026-07 | raw_notional | -5 | 0.02673128 | 1170180 |
| 2026-07 | raw_notional | -4 | 0.02045299 | 1170182 |
| 2026-07 | raw_notional | -3 | 0.04992837 | 1170184 |
| 2026-07 | raw_notional | -2 | 0.03378305 | 1170186 |
| 2026-07 | raw_notional | -1 | 0.05370316 | 1170188 |
| 2026-07 | raw_notional | 0 | 0.18630669 | 1170190 |
| 2026-07 | raw_notional | 1 | 0.13536850 | 1170188 |
| 2026-07 | raw_notional | 2 | 0.04297120 | 1170186 |
| 2026-07 | raw_notional | 3 | 0.03980750 | 1170184 |
| 2026-07 | raw_notional | 4 | 0.02677821 | 1170182 |
| 2026-07 | raw_notional | 5 | 0.09082646 | 1170180 |
| 2026-07 | raw_notional | 6 | 0.03194253 | 1170178 |
| 2026-07 | raw_notional | 7 | 0.02115444 | 1170176 |
| 2026-07 | raw_notional | 8 | 0.00906996 | 1170174 |
| 2026-07 | raw_notional | 9 | 0.00481887 | 1170172 |
| 2026-07 | raw_notional | 10 | 0.00815800 | 1170170 |
| 2026-07 | raw_notional | 11 | 0.00650313 | 1170168 |
| 2026-07 | raw_notional | 12 | 0.00909669 | 1170166 |
| 2026-07 | raw_notional | 13 | 0.00692242 | 1170164 |
| 2026-07 | raw_notional | 14 | 0.00481970 | 1170162 |
| 2026-07 | raw_notional | 15 | 0.02366787 | 1170160 |
| 2026-07 | raw_notional | 16 | 0.01391330 | 1170158 |
| 2026-07 | raw_notional | 17 | 0.03028479 | 1170156 |
| 2026-07 | raw_notional | 18 | 0.01459170 | 1170154 |
| 2026-07 | raw_notional | 19 | 0.00960764 | 1170152 |
| 2026-07 | raw_notional | 20 | 0.01079151 | 1170150 |
| 2026-07 | raw_notional | 21 | 0.00893460 | 1170148 |
| 2026-07 | raw_notional | 22 | 0.00585881 | 1170146 |
| 2026-07 | raw_notional | 23 | 0.00640359 | 1170144 |
| 2026-07 | raw_notional | 24 | 0.00392179 | 1170142 |
| 2026-07 | raw_notional | 25 | 0.00299026 | 1170140 |
| 2026-07 | raw_notional | 26 | 0.00273659 | 1170138 |
| 2026-07 | raw_notional | 27 | 0.00465756 | 1170136 |
| 2026-07 | raw_notional | 28 | 0.00471358 | 1170134 |
| 2026-07 | raw_notional | 29 | 0.00154417 | 1170132 |
| 2026-07 | raw_notional | 30 | 0.00207199 | 1170130 |
| 2026-07 | log1p_notional | -30 | 0.04466568 | 1170130 |
| 2026-07 | log1p_notional | -29 | 0.04232130 | 1170132 |
| 2026-07 | log1p_notional | -28 | 0.04571425 | 1170134 |
| 2026-07 | log1p_notional | -27 | 0.04571674 | 1170136 |
| 2026-07 | log1p_notional | -26 | 0.04747734 | 1170138 |
| 2026-07 | log1p_notional | -25 | 0.04774040 | 1170140 |
| 2026-07 | log1p_notional | -24 | 0.04919845 | 1170142 |
| 2026-07 | log1p_notional | -23 | 0.04967612 | 1170144 |
| 2026-07 | log1p_notional | -22 | 0.05335951 | 1170146 |
| 2026-07 | log1p_notional | -21 | 0.05669168 | 1170148 |
| 2026-07 | log1p_notional | -20 | 0.05760409 | 1170150 |
| 2026-07 | log1p_notional | -19 | 0.05858432 | 1170152 |
| 2026-07 | log1p_notional | -18 | 0.06513171 | 1170154 |
| 2026-07 | log1p_notional | -17 | 0.06336695 | 1170156 |
| 2026-07 | log1p_notional | -16 | 0.06656175 | 1170158 |
| 2026-07 | log1p_notional | -15 | 0.06873462 | 1170160 |
| 2026-07 | log1p_notional | -14 | 0.06698416 | 1170162 |
| 2026-07 | log1p_notional | -13 | 0.06894090 | 1170164 |
| 2026-07 | log1p_notional | -12 | 0.07391882 | 1170166 |
| 2026-07 | log1p_notional | -11 | 0.08074546 | 1170168 |
| 2026-07 | log1p_notional | -10 | 0.08538079 | 1170170 |
| 2026-07 | log1p_notional | -9 | 0.08879269 | 1170172 |
| 2026-07 | log1p_notional | -8 | 0.09886769 | 1170174 |
| 2026-07 | log1p_notional | -7 | 0.10371157 | 1170176 |
| 2026-07 | log1p_notional | -6 | 0.11290054 | 1170178 |
| 2026-07 | log1p_notional | -5 | 0.12239913 | 1170180 |
| 2026-07 | log1p_notional | -4 | 0.13229288 | 1170182 |
| 2026-07 | log1p_notional | -3 | 0.15742496 | 1170184 |
| 2026-07 | log1p_notional | -2 | 0.18554666 | 1170186 |
| 2026-07 | log1p_notional | -1 | 0.22945256 | 1170188 |
| 2026-07 | log1p_notional | 0 | 0.39451740 | 1170190 |
| 2026-07 | log1p_notional | 1 | 0.23311181 | 1170188 |
| 2026-07 | log1p_notional | 2 | 0.18238870 | 1170186 |
| 2026-07 | log1p_notional | 3 | 0.14908813 | 1170184 |
| 2026-07 | log1p_notional | 4 | 0.12837354 | 1170182 |
| 2026-07 | log1p_notional | 5 | 0.11696753 | 1170180 |
| 2026-07 | log1p_notional | 6 | 0.10793560 | 1170178 |
| 2026-07 | log1p_notional | 7 | 0.10066538 | 1170176 |
| 2026-07 | log1p_notional | 8 | 0.09420592 | 1170174 |
| 2026-07 | log1p_notional | 9 | 0.09006268 | 1170172 |
| 2026-07 | log1p_notional | 10 | 0.08576215 | 1170170 |
| 2026-07 | log1p_notional | 11 | 0.08042286 | 1170168 |
| 2026-07 | log1p_notional | 12 | 0.07761358 | 1170166 |
| 2026-07 | log1p_notional | 13 | 0.07332391 | 1170164 |
| 2026-07 | log1p_notional | 14 | 0.06870776 | 1170162 |
| 2026-07 | log1p_notional | 15 | 0.07151126 | 1170160 |
| 2026-07 | log1p_notional | 16 | 0.06643797 | 1170158 |
| 2026-07 | log1p_notional | 17 | 0.06230572 | 1170156 |
| 2026-07 | log1p_notional | 18 | 0.06291351 | 1170154 |
| 2026-07 | log1p_notional | 19 | 0.05968600 | 1170152 |
| 2026-07 | log1p_notional | 20 | 0.05874548 | 1170150 |
| 2026-07 | log1p_notional | 21 | 0.05331058 | 1170148 |
| 2026-07 | log1p_notional | 22 | 0.05207569 | 1170146 |
| 2026-07 | log1p_notional | 23 | 0.04985018 | 1170144 |
| 2026-07 | log1p_notional | 24 | 0.04914394 | 1170142 |
| 2026-07 | log1p_notional | 25 | 0.04348713 | 1170140 |
| 2026-07 | log1p_notional | 26 | 0.04601017 | 1170138 |
| 2026-07 | log1p_notional | 27 | 0.04628705 | 1170136 |
| 2026-07 | log1p_notional | 28 | 0.04405800 | 1170134 |
| 2026-07 | log1p_notional | 29 | 0.03955542 | 1170132 |
| 2026-07 | log1p_notional | 30 | 0.03976779 | 1170130 |
| 2026-07 | signed_notional | -30 | 0.00155643 | 1170130 |
| 2026-07 | signed_notional | -29 | 0.00194128 | 1170132 |
| 2026-07 | signed_notional | -28 | 0.00125621 | 1170134 |
| 2026-07 | signed_notional | -27 | 0.00556839 | 1170136 |
| 2026-07 | signed_notional | -26 | 0.00424818 | 1170138 |
| 2026-07 | signed_notional | -25 | 0.00390277 | 1170140 |
| 2026-07 | signed_notional | -24 | 0.00256455 | 1170142 |
| 2026-07 | signed_notional | -23 | 0.00242365 | 1170144 |
| 2026-07 | signed_notional | -22 | 0.00364939 | 1170146 |
| 2026-07 | signed_notional | -21 | 0.00197828 | 1170148 |
| 2026-07 | signed_notional | -20 | 0.00236876 | 1170150 |
| 2026-07 | signed_notional | -19 | 0.00341041 | 1170152 |
| 2026-07 | signed_notional | -18 | 0.00320887 | 1170154 |
| 2026-07 | signed_notional | -17 | 0.01698150 | 1170156 |
| 2026-07 | signed_notional | -16 | 0.00594678 | 1170158 |
| 2026-07 | signed_notional | -15 | 0.00693650 | 1170160 |
| 2026-07 | signed_notional | -14 | 0.01181108 | 1170162 |
| 2026-07 | signed_notional | -13 | 0.01053940 | 1170164 |
| 2026-07 | signed_notional | -12 | 0.00659720 | 1170166 |
| 2026-07 | signed_notional | -11 | 0.01460308 | 1170168 |
| 2026-07 | signed_notional | -10 | 0.00559118 | 1170170 |
| 2026-07 | signed_notional | -9 | 0.00702060 | 1170172 |
| 2026-07 | signed_notional | -8 | 0.00907842 | 1170174 |
| 2026-07 | signed_notional | -7 | 0.00872156 | 1170176 |
| 2026-07 | signed_notional | -6 | 0.04956214 | 1170178 |
| 2026-07 | signed_notional | -5 | 0.02670316 | 1170180 |
| 2026-07 | signed_notional | -4 | 0.02062111 | 1170182 |
| 2026-07 | signed_notional | -3 | 0.05010752 | 1170184 |
| 2026-07 | signed_notional | -2 | 0.03381130 | 1170186 |
| 2026-07 | signed_notional | -1 | 0.05284043 | 1170188 |
| 2026-07 | signed_notional | 0 | 0.17031684 | 1170190 |
| 2026-07 | signed_notional | 1 | 0.13483643 | 1170188 |
| 2026-07 | signed_notional | 2 | 0.04228942 | 1170186 |
| 2026-07 | signed_notional | 3 | 0.03987346 | 1170184 |
| 2026-07 | signed_notional | 4 | 0.02687440 | 1170182 |
| 2026-07 | signed_notional | 5 | 0.09090635 | 1170180 |
| 2026-07 | signed_notional | 6 | 0.03210249 | 1170178 |
| 2026-07 | signed_notional | 7 | 0.02122048 | 1170176 |
| 2026-07 | signed_notional | 8 | 0.00920136 | 1170174 |
| 2026-07 | signed_notional | 9 | 0.00502792 | 1170172 |
| 2026-07 | signed_notional | 10 | 0.00833665 | 1170170 |
| 2026-07 | signed_notional | 11 | 0.00672675 | 1170168 |
| 2026-07 | signed_notional | 12 | 0.00931710 | 1170166 |
| 2026-07 | signed_notional | 13 | 0.00713641 | 1170164 |
| 2026-07 | signed_notional | 14 | 0.00503081 | 1170162 |
| 2026-07 | signed_notional | 15 | 0.02385519 | 1170160 |
| 2026-07 | signed_notional | 16 | 0.01412822 | 1170158 |
| 2026-07 | signed_notional | 17 | 0.03047947 | 1170156 |
| 2026-07 | signed_notional | 18 | 0.01471524 | 1170154 |
| 2026-07 | signed_notional | 19 | 0.00938678 | 1170152 |
| 2026-07 | signed_notional | 20 | 0.01091405 | 1170150 |
| 2026-07 | signed_notional | 21 | 0.00910766 | 1170148 |
| 2026-07 | signed_notional | 22 | 0.00604908 | 1170146 |
| 2026-07 | signed_notional | 23 | 0.00636028 | 1170144 |
| 2026-07 | signed_notional | 24 | 0.00415136 | 1170142 |
| 2026-07 | signed_notional | 25 | 0.00320684 | 1170140 |
| 2026-07 | signed_notional | 26 | 0.00289012 | 1170138 |
| 2026-07 | signed_notional | 27 | 0.00489025 | 1170136 |
| 2026-07 | signed_notional | 28 | 0.00494857 | 1170134 |
| 2026-07 | signed_notional | 29 | 0.00177125 | 1170132 |
| 2026-07 | signed_notional | 30 | 0.00228582 | 1170130 |

### Argmax Lag Distribution

| variant | argmax_lag_sec_btc_leads_positive | block_count |
|---|---|---|
| log1p_notional | 0 | 5 |
| raw_notional | 0 | 5 |
| signed_notional | 0 | 5 |

## Step 4: Conditional Large-BTC Check

The full-population BTC raw-flow 99th percentile is 292.30297200. The restriction is strict (`btc_flow > percentile`) and N is the number of selected BTC cascade bins available at each lag. **Positive lag means BTC leads ETH; `r = corr(BTC[t], ETH[t + lag])`.**

| lag_sec_btc_leads_positive | pearson_r | N |
|---|---|---|
| -30 | 0.01472249 | 93047 |
| -29 | 0.01736461 | 93047 |
| -28 | 0.05646041 | 93047 |
| -27 | 0.02835794 | 93047 |
| -26 | 0.01151818 | 93047 |
| -25 | 0.02271666 | 93047 |
| -24 | 0.01141121 | 93047 |
| -23 | 0.01553407 | 93047 |
| -22 | 0.02376499 | 93047 |
| -21 | 0.01478345 | 93047 |
| -20 | 0.01155010 | 93047 |
| -19 | 0.02821508 | 93047 |
| -18 | 0.01532617 | 93047 |
| -17 | 0.01882145 | 93047 |
| -16 | 0.01675337 | 93047 |
| -15 | 0.01413298 | 93047 |
| -14 | 0.01622024 | 93047 |
| -13 | 0.01580353 | 93047 |
| -12 | 0.01857764 | 93047 |
| -11 | 0.01733743 | 93047 |
| -10 | 0.01925967 | 93047 |
| -9 | 0.01916089 | 93047 |
| -8 | 0.01018884 | 93047 |
| -7 | 0.01288252 | 93047 |
| -6 | 0.02285419 | 93047 |
| -5 | 0.02777262 | 93047 |
| -4 | 0.02188146 | 93047 |
| -3 | 0.02417700 | 93047 |
| -2 | 0.05734899 | 93047 |
| -1 | 0.09565591 | 93047 |
| 0 | 0.19039026 | 93047 |
| 1 | 0.12818827 | 93044 |
| 2 | 0.04728125 | 93042 |
| 3 | 0.04023969 | 93040 |
| 4 | 0.03924532 | 93040 |
| 5 | 0.03898204 | 93040 |
| 6 | 0.02200674 | 93040 |
| 7 | 0.01303100 | 93040 |
| 8 | 0.02954468 | 93040 |
| 9 | 0.03695199 | 93040 |
| 10 | 0.01740586 | 93039 |
| 11 | 0.09433286 | 93038 |
| 12 | 0.01597746 | 93038 |
| 13 | 0.01543246 | 93038 |
| 14 | 0.00776206 | 93038 |
| 15 | 0.01777649 | 93038 |
| 16 | 0.01682454 | 93038 |
| 17 | 0.02960320 | 93038 |
| 18 | 0.02292030 | 93037 |
| 19 | 0.02190794 | 93037 |
| 20 | 0.01861281 | 93037 |
| 21 | 0.00996235 | 93037 |
| 22 | 0.01340021 | 93036 |
| 23 | 0.01220987 | 93036 |
| 24 | 0.00674225 | 93036 |
| 25 | 0.01518999 | 93036 |
| 26 | 0.00901033 | 93036 |
| 27 | 0.01817097 | 93036 |
| 28 | 0.03459393 | 93036 |
| 29 | 0.01723645 | 93035 |
| 30 | 0.01030860 | 93034 |

## Step 5: Circular-Shift Negative Control

Twenty deterministic random circular offsets were drawn without replacement from offsets at least 604,800 bins (7 days) away from zero in either circular direction. For each shuffle and transform, the table reports the peak absolute Pearson correlation across -30s..+30s. This seed fixes reproducibility only (`20260718`); it is not tuned.

| shuffle | offset_bins | variant | peak_abs_r | peak_abs_lag_sec_btc_leads_positive |
|---|---|---|---|---|
| 1 | 4468528 | raw_notional | 0.00072602 | -5 |
| 1 | 4468528 | log1p_notional | 0.00136077 | -3 |
| 1 | 4468528 | signed_notional | 0.00079844 | -5 |
| 2 | 1440345 | raw_notional | 0.00380211 | 17 |
| 2 | 1440345 | log1p_notional | 0.00077089 | 19 |
| 2 | 1440345 | signed_notional | 0.00367050 | 17 |
| 3 | 4748221 | raw_notional | 0.00055041 | -3 |
| 3 | 4748221 | log1p_notional | 0.00139038 | -24 |
| 3 | 4748221 | signed_notional | 0.00056469 | -3 |
| 4 | 4229471 | raw_notional | 0.00019601 | 29 |
| 4 | 4229471 | log1p_notional | 0.00180319 | 9 |
| 4 | 4229471 | signed_notional | 0.00028444 | 29 |
| 5 | 6071922 | raw_notional | 0.00062264 | 21 |
| 5 | 6071922 | log1p_notional | 0.00110607 | -29 |
| 5 | 6071922 | signed_notional | 0.00062458 | 21 |
| 6 | 4584047 | raw_notional | 0.00017795 | 3 |
| 6 | 4584047 | log1p_notional | 0.00078906 | 0 |
| 6 | 4584047 | signed_notional | 0.00018965 | 30 |
| 7 | 3083060 | raw_notional | 0.00343852 | 12 |
| 7 | 3083060 | log1p_notional | 0.00127201 | 24 |
| 7 | 3083060 | signed_notional | 0.00349581 | 12 |
| 8 | 3532555 | raw_notional | 0.00017775 | 5 |
| 8 | 3532555 | log1p_notional | 0.00096267 | 29 |
| 8 | 3532555 | signed_notional | 0.00021985 | 7 |
| 9 | 893911 | raw_notional | 0.00103333 | -7 |
| 9 | 893911 | log1p_notional | 0.00070238 | 6 |
| 9 | 893911 | signed_notional | 0.00103870 | -7 |
| 10 | 7209783 | raw_notional | 0.00017932 | 2 |
| 10 | 7209783 | log1p_notional | 0.00237940 | -4 |
| 10 | 7209783 | signed_notional | 0.00027095 | 2 |
| 11 | 1372402 | raw_notional | 0.00070133 | 22 |
| 11 | 1372402 | log1p_notional | 0.00246196 | -19 |
| 11 | 1372402 | signed_notional | 0.00054946 | 22 |
| 12 | 7347298 | raw_notional | 0.00101879 | 10 |
| 12 | 7347298 | log1p_notional | 0.00088106 | -19 |
| 12 | 7347298 | signed_notional | 0.00107352 | 10 |
| 13 | 7080734 | raw_notional | 0.00092237 | 23 |
| 13 | 7080734 | log1p_notional | 0.00118997 | 3 |
| 13 | 7080734 | signed_notional | 0.00084139 | -1 |
| 14 | 8103825 | raw_notional | 0.00210729 | 20 |
| 14 | 8103825 | log1p_notional | 0.00211562 | 22 |
| 14 | 8103825 | signed_notional | 0.00178032 | 20 |
| 15 | 2592616 | raw_notional | 0.00162807 | -8 |
| 15 | 2592616 | log1p_notional | 0.00288110 | -10 |
| 15 | 2592616 | signed_notional | 0.00166559 | -8 |
| 16 | 4754328 | raw_notional | 0.00013730 | 17 |
| 16 | 4754328 | log1p_notional | 0.00070921 | 7 |
| 16 | 4754328 | signed_notional | 0.00023406 | 17 |
| 17 | 2224622 | raw_notional | 0.00058869 | -22 |
| 17 | 2224622 | log1p_notional | 0.00147763 | -8 |
| 17 | 2224622 | signed_notional | 0.00031138 | -22 |
| 18 | 8401745 | raw_notional | 0.00016890 | -11 |
| 18 | 8401745 | log1p_notional | 0.00109458 | 16 |
| 18 | 8401745 | signed_notional | 0.00023853 | -11 |
| 19 | 6130890 | raw_notional | 0.00067080 | -16 |
| 19 | 6130890 | log1p_notional | 0.00122518 | 11 |
| 19 | 6130890 | signed_notional | 0.00066267 | -16 |
| 20 | 1602905 | raw_notional | 0.00056457 | 6 |
| 20 | 1602905 | log1p_notional | 0.00127656 | 1 |
| 20 | 1602905 | signed_notional | 0.00042972 | 6 |

### Shuffle Null Distribution Summary

| variant | null_min | null_median | null_p95 | null_max | real_peak_abs_r | real_peak_abs_lag_sec_btc_leads_positive | real_exceeds_null_max |
|---|---|---|---|---|---|---|---|
| log1p_notional | 0.00070238 | 0.00124860 | 0.00248291 | 0.00288110 | 0.39597000 | 0 | True |
| raw_notional | 0.00013730 | 0.00064672 | 0.00345670 | 0.00380211 | 0.15913715 | 0 | True |
| signed_notional | 0.00018965 | 0.00059463 | 0.00350454 | 0.00367050 | 0.15688006 | 0 | True |

## Microstructure Research Seeds

These are research-only follow-ups planted by the measured zero-lag result. They are not signals, trading rules, optimized variants, or promotion candidates.

| seed_id | question | required measurement | falsifier / guardrail |
|---|---|---|---|
| MS-SEED-BE-001 | Does a sub-second BTC→ETH ordering exist inside the stable 0s bin? | Rebuild event-time cross-correlation at millisecond resolution using `trade_time_ms`, while reporting timestamp ties and collector latency separately. | Reject if the monthly ordering changes sign, remains tied, or sits inside a timestamp-jitter control band. |
| MS-SEED-BE-002 | Is the zero-lag relationship a common market shock rather than directed transmission? | Condition BTC/ETH co-liquidation intensity on all-market forced-order intensity and independently measured market-wide stress. | Reject directed transmission if the BTC→ETH component vanishes after the common-shock control. Do not use price outcomes. |
| MS-SEED-BE-003 | Does zero-lag coupling differ between long- and short-liquidation cascades? | Pre-register separate `SELL`/long-liquidation and `BUY`/short-liquidation descriptive correlograms with the same gaps, lags, blocks, and shuffle controls. | No pooling or threshold search; reject any side-specific claim that is unstable across calendar blocks. |
| MS-SEED-BE-004 | Can symbol-specific timestamp latency create an artificial 0s peak? | Compare `ts_ms - trade_time_ms` distributions for BTC and ETH by month and collector regime, including tie rates. | Treat any inferred sub-second ordering as instrumentation if latency differences are of comparable magnitude. |
| MS-SEED-BE-005 | Negative knowledge: should +1s..+30s BTC-leading liquidation rules be pursued on this dataset? | Preserve this report as the baseline nullifier for future proposals. | Do not reopen the family without a new source, finer timestamp evidence, or a pre-registered mechanism that directly addresses the stable 0s result. |

## VERDICT

**LEAD_LAG_REJECTED**

Raw notional peaks at 0s with r=0.15913715, and all 5 monthly raw blocks also peak at 0s; the stable result is contemporaneous coupling, not a positive BTC-leading interval. The peak clears the 20-shuffle raw null maximum (0.00380211) and the large-BTC conditional peak strengthens to 0.19039026, but those checks strengthen only the zero-lag relationship. Because the mandatory positive BTC-lead condition fails, the BTC-to-ETH lead-lag finding is dead.
