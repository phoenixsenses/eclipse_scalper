Perfect timing. A \*\*README.md\*\* is the front door of the project.

Below is a \*\*clean, honest, first-version README\*\* you can paste as-is.

It explains what this is, how to run it, and—crucially—how \*\*not\*\* to hurt yourself.



---



```markdown

\# Eclipse Scalper



A modular, safety-first crypto trading bot built in Python.



\*\*Eclipse Scalper\*\* is a research-driven scalping framework designed for

Binance (spot \& futures), with a strong emphasis on:

\- risk control

\- dry-run safety

\- clean architecture

\- debuggability

\- and gradual escalation from simulation → micro → live



This repository contains the \*\*core engine\*\*, not exchange secrets or runtime state.



---



\## ⚠️ Disclaimer (Read This First)



This project is \*\*experimental software\*\*.



\- It can lose money.

\- It will behave exactly as you tell it to.

\- There are no guarantees of profitability.



Use \*\*dry-run mode\*\* first.  

Use \*\*micro capital\*\* next.  

Only then consider real exposure.



You are responsible for every trade executed.



---



\## Project Structure



```



eclipse\_scalper/

│

├── bot/                # Core runner and orchestration

├── execution/          # Entry, exit, order routing, guardian loops

├── strategies/         # Signal logic (Eclipse Scalper strategy)

├── risk/               # Kill-switches and safety logic

├── exchanges/          # Exchange adapters (Binance)

├── notifications/      # Telegram, alerts

├── config/             # Static configuration helpers

├── utils/              # Logging, helpers

├── tools/              # Smoke tests \& diagnostics

│

├── main.py             # Main entry point

├── guardian.py         # Global safety guardian

├── signal\_check.py     # Signal diagnostics

├── settings.py         # Runtime settings

└── requirements.txt    # Python dependencies



````



---



\## What Is \*Not\* in This Repo (On Purpose)



The following are \*\*ignored by Git\*\* for safety:



\- `.env` (API keys, secrets)

\- `logs/`

\- `brain/` (state / persistence)

\- `data/`

\- runtime logs and artifacts



You must create these locally.



---



\## Requirements



\- Python \*\*3.10+\*\*

\- Git

\- Binance account (for live trading)

\- Telegram bot token (optional, for alerts)



---



\## Installation



Clone the repository:



```bash

git clone https://github.com/phoenixsenses/eclipse\_scalper.git

cd eclipse\_scalper

````



Create a virtual environment (recommended):



```bash

python -m venv .venv

.venv\\Scripts\\activate   # Windows

```



Install dependencies:



```bash

pip install -r requirements.txt

```



---



\## Environment Variables (`.env`)



Create a `.env` file in the project root.



Example (DO NOT COMMIT THIS):



```env

BINANCE\_API\_KEY=your\_key\_here

BINANCE\_API\_SECRET=your\_secret\_here



SCALPER\_DRY\_RUN=1

SCALPER\_SIGNAL\_PROFILE=micro

TELEGRAM\_TOKEN=optional

TELEGRAM\_CHAT\_ID=optional

```



\*\*Important flags:\*\*



\* `SCALPER\_DRY\_RUN=1` → no real orders

\* `SCALPER\_DRY\_RUN=0` → live trading (dangerous)



---



\## Running the Bot (Dry Run)



Start in dry-run mode:



```bash

python main.py

```



Watch the logs carefully.

No orders should hit the exchange when `SCALPER\_DRY\_RUN=1`.



---



\## Safety Philosophy



Eclipse Scalper is designed around layered defense:



\* Signal confidence thresholds

\* Position sizing limits

\* Guardian loops

\* Kill switches

\* Dry-run first, always



If something looks wrong — \*\*it probably is\*\*.

Stop the bot, inspect logs, adjust, retry.



---



\## Development Status



\* ✅ Core execution engine

\* ✅ Signal pipeline

\* ✅ Order router

\* ✅ Risk kill-switches

\* ⚠️ Strategy tuning ongoing

\* ⚠️ Live trading requires caution



This project is under \*\*active development\*\*.



---



\## License



Private / personal use for now.

License to be defined later.



---



\## Final Note



Treat this bot like a sharp instrument.



Used carefully, it teaches you a lot.

Used carelessly, it teaches you faster.



Proceed deliberately.

