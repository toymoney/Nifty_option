# Nifty Options Research Bot — Upstox + Telegram

A GitHub-ready Python collector for Ajay's three strategies. No broker orders are placed.
Uses Upstox's read-only data API, native Kronos-mini inference, and Telegram.
The main program is `nifty_options_bot.py`; the workflow is needed for automatic GitHub running.

## Setup — do these once

1. Create an empty **public** GitHub repository. Extract this ZIP and upload its contents into the repository root, not a nested folder. Include `.github/workflows/nifty.yml` (a hidden folder on some systems), `requirements.txt`, `tests/`, and the Python file.
2. In Upstox Developer Apps, open Analytics and generate an Analytics Token.
3. GitHub repository → Settings → Secrets and variables → Actions → **New repository secret**. Create exactly:

   | Secret | Value |
   |---|---|
   | `UPSTOX_ACCESS_TOKEN` | Your Upstox Analytics Token |
   | `TELEGRAM_BOT_TOKEN` | Your Telegram bot token |
   | `TELEGRAM_CHAT_ID` | Your personal/group chat ID, including a leading minus sign if present |

   Start a conversation with your Telegram bot, or add it to your group and allow it to send messages. Never put these values in Python, YAML, screenshots, logs, or commits.
4. Settings → Actions → General → Workflow permissions → **Read and write permissions**. Save. The bot needs to write its state branch.
5. Actions → **Nifty options alerts** → Run workflow → choose **test**. A successful test sends an Upstox/Telegram connection message. This is a connectivity test, not a live strategy validation.
6. Run workflow again with **run** during market hours to start immediately. Thereafter weekday schedules run automatically. If first started after 09:25, strategy 2 intentionally skips that day's entry.
7. Look for the **bot-state** branch. It contains `state.json` and dated reports. Do not delete it: it is the learning/outcome memory. A public repository makes this research history public; credentials and chat ID are never written there.

Do not upload the ZIP itself as the only repository file. Upload the extracted contents.

## What each strategy does

### 1. Kronos prediction + learning from real forward outcomes

- Loads the official pretrained **Kronos-mini** model and matching 2k tokenizer. The official source is pinned to commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.
- Uses up to 256 **completed Nifty spot five-minute candles**, including previous sessions, to forecast three candles ahead. Minimum context: 120 candles. Nifty index has no traded volume: no volume is fabricated.
- Runs inference in a background thread. Price/exit monitoring continues while the model loads or predicts. Forecasts older than three minutes are discarded. Failed or disabled Kronos means **S1 skips**; it never silently substitutes another model.
- Evaluates every half-hour from 09:35–14:45. A forecast move of at least 0.10% selects ATM CE for up / ATM PE for down. Liquidity and freshness checks apply. Maximum three main S1 alerts/day, one open S1 alert at a time. This is a research baseline, not a validated profitable setup.
- Default paper target **+15%**, stop **−10%**, no trailing stop. These are assumptions for S1; your explicit S2 levels are unchanged.
- Simultaneously records a **shadow purchase of the opposite option at the same strike** when liquid. This is the counter trade: e.g. CE alert versus hypothetical PE purchase. It does not reverse an existing position or send an execution instruction.
- Stores entry features: forecast strength, IV, Delta, Gamma, Theta, Vega, OI change, spread, and time of day. A small online logistic ranking model updates at the end of the day from resolved main S1 target/stop outcomes. Ambiguous, missing, unresolved and shadow results are excluded.
- The first 30 resolved S1 outcomes are warm-up. Afterward scores below 0.45 suppress candidates. An additional side/IV bucket needs at least 20 outcomes and a target rate under 35% before it is suppressed. These are configurable code defaults, not validated optimal thresholds.
- Saves the score **before the outcome** and reports forward Brier score, win/loss feature associations, suppressed setups and counter outcomes. It also grades each saved Nifty forecast against the subsequent completed candle and reports direction accuracy and absolute error. A low sample size is labelled. It does not claim causal reasons for market movement.
- **The online ranker learns; the Kronos foundation weights are NOT fine-tuned.** Model training on a small number of alerts would not justify a promise of improved accuracy. Losses can recur. The 15-minute forecast is used for direction; option barriers are monitored through the session, not treated as a 15-minute forecast-accuracy label.

### 2. Previous-close ATM opening strategy

- Fetches the previous **trading session's Nifty daily closing price**. Rounds to the nearest 50: 25,626 → 25,650. It fixes that strike for today's opening test.
- Uses the nearest currently active expiry; compares today's opening price against the previous daily close of the **same contract and expiry**. On expiry rollover it never compares two different expiries.
- Checks both CE and PE using the completed 09:15–09:16 candle. If the opening premium is strictly above the same contract's previous close, it sends a research alert with strike, side, expiry, current LTP, opening price, previous close, **+15% target and −10% stop**.
- If both sides qualify, both are sent. If neither qualifies, no opening alert. A missing previous close is never invented.
- Entry/target reference is **LTP when the alert is generated**, not an unavailable retroactive opening fill. It retries data/liquidity until 09:25, then skips. The qualifying opening price itself remains the 09:15 price.
- Each contract alerts once/day. Target and stop stay fixed. One terminal outcome notification; no trailing SL.

### 3. Relative fair-value watch

- Scans **ATM + five strikes below + five above**, both CE and PE (up to 22 contracts), using the nearest expiry. The scan band follows current ATM; an already recorded alert stays monitored even if it moves outside the band.
- IV inferred from an option's own market price cannot independently prove that same option is mispriced. Instead, the code excludes its own IV and interpolates neighbouring liquid same-side IVs. OI and bid/ask spread weight those peers.
- Values the option with spot Black-Scholes, peer IV, strike, actual seconds to expiry, risk-free rate and dividend yield. Defaults: annual rate 6%, yield 1.2%; these are **assumptions**, not fetched market facts. Configure environment variables `RISK_FREE_RATE` and `DIVIDEND_YIELD` as decimals.
- Delta/Gamma/Theta/Vega are recorded and shown as **risk sensitivities**, with Greek shock scenarios saved in the journal. They are not added to a price as though they were independent valuation inputs. OI supports liquidity/peer weighting; it is not an intrinsic-value term.
- Signals both below-model and above-model premium differences of **at least 15%**; differences of 25% or more get a stronger category. Percentage gap is `(fair value − LTP) / fair value`. It must also clear the threshold at ask for below-model / bid for above-model.
- The fair-value target is **frozen at alert time**. Later movement of the estimate is not counted as price convergence. A further alert reports when LTP/candle reaches that fixed value. Overpriced contracts are watch alerts, not naked option-selling instructions.
- No extra S3 stop rule was requested, so this is a valuation watch, with no invented stop. Unreached watches end as EOD_OPEN or DATA_MISSING and are listed in the report. They do not silently carry overnight.
- Liquidity defaults: OI ≥1,000, volume ≥100, LTP ≥₹10, bid >0, valid spread ≤5%. No unsupported IV wing extrapolation; boundary strikes without peers on both sides may be skipped. No new valuations in the last 30 minutes before expiry, or after 15:00.
- Model value is **not a guaranteed actual/fair price or an arbitrage claim**. Dividends, volatility skew, quote quality, spreads and model assumptions can explain differences.

## Monitoring, reports and timing

- Polls roughly every 30 seconds; provider latency and API work can extend that interval. Completed one-minute candles recover intervening movements and handover gaps.
- Both target and stop in one candle → **AMBIGUOUS**, never a fabricated winner. The entry minute's historical high/low is excluded because it may precede the alert. S1/S2 outcome alerts wait for completed candles (typically 1–2 minutes); movements confined to the excluded entry minute can be missed. S3 has only one barrier and can also use a polled LTP touch. This is not tick-exact monitoring or an exchange stop order.
- Gaps below stops use the lower candle open for estimated exit. LTP paper entries and candle touches do not guarantee executable fills. Default round-trip cost/slippage assumption is 0.5% of entry premium (`ROUND_TRIP_COST_PCT=0.5`), not actual broker charges. Returns exclude position sizing and taxes not represented by this assumption.
- Approximately **16:15 IST**: Telegram report lists each strategy's outcomes, counter trades, valuation targets reached, remaining gaps, learning evidence and diagnostics. GitHub may delay it; a 16:17 fallback job is included. Reports also remain in `bot-state/reports`.
- Jobs are scheduled at 08:57, 14:07 and 16:17 IST. Morning monitoring ends at 14:10, the handover waits if necessary; afternoon monitoring/reporting ends at 16:25. Each job stays below GitHub's six-hour job limit. Weekends skip; on exchange holidays/stale data, freshness checks suppress entries. Special weekend sessions are not covered automatically.
- State saves locally after alerts; commits to `bot-state` every five minutes and on normal exit. Interrupted previous sessions are recovered using historical candles on the next run, if the provider still serves the contracts. Missing history stays DATA_MISSING.
- Telegram uses a persisted outbox and logical event IDs. Normal restarts deduplicate; **exactly-once delivery is not guaranteed** if the process dies after Telegram accepts a message but before state commits, or if the response is lost. A forced runner loss may lose up to five minutes of uncommitted state.
- GitHub schedules can be delayed/dropped and scheduled workflows can be disabled after inactivity. They are not suitable for guaranteed 09:15/tick-exact execution. This is the trade-off for free hosted research monitoring.

## Costs and controls

Upstox currently documents free data API access. Standard GitHub-hosted runners in public repositories are currently free under GitHub's policies; this package does not select larger paid runners or external paid hosting. Policies can change. You need not enable any paid service for this package. Large local model downloads happen on the runner and are cached; the model files are not committed.

- Stop: Actions → disable this workflow. To stop a currently running job, cancel it too.
- Pause Kronos only: repository variable `ENABLE_KRONOS=false`; S2 and S3 remain available.
- Test: workflow mode `test`.
- One poll/recovery: workflow mode `once`.
- Token expiration: renew your Upstox secret when needed; Analytics tokens currently document one-year validity.
- No `pull_request`, `pull_request_target` or push workflow is configured. Never add a workflow that exposes these secrets to untrusted contributions.

## Local usage

Python 3.11+, install `requirements.txt` and CPU PyTorch as shown in the workflow, then clone the pinned Kronos checkout to `vendor/Kronos`. Set the three environment variables above. Run:

```sh
python nifty_options_bot.py --mode test
python nifty_options_bot.py --mode run
python -m unittest discover -s tests -v
python tests/kronos_smoke.py  # optional native model download/inference test
```

On a local Windows machine, install `tzdata` if ZoneInfo cannot find Asia/Kolkata. `GIT_STATE` defaults off locally. Local state goes in `state/`; keep it between runs.

## Validation boundary

Automated synthetic tests cover calculations, same-contract opening logic, first-touch ambiguity, entry-minute exclusion, recovery, counter exclusions, deduplication, learning idempotence and Telegram failure retention. They do not establish market profitability. No user credentials were available during development, so authenticated Upstox responses, your Telegram delivery and a full live session must be checked by the connection test and the first paper session. Native Kronos execution status is recorded separately in VALIDATION.md.

## Official references (checked 18 November 2026)

- https://upstox.com/developer/api-documentation/analytics-token/
- https://upstox.com/developer/api-documentation/get-pc-option-chain/
- https://upstox.com/developer/api-documentation/get-option-contracts/
- https://upstox.com/developer/api-documentation/v3/get-intra-day-candle-data/
- https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/
- https://github.com/shiyu-coder/Kronos
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
