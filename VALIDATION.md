# Validation — 18 November 2026

- **26 automated tests passed**. These use synthetic fixtures and mocked APIs, not historical profitability backtests.
- Native official Kronos-mini and the 2k tokenizer downloaded successfully from Hugging Face; CPU inference on a synthetic 120-candle OHLC/zero-volume input returned three finite predictions. Three-sample inference plus loading completed in about 33.5 seconds in the development environment. GitHub performance may differ.
- Python syntax compilation passed.
- Workflow YAML parsed; schedule, inputs and permissions checked. The workflow itself has not run in the user's GitHub account.
- Local bare-Git simulation verified state branch creation, checkpoint push and worktree restore without modifying source files.
- No Upstox credential or Telegram credential was supplied. Authenticated live-chain data, actual Telegram delivery and end-to-end GitHub scheduling remain untested. Run the included connection test, then observe a full paper session.

Tests include ATM rounding, put-call parity, leave-one-out IV, unsupported wing rejection, fixed 15%/10% barriers, first-touch ambiguity, chronological target/stop handling, gap stops, exclusion of entry-minute history, persistence/dedup, shadow exclusion, idempotent training, delayed opening skip, same-contract opening comparisons, fixed model targets, scan-band limits, missing-data handling, Telegram failure queue retention, main/counter generation and forward forecast grading.

The code is a research implementation, not an established profitable strategy. Learning can reduce or worsen future performance. No win-rate claim is made. Read README for timing, polling and fair-value assumptions.
