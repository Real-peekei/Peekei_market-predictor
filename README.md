# Market Predictor

A predictive model for **any tradable product** — not just gold. Same
architecture as the football predictor: Python backend (data → features →
model → live API) + a static frontend served via Live Server, calling the
API in real time.

Point it at gold, a stock, a forex pair, crypto, another commodity —
whatever your data source supports. Every asset you fetch/train gets its
own namespace, so multiple products' data and models coexist without
colliding, and the site's **ASSET** dropdown lets you switch between
whichever ones you've trained.

Works across any common timeframe: `1m 2m 5m 15m 30m 45m 1h 90m 2h 4h 8h 1d
1wk 1mo` (exact set depends on your data source — see below) — train a
separate model per timeframe you care about, and the **TIMEFRAME** dropdown
lists them in proper chronological order (shortest to longest), not
alphabetically.

## What it predicts

For the next bar (next day, next hour, etc. depending on timeframe):
- **Direction** — probability of the next close being higher (classifier)
- **Expected range** — 20th-80th percentile price range (quantile regression,
  not a single point guess — markets are noisy, a range is honest)
- **Walk-forward backtest** — the model is retrained periodically through
  history and evaluated bar-by-bar on data it hadn't seen yet, then compared
  to plain buy-and-hold over the same stretch. This is slower to compute than
  a single train/test split but it's the only way to get a number you should
  actually trust.

## 1. Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get data — for any symbol

**Option A — yfinance (free, no API key needed):**

```bash
python3 data_fetch.py --symbol GC=F     --interval 1d --period 10y    # gold futures
python3 data_fetch.py --symbol AAPL     --interval 1h --period 730d   # a stock
python3 data_fetch.py --symbol BTC-USD  --interval 15m --period 60d   # crypto
python3 data_fetch.py --symbol EURUSD=X --interval 1d --period 10y    # forex
```

Output files are automatically namespaced by symbol — `--symbol GC=F
--interval 1d` writes `data/GC_F_1d.csv` (plus a small `GC_F_meta.json`
recording the original symbol), `--symbol AAPL --interval 1h` writes
`data/AAPL_1h.csv`, and so on. This is what lets multiple assets' data live
side by side without overwriting each other.

Note Yahoo Finance's own limits on intraday history: `2m/5m/15m/30m/90m`
data only goes back ~60 days, `1m` only ~7 days, `1h` goes back ~730 days,
daily+ goes back essentially forever. This isn't a limitation of this
project — it's a Yahoo Finance constraint on free intraday data.

**A real caveat with yfinance's OTC/spot-style tickers** (e.g. `XAUUSD=X`,
`EURUSD=X`): they're unofficial, not maintained products — data through
them can be unreliable (missing bars, outright failures). yfinance is solid
for genuine exchange-traded instruments (`GC=F` futures, `AAPL` stock,
`BTC-USD`) but not always the best choice for spot/OTC pairs specifically.
See Option B below for a more reliable alternative for those.

**Option B — Twelve Data (better for spot/OTC instruments specifically):**

Twelve Data has dedicated, documented symbols for spot commodities, forex
pairs, stocks, and crypto — more reliable than yfinance's unofficial OTC
tickers for those instrument types. Needs a free API key (no credit card):

1. Sign up at https://twelvedata.com
2. Copy your API key from the dashboard
3. Set it as an environment variable (never hardcode it):
   ```bash
   export TWELVEDATA_API_KEY="your_key_here"        # Mac/Linux
   $env:TWELVEDATA_API_KEY = "your_key_here"          # Windows PowerShell
   set TWELVEDATA_API_KEY=your_key_here                # Windows cmd
   ```
4. Fetch:
   ```bash
   python3 twelvedata_fetch.py --symbol XAU/USD --interval 1d  --period 10y
   python3 twelvedata_fetch.py --symbol EUR/USD --interval 1h  --period 730d
   ```
   Or fetch every timeframe at once through `fetch_all.py`:
   ```bash
   python3 fetch_all.py --source twelvedata --symbol XAU/USD
   ```
   Twelve Data supports **12 intervals**: `1m, 5m, 15m, 30m, 45m, 1h, 2h,
   4h, 8h, 1d, 1wk, 1mo`. yfinance supports its own **10 intervals**: `1m,
   2m, 5m, 15m, 30m, 90m, 1h, 1d, 1wk, 1mo` (`60m` is also valid on
   yfinance but identical to `1h`, so it's left out to avoid fetching the
   same data twice under two names).

Free tier: **8 requests/minute, 800/day** (resets midnight UTC) — plenty
for normal use; the script backs off and retries automatically if you hit
the per-minute cap.

**Note on volume for spot instruments:** Twelve Data's spot/OTC feeds
(commodities, forex) don't report trading volume — there's no single
central exchange for OTC instruments to report it from — so the `Volume`
column comes back as 0. This makes one of the model's ~20 features
(`vol_ratio`) uninformative for spot-sourced data — a real but minor
limitation, handled gracefully (set to a neutral constant rather than
causing errors), not a bug.

**Worked example — finding futures instead of spot on Twelve Data:**

If you want a genuinely exchange-traded product (real volume) rather than
spot/OTC, you need the right symbol first. Twelve Data's dedicated
Commodities product (the one behind `XAU/USD`) is **explicitly documented
as spot-only** — there's no "Futures" product listed alongside it, and no
stable published symbol format for e.g. COMEX gold futures in their public
docs. Don't guess a symbol (e.g. `GC1!`, which is TradingView's notation,
not necessarily Twelve Data's) — ask Twelve Data directly what it actually
has:

```bash
python3 twelvedata_symbol_search.py --query gold
python3 twelvedata_symbol_search.py --query "apple stock"
```

This hits Twelve Data's own `symbol_search` reference endpoint and prints
every matching instrument, with its exact symbol, exchange, and instrument
type. Once you find the right one, use it directly:

```bash
python3 twelvedata_fetch.py --symbol "<symbol from search results>" --interval 1d
python3 fetch_all.py --source twelvedata --symbol "<symbol from search results>"
```

If nothing relevant shows up, that instrument type likely isn't on your
current Twelve Data plan tier — check https://twelvedata.com/pricing.

(For gold specifically: yfinance's `GC=F` is already a genuine,
exchange-traded futures contract with real reported volume, free, no API
key needed — you may not need Twelve Data for gold futures at all. Twelve
Data mainly earns its keep here for spot/OTC reliability.)

**Option C — bundled synthetic sample (offline testing only):**

```bash
python3 generate_sample_data.py
```

Writes `data/SAMPLE_1d.csv` — a random-walk-like series with realistic
volatility clustering. **This has no real market structure** and exists only
to confirm the pipeline runs, for any asset workflow, before you point it at
real data. On this data the model should land close to a 50% hit rate — if
it claims a much higher edge on synthetic random-walk data, that's a red
flag the backtest is leaking future information somewhere, not a sign of a
good model.

## 3. Train (per asset, per timeframe)

```bash
python3 train_model.py --data data/GC_F_1d.csv --interval 1d
python3 train_model.py --data data/AAPL_1h.csv --interval 1h
```

The asset (`GC_F`, `AAPL`, ...) is automatically inferred from the data
filename — `data_fetch.py`/`twelvedata_fetch.py` always name files
`{ASSET}_{interval}.csv`, so this "just works" if you used those. Pass
`--asset` explicitly to override if you renamed the file.

Run once per asset+timeframe combination you want available in the
dropdowns. Each run prints the walk-forward backtest results to the
console — read those numbers before trusting anything the live predictor
says.

### If training feels slow (especially on shorter/high-volume timeframes)

The walk-forward backtest is the slow part — it retrains the model many
times through history rather than testing once, because that's the only
honest way to simulate how it would've performed if traded live. By default
this is now capped so it doesn't blow up regardless of how much data you
feed it:

```bash
python3 train_model.py --data data/GC_F_1h.csv --interval 1h
# uses defaults: retrain every 40 bars, 1500-bar training window, tests the most recent 2500 bars
```

If it's still too slow, or you just want a quick rough pass while
iterating:

```bash
python3 train_model.py --data data/GC_F_1h.csv --interval 1h --fast
```

Or tune it directly:
```bash
python3 train_model.py --data data/GC_F_1h.csv --interval 1h \
  --retrain-every 60 --max-train-window 1000 --max-test-bars 1500
```

- `--retrain-every N` — retrain less often (higher N = faster, less precise)
- `--max-train-window N` — cap how much history each retrain uses (0 = uncapped/expanding, much slower on long histories)
- `--max-test-bars N` — only backtest the most recent N bars instead of the whole history (0 = full history, much slower)

`train_all.py --fast` applies the fast preset across every asset+timeframe
combination in one go.

The console prints progress (`...800/2500 bars tested (32%)`) so you can see
it's actually working rather than frozen, plus how long the backtest took.

## 4. Run the live API

```bash
uvicorn server:app --reload --port 8001
```

(Port 8001, not 8000, so this can run alongside the football predictor if
you want both open at once.) Check it's up at http://127.0.0.1:8001/docs

## 5. Run the frontend

Open `frontend/index.html` with the VS Code **Live Server** extension. Pick
an **asset**, then a **timeframe**, hit **RUN PREDICTION**.

## How to actually judge whether this is any good

Don't just look at the direction call — look at the backtest panel:

- **Hit rate near 50%** → the model has little to no edge at that
  asset+timeframe combination. This is common and expected, especially on
  shorter timeframes and heavily-traded, liquid instruments.
- **Strategy Sharpe below buy-and-hold Sharpe** → the model isn't adding
  value over doing nothing extra.
- **Strategy return below buy-and-hold** → same story, in blunter terms.
- A model clearing all three by a meaningful, consistent margin across
  multiple timeframes is the bar for "this might actually be worth watching"
  — and even then, this backtest ignores fees, slippage, and the fact that
  most liquid instruments are heavily traded by professionals with far more
  data and speed. Treat any edge found here as a hypothesis to keep testing,
  not a result to act on with real money.

## Model selection (which model gets used)

Same idea as the football predictor: `train_model.py` trains three
candidate direction-classifiers (gradient boosting, random forest, logistic
regression) per asset+timeframe, scores each on a chronological hold-out,
and keeps the winner — used for both the live model and inside the
walk-forward backtest loop. It prints the comparison so you can see how
close it was.

## Fetch, train, and predict across every asset and timeframe at once

```bash
python fetch_all.py --symbol GC=F                       # all 10 yfinance-supported timeframes, one asset
python fetch_all.py --symbol AAPL --intervals 1d 1h 15m  # a subset of timeframes
python fetch_all.py --source twelvedata --symbol XAU/USD # all 12 Twelve Data timeframes

python train_all.py                    # no --asset given: trains EVERY asset you have data for
python train_all.py --asset GC_F       # just one asset, all its available timeframes

python predict_all.py --asset GC_F     # prints every trained timeframe for one asset, chronologically sorted
```

`train_all.py` auto-discovers every asset with data files in `data/` and
skips any asset+timeframe combination it doesn't have data for (with a
note), rather than failing the whole run. Same idea via the site: pick an
asset, click **"ALL TIMEFRAMES"** to see every trained timeframe's
prediction for that asset, side by side, sorted shortest-to-longest. Or hit
the API directly: `GET /api/predict-all?asset=GC_F`.

## When to trust the direction call and price range — in plain terms

Every prediction comes with a **trust read**, computed from that specific
asset+timeframe's own walk-forward backtest — not generic advice, a
judgment based on what that model has actually shown on real history:

- **"Worth watching"** - hit rate was clearly above a coin flip (>53%) AND it
  beat plain buy-and-hold on both total return and risk-adjusted return
  (Sharpe) in the backtest. This means the model has shown *some* real skill
  historically. Still not a green light to trade on its own - past edge
  doesn't guarantee future edge - but it's a real signal worth weighing
  alongside your own judgment.
- **"Mixed"** - some metrics beat buy-and-hold, some didn't. This is the most
  common result and the hardest to act on. Treat the direction call as one
  minor input, not something to rely on by itself.
- **"No real edge shown"** - hit rate near 50% (a coin flip) and it
  underperformed buy-and-hold. Treat any single prediction from this model as
  noise. This is a completely normal result, especially on short/liquid
  timeframes - it doesn't mean something is broken.
- **"Too little history"** - fewer than 100 bars were backtested, not enough
  to tell skill from luck either way. Fetch more history and retrain.

**Short version: trust it more when the backtest panel shows the model
actually beating buy-and-hold on return AND Sharpe AND hit rate, all three,
not just one of them. Trust it less - or not at all - the moment any of
those three go the other way.**

## How asset/timeframe namespacing works, if you're curious

Every symbol gets slugified into a filesystem-safe **asset key** (e.g.
`GC=F` → `GC_F`, `XAU/USD` → `XAU_USD`, `AAPL` → `AAPL`) via `assets.py`.
Data files are `data/{ASSET}_{interval}.csv`, model files are
`models/{ASSET}_clf_{interval}.pkl` (and similarly for the regressors,
state, backtest, and model-selection files). This is what lets you fetch
and train as many different products as you want without them colliding —
and it's how the API's `/api/assets` endpoint discovers what's available
for the frontend's ASSET dropdown.

Timeframes are sorted chronologically (not alphabetically — plain string
sort would put `"1d"` before `"1h"` before `"1m"`, which is meaningless)
via `intervals.py` on the backend and an equivalent mapping in
`script.js` on the frontend, so the TIMEFRAME dropdown and the "All
Timeframes" results always read shortest-to-longest.
