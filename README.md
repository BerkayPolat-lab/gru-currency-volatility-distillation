# GRU Volatility Distillation (EUR/USD)

A compact one-layer GRU student is distilled from a deeper GRU teacher to forecast EUR/USD volatility.
The trained models are published to Hugging Face, served by a FastAPI Space, and displayed in a Next.js app on Vercel.

The full research protocol is in [`docs/03_GRU_EURUSD_Volatility_Distillation.pdf`](docs/03_GRU_EURUSD_Volatility_Distillation.pdf) (ECE 662, Fall 2026).
This README summarizes it and adds the deployment plan.

## Investigative question

Can a distilled one-layer GRU preserve a deeper GRU's volatility-forecasting quality at lower inference cost, including during high-volatility periods?
Does distillation outperform direct training of the same student?

The task is volatility forecasting, not trading.
Forecast-error improvements must never be read as trading profitability.

## Phases

| Phase | Where | Output | Status |
| --- | --- | --- | --- |
| 1. Model training and testing | Google Colab (`model/notebooks`) | Checkpoints, manifests, four metric tables | In progress |
| 2. Model publishing | Hugging Face Hub (model repo) | Student and teacher weights with scaling metadata | Not started |
| 3. Inference service | Hugging Face Space, FastAPI (`inference/`) | `/predict` endpoint | Not started |
| 4. Web application | Next.js on Vercel (`web/`) | Daily forecast dashboard with live spot | Not started |

Phases 3 and 4 begin only after phase 1 is complete.

## Repository layout

```
.
├── docs/                          Project proposal (source of truth for the protocol)
├── model/
│   ├── configs/base.yaml          Frozen experiment settings
│   ├── notebooks/                 Colab notebooks, run in order 01 to 08
│   ├── src/gru_vol/               Shared code imported by every notebook
│   ├── data/{raw,processed}/      Local data (git-ignored)
│   ├── artifacts/                 Checkpoints, manifests, teacher cache, results (git-ignored)
│   └── requirements.txt
├── inference/                     Hugging Face Space (FastAPI)
└── web/                           Next.js app for Vercel
```

## Phase 1: model training and testing

### Data

- Source: FRED `DEXUSEU`, daily U.S. dollars per euro, 2000-2025, frozen as a dated snapshot.
- Only observed dates are kept, with no interpolation.
- Log return: `r(t) = log(P(t) / P(t-1))`.
- Inputs: the previous 60 observations of returns, squared returns, and trailing 5- and 20-observation volatility.
- Target: `v(t,h) = (1/h) * sum of r(t+j)^2 for j = 1..h`, with `h = 5` (core) and `h = 20` (optional extension).
- Target scaling divides by the positive training-period mean without centering.
- Input scaling is fitted on training data only.

### Splits

| Split | Years |
| --- | --- |
| Train | 2000-2017 |
| Validation | 2018-2020 |
| Test | 2021-2025 |

Examples whose forward target crosses a split boundary are dropped.

### Models

| Model | Architecture | Loss |
| --- | --- | --- |
| Deep teacher | 3 GRU layers x 128 units, dropout, nonnegative head | MSE |
| Direct student | 1 GRU layer x 32 units | MSE (alpha = 0) |
| Distilled student | Same as direct student | `(1 - alpha) * MSE(student, target) + alpha * MSE(student, teacher)` |
| Historical control | Trailing 20-observation mean squared return | None |

- Alpha is chosen from {0.25, 0.50, 0.75} on validation data.
- Teacher predictions are cached on training inputs only.
- Both students share targets, batches, early stopping, tuning budget, and starting weights per seed.
- AdamW, at most 100 epochs, early stopping on validation MAE.
- Three final seed triplets give nine main neural fits.

### Metrics

1. Forecast MAE in original variance-proxy units, per seed and as mean with standard deviation.
2. High-volatility MAE on test targets above the training 90th percentile, with the qualifying-date count.
3. Batch-1 inference latency, median and p95, with CPU and GPU reported separately.
4. Serialized model size of the deployment state dict with scaling metadata.

Training, caching, and search time are also recorded as compute accounting.
A student within 5% of teacher MAE with lower latency and size is the predeclared research target, not an expected result.

### Notebooks

| Notebook | Purpose |
| --- | --- |
| `01_data_pipeline` | Snapshot, features, targets, splits, scaling, manifests |
| `02_historical_control` | Trailing-20 baseline and high-volatility cutoff |
| `03_teacher_training` | Teacher pilot and seeded training |
| `04_teacher_cache` | Frozen teacher predictions on training inputs |
| `05_student_training` | Alpha search, direct and distilled students |
| `06_evaluation` | MAE tables, high-volatility analysis, figures |
| `07_benchmark` | Latency and model size |
| `08_export_to_huggingface` | Push deployment checkpoints to the Hub |

### Running in Colab

1. Add two Colab Secrets: `GITHUB_TOKEN` (read access to this private repo) and `HF_TOKEN` (write access, needed for notebook 08).
2. Open a notebook from GitHub in Colab and run the setup cell.
   It clones the repo, installs `model/requirements.txt`, and mounts Google Drive.
3. Checkpoints, manifests, and caches are saved to `MyDrive/gru_vol_artifacts` so they survive Colab session resets.

## Phase 2: Hugging Face model repo

- Weights for the distilled student (deployment model) and teacher (comparison) in `safetensors`.
- A `config.json` with feature scaling, the target scaling constant, window length, and horizon.
- A model card with the protocol, test metrics, and the not-a-trading-signal disclaimer.

## Phase 3: inference service

A FastAPI app in a Docker Hugging Face Space on free CPU hardware.

- Loads both models from the Hub at startup.
- `POST /predict` takes recent daily closes and returns the student forecast, the teacher forecast, and the historical control.
- Free Spaces sleep when idle, so the first request after a pause is slow.

## Phase 4: web application

A Next.js app on Vercel.
"Real-time" means two things here:

- **Daily forecast:** inference runs once per trading day on daily closes, matching how the model was trained.
- **Live spot:** the latest EUR/USD rate is shown next to the forecast for context only and never changes the forecast.

The dashboard shows the distilled student forecast with the teacher forecast and historical control side by side.

### Live data source: FXAPI

The free FXAPI plan allows 300 requests per month and 10 per minute, and includes historical data.
Estimated usage:

- Bootstrap: the window needs about 81 trading days of history.
  Most of it comes from FRED, so FXAPI only fills the days FRED has not published yet.
- Daily top-up: about 22 requests per month, one per trading day, made by a Vercel Cron job.
- Live spot: the free plan updates rates once a day, so the daily request also provides the spot value.

Results are cached on the server so page views never call FXAPI directly.

## Open questions

- **Intraday spot:** FXAPI's free plan updates daily, so the "live spot" will not move during the day.
  A true intraday ticker needs a paid FXAPI plan or a second source.
- **Source mismatch:** the model is trained on FRED's noon New York buying rates, but live inputs come from FXAPI.
  Measure the difference on overlapping dates before relying on the live forecast.
- **Calendar alignment:** FXAPI may quote weekend dates.
  Live inputs must keep only the business days that FRED would have observed.
- **Rolling price storage:** the app needs a small persistent store for the rolling price window, such as Vercel KV or Blob.
- **Hyperparameters not fixed by the proposal:** teacher dropout, learning rate, weight decay, and patience in `base.yaml` are starting points to confirm in the pilot.
