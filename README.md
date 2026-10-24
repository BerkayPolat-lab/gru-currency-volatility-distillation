# GRU Volatility Distillation (EUR/USD)

A compact one-layer GRU student is distilled from a deeper GRU teacher to forecast EUR/USD volatility.
The trained models are published to Hugging Face, served by a FastAPI Space, and displayed in a Next.js app on Vercel.

The full research protocol is in [`docs/03_GRU_EURUSD_Volatility_Distillation.pdf`](docs/03_GRU_EURUSD_Volatility_Distillation.pdf) (ECE 662, Fall 2026, revised 23 October 2026).
Its source is `docs/source/build_proposal.py`; edit the text there and rebuild the PDF rather than editing the PDF directly.
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
├── docs/                          Project proposal (source of truth) and its build script
├── model/
│   ├── configs/base.yaml          Frozen experiment settings
│   ├── notebooks/                 Colab notebooks, run in order 01 to 08
│   ├── src/gru_vol/               Shared code imported by every notebook
│   ├── data/raw/                  Frozen DEXUSEU snapshot
│   ├── data/processed/            Model-ready arrays (splits.npz) and the example table (examples.csv)
│   ├── artifacts/                 Manifests, results, and checkpoints, committed so the team shares them
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

### Scaling

Both transforms use only the trailing 60 observations ending at each forecast origin, so no statistic fitted on the whole training period is involved.
The same rule applies in training, evaluation, and deployment, and nothing has to be stored to reproduce it.

- **Inputs:** each feature in a 60-step window is z-scored with that feature's mean and standard deviation over the trailing 60 observations.
- **Target:** models predict the ratio `y(t) = v(t,5) / s(t)`, where `s(t)` is the trailing 60-observation mean squared return.
  `s(t)` is also the historical-control forecast, so a ratio of 1 reproduces the control.
- **Back to original units:** `forecast = predicted ratio * s(t)`.
  All metrics, early stopping, and alpha selection use original units.

Known costs, listed as risks in the proposal: per-window scaling hides the absolute volatility level from the model, and MSE on the ratio weights calm periods more heavily than MSE in original units.

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
| Historical control | Trailing 60-observation mean squared return | None |

- Alpha is chosen from {0.25, 0.50, 0.75} by validation MSE.
- Teacher predictions are cached on training inputs only.
- Both students share targets, batches, early stopping, tuning budget, and starting weights per seed.
- AdamW, at most 100 epochs, gradient-norm clipping at 1.0, early stopping on validation MSE.
- Optimizer settings are per model: `models.teacher` and `models.student` in `base.yaml` each hold `learning_rate`, `weight_decay`, and `dropout`.
  `training:` holds only the shared budget (batch size, epochs, patience, clipping).
- The teacher's settings come from the validation grid in `tuning.teacher` (12 configurations, seed 0).
  The winner (learning rate 3e-4, dropout 0.4) is copied into `models.teacher`, and notebook 03 asserts that the config matches the grid's top row before training seeds 0, 1, and 2.
  Both students get the same tuning budget as each other.
- Three final seed triplets give nine main neural fits.

### Metrics

1. Forecast MSE in original variance-proxy units, per seed and as mean with standard deviation.
2. High-volatility MSE on test targets above the training 90th percentile, with the qualifying-date count.
   Selecting dates on realized targets favors over-forecasts, so this metric is descriptive, not robust evidence.
3. Batch-1 inference latency, median and p95, with CPU and GPU reported separately.
4. Serialized model size of the deployment state dict with scaling metadata.

Training, caching, and search time are also recorded as compute accounting.
Errors are scored with MSE rather than MAE because daily squared returns are a noisy proxy.
With a noisy but unbiased proxy, MSE ranks forecasts the same way the true variance would (Patton, 2011).

Proxy noise adds the same large error to every model, so models are compared by their MSE gain over the historical control, not by raw error ratios.
The predeclared research target is a student that retains at least 90% of the teacher's gain, with lower latency and size:

```
retained = (MSE_control - MSE_student) / (MSE_control - MSE_teacher) >= 0.90
```

This is a research target, not an expected result.

### Notebooks

| Notebook | Purpose |
| --- | --- |
| `01_data_pipeline` | Snapshot, features, targets, splits, trailing scaling, processed arrays, manifest |
| `02_historical_control` | Trailing-60 baseline, high-volatility cutoff check, control metrics |
| `03_teacher_training` | Teacher pilot, hyperparameter grid, three-seed training |
| `03b_preliminary_table_metrics` | Params, size, and CPU latency (no student training needed), plus control and teacher validation metrics |
| `04_teacher_cache` | Frozen teacher predictions on training inputs |
| `05_student_training` | Alpha search, direct and distilled students |
| `06_evaluation` | MSE tables, gain over the control, high-volatility analysis, figures |
| `07_benchmark` | Latency and model size |
| `08_export_to_huggingface` | Push deployment checkpoints to the Hub |

### Running in Colab

1. Add a Colab Secret named `HF_TOKEN` with write access, needed for notebook 08.
   The repository is public, so cloning needs no token.
2. Open a notebook from GitHub in Colab and run the setup cell.
   It clones or pulls the repo, installs `model/requirements.txt`, and mounts Google Drive.
3. Outputs are written to the repo folder and committed so the team shares them, and copies go to `MyDrive/gru_vol_artifacts`.

## Phase 2: Hugging Face model repo

- Weights for the distilled student (deployment model) and teacher (comparison) in `safetensors`.
- A `config.json` with the input window, the trailing scaling window, the target transform, and the horizon.
  Trailing scaling needs no stored constants.
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
- **Grid seed overlaps a final seed:** the teacher grid ran with seed 0, which is also a final seed, so seed 0's validation result is slightly optimistic.
  Use a separate grid seed, or average each configuration over several seeds, for the student grid.
