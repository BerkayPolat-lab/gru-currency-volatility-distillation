# CLAUDE.md

EUR/USD volatility forecasting with GRU distillation.
`README.md` has the full plan; `docs/03_GRU_EURUSD_Volatility_Distillation.pdf` is the source of truth for the research protocol.
Change the protocol in `docs/source/build_proposal.py` and rebuild the PDF; add each change to its revision history.

## Workflow

- Phases run in order: model training and testing in Google Colab, then Hugging Face (model repo and FastAPI Space), then the Next.js app on Vercel.
- Do not build the inference Space or web app until the model phase is complete.
- Notebooks must run in Colab: the repo is public, so clone without a token, and never hard-code tokens.
- Manifests, processed data, results, and checkpoints are committed to git so the team shares them; Google Drive holds backup copies.
- Shared logic lives in `model/src/gru_vol` and experiment settings in `model/configs/base.yaml`, not inline in notebooks.

## Protocol rules

- No look-ahead: features end at the forecast origin, and no interpolation is done with future prices.
- Scaling uses only the trailing 60 observations at each origin: inputs are z-scored per window, and the target is the ratio of future variance to trailing 60-observation variance.
  Never fit scaling statistics on a whole split.
- The historical control is the trailing 60-observation mean squared return (a ratio of 1).
- Never use the test period (2021-2025) for training, checkpoint selection, alpha selection, or tuning.
- Teacher predictions used for student training are cached on training inputs only.
- Direct and distilled students must be matched: same data, batches, early stopping, budget, and initial weights per seed.
- Score forecasts with MSE (not MAE) in original variance-proxy units (predicted ratio times trailing variance); this also applies to early stopping and alpha selection.
- Keep CPU and GPU latency as separate comparisons.

## Deployment

- The web app shows the distilled student forecast, with the teacher and historical control for comparison.
- "Real-time" means a daily forecast plus a live spot rate shown for context; spot never changes the forecast.
- Live rates come from FXAPI (free plan: 300 requests per month), always through a server-side cache.
