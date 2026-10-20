# Inference service (Hugging Face Space)

FastAPI app deployed as a Docker Space on Hugging Face.

Not started.
It is built after the model phase finishes and the checkpoints are on the Hugging Face Hub.

Planned scope:

- Load the distilled student and the teacher from the Hugging Face model repo at startup.
- `POST /predict`: accept the latest EUR/USD daily closes, rebuild the 60-step feature window with the saved scaling metadata, and return the 5-day variance forecast from each model plus the trailing-20 historical control.
- `GET /health`: liveness check used by the web app to wake the Space.
