"""Build docs/03_GRU_EURUSD_Volatility_Distillation.pdf, the protocol source of truth.

Edit the text below and rebuild:
    uv run --no-project --with reportlab --with matplotlib python docs/source/build_proposal.py
"""

from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "03_GRU_EURUSD_Volatility_Distillation.pdf"
HEADER = "03  GRU volatility distillation  |  ECE 662  |  FALL 2026"
FOOTER = "Project proposal  |  25 September 2026  |  revised 23 October 2026"

INK = colors.HexColor("#152333")
MUTED = colors.HexColor("#526171")
ACCENT = colors.HexColor("#234F72")
STRIPE = colors.HexColor("#F2F5F8")
RULE = colors.HexColor("#D9DFE5")

fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", fonts / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", fonts / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

body = ParagraphStyle("body", fontName="DejaVu", fontSize=10.2, leading=14.2, textColor=INK, spaceAfter=6, alignment=TA_LEFT, allowWidows=0)
title = ParagraphStyle("title", parent=body, fontName="DejaVu-Bold", fontSize=22, leading=27, spaceAfter=8)
subtitle = ParagraphStyle("subtitle", parent=body, fontSize=11, leading=15, textColor=MUTED, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=body, fontName="DejaVu-Bold", fontSize=22, leading=27, spaceBefore=8, spaceAfter=8, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="DejaVu-Bold", fontSize=12.2, leading=16, spaceBefore=8, spaceAfter=5, keepWithNext=1)
note = ParagraphStyle("note", parent=body, fontSize=8.8, leading=12, textColor=MUTED)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, firstLineIndent=-8, spaceAfter=4)
cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=12, spaceAfter=0)
head_cell = ParagraphStyle("head_cell", parent=cell, fontName="DejaVu-Bold", textColor=colors.white)


def P(text, style=body):
    return Paragraph(text, style)


def bullets(items):
    return [P(f"- {item}", bullet) for item in items]


def link(text, url):
    return f'<link href="{url}" color="#234F72">{text}</link>'


def table(header, rows, widths):
    data = [[P(h, head_cell) for h in header]] + [[P(c, cell) for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.45, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]
    style += [("BACKGROUND", (0, i), (-1, i), STRIPE) for i in range(2, len(data), 2)]
    t.setStyle(TableStyle(style))
    return t


story = [
    P("GRU volatility distillation", title),
    P("Learning a compact currency-risk forecaster from a deeper recurrent teacher", subtitle),
    P("This proposal follows the course guidance: one focused efficiency question, an explicit baseline and team "
      "contribution, a controlled test case, and four primary metrics. A supported negative result is a valid outcome. "
      "Architecture sizes, resource budgets, and decision thresholds below are proposed starting points, not measured results."),

    P("The idea and financial task", h2),
    P("We will forecast future exchange-rate variability for the euro relative to the U.S. dollar (EUR/USD). First train a "
      "deeper GRU teacher on this series. Then train a smaller GRU student using a combination of observed outcomes and "
      "teacher predictions. The financial task is volatility forecasting, not predicting a profitable trade."),

    P("Investigative question", h2),
    P("<b>Can a distilled one-layer GRU preserve a deeper GRU's volatility-forecasting quality at lower inference cost, "
      "including during high-volatility periods, and does distillation outperform direct training of the same student?</b>"),

    P("Baselines and model design", h2),
    table(
        ["Model", "Proposed architecture", "Purpose"],
        [
            ["Deep teacher", "3 GRU layers, 128 units per layer; single-output nonnegative regression head", "Main quality reference"],
            ["Direct student", "1 GRU layer, 32 units; identical single-output task", "Essential control for student size"],
            ["Distilled student", "Exactly the direct student architecture", "Isolates the value of teacher supervision"],
            ["Historical control", "Trailing 60-observation mean squared EUR/USD return",
             "Checks whether learned forecasting adds useful value"],
        ],
        [95, 250, 171],
    ),
    Spacer(1, 6),

    P("Our contribution", h2),
    P("We will implement and evaluate regression distillation for a single-series EUR/USD forecasting task, holding student "
      "capacity fixed. The main contribution is the quality-versus-inference-cost comparison and an analysis of whether the "
      "student retains the teacher's behavior during volatile periods. We will also document the extra training work "
      "required to obtain deployment savings."),

    P("Training objective", h2),
    P("Train the teacher and direct student with MSE on the ratio target defined below. Train the distilled student with "
      "<b>L = (1 - alpha) x MSE(student, observed ratio) + alpha x MSE(student, teacher ratio)</b>. Freeze the trained "
      "teacher. Choose alpha from a small grid such as 0.25, 0.50, and 0.75 by validation MSE in original units; alpha = 0 "
      "is the direct-training control. This is regression distillation, so no classification temperature is required."),
    P("Use the same nonnegative output transform, target transform, optimization budget, gradient-norm clipping, and "
      "validation criterion for both students. The teacher may be larger without being better; retain the simple control "
      "and a smaller GRU pilot to test whether its depth is worthwhile.", note),

    P("Dataset and forecasting tests", h1),
    P("Dataset and input construction", h2),
    P("Use FRED DEXUSEU, daily U.S. dollars per euro, for 2000-2025 [1]. Freeze a dated snapshot and keep only observed "
      "dates. Do not interpolate using future exchange rates. Compute the log return as <b>r(t) = log(P(t) / P(t-1))</b>."),
    P("Each example contains the previous 60 observations of EUR/USD returns, squared returns, and trailing 5- and "
      "20-observation volatility features. All rolling features end at the forecast origin. The model forecasts after the "
      "corresponding source observations are available."),
    P("Inputs are scaled per example. Each feature in a window is z-scored with that feature's mean and standard deviation "
      "over the trailing 60 observations ending at the forecast origin. Statistics fitted once on the whole training period "
      "would scale early training examples with observations from later in that period. Trailing statistics use only data "
      "observed by each origin, need no stored constants, and apply the same rule in training, evaluation, and deployment."),

    P("Target definition", h2),
    P("For horizon h, use <b>v(t,h) = (1/h) x sum of r(t+j)^2 for j = 1,...,h</b>. This is a daily-return-based variance "
      "proxy, not intraday realized volatility. The core uses h = 5; a later extension uses h = 20."),
    P("Models are trained on the ratio <b>y(t) = v(t,h) / s(t)</b>, where s(t) is the mean squared return over the "
      "trailing 60 observations ending at t. Because s(t) is the historical-control forecast, a ratio of 1 reproduces the "
      "control. The ratio is nonnegative and needs no fitted constant. Convert forecasts to original units as "
      "<b>v-hat(t,h) = y-hat(t) x s(t)</b>, and report forecast errors, early stopping, and alpha selection in these "
      "original units. Use the same target transform for teacher and students."),

    P("Chronological protocol", h2),
    *bullets([
        "<b>Train:</b> 2000-2017. <b>Validate:</b> 2018-2020. <b>Test:</b> 2021-2025. Remove examples whose forward target "
        "extends outside its assigned split.",
        "<b>Teacher:</b> fit only on training examples, with checkpoint selection using validation data. Never retrain "
        "using the test period.",
        "<b>Student supervision:</b> cache teacher predictions for training inputs only. Teacher predictions on validation "
        "or test inputs are for evaluation, never student training.",
        "<b>Matched comparison:</b> both students see the same observed targets, batches, early-stopping rule, and tuning "
        "budget. For a given seed, use the same starting student weights where practical.",
        "<b>Volatile-period subset:</b> fix the 90th-percentile future-variance cutoff from training targets in original "
        "units; report errors on test targets exceeding it. This is a retrospective evaluation subset, not information "
        "given to the model.",
    ]),

    P("Test cases", h2),
    table(
        ["Test", "Purpose"],
        [
            ["Teacher vs historical control", "Establish whether the teacher is a credible quality reference."],
            ["Direct vs distilled student", "Measure the effect of distillation at identical deployment size."],
            ["Teacher vs distilled student", "Measure forecast-quality loss or gain against latency and storage savings."],
            ["Volatile-period subset", "Check whether aggregate improvements hide tail-period failures."],
            ["Optional 20-observation horizon", "Retrain with the same frozen protocol to test horizon sensitivity."],
        ],
        [185, 331],
    ),

    P("Metrics and resource needs", h1),
    P("Four primary metrics", h2),
    table(
        ["Metric", "Definition and reporting"],
        [
            ["Forecast MSE", "Mean squared error against the variance proxy in its original units, for each horizon. With a "
             "noisy but unbiased proxy, MSE ranks forecasts as the true variance would [4]. Report each seed and the mean "
             "with standard deviation."],
            ["High-volatility MSE", "The same error on the predeclared high-volatility subset. Show the number of qualifying "
             "dates. Selecting dates on realized targets favors over-forecasts, so treat this as descriptive, not robust "
             "evidence."],
            ["Inference latency", "Batch-1 full-window forward pass including the feature-to-output model. Time teacher and "
             "students on the same device; report CPU and GPU separately."],
            ["Serialized model size", "Size of a deployment state dictionary with weights and required scaling metadata. "
             "Exclude optimizer states; keep dtype and serialization format identical."],
        ],
        [123, 393],
    ),
    Spacer(1, 6),

    P("Benchmark protocol", h2),
    P("Use identical held-out inputs, device, thread count, numerical precision, and software versions for each comparison. "
      "Warm up models, disable gradients, exclude file loading from inference timing, and synchronize CUDA before and after "
      "timed GPU calls. Report median and 95th-percentile latency over repeated trials. CPU and GPU results belong in "
      "separate comparisons; Colab device changes must not be mistaken for model improvements."),
    P("Record teacher training, cached-prediction generation, student training, and hyperparameter-search time as "
      "supporting compute accounting. Distillation adds up-front cost, so a faster student does not imply lower total cost "
      "for a small number of predictions. CPU latency is especially relevant to deploying the student without a GPU."),

    P("Minimum practical resources", h2),
    table(
        ["Activity", "Planning floor", "Conditions"],
        [
            ["Teacher training", "1 CUDA GPU with about 4 GB VRAM; 2 CPU cores; 8 GB system RAM",
             "For the proposed 3 x 128 GRU and 60-step sequences, start at batch 16-32. Confirm actual peak memory in a pilot."],
            ["Student training", "CPU with 2 cores and 8 GB RAM; GPU optional",
             "Cache teacher outputs first. Student fits can use a CPU; the GPU reduces iteration time."],
            ["Comfortable Colab setup", "1 GPU with 12-16 GB VRAM; standard system RAM",
             "Provides room for larger batches and repeated seeds; no multi-GPU setup is needed."],
            ["Inference and storage", "CPU with 2 cores and 2-4 GB RAM; allow 1-2 GB disk for the project",
             "One loaded teacher or student. Disk allowance includes snapshots, checkpoints, cached predictions, and logs."],
        ],
        [114, 187, 215],
    ),
    Spacer(1, 6),
    P("<b>Resource figures are planning estimates, not measured minimums.</b> CPU-only teacher training is technically "
      "possible on this daily dataset but may lengthen experiments. Pilot one full teacher epoch and one student epoch "
      "before allocating the complete run budget. Do not increase depth simply to use available GPU memory."),
    P("Colab hardware, memory availability, and session limits vary [3]. Save checkpoints, configuration files, and split "
      "manifests after each run. Do not make access to an A100, multiple GPUs, paid market data, or a continuously running "
      "service a project dependency.", note),

    P("Contribution and execution plan", h1),
    P("How our team will carry out the work", h2),
    *bullets([
        "<b>Data and targets:</b> clean the EUR/USD series, build past-only features and forward variance targets, freeze "
        "splits, and implement the historical-volatility control.",
        "<b>Teacher and distillation:</b> train and validate the teacher, generate cached training predictions, implement "
        "the combined loss, and pair direct and distilled student experiments.",
        "<b>Efficiency and robustness:</b> profile the deployed checkpoints, analyze volatile periods, track training "
        "overhead, and produce the final comparison figures.",
    ]),

    P("Milestones and experiment budget", h2),
    P("<b>By October 21:</b> have the data pipeline, historical control, one working teacher, and one direct student; "
      "obtain a first distilled-student result if the teacher is ready. Select one alpha from the grid by validation MSE. "
      "Then run three final seed triplets: teacher, direct student, and distilled student. This gives nine main neural "
      "fits, with a small separate alpha-search budget."),
    P("Use AdamW, an initial maximum of 100 epochs, dropout in the teacher, gradient-norm clipping at 1.0, and early "
      "stopping on validation MSE. Select the teacher's learning rate, dropout, and weight decay from a small validation "
      "grid with one seed, then train the chosen configuration with three seeds; give both students the same tuning "
      "budget. Save the best checkpoint and log exact settings. If the teacher is consistently worse than the smaller "
      "reference, reduce teacher depth or report that distillation lacks a useful teacher; do not search the test set for "
      "a favorable configuration."),

    P("Core contribution and optional extension", h2),
    P("<b>Core:</b> one currency pair (EUR/USD), one five-observation horizon, one teacher architecture, one student "
      "architecture, and the with/without-distillation control. <b>Extension:</b> add the 20-observation horizon OR "
      "introduce training weights for volatile examples. For weighted training, give the direct student the same weighting "
      "so any benefit is not falsely attributed to distillation. Complete the core report before either extension."),

    P("What would constitute a useful finding", h2),
    P("Plot overall and high-volatility errors against latency and model size. Proxy noise adds the same large error to "
      "every model, so compare gains over the historical control rather than raw error ratios. A possible predeclared "
      "target is a student that retains at least 90% of the teacher's forecast-MSE improvement over the historical control, "
      "with lower latency and size. Treat this as a research target, not a forecast of success. Even if direct training "
      "wins, explaining when teacher supervision hurts is a valid result."),

    P("Risks and final deliverables", h2),
    P("Daily squared returns are noisy volatility proxies, which is why errors are scored with MSE rather than MAE [4]; "
      "overlapping horizons limit independence. Use time-block methods if estimating uncertainty in error differences. A "
      "large teacher can overfit, and a student can inherit its mistakes. Per-window scaling removes the absolute volatility "
      "level from the inputs, and training on the ratio weights calm periods more heavily than the original-unit MSE does; "
      "if either appears to limit accuracy, compare against a train-fitted scaling variant on validation data only."),
    P("Deliver a reproducible Colab notebook, feature/target and split manifests, teacher and student checkpoints, "
      "cached-output generation code, all four metric tables, and a supported explanation of tradeoffs. Do not infer "
      "trading profitability from forecast-error improvements."),

    KeepTogether([
        P("Revision history", h2),
        *bullets([
            "<b>21 October 2026:</b> forecast and high-volatility errors are scored with MSE instead of MAE, and the research "
            "target is stated as retained gain over the historical control.",
            "<b>23 October 2026:</b> inputs are scaled per window with trailing 60-observation statistics; models predict the "
            "ratio of future variance to trailing 60-observation variance; the historical control is lengthened from 20 to "
            "60 observations; teacher hyperparameters come from a small validation grid; gradients are clipped.",
        ]),
    ]),

    KeepTogether([
    P("Sources and starting points", h2),
    P(f"[1] {link('FRED DEXUSEU', 'https://fred.stlouisfed.org/series/DEXUSEU')}. U.S. dollars per euro, daily.", note),
    P(f"[2] {link('Hinton et al. Distilling the Knowledge in a Neural Network', 'https://arxiv.org/abs/1503.02531')}. "
      "Foundational teacher-student concept; the proposed regression loss is a project design choice.", note),
    P(f"[3] {link('Google Colab FAQ', 'https://research.google.com/colaboratory/faq.html')}. Hardware and runtime "
      "availability.", note),
    P(f"[4] {link('Patton. Volatility Forecast Comparison Using Imperfect Volatility Proxies', 'https://doi.org/10.1016/j.jeconom.2010.03.034')}. "
      "Journal of Econometrics 160(1), 2011. Basis for scoring with MSE.", note),
    P("Course reference: Duke ECE 662, Fall 2026, 01A - Project Guidance.docx, supplied by the team. Architecture and "
      "resource choices remain subject to the initial pilot.", note),
    ]),
]


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(48, letter[1] - 28, HEADER)
    canvas.drawString(48, 26, FOOTER)
    canvas.drawRightString(letter[0] - 48, 26, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=54, rightMargin=42, topMargin=56, bottomMargin=56,
                        title="GRU volatility distillation", author="ECE 662 team 03")
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
print(f"wrote {OUT}")
