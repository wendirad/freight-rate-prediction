import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report.pdf"
RENDER = Path("/tmp/freight-report-pages")
OUT.parent.mkdir(exist_ok=True)
RENDER.mkdir(exist_ok=True)

NAVY = "#17324D"
TEAL = "#087E8B"
GOLD = "#E0A126"
INK = "#24313B"
MUTED = "#667681"
LIGHT = "#F2F6F7"
RED = "#B64B4B"
WHITE = "#FFFFFF"
plt.rcParams.update(
    {
        "font.family": "Georgia",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#B7C4C9",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def base(page, section):
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0.972), 1, 0.028, color=NAVY, transform=ax.transAxes))
    ax.text(
        0.07,
        0.947,
        section.upper(),
        fontsize=8.5,
        color=TEAL,
        weight="bold",
        va="top",
        transform=ax.transAxes,
    )
    ax.plot(
        [0.07, 0.93], [0.925, 0.925], color="#D8E1E4", lw=0.8, transform=ax.transAxes
    )
    ax.text(
        0.07,
        0.035,
        "FREIGHT RATE PREDICTION  •  TECHNICAL REPORT",
        fontsize=7.4,
        color=MUTED,
        transform=ax.transAxes,
    )
    ax.text(
        0.93,
        0.035,
        str(page),
        fontsize=8,
        color=MUTED,
        ha="right",
        transform=ax.transAxes,
    )
    return fig, ax


def title(ax, text, subtitle=None, size=20):
    ax.text(
        0.07,
        0.885,
        text,
        fontsize=size,
        color=NAVY,
        weight="bold",
        va="top",
        transform=ax.transAxes,
    )
    if subtitle:
        ax.text(
            0.07,
            0.84,
            subtitle,
            fontsize=10.5,
            color=MUTED,
            va="top",
            transform=ax.transAxes,
        )


def para(
    ax, x, y, text, width=92, size=10.2, color=INK, leading=0.024, weight="normal"
):
    lines = textwrap.wrap(text, width=width, break_long_words=False)
    ax.text(
        x,
        y,
        "\n".join(lines),
        fontsize=size,
        color=color,
        va="top",
        weight=weight,
        linespacing=1.42,
        transform=ax.transAxes,
    )
    return y - leading * len(lines)


def heading(ax, x, y, text):
    ax.text(
        x,
        y,
        text,
        fontsize=13,
        color=TEAL,
        weight="bold",
        va="top",
        transform=ax.transAxes,
    )
    return y - 0.038


def metric(ax, x, y, w, label, value, note="", color=TEAL):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            0.115,
            boxstyle="round,pad=0.012,rounding_size=.012",
            facecolor=WHITE,
            edgecolor="#D4E1E4",
            linewidth=0.9,
            transform=ax.transAxes,
        )
    )
    ax.add_patch(
        Rectangle(
            (x + 0.006, y + 0.115 - 0.014),
            w - 0.012,
            0.009,
            facecolor=color,
            edgecolor="none",
            transform=ax.transAxes,
        )
    )
    ax.text(
        x + 0.018,
        y + 0.086,
        label.upper(),
        fontsize=7.5,
        color=MUTED,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.018,
        y + 0.043,
        value,
        fontsize=21,
        color=color,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.018, y + 0.014, note, fontsize=7.3, color=MUTED, transform=ax.transAxes
    )


def callout(ax, x, y, w, h, text, color=TEAL):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=.01",
            facecolor=LIGHT,
            edgecolor="#D4E1E4",
            transform=ax.transAxes,
        )
    )
    ax.add_patch(Rectangle((x, y), 0.009, h, color=color, transform=ax.transAxes))
    para(
        ax,
        x + 0.025,
        y + h - 0.022,
        text,
        width=max(35, int(w * 108)),
        size=9.4,
        leading=0.022,
    )


def table(ax, bbox, cols, rows, widths=None, font=8.3):
    t = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="left",
        colLoc="left",
        bbox=bbox,
        colWidths=widths,
    )
    t.auto_set_font_size(False)
    t.set_fontsize(font)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#DCE4E7")
        cell.set_linewidth(0.7)
        cell.PAD = 0.025
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(WHITE if r % 2 else "#F7F9FA")
    return t


def save(pdf, fig, page):
    pdf.savefig(fig, bbox_inches=None)
    fig.savefig(RENDER / f"page-{page:02d}.png", dpi=150, facecolor="white")
    plt.close(fig)


with PdfPages(OUT) as pdf:
    # 1 Cover
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, color=NAVY, transform=ax.transAxes))
    ax.add_patch(Rectangle((0, 0.885), 1, 0.115, color=TEAL, transform=ax.transAxes))
    ax.text(
        0.09,
        0.79,
        "FREIGHT RATE\nPREDICTION",
        fontsize=38,
        color=WHITE,
        weight="bold",
        va="top",
        linespacing=0.95,
        transform=ax.transAxes,
    )
    ax.plot([0.095, 0.43], [0.585, 0.585], color=GOLD, lw=4, transform=ax.transAxes)
    ax.text(
        0.095,
        0.52,
        "TECHNICAL REPORT",
        fontsize=11,
        color=GOLD,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.095,
        0.45,
        "Experimental evidence, algorithmic methods,\nand final evaluation findings.",
        fontsize=13,
        color=WHITE,
        linespacing=1.55,
        transform=ax.transAxes,
    )
    ax.text(
        0.095,
        0.075,
        "Wendirad Demelash  •  wendirad.work@gmail.com  •  September 2026",
        fontsize=9,
        color="#B9CDD8",
        transform=ax.transAxes,
    )
    save(pdf, fig, 1)

    # 2 Executive summary
    fig, ax = base(2, "Executive Summary")
    title(ax, "Executive Summary")
    y = 0.79
    y = para(
        ax,
        0.07,
        y,
        "The objective was to predict posted freight rates from route, shipment, market, and calendar information while preserving the temporal conditions under which the model would be used. Because volumes and pricing visibly drift month over month, a random split was ruled out early, since it would let the model see fragments of the future during training and overstate how well it generalizes. The central requirement was therefore not model complexity but honest evaluation, with every preprocessing step fitted only inside each historical training window and judged by MAE, so a handful of extreme loads could not dominate the score.",
    )
    metric(
        ax,
        0.07,
        0.52,
        0.25,
        "Best CV MAE",
        "95.12",
        "20-fold expanding window",
        color=TEAL,
    )
    metric(
        ax,
        0.375,
        0.52,
        0.25,
        "October MAE",
        "168.48",
        "untouched temporal holdout",
        color=GOLD,
    )
    metric(
        ax,
        0.68,
        0.52,
        0.25,
        "October MAPE",
        "8.14%",
        "rate-scale interpretation",
        color=GOLD,
    )
    y = 0.48
    y = heading(ax, 0.07, y, "Main conclusions")
    bullets = [
        "CatBoost was the strongest model family under the same 20-fold chronological protocol, ahead of LightGBM, XGBoost, and linear regression.",
        "Equipment and calendar information supplied the most reliable engineered signal. Lane identity, market-index EMA, and quote z-score did not improve the selected combination.",
        "Direct rate prediction was retained. Rate-per-kilometre targets slightly improved average error in one comparison but produced substantially worse fold stability and tail risk.",
        "The October deterioration is the most important result: it indicates temporal distribution shift and sets a more realistic expectation than cross-validation alone.",
    ]
    for b in bullets:
        ax.text(
            0.085, y, "•", fontsize=15, color=GOLD, va="top", transform=ax.transAxes
        )
        y = para(ax, 0.112, y, b, width=88, size=9.1, leading=0.020) - 0.009
    callout(
        ax,
        0.07,
        0.065,
        0.86,
        0.09,
        "Decision: use CatBoost with the direct freight-rate target, 500 boosting iterations, depth 6, learning rate 0.05, MAE loss, and the selected 14-feature representation. Treat October performance, not training fit, as the deployment-risk signal.",
        TEAL,
    )
    save(pdf, fig, 2)

    # 3 Data and evidence
    fig, ax = base(3, "Evidence Base")
    title(
        ax,
        "Data Characteristics and Exploratory Evidence",
        "The signals that shaped the modelling strategy",
        20,
    )
    table(
        ax,
        [0.07, 0.61, 0.86, 0.19],
        ["Property", "Observed evidence", "Modelling implication"],
        [
            [
                "Volume / period",
                "48,000 loads; Jan–Oct 2025",
                "Use chronological rather than random validation",
            ],
            [
                "Target",
                "Median 2,031; mean 2,374; skew 1.90",
                "Report MAE; examine robust target alternatives",
            ],
            [
                "Distance",
                "Median 953 km; r = 0.909 with rate",
                "Primary continuous rate driver",
            ],
            [
                "Network",
                "64 locations; 4,014 observed lanes",
                "Lane identity is sparse/high-cardinality",
            ],
            [
                "Missingness",
                "300 weights; 374 market-index values",
                "Fit imputation only on historical training data",
            ],
        ],
        [0.21, 0.30, 0.49],
        7.3,
    )
    # charts from data
    df = pd.read_csv(ROOT / "data/raw/train-test.csv")
    df["date"] = pd.to_datetime(df["date"])
    c1 = fig.add_axes([0.09, 0.33, 0.38, 0.21])
    c2 = fig.add_axes([0.56, 0.33, 0.35, 0.21])
    sample = df.sample(min(12000, len(df)), random_state=42)
    c1.hexbin(sample.distance, sample.posted_rate, gridsize=38, cmap="Blues", mincnt=1)
    c1.set_xlabel("Distance (km)", fontsize=8)
    c1.set_ylabel("Posted rate", fontsize=8)
    c1.tick_params(labelsize=7)
    c1.set_title(
        "Distance dominates the marginal signal", fontsize=9, color=NAVY, weight="bold"
    )
    eq = df.groupby("equipment").posted_rate.mean().sort_values()
    c2.barh(eq.index, eq.values, color=[TEAL, GOLD, NAVY])
    c2.set_xlabel("Mean posted rate", fontsize=8)
    c2.tick_params(labelsize=7)
    c2.set_title(
        "Equipment separates price levels", fontsize=9, color=NAVY, weight="bold"
    )
    ax.text(
        0.09,
        0.27,
        "Figure 1. Distance-rate structure and equipment-level rate differences.",
        fontsize=7.6,
        color=MUTED,
        transform=ax.transAxes,
    )
    y = 0.25
    y = heading(ax, 0.07, y, "Interpretation")
    y = para(
        ax,
        0.07,
        y,
        "Distance supplies the strongest first-order relationship, but it does not explain the full price. Equipment changes the cost structure, calendar variables represent periodic demand, and coordinates model regional effects without a brittle lane label. Quote signal is weak marginally (correlation −0.04); market-index seasonality also does not guarantee incremental value once stronger variables are present.",
        width=101,
        size=9.1,
        leading=0.019,
    )
    callout(
        ax,
        0.07,
        0.06,
        0.86,
        0.055,
        "Data quality controls removed 12 lane-distance anomalies and 480 target outliers. The final fitted sample contained 47,508 valid training rows.",
        GOLD,
    )
    save(pdf, fig, 3)

    # 4 Evaluation method
    fig, ax = base(4, "Evaluation Design")
    title(
        ax,
        "Leakage-Safe Temporal Evaluation",
        "The experimental protocol mirrors prediction into the future",
    )
    y = 0.80
    y = para(
        ax,
        0.07,
        y,
        "Random train/test splitting would mix future market conditions into earlier training periods and produce optimistic results. The study instead used expanding-window backtesting on January–September data. Each fold trained on all preceding time blocks and validated on the next block. October remained untouched until the model and feature set were selected.",
    )
    # timeline
    tax = fig.add_axes([0.08, 0.52, 0.84, 0.16])
    tax.axis("off")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    for i, m in enumerate(months):
        color = "#A8DADC" if i < 9 else GOLD
        tax.add_patch(
            Rectangle(
                (i * 0.098, 0.36), 0.088, 0.28, facecolor=color, edgecolor="white"
            )
        )
        tax.text(
            i * 0.098 + 0.044,
            0.50,
            m,
            ha="center",
            va="center",
            fontsize=8,
            color=NAVY,
            weight="bold",
        )
    tax.annotate(
        "Expanding-window model selection",
        (0.40, 0.78),
        ha="center",
        fontsize=9,
        color=TEAL,
        weight="bold",
    )
    tax.annotate(
        "Reserved holdout",
        (0.925, 0.13),
        ha="center",
        fontsize=8,
        color="#8A6417",
        weight="bold",
    )
    ax.text(
        0.08,
        0.49,
        "Figure 2. Time ordering used for model selection and final confirmation.",
        fontsize=7.6,
        color=MUTED,
        transform=ax.transAxes,
    )
    y = 0.43
    y = heading(ax, 0.07, y, "Fold-local fitting rule")
    y = para(
        ax,
        0.07,
        y,
        "For every fold, cleaning and feature engineering were fitted only on that fold’s historical training rows. The resulting frozen transformations were then applied to its validation rows. No medians, outlier thresholds, category state, or engineered temporal statistics were learned from the validation period.",
    )
    y -= 0.015
    y = heading(ax, 0.07, y, "Metrics")
    y = para(
        ax,
        0.07,
        y,
        "MAE was chosen over RMSE as the selection metric because a handful of very long or short hauls should not dominate model selection the way squared error would let them, and it stays in the original rate units, so a fold's error reads directly as dollars. MAPE was kept as a secondary, scale-relative read on October. Fold standard deviation and worst-fold MAE sat alongside the mean because a model that looks good on average but falls apart in one fold is not worth deploying.",
        width=98,
        size=9.6,
        leading=0.021,
    )
    callout(
        ax,
        0.07,
        0.055,
        0.86,
        0.07,
        "Twenty folds were selected for their balance of average error, variability, and worst-period error. Fold-count results are not interchangeable: changing the number of folds also changes the validation windows.",
        TEAL,
    )
    save(pdf, fig, 4)

    # 5 Algorithm
    fig, ax = base(5, "Algorithmic Method")
    title(
        ax,
        "Representation and Learning Algorithm",
        "A compact tabular model built around stable signals",
        21,
    )
    y = 0.80
    y = heading(ax, 0.07, y, "Selected representation: 14 model inputs")
    table(
        ax,
        [0.07, 0.57, 0.86, 0.17],
        ["Feature group", "Variables", "Rationale"],
        [
            ["Shipment scale", "distance, weight", "Core physical cost drivers"],
            [
                "Geography",
                "pickup/delivery latitude and longitude",
                "Regional structure without lane memorization",
            ],
            [
                "Market context",
                "market_index, quote_signal",
                "Contemporaneous market state",
            ],
            [
                "Calendar",
                "month sine/cosine, day of week",
                "Periodic and weekly demand effects",
            ],
            [
                "Equipment",
                "Dry Van, Flatbed, Reefer indicators",
                "Mode-specific cost and capacity effects",
            ],
        ],
        [0.18, 0.43, 0.39],
        7.8,
    )
    y = 0.52
    y = heading(ax, 0.07, y, "Why CatBoost")
    y = para(
        ax,
        0.07,
        y,
        "Freight pricing contains nonlinear interactions: the cost effect of distance changes with equipment, region, season, and market conditions. Gradient-boosted decision trees represent these thresholds and interactions directly without requiring a rigid functional form. CatBoost produced the lowest chronological validation error of the tested families and remained stable across random seeds.",
    )
    y -= 0.012
    y = heading(ax, 0.07, y, "Learning objective and capacity")
    y = para(
        ax,
        0.07,
        y,
        "The selected learner minimizes MAE on the direct posted-rate target. It uses 500 boosting iterations, depth 6, learning rate 0.05, L2 regularization at the model default, and random seed 42. Diagnostic runs with early stopping showed best fold iterations ranging from 216 to 571, supporting a moderate fixed capacity near 500 rather than an aggressively deep or very long fit.",
    )
    # pipeline diagram
    yy = 0.105
    labels = [
        "Historical loads",
        "Fold-local cleaning",
        "14-feature representation",
        "CatBoost MAE model",
        "Rate prediction",
    ]
    xs = np.linspace(0.07, 0.79, 5)
    for i, (x, l) in enumerate(zip(xs, labels)):
        ax.add_patch(
            FancyBboxPatch(
                (x, yy),
                0.14,
                0.075,
                boxstyle="round,pad=.008",
                facecolor=LIGHT,
                edgecolor=TEAL,
                transform=ax.transAxes,
            )
        )
        ax.text(
            x + 0.07,
            yy + 0.037,
            l,
            fontsize=7.2,
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
        )
        if i < 4:
            ax.annotate(
                "",
                xy=(x + 0.175, yy + 0.037),
                xytext=(x + 0.142, yy + 0.037),
                arrowprops={"arrowstyle": "->", "color": GOLD},
                xycoords=ax.transAxes,
            )
    save(pdf, fig, 5)

    # 6 Model comparison
    fig, ax = base(6, "Model Selection")
    title(
        ax,
        "Model-Family Comparison",
        "Identical features and 20-fold chronological evaluation",
    )
    models = ["CatBoost", "LightGBM", "XGBoost", "Linear regression"]
    means = [95.12, 120.57, 122.60, 147.82]
    std = [14.09, 15.82, 14.63, 23.64]
    worst = [135.96, 160.38, 165.33, 213.73]
    bax = fig.add_axes([0.16, 0.55, 0.72, 0.22])
    ypos = np.arange(4)
    bax.barh(
        ypos,
        means,
        xerr=std,
        color=[TEAL, "#759AA5", "#9DB3BA", "#C5D0D4"],
        ecolor="#54656C",
        capsize=3,
    )
    bax.set_yticks(ypos, models)
    bax.invert_yaxis()
    bax.set_xlabel("Mean MAE (error bars: fold standard deviation)", fontsize=8)
    bax.tick_params(labelsize=8)
    bax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(means):
        bax.text(
            v + 2, i, f"{v:.2f}", va="center", fontsize=8, color=NAVY, weight="bold"
        )
    ax.text(
        0.12,
        0.51,
        "Figure 3. CatBoost achieved the lowest mean error under the common protocol.",
        fontsize=7.6,
        color=MUTED,
        transform=ax.transAxes,
    )
    table(
        ax,
        [0.07, 0.25, 0.86, 0.18],
        ["Model", "Mean MAE", "Fold std", "Worst fold"],
        [
            ["CatBoost", "95.12", "14.09", "135.96"],
            ["LightGBM", "120.57", "15.82", "160.38"],
            ["XGBoost", "122.60", "14.63", "165.33"],
            ["Linear regression", "147.82", "23.64", "213.73"],
        ],
        [0.36, 0.21, 0.21, 0.22],
        8.2,
    )
    callout(
        ax,
        0.07,
        0.09,
        0.86,
        0.10,
        "CatBoost reduced mean MAE by 21.1% relative to LightGBM and 35.7% relative to linear regression, and it also had the smallest worst-fold error, a margin consistent enough across folds that a lucky split could not explain it. Linear regression was kept as a sanity floor: its much larger error confirms the nonlinear interactions the tree models capture are earning their complexity.",
        TEAL,
    )
    save(pdf, fig, 6)

    # 7 ablations
    fig, ax = base(7, "Feature Evidence")
    title(
        ax,
        "Feature Ablation Findings",
        "Which engineered signals improved generalization",
    )
    labels = [
        "Minimal baseline",
        "+ Calendar",
        "+ Market EMA",
        "+ Quote z-score",
        "+ Equipment",
        "All 4 families",
        "Calendar + equipment",
    ]
    vals = [192.27, 189.68, 187.99, 170.99, 153.04, 126.08, 121.03]
    colors = ["#C5D0D4", "#A7BDC3", "#93AFB6", "#779FA7", "#4C929A", GOLD, TEAL]
    cax = fig.add_axes([0.27, 0.45, 0.62, 0.34])
    y = np.arange(len(labels))
    cax.barh(y, vals, color=colors)
    cax.set_yticks(y, labels)
    cax.invert_yaxis()
    cax.set_xlabel("Mean MAE (5-fold experiments)", fontsize=8)
    cax.tick_params(labelsize=8)
    cax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(vals):
        cax.text(v + 2, i, f"{v:.2f}", va="center", fontsize=8)
    ax.text(
        0.09,
        0.405,
        "Figure 4. Calendar plus equipment was the strongest compact engineered set.",
        fontsize=7.6,
        color=MUTED,
        transform=ax.transAxes,
    )
    y = 0.36
    y = heading(ax, 0.07, y, "Removal evidence")
    y = para(
        ax,
        0.07,
        y,
        "Starting from the four-family model (MAE 126.08), removing the quote z-score improved MAE to 122.79 and removing the market-index EMA improved it to 122.42. Removing both produced 121.03. By contrast, removing calendar worsened MAE to 131.10, and removing equipment sharply degraded it to 167.60. These paired results identify equipment as the dominant engineered contributor and calendar as a smaller but consistent complement.",
    )
    y -= 0.01
    y = heading(ax, 0.07, y, "Why lane identity was rejected")
    y = para(
        ax,
        0.07,
        y,
        "Native categorical lane handling did not improve the minimal baseline (192.70 versus 192.27) and harmed the combined representation (142.17 versus 126.08). With 4,014 observed lanes, the raw label is sparse, redundant with geographic inputs, and vulnerable to poor transfer to unseen or infrequent routes.",
    )
    save(pdf, fig, 7)

    # 8 robustness
    fig, ax = base(8, "Robustness")
    title(
        ax,
        "Target, Capacity, and Stability Tests",
        "Improvement was judged by temporal robustness, not training intensity",
    )
    table(
        ax,
        [0.07, 0.62, 0.86, 0.17],
        ["Target formulation", "Mean MAE", "Fold std", "Worst fold", "Decision"],
        [
            ["Direct posted rate", "118.89", "22.10", "147.15", "Selected"],
            ["Rate per kilometre", "117.52", "42.82", "196.15", "Rejected: unstable"],
            ["Log rate/km, detrended", "124.44", "37.68", "176.52", "Rejected"],
        ],
        [0.31, 0.15, 0.15, 0.17, 0.22],
        8.1,
    )
    y = 0.56
    y = heading(ax, 0.07, y, "Harder training did not reliably help")
    y = para(
        ax,
        0.07,
        y,
        "Depth, learning rate, regularization, and iteration-count comparisons showed a flat performance region rather than a monotonic benefit from more capacity. Examples ranged from MAE 118.25 at 700 iterations/depth 4 to 122.82 at 1,000 iterations/depth 7. Deeper, slower models increased computation without improving temporal generalization.",
    )
    # seed chart
    sax = fig.add_axes([0.12, 0.255, 0.76, 0.16])
    seeds = ["42", "123", "2026"]
    seedm = [121.03, 122.08, 121.77]
    sax.plot(seeds, seedm, marker="o", lw=2.5, color=TEAL)
    sax.fill_between(np.arange(3), [120.7] * 3, [122.4] * 3, color=TEAL, alpha=0.08)
    sax.set_ylim(119.8, 123.2)
    sax.set_ylabel("Mean MAE", fontsize=8)
    sax.set_xlabel("Random seed", fontsize=8)
    sax.tick_params(labelsize=8)
    sax.grid(axis="y", alpha=0.2)
    sax.set_title(
        "Selected feature set remained stable across seeds",
        fontsize=9,
        color=NAVY,
        weight="bold",
    )
    ax.text(
        0.12,
        0.215,
        "Figure 5. Average MAE 121.63; standard deviation across seed means 0.44.",
        fontsize=7.6,
        color=MUTED,
        transform=ax.transAxes,
    )
    callout(
        ax,
        0.07,
        0.085,
        0.86,
        0.075,
        "The direct target was retained despite a 1.37-point average advantage for rate/km because its worst fold was 49 points better and its fold variability was roughly half as large.",
        GOLD,
    )
    save(pdf, fig, 8)

    # 9 final validation
    fig, ax = base(9, "Final Evaluation")
    title(
        ax,
        "October Holdout and External Validation",
        "The final test exposes a meaningful temporal gap",
        21,
    )
    metric(
        ax,
        0.07,
        0.68,
        0.25,
        "Cross-val MAE",
        "95.12",
        "Jan-Sep model selection",
        color=TEAL,
    )
    metric(
        ax,
        0.375,
        0.68,
        0.25,
        "October MAE",
        "168.48",
        "untouched holdout",
        color=GOLD,
    )
    metric(
        ax,
        0.68,
        0.68,
        0.25,
        "October MAPE",
        "8.14%",
        "relative absolute error",
        color=GOLD,
    )
    y = 0.62
    y = para(
        ax,
        0.07,
        y,
        "After model and feature selection, October was evaluated exactly once, on purpose: touching it earlier or refitting afterward to close the gap would have turned a genuine holdout into another tuning fold and defeated the point of reserving it. Its MAE was 73.36 points higher than the 20-fold cross-validation mean. That gap is too large to dismiss as random fold noise and is the clearest evidence of temporal distribution shift in the data. The in-sample MAE of 54.79 measures fit to historical data and must not be interpreted as expected future performance.",
    )
    december = pd.read_csv(ROOT / "data/raw/december-chart-inputs.csv")
    december["date"] = pd.to_datetime(december["date"])
    iax = fig.add_axes([0.15, 0.25, 0.70, 0.20])
    iax.plot(
        december["date"],
        december["predicted_rate"],
        color=TEAL,
        lw=2,
        marker="o",
        ms=2.5,
    )
    iax.fill_between(
        december["date"],
        december["predicted_rate"],
        december["predicted_rate"].min() - 2,
        color=TEAL,
        alpha=0.08,
    )
    iax.set_title(
        "Fixed December 2025 predicted load rate", fontsize=9, color=NAVY, weight="bold"
    )
    iax.set_ylabel("Predicted rate", fontsize=8)
    iax.tick_params(axis="both", labelsize=7)
    iax.tick_params(axis="x", rotation=28)
    iax.grid(axis="y", alpha=0.2)
    iax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.15,
        0.205,
        "Lexington to Fort Wayne  •  distance 360 km  •  Dry Van  •  weight 32,000",
        fontsize=7.2,
        color=MUTED,
        transform=ax.transAxes,
    )
    ax.text(
        0.15,
        0.18,
        "Figure 6. Fixed December visualization generated from the final validation predictions.",
        fontsize=7.6,
        color=MUTED,
        transform=ax.transAxes,
    )
    callout(
        ax,
        0.07,
        0.075,
        0.86,
        0.075,
        "The 12,000-row validation prediction file passed the official scorer. December labels are not available locally, so the chart verifies output structure and temporal pattern, not predictive accuracy.",
        TEAL,
    )
    save(pdf, fig, 9)

    # 10 conclusions
    fig, ax = base(10, "Conclusion")
    title(
        ax,
        "Conclusions and Limitations",
        "What is supported by the evidence, and what remains uncertain",
    )
    y = 0.80
    y = heading(ax, 0.07, y, "Supported conclusions")
    for n, t in enumerate(
        [
            "A leakage-safe chronological protocol materially changes the credibility of the result; preprocessing must remain fold-local.",
            "CatBoost is the strongest tested model family for this tabular freight problem under the common 20-fold comparison.",
            "The compact calendar-plus-equipment augmentation is better supported than adding every available engineered family.",
            "Direct rate prediction offers a better stability profile than rate-per-kilometre transformations, despite a small average-error trade-off in one experiment.",
            "October error indicates that future-period risk is substantially higher than the headline cross-validation mean.",
        ],
        1,
    ):
        ax.text(
            0.075,
            y,
            f"{n}",
            fontsize=10,
            color=WHITE,
            weight="bold",
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox={"boxstyle": "circle,pad=.35", "facecolor": TEAL, "edgecolor": "none"},
        )
        y = para(ax, 0.11, y + 0.008, t, width=82, size=9.6, leading=0.022) - 0.018
    y -= 0.005
    y = heading(ax, 0.07, y, "Limitations")
    y = para(
        ax,
        0.07,
        y,
        "The observations cover one synthetic-looking annual period, and the near-deterministic relationship between supplied distance and coordinate-derived distance limits geographic realism. Sparse lanes and locations may behave differently in a live network. The October shift shows that one holdout month cannot characterize all future regimes. December ground truth is hidden, preventing local accuracy measurement on the final submission period.",
    )
    y -= 0.012
    y = heading(ax, 0.07, y, "Recommended interpretation")
    y = para(
        ax,
        0.07,
        y,
        "The selected model is a defensible assessment solution and a strong experimental baseline, not a guarantee of production accuracy. The next research priority is not indiscriminately harder training; it is monitoring temporal drift, evaluating additional labelled periods, and introducing new operational signals only when they improve chronological holdouts.",
    )
    callout(
        ax,
        0.07,
        0.075,
        0.86,
        0.085,
        "Final finding: model quality came primarily from honest validation, strong base variables, and disciplined feature removal. More features and more boosting were not consistently better.",
        GOLD,
    )
    save(pdf, fig, 10)

print(f"{OUT}\n10 pages\nrendered to {RENDER}")
