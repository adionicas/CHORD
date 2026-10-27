"""
plots.py
Plotly figures for all harmonization evaluation metrics.
All figures have white background and are returned as plotly Figure objects.
"""

import warnings

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.naming import natural_sorted

BLUE   = "#2A6EBB"
ORANGE = "#EB6400"
GREEN  = "#2A9D8F"
GREY   = "#6C757D"
SEP_COLOR = "#2A6EBB"   # batch and group boundaries on the data maps (dotted, never black)

BEFORE_COLOR = ORANGE
AFTER_EBT    = BLUE
AFTER_EBF    = GREEN

# A legend under the plot is anchored to the bottom of the whole figure (not to the
# plot area), with its own space reserved in the bottom margin, so it never runs into
# the x-axis tick labels or the axis title, whatever their size.
BOTTOM_LEGEND = dict(orientation="h", yref="container", yanchor="bottom", y=0.01,
                     xanchor="center", x=0.5)

WHITE_BG = dict(
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    font          = dict(color="black", family="Arial"),
)


# ---------------------------------------------------------------------------
# Site mean z-score deviation
# ---------------------------------------------------------------------------
def plot_site_deviation(
    dev_before: pd.DataFrame,
    dev_after_ebt: pd.DataFrame,
    dev_after_ebf: pd.DataFrame | None = None,
    site_n: dict | None = None,
) -> go.Figure:
    """
    Box + jitter of site mean z-scores per site.
    One panel per harmonization condition.
    """
    datasets = [(dev_before, "Before harmonization", BEFORE_COLOR)]
    if dev_after_ebt is not None and len(dev_after_ebt) > 0:
        datasets.append((dev_after_ebt, "After (EB=TRUE)", AFTER_EBT))
    if dev_after_ebf is not None and len(dev_after_ebf) > 0:
        datasets.append((dev_after_ebf, "After (EB=FALSE)", AFTER_EBF))

    n_panels = len(datasets)
    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=[d[1] for d in datasets],
                        shared_yaxes=True)

    sites    = natural_sorted(dev_before["site"].unique())
    site_pos = {s: i for i, s in enumerate(sites)}
    n_sites  = len(sites)
    # Use numbered "Batch N" labels plus a key when there are many batches OR when
    # any batch name is long enough to crowd the axis; short-named, few-batch
    # datasets keep their full names on the axis.
    _max_name_len = max((len(str(s)) for s in sites), default=0)
    use_numbers = n_sites > 12 or _max_name_len > 16
    tick_size   = max(7, 11 - max(0, n_sites - 10) // 3)

    def _full_label(s):
        return f"{s} (n={site_n[s]})" if site_n and s in site_n else str(s)

    for col_idx, (df_dev, label, color) in enumerate(datasets, start=1):
        for s in sites:
            sub = df_dev[df_dev["site"] == s]["site_mean_z"].values
            if len(sub) == 0:
                continue
            xpos = site_pos[s]
            fig.add_trace(go.Box(
                x=[xpos] * len(sub), y=sub,
                name=f"Batch {xpos + 1}", showlegend=False,
                marker_color=color, line_color="black",
                boxpoints="all", jitter=0.4, pointpos=0,
                marker=dict(size=5, opacity=0.55, color=color),
                fillcolor="rgba(255,255,255,0.5)", width=0.5,
            ), row=1, col=col_idx)

        fig.add_hline(y=0, line_dash="dash", line_color="black",
                      opacity=0.4, row=1, col=col_idx)
        fig.update_xaxes(
            tickmode="array", tickvals=list(range(n_sites)),
            ticktext=([f"Batch {i + 1}" for i in range(n_sites)] if use_numbers
                      else [_full_label(s) for s in sites]),
            tickangle=(-90 if use_numbers else 45),
            tickfont=dict(size=tick_size), row=1, col=col_idx,
        )

    n_key_lines = 0
    if use_numbers:
        parts = [f"Batch {i + 1} = {s}" + (f" (n={site_n[s]})" if site_n and s in site_n else "")
                 for i, s in enumerate(sites)]
        key_lines = ["      ".join(parts[k:k + 2]) for k in range(0, len(parts), 2)]
        n_key_lines = len(key_lines)
        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=-0.34, xanchor="left", yanchor="top",
            text="<b>Batch key</b><br>" + "<br>".join(key_lines),
            showarrow=False, align="left", font=dict(size=8, color="black"),
        )

    fig.update_yaxes(title_text="Batch mean z-score", row=1, col=1)
    fig.update_layout(
        title="Batch mean deviation from grand mean (z-scored per feature)",
        height=(430 if use_numbers else 480) + (13 * n_key_lines if use_numbers else 0), **WHITE_BG,
        showlegend=False,
        margin=dict(b=(150 + 13 * n_key_lines) if use_numbers else 90),
    )
    return fig


# ---------------------------------------------------------------------------
# Spearman r — violin
# ---------------------------------------------------------------------------
def plot_spearman(spm_df: pd.DataFrame) -> go.Figure:
    """
    Violin + box of Spearman r values per feature, one violin per condition.
    """
    if "harmonization" not in spm_df.columns:
        spm_df = spm_df.copy()
        spm_df["harmonization"] = "EB=TRUE"

    conditions = spm_df["harmonization"].unique()
    colors = [AFTER_EBT, AFTER_EBF, GREEN]

    fig = go.Figure()
    for i, cond in enumerate(conditions):
        sub = spm_df[spm_df["harmonization"] == cond]["spearman_r"].dropna()
        color = colors[i % len(colors)]
        med   = sub.median()
        fig.add_trace(go.Violin(
            y=sub, name=cond, box_visible=True,
            meanline_visible=False,
            points="all", jitter=0.3, pointpos=0,
            marker=dict(size=5, opacity=0.55, color=color),
            fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if color.startswith("rgb") else color,
            line_color=color,
            opacity=0.85,
            spanmode="hard",
        ))
        fig.add_annotation(
            x=cond, y=med,
            text=f"Median: {med:.3f}",
            showarrow=False, yshift=14,
            font=dict(size=11, color="black", family="Arial"),
            bgcolor="white", bordercolor="grey", borderwidth=1,
        )

    fig.add_hline(y=0.80, line_dash="dot", line_color=GREY, opacity=0.55, line_width=1.2,
                  annotation_text="0.80", annotation_position="right",
                  annotation_font_size=10, annotation_font_color=GREY)
    fig.add_hline(y=0.90, line_dash="dot", line_color=GREY, opacity=0.55, line_width=1.2,
                  annotation_text="0.90", annotation_position="right",
                  annotation_font_size=10, annotation_font_color=GREY)
    fig.add_hline(y=0.95, line_dash="dot", line_color=GREY, opacity=0.55, line_width=1.2,
                  annotation_text="0.95", annotation_position="right",
                  annotation_font_size=10, annotation_font_color=GREY)

    fig.update_layout(
        title="Within-batch consistency: Spearman r (raw vs harmonized)",
        yaxis_title="Spearman r",
        yaxis_range=[-0.1, 1.05],
        height=480, **WHITE_BG,
    )
    return fig


# ---------------------------------------------------------------------------
# ICC3 — category count bar chart
# ---------------------------------------------------------------------------
def plot_icc(icc_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: number of features in each Koo & Li (2016) ICC3 category.
    One group per harmonization condition. Clear and direct.
    """
    if "harmonization" not in icc_df.columns:
        icc_df = icc_df.copy()
        icc_df["harmonization"] = "EB=TRUE"

    conditions = list(icc_df["harmonization"].unique())
    colors_map = {"EB=TRUE": AFTER_EBT, "EB=FALSE": AFTER_EBF}

    categories  = ["Poor\n(< 0.50)", "Moderate\n(0.50–0.75)",
                   "Good\n(0.75–0.90)", "Excellent\n(≥ 0.90)"]
    cat_colors  = ["rgba(210,50,50,0.75)", "rgba(220,130,30,0.75)",
                   "rgba(60,160,60,0.75)",  "rgba(20,100,20,0.75)"]

    fig = go.Figure()
    for cond in conditions:
        sub = icc_df[icc_df["harmonization"] == cond]["icc3"].dropna()
        counts = [
            int((sub < 0.50).sum()),
            int(((sub >= 0.50) & (sub < 0.75)).sum()),
            int(((sub >= 0.75) & (sub < 0.90)).sum()),
            int((sub >= 0.90).sum()),
        ]
        color = colors_map.get(cond, BLUE)
        fig.add_trace(go.Bar(
            name=cond,
            x=categories,
            y=counts,
            marker_color=color,
            opacity=0.80,
            text=counts,
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title="Within-batch consistency: ICC3 feature counts by category — Koo & Li (2016)",
        xaxis_title="ICC3 category",
        yaxis_title="Number of features",
        height=420, **WHITE_BG,
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5),
        margin=dict(b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Age correlations — scatter before vs after (one point per feature)
# ---------------------------------------------------------------------------
def plot_age_correlations(age_df: pd.DataFrame) -> go.Figure:
    """
    Scatter: Pearson r before harmonization (x) vs after (y), one point per feature.
    Points above diagonal = correlation increased after harmonization.
    Color encodes FDR significance after harmonization.
    """
    colors_map = {"After (EB=TRUE)": AFTER_EBT, "After (EB=FALSE)": AFTER_EBF}
    before_df  = age_df[age_df["harmonization"] == "Before harmonization"]
    after_conds = [c for c in age_df["harmonization"].unique() if c != "Before harmonization"]

    if not after_conds or len(before_df) == 0:
        return go.Figure()

    n_panels = len(after_conds)
    fig = make_subplots(rows=1, cols=n_panels,
                        subplot_titles=after_conds,
                        shared_xaxes=True, shared_yaxes=True)

    for col_i, cond in enumerate(after_conds, start=1):
        after_df = age_df[age_df["harmonization"] == cond]
        merged   = before_df[["feature", "r", "sig_fdr"]].merge(
            after_df[["feature", "r", "sig_fdr"]], on="feature", suffixes=("_bef", "_aft")
        )
        if len(merged) == 0:
            continue

        # single consistent colour for "significant after only" so the legend
        # (built from the first panel only) matches the markers in every panel
        color = AFTER_EBT
        show  = col_i == 1

        # Four mutually exclusive categories — each with a distinct color AND shape
        ns       = merged[~merged["sig_fdr_bef"] & ~merged["sig_fdr_aft"]]   # neither
        new_sig  = merged[~merged["sig_fdr_bef"] &  merged["sig_fdr_aft"]]   # newly sig after
        lost_sig = merged[ merged["sig_fdr_bef"] & ~merged["sig_fdr_aft"]]   # lost after
        both_sig = merged[ merged["sig_fdr_bef"] &  merged["sig_fdr_aft"]]   # sig in both

        # color:  grey | condition color | orange | dark purple
        # symbol: circle | circle | diamond | square
        # — every combination is unique on both dimensions
        PURPLE = "#6A0DAD"
        for sub, label, mc, sym, sz, op, border in [
            (ns,       "Not significant (neither)",       GREY,   "circle",  7,  0.40, "rgba(150,150,150,0.3)"),
            (new_sig,  "FDR significant after only",      color,  "circle",  9,  0.85, "white"),
            (lost_sig, "FDR significant before only",     ORANGE, "diamond", 9,  0.85, "white"),
            (both_sig, "FDR significant before and after",PURPLE, "square",  9,  0.85, "white"),
        ]:
            if len(sub) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=sub["r_bef"], y=sub["r_aft"],
                mode="markers", name=label,
                legendgroup=label, showlegend=show,
                marker=dict(color=mc, symbol=sym, size=sz, opacity=op,
                            line=dict(color=border, width=0.8)),
                text=sub["feature"],
                hovertemplate="<b>%{text}</b><br>Before: %{x:.3f}<br>After: %{y:.3f}<extra></extra>",
            ), row=1, col=col_i)

        # Diagonal (no change) and zero lines
        r_vals = pd.concat([merged["r_bef"], merged["r_aft"]]).dropna()
        lo, hi = r_vals.min() - 0.05, r_vals.max() + 0.05
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            name="No change", legendgroup="diag",
            showlegend=show,
            line=dict(color=GREY, dash="dash", width=1.2),
        ), row=1, col=col_i)
        fig.add_vline(x=0, line_dash="dot", line_color=GREY, opacity=0.35, row=1, col=col_i)
        fig.add_hline(y=0, line_dash="dot", line_color=GREY, opacity=0.35, row=1, col=col_i)

    fig.update_xaxes(title_text="Pearson r (before harmonization)")
    fig.update_yaxes(title_text="Pearson r (after harmonization)", row=1, col=1)
    fig.update_layout(
        title="Age associations: Pearson r before vs after harmonization (each point = one feature)",
        height=520, **WHITE_BG,
        legend=BOTTOM_LEGEND,
        margin=dict(b=130),
    )
    return fig


# ---------------------------------------------------------------------------
# ANCOVA Cohen's f — scatter before vs after
# ---------------------------------------------------------------------------
def plot_cohens_f(anc_before: pd.DataFrame, anc_ebt: pd.DataFrame,
                  anc_ebf: pd.DataFrame | None = None) -> go.Figure:
    """
    Scatter: Cohen's f before (x) vs after (y) harmonization.
    Each point = one feature. Significance of the site effect (uncorrected
    ANCOVA p < 0.05) is encoded on two axes:
      colour  = after-harmonization status (red = p < 0.05, green = p >= 0.05)
      fill    = before-harmonization status (filled = p < 0.05, open = p >= 0.05)
    The legend is drawn as a 2 x 2 grid (before vs after).
    """
    RED   = "#E74C3C"
    GREEN = "#27AE60"
    P     = 0.05

    def _merge(df_before, df_after):
        cols = ["feature", "cohens_f", "p_value"]
        return df_before[cols].merge(
            df_after[cols], on="feature", suffixes=("_before", "_after")
        )

    def _add_condition(fig, merged, base_symbol, eb_label):
        sig_b = merged["p_value_before"] < P
        sig_a = merged["p_value_after"]  < P
        # (mask, after-colour, filled?)  -> 2 x 2 = 4 groups
        groups = [
            (sig_b  & sig_a,  RED,   True),    # sig before, sig after   (persists)
            (sig_b  & ~sig_a, GREEN, True),    # sig before, ns after     (removed)
            (~sig_b & sig_a,  RED,   False),   # ns before, sig after     (introduced)
            (~sig_b & ~sig_a, GREEN, False),   # ns before, ns after      (neither)
        ]
        for mask, color, filled in groups:
            sub = merged[mask]
            if len(sub) == 0:
                continue
            symbol = base_symbol if filled else base_symbol + "-open"
            fig.add_trace(go.Scatter(
                x=sub["cohens_f_before"], y=sub["cohens_f_after"],
                mode="markers", showlegend=False,
                marker=dict(color=color, symbol=symbol, size=9, opacity=0.85,
                            line=dict(color="rgba(0,0,0,0.35)", width=0.7)),
                text=sub["feature"],
                hovertemplate=(
                    f"<b>%{{text}}</b> ({eb_label})<br>"
                    f"Before: f = %{{x:.3f}}<br>After: f = %{{y:.3f}}<extra></extra>"
                ),
            ))

    fig = go.Figure()
    conditions = []
    if anc_ebt is not None and len(anc_ebt) > 0:
        conditions.append((anc_ebt, "EB=TRUE"))
    if anc_ebf is not None and len(anc_ebf) > 0:
        conditions.append((anc_ebf, "EB=FALSE"))

    all_vals = anc_before["cohens_f"].to_numpy(dtype=float)
    for i, (anc_after, eb_label) in enumerate(conditions):
        merged   = _merge(anc_before, anc_after)
        all_vals = np.concatenate([all_vals,
                                   merged[["cohens_f_before", "cohens_f_after"]].values.ravel()])
        base_symbol = "diamond" if (len(conditions) > 1 and i == 1) else "circle"
        _add_condition(fig, merged, base_symbol, eb_label)

    has_ebf = len(conditions) > 1
    max_val = max(float(np.nanmax(all_vals)) * 1.05, 0.5)

    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(color=GREY, dash="dash", width=1.2),
        showlegend=False, hoverinfo="skip",
    ))

    for val, lbl in [(0.10, "small"), (0.25, "medium"), (0.40, "large")]:
        for add_line in [fig.add_vline, fig.add_hline]:
            add_line(val, line_dash="dot", line_color=GREY, opacity=0.45, line_width=1.1,
                     annotation_text=lbl,
                     annotation_position="top right" if add_line == fig.add_vline else "right",
                     annotation_font_size=9, annotation_font_color=GREY)

    _add_cohens_f_grid_legend(fig, RED, GREEN, has_ebf)

    fig.update_layout(
        title="Batch effect size: Cohen's f before vs after harmonization",
        xaxis_title="Cohen's f (before harmonization)",
        yaxis_title="Cohen's f (after harmonization)",
        height=560, **WHITE_BG,
        showlegend=False,
        xaxis_range=[0, max_val],
        yaxis_range=[0, max_val],
        margin=dict(r=70, b=70),
    )
    return fig


def _add_cohens_f_grid_legend(fig, red, green, has_ebf=False):
    """
    Draw the 2 x 2 contingency legend in the upper-right corner of the plot,
    over a light background box (that region is usually empty because
    harmonization moves points toward the lower-left). Each cell holds the
    actual marker a feature of that type carries, so a point can be read
    directly:
      rows    = before harmonization (p < 0.05 / p >= 0.05)  -> fill
      columns = after  harmonization (p < 0.05 / p >= 0.05)  -> colour
    """
    def ann(x, y, text, size=10, color="#333", xanchor="center"):
        fig.add_annotation(x=x, y=y, xref="paper", yref="paper", text=text,
                           showarrow=False, font=dict(size=size, color=color,
                                                      family="Arial"),
                           xanchor=xanchor, yanchor="middle")

    # Compact box hugging the content, with larger symbols and text so the grid
    # reads clearly rather than sitting tiny inside an oversized square.
    y_bottom = 0.66 if has_ebf else 0.74
    fig.add_shape(type="rect", xref="paper", yref="paper",
                  x0=0.60, x1=0.992, y0=y_bottom, y1=0.992,
                  fillcolor="rgba(255,255,255,0.92)", line=dict(color="#cccccc", width=1),
                  layer="above")

    cx1, cx2 = 0.86, 0.955          # column x-positions (after p<0.05 / p>=0.05)
    r1, r2   = (0.885, 0.805) if has_ebf else (0.87, 0.79)   # row y-positions

    # column header (after harmonization)
    ann((cx1 + cx2) / 2, 0.955, "<b>After harmonization</b>", size=10, color="#111")
    ann(cx1, 0.918, "p &lt; 0.05",   size=9, color="#444")
    ann(cx2, 0.918, "p &#8805; 0.05", size=9, color="#444")

    # row header (before harmonization)
    ann(0.615, (r1 + r2) / 2, "<b>Before<br>harmonization</b>", size=10,
        color="#111", xanchor="left")
    ann(0.80, r1, "p &lt; 0.05",   size=9, color="#444", xanchor="right")
    ann(0.80, r2, "p &#8805; 0.05", size=9, color="#444", xanchor="right")

    # cells: exactly the marker a feature of that type carries
    ann(cx1, r1, "&#9679;", size=24, color=red)     # filled red   = sig before & after
    ann(cx2, r1, "&#9679;", size=24, color=green)   # filled green = sig before, ns after
    ann(cx1, r2, "&#9675;", size=24, color=red)     # open red     = ns before, sig after
    ann(cx2, r2, "&#9675;", size=24, color=green)   # open green   = ns before & after

    if has_ebf:
        y_s = 0.71
        ann(0.635, y_s, "&#9679;", size=16, color="#333")
        ann(0.665, y_s, "EB=TRUE",  size=9, color="#444", xanchor="left")
        ann(0.83, y_s, "&#9670;", size=16, color="#333")
        ann(0.86, y_s, "EB=FALSE", size=9, color="#444", xanchor="left")


# ---------------------------------------------------------------------------
# ICC by site — box + jitter per site, one panel per harmonization mode
# ---------------------------------------------------------------------------
def _icc_band_shapes():
    """Reusable Koo & Li band shapes for by-site plots."""
    return [
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
             y0=0, y1=0.50, fillcolor="rgba(210,50,50,0.06)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
             y0=0.50, y1=0.75, fillcolor="rgba(220,130,30,0.06)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
             y0=0.75, y1=0.90, fillcolor="rgba(60,160,60,0.06)", line_width=0, layer="below"),
        dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
             y0=0.90, y1=1.08, fillcolor="rgba(20,100,20,0.06)", line_width=0, layer="below"),
    ]


def plot_icc_by_site(
    icc_by_site_ebt: pd.DataFrame,
    icc_by_site_ebf: pd.DataFrame | None = None,
    site_n: dict | None = None,
) -> go.Figure:
    """
    site_n: dict mapping site name -> participant count, used for x-axis labels.
    Points are colored by ICC3 threshold category (Koo & Li 2016).
    Hover shows the brain ROI (feature) name and ICC3 value.
    """
    CAT_BINS = [
        ("Poor (< 0.50)",        "#E74C3C", -np.inf, 0.50),
        ("Moderate (0.50-0.75)", "#E67E22",  0.50,   0.75),
        ("Good (0.75-0.90)",     "#2980B9",  0.75,   0.90),
        ("Excellent (>= 0.90)",  "#27AE60",  0.90,   np.inf),
    ]

    datasets = []
    if icc_by_site_ebt is not None and len(icc_by_site_ebt) > 0:
        datasets.append((icc_by_site_ebt, "EB=TRUE", AFTER_EBT))
    if icc_by_site_ebf is not None and len(icc_by_site_ebf) > 0:
        datasets.append((icc_by_site_ebf, "EB=FALSE", AFTER_EBF))

    n_panels = len(datasets)
    titles   = [d[1] for d in datasets]
    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=titles, shared_yaxes=True)

    sites    = natural_sorted(set().union(*[set(d[0]["site"].unique()) for d in datasets if len(d[0]) > 0]))
    site_pos = {s: i for i, s in enumerate(sites)}

    # When there are many sites, long batch labels overlap. Above a threshold we
    # number the sites 1..N on the x-axis and print a key mapping numbers to
    # batch names below the figure. Otherwise labels are shown in full, angled
    # 45 degrees, with the font size shrinking as the number of sites grows.
    n_sites     = len(sites)
    use_numbers = n_sites > 12
    tick_size   = max(7, 11 - max(0, n_sites - 10) // 3)

    def _site_label(s):
        if site_n and s in site_n:
            return f"{s}<br>(n={site_n[s]})"
        return s

    rng          = np.random.default_rng(42)
    legend_shown = set()

    for col_i, (df_s, label, _) in enumerate(datasets, start=1):
        # Box traces (distribution shape only, no individual points)
        for site in sites:
            sd = df_s[df_s["site"] == site][["icc3", "feature"]].dropna(subset=["icc3"])
            if len(sd) == 0:
                continue
            xpos = site_pos[site]
            fig.add_trace(go.Box(
                x=[xpos] * len(sd),
                y=sd["icc3"].values,
                name=_site_label(site),
                showlegend=False,
                boxpoints=False,
                line_color="black",
                line_width=1.0,
                fillcolor="rgba(200,200,200,0.15)",
                width=0.45,
                hoverinfo="skip",
            ), row=1, col=col_i)

        # Scatter traces: one per ICC category, colored by category
        for cat_name, cat_color, lo, hi in CAT_BINS:
            x_all, y_all, text_all = [], [], []
            for site in sites:
                sd = df_s[df_s["site"] == site][["icc3", "feature"]].dropna(subset=["icc3"])
                mask = (sd["icc3"] >= lo) & (sd["icc3"] < hi)
                cd   = sd[mask]
                if len(cd) == 0:
                    continue
                n      = len(cd)
                jitter = rng.uniform(-0.12, 0.12, n)
                x_all.extend([site_pos[site] + j for j in jitter])
                y_all.extend(cd["icc3"].values)
                text_all.extend([
                    f"<b>{feat}</b><br>ICC(3,1): {icc:.3f}<br>Batch: {site}"
                    for feat, icc in zip(cd["feature"].values, cd["icc3"].values)
                ])

            if not x_all:
                continue

            show = cat_name not in legend_shown
            if show:
                legend_shown.add(cat_name)

            fig.add_trace(go.Scatter(
                x=x_all, y=y_all,
                mode="markers",
                name=cat_name,
                legendgroup=cat_name,
                showlegend=show,
                marker=dict(
                    color=cat_color, size=6, opacity=0.85,
                    line=dict(color="rgba(0,0,0,0.15)", width=0.5),
                ),
                text=text_all,
                hovertemplate="%{text}<extra></extra>",
            ), row=1, col=col_i)

        # Threshold dashed lines
        for yval in [0.50, 0.75, 0.90]:
            fig.add_hline(y=yval, line_dash="dash", line_color=GREY,
                          opacity=0.45, line_width=0.8, row=1, col=col_i)

        # x-axis: numeric positions mapped to site labels (or 1..N numbers)
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(sites))),
            ticktext=([str(i + 1) for i in range(n_sites)] if use_numbers
                      else [_site_label(s) for s in sites]),
            tickangle=0,
            tickfont=dict(size=tick_size),
            row=1, col=col_i,
        )

    # Site key when the sites are numbered
    n_key_lines = 0
    if use_numbers:
        parts = [
            f"Batch {i + 1} = {s}" + (f" (n={site_n[s]})" if site_n and s in site_n else "")
            for i, s in enumerate(sites)
        ]
        per_line = 2
        key_lines = ["      ".join(parts[k:k + per_line]) for k in range(0, len(parts), per_line)]
        n_key_lines = len(key_lines)
        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=-0.12, xanchor="left", yanchor="top",
            text="<b>Batch key</b><br>" + "<br>".join(key_lines),
            showarrow=False, align="left", font=dict(size=8, color="black"),
        )

    bottom   = (150 + 22 * n_key_lines) if use_numbers else 140
    fig.update_yaxes(title_text="ICC(3,1)", range=[0, 1.08], row=1, col=1)
    fig.update_layout(
        title="Within-batch consistency by batch, ICC(3,1) (Koo & Li 2016)",
        height=520 + (18 * n_key_lines if use_numbers else 0), **WHITE_BG,
        margin=dict(r=40, b=bottom),
        legend=dict(BOTTOM_LEGEND, title_text="ICC(3,1) category"),
    )
    return fig


def plot_spearman_by_site(
    spm_by_site_ebt: pd.DataFrame,
    spm_by_site_ebf: pd.DataFrame | None = None,
) -> go.Figure:
    datasets = [(spm_by_site_ebt, "EB=TRUE", AFTER_EBT)]
    if spm_by_site_ebf is not None and len(spm_by_site_ebf) > 0:
        datasets.append((spm_by_site_ebf, "EB=FALSE", AFTER_EBF))

    n_panels = len(datasets)
    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=[d[1] for d in datasets],
                        shared_yaxes=True)

    sites = natural_sorted(set().union(*[set(d[0]["site"].unique()) for d in datasets if len(d[0]) > 0]))
    site_palette_local = px.colors.qualitative.Safe[:len(sites)]
    site_color = dict(zip(sites, site_palette_local))

    for col_i, (df_s, label, _) in enumerate(datasets, start=1):
        for site in sites:
            sub = df_s[df_s["site"] == site]["spearman_r"].dropna()
            if len(sub) == 0:
                continue
            fig.add_trace(go.Box(
                y=sub, name=site, legendgroup=site,
                showlegend=(col_i == 1),
                marker_color=site_color.get(site, "#888"),
                line_color=site_color.get(site, "#888"),
                boxpoints="all", jitter=0.35, pointpos=0,
                marker=dict(size=4, opacity=0.55),
                fillcolor="rgba(255,255,255,0.5)",
                hovertemplate=f"<b>{site}</b><br>Spearman r: %{{y:.3f}}<extra></extra>",
            ), row=1, col=col_i)

        for ref in [0.80, 0.90, 0.95]:
            fig.add_hline(y=ref, line_dash="dot", line_color=GREY,
                          opacity=0.50, line_width=0.9,
                          annotation_text=str(ref), annotation_position="right",
                          annotation_font_size=8, annotation_font_color=GREY,
                          row=1, col=col_i)

    fig.update_yaxes(title_text="Spearman r", range=[-0.1, 1.05], row=1, col=1)
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        title="Within-batch consistency by batch: Spearman r (raw vs harmonized)",
        height=500, **WHITE_BG,
        margin=dict(r=60, b=120),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.25,
            xanchor="center", x=0.5,
            title_text="Batch",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Extra variable associations — effect size before vs after harmonization
# ---------------------------------------------------------------------------
def plot_extra_associations(assoc_df: pd.DataFrame) -> go.Figure | None:
    """
    Scatter of effect size (Pearson r for continuous, Cohen's f for categorical)
    before vs after harmonization. One panel per variable x EB condition.
    Points colored by FDR significance category (same scheme as age correlations).
    """
    if assoc_df is None or len(assoc_df) == 0:
        return None

    before_df   = assoc_df[assoc_df["harmonization"] == "Before harmonization"]
    after_conds = [c for c in assoc_df["harmonization"].unique() if c != "Before harmonization"]
    variables   = sorted(assoc_df["variable"].unique())

    if len(before_df) == 0 or not after_conds or not variables:
        return None

    # One ROW per variable so the layout scales cleanly with many variables.
    n_rows = len(variables)
    n_cols = len(after_conds)
    titles = [f"{v}<br><sub>{c}</sub>" for v in variables for c in after_conds]
    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=titles,
        vertical_spacing=min(0.12, 0.55 / max(n_rows, 1)),
        horizontal_spacing=0.14,
    )

    colors_map = {"After (EB=TRUE)": AFTER_EBT, "After (EB=FALSE)": AFTER_EBF}
    PURPLE = "#6A0DAD"

    for row_i, var in enumerate(variables, start=1):
        var_rows = assoc_df[assoc_df["variable"] == var]
        vtype    = var_rows["var_type"].iloc[0] if len(var_rows) > 0 else "continuous"
        eff_lbl  = "Pearson r" if vtype == "continuous" else "Cohen's f"

        for col_i, cond in enumerate(after_conds, start=1):
            color = colors_map.get(cond, BLUE)

            b = before_df[before_df["variable"] == var][["feature", "effect_size", "sig_fdr"]].rename(
                columns={"effect_size": "eff_b", "sig_fdr": "sig_b"})
            a_rows = assoc_df[(assoc_df["variable"] == var) & (assoc_df["harmonization"] == cond)]
            a = a_rows[["feature", "effect_size", "sig_fdr"]].rename(
                columns={"effect_size": "eff_a", "sig_fdr": "sig_a"})
            merged = b.merge(a, on="feature")
            if len(merged) == 0:
                continue

            ns      = merged[~merged["sig_b"] & ~merged["sig_a"]]
            new_sig = merged[~merged["sig_b"] &  merged["sig_a"]]
            lost    = merged[ merged["sig_b"] & ~merged["sig_a"]]
            both    = merged[ merged["sig_b"] &  merged["sig_a"]]

            for sub, lbl, mc, sym, sz, op in [
                (ns,      "Not significant (neither)",        GREY,   "circle",  7,  0.40),
                (new_sig, "FDR significant after only",       color,  "circle",  9,  0.85),
                (lost,    "FDR significant before only",      ORANGE, "diamond", 9,  0.85),
                (both,    "FDR significant before and after", PURPLE, "square",  9,  0.85),
            ]:
                if len(sub) == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["eff_b"], y=sub["eff_a"],
                    mode="markers", name=lbl,
                    legendgroup=lbl, showlegend=False,
                    marker=dict(color=mc, symbol=sym, size=sz, opacity=op,
                                line=dict(color="white", width=0.8)),
                    text=sub["feature"],
                    hovertemplate="<b>%{text}</b><br>Before: %{x:.3f}<br>After: %{y:.3f}<extra></extra>",
                ), row=row_i, col=col_i)

            vals = pd.concat([merged["eff_b"], merged["eff_a"]]).dropna()
            if len(vals) == 0:
                continue
            lo, hi = float(vals.min()) - 0.05, float(vals.max()) + 0.05
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                name="No change", legendgroup="diag", showlegend=False,
                line=dict(color=GREY, dash="dash", width=1.2),
            ), row=row_i, col=col_i)

            fig.update_xaxes(title_text=f"{eff_lbl} (before)", row=row_i, col=col_i)
            fig.update_yaxes(title_text=f"{eff_lbl} (after)", row=row_i, col=col_i)

    # Fixed legend: always show every category (table-like), even those with no
    # points in this run, so the full scheme is visible.
    for lbl, mc, sym in [
        ("Not significant (neither)",        GREY,      "circle"),
        ("FDR significant after only",       AFTER_EBT, "circle"),
        ("FDR significant before only",      ORANGE,    "diamond"),
        ("FDR significant before and after", PURPLE,    "square"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=lbl, legendgroup=lbl,
            marker=dict(color=mc, symbol=sym, size=9, line=dict(color="white", width=0.8)),
            showlegend=True,
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name="No change", legendgroup="diag",
        line=dict(color=GREY, dash="dash", width=1.2), showlegend=True,
    ), row=1, col=1)

    fig.update_layout(
        title="Additional variable associations: effect size before vs after harmonization",
        height=300 * n_rows + 170, **WHITE_BG,
        legend=BOTTOM_LEGEND,
        margin=dict(b=130),
    )
    return fig


def plot_single_association(assoc_df: pd.DataFrame, variable: str) -> go.Figure | None:
    """
    Before-versus-after effect-size scatter for ONE variable, one panel per
    after-harmonization condition (EB=TRUE, EB=FALSE). Continuous variables use
    Pearson r; categorical variables use Cohen's f. Points are coloured and
    shaped by their FDR significance category, with an always-visible legend.

    This is the per-variable figure used for the results tabs, so age, sex, and
    every additional variable are shown in exactly the same grammar.
    """
    if assoc_df is None or len(assoc_df) == 0:
        return None
    sub_all = assoc_df[assoc_df["variable"] == variable]
    if len(sub_all) == 0:
        return None

    before_df   = sub_all[sub_all["harmonization"] == "Before harmonization"]
    after_conds = [c for c in ["After (EB=TRUE)", "After (EB=FALSE)"]
                   if c in set(sub_all["harmonization"].unique())]
    if len(before_df) == 0 or not after_conds:
        return None

    vtype   = sub_all["var_type"].iloc[0]
    eff_lbl = "Pearson r" if vtype == "continuous" else "Cohen's f"
    # significance labelling reflects whether FDR correction was applied
    _corr = str(sub_all["correction"].iloc[0]) if "correction" in sub_all.columns else "FDR (Benjamini-Hochberg)"
    _fdr  = _corr.startswith("FDR")
    _sig  = "FDR significant" if _fdr else "Significant (uncorrected)"
    LBL_NS, LBL_NEW    = "Not significant (neither)", f"{_sig} after only"
    LBL_LOST, LBL_BOTH = f"{_sig} before only", f"{_sig} before and after"
    _pcol = "p_fdr" if _fdr else "p_value"
    _plab = "FDR-corrected p (after)" if _fdr else "Uncorrected p (after)"
    # linear-model statistic surfaced in the hover, per variable type
    lm_col  = "beta" if vtype == "continuous" else "eta_sq"
    lm_name = ("Linear-model coefficient (after)" if vtype == "continuous"
               else "Partial eta squared (after)")
    colors_map = {"After (EB=TRUE)": AFTER_EBT, "After (EB=FALSE)": AFTER_EBF}
    PURPLE = "#6A0DAD"

    n_cols = len(after_conds)
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=after_conds,
                        horizontal_spacing=0.14)

    for col_i, cond in enumerate(after_conds, start=1):
        color = colors_map.get(cond, BLUE)
        b = before_df[["feature", "effect_size", "sig_fdr"]].rename(
            columns={"effect_size": "eff_b", "sig_fdr": "sig_b"})
        _after = sub_all[sub_all["harmonization"] == cond]
        a_cols = ["feature", "effect_size", "sig_fdr"]
        if _pcol in _after.columns:
            a_cols.append(_pcol)
        if lm_col in _after.columns:
            a_cols.append(lm_col)
        a = _after[a_cols].rename(columns={"effect_size": "eff_a", "sig_fdr": "sig_a"})
        merged = b.merge(a, on="feature")
        if len(merged) == 0:
            continue
        _has_lm = (lm_col in merged.columns) and (_pcol in merged.columns)

        ns      = merged[~merged["sig_b"] & ~merged["sig_a"]]
        new_sig = merged[~merged["sig_b"] &  merged["sig_a"]]
        lost    = merged[ merged["sig_b"] & ~merged["sig_a"]]
        both    = merged[ merged["sig_b"] &  merged["sig_a"]]

        for grp, lbl, mc, sym, sz, op in [
            (ns,      LBL_NS,   GREY,   "circle",  7,  0.40),
            # one colour in every panel for "significant after only", so the
            # legend (built from the first panel) matches the markers everywhere
            (new_sig, LBL_NEW,  AFTER_EBT, "circle",  9,  0.85),
            (lost,    LBL_LOST, ORANGE, "diamond", 9,  0.85),
            (both,    LBL_BOTH, PURPLE, "square",  9,  0.85),
        ]:
            if len(grp) == 0:
                continue
            if _has_lm:
                cdata = grp[[lm_col, _pcol]].to_numpy()
                htmpl = ("<b>%{text}</b><br>Before: %{x:.3f}<br>After: %{y:.3f}<br>"
                         + lm_name + ": %{customdata[0]:.3f}<br>"
                         + _plab + ": %{customdata[1]:.3g}<extra></extra>")
            else:
                cdata = None
                htmpl = "<b>%{text}</b><br>Before: %{x:.3f}<br>After: %{y:.3f}<extra></extra>"
            fig.add_trace(go.Scatter(
                x=grp["eff_b"], y=grp["eff_a"],
                mode="markers", name=lbl, legendgroup=lbl, showlegend=False,
                marker=dict(color=mc, symbol=sym, size=sz, opacity=op,
                            line=dict(color="white", width=0.8)),
                text=grp["feature"], customdata=cdata,
                hovertemplate=htmpl,
            ), row=1, col=col_i)

        vals = pd.concat([merged["eff_b"], merged["eff_a"]]).dropna()
        if len(vals) == 0:
            continue
        lo, hi = float(vals.min()) - 0.05, float(vals.max()) + 0.05
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            name="No change", legendgroup="diag", showlegend=False,
            line=dict(color=GREY, dash="dash", width=1.2),
        ), row=1, col=col_i)
        fig.update_xaxes(title_text=f"{eff_lbl} (before)", row=1, col=col_i)
        fig.update_yaxes(title_text=f"{eff_lbl} (after)", row=1, col=col_i)

    for lbl, mc, sym in [
        (LBL_NS,   GREY,      "circle"),
        (LBL_NEW,  AFTER_EBT, "circle"),
        (LBL_LOST, ORANGE,    "diamond"),
        (LBL_BOTH, PURPLE,    "square"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=lbl, legendgroup=lbl,
            marker=dict(color=mc, symbol=sym, size=9, line=dict(color="white", width=0.8)),
            showlegend=True,
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name="No change", legendgroup="diag",
        line=dict(color=GREY, dash="dash", width=1.2), showlegend=True,
    ), row=1, col=1)

    fig.update_layout(
        title=f"{variable} association ({eff_lbl}) before versus after harmonization",
        height=500, **WHITE_BG,
        legend=BOTTOM_LEGEND,
        margin=dict(b=130),
    )
    return fig


def plot_icc_ci_compare(icc_ebt, icc_ebf):
    """
    Within-batch ICC(3,1) per feature for EB=TRUE vs EB=FALSE. Each marker is the
    across-batch median and the segment is the interquartile range (Q1 to Q3),
    drawn as coloured vertical segments side by side so their overlap is visible.
    Features are sorted by the EB=TRUE median so the plot reads as two ordered
    bands; where the two coloured segments overlap in height, the interquartile
    ranges overlap.
    """
    if (icc_ebt is None or icc_ebf is None
            or len(icc_ebt) == 0 or len(icc_ebf) == 0):
        return None
    need = {"feature", "icc3", "icc3_lower", "icc3_upper"}
    if not need.issubset(icc_ebt.columns) or not need.issubset(icc_ebf.columns):
        return None
    m = icc_ebt.merge(icc_ebf, on="feature", suffixes=("_t", "_f"))
    if len(m) == 0:
        return None
    m = m.sort_values("icc3_t").reset_index(drop=True)
    x = np.arange(len(m))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x - 0.2, y=m["icc3_t"],
        error_y=dict(type="data", symmetric=False,
                     array=m["icc3_upper_t"] - m["icc3_t"],
                     arrayminus=m["icc3_t"] - m["icc3_lower_t"],
                     color="rgba(42,110,187,0.35)", thickness=1.0, width=0),
        mode="markers", marker=dict(color=AFTER_EBT, size=3),
        name="EB=TRUE", text=m["feature"],
        hovertemplate="<b>%{text}</b><br>EB=TRUE within-batch ICC(3,1) median: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x + 0.2, y=m["icc3_f"],
        error_y=dict(type="data", symmetric=False,
                     array=m["icc3_upper_f"] - m["icc3_f"],
                     arrayminus=m["icc3_f"] - m["icc3_lower_f"],
                     color="rgba(42,157,143,0.35)", thickness=1.0, width=0),
        mode="markers", marker=dict(color=AFTER_EBF, size=3),
        name="EB=FALSE", text=m["feature"],
        hovertemplate="<b>%{text}</b><br>EB=FALSE within-batch ICC(3,1) median: %{y:.3f}<extra></extra>",
    ))
    for yval in [0.50, 0.75, 0.90]:
        fig.add_hline(y=yval, line_dash="dash", line_color=GREY, opacity=0.4, line_width=0.8)
    fig.update_layout(
        title="Within-batch ICC(3,1) per feature, median and interquartile range, EB=TRUE versus EB=FALSE",
        xaxis_title="feature (sorted by EB=TRUE median)",
        yaxis_title="ICC(3,1)",
        height=520, **WHITE_BG,
        xaxis=dict(showticklabels=False),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(b=60, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Data matrix preview (carpet plot)
# ---------------------------------------------------------------------------
def _proportional_blocks(sizes, max_total):
    """
    Split consecutive groups of the given sizes into display blocks, at most
    about max_total blocks in all, never crossing a group boundary. Each group
    gets a share of blocks proportional to its size (at least one). Returns a
    list of (start, end, group index) over the concatenated groups.
    """
    total = int(sum(sizes))
    blocks, start = [], 0
    for gi, size in enumerate(sizes):
        size = int(size)
        if size == 0:
            continue
        k = size if total <= max_total else max(1, min(size, round(size * max_total / total)))
        edges = np.linspace(0, size, k + 1).round().astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if b > a:
                blocks.append((start + a, start + b, gi))
        start += size
    return blocks


def _cluster_order(X: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Order of the rows of a 0/1 matrix that puts similar rows next to each
    other (average-linkage clustering; weights count each column that many times)."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    if len(X) <= 2:
        return np.arange(len(X))
    Xw = X * np.sqrt(weights) if weights is not None else X
    return leaves_list(linkage(Xw, method="average", metric="euclidean"))


def _order_cols_by_missing(d: pd.DataFrame, cols: list) -> list:
    """Feature columns reordered so that columns missing in similar participants
    sit together; columns with the same pattern keep their original order."""
    if len(cols) < 3:
        return list(cols)
    M = d[cols].isna().to_numpy().T
    U, inv = np.unique(M, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    order_u = (_cluster_order(U.astype(float)) if len(U) <= 1500
               else np.argsort(-U.sum(axis=1), kind="stable"))
    rank = np.empty(len(U), dtype=int)
    rank[order_u] = np.arange(len(U))
    return [c for _, c in sorted(zip(rank[inv], cols), key=lambda t: t[0])]


def _order_rows_by_missing(d: pd.DataFrame, site_col: str, feats: list) -> pd.DataFrame:
    """Participants reordered within each batch so that those missing similar
    features sit together (batches keep their order)."""
    if not feats:
        return d
    M = d[feats].isna().to_numpy()
    U, cnt = np.unique(M.T, axis=0, return_counts=True)       # one column per distinct feature pattern
    R, w = U.T.astype(float), cnt.astype(float)
    sites = d[site_col].astype(str).to_numpy()
    new_idx = []
    for s in dict.fromkeys(sites):
        idx = np.flatnonzero(sites == s)
        if 2 < len(idx) <= 3000 and R.shape[1] <= 2000:
            o = _cluster_order(R[idx], w)
        else:
            o = np.argsort(-(R[idx] @ w), kind="stable")
        new_idx.extend(idx[o])
    return d.iloc[new_idx].reset_index(drop=True)


def plot_data_matrix(
    df: pd.DataFrame,
    site_col: str,
    covariate_cols: list,
    feature_cols: list,
    modality_groups: dict | None = None,
    marked_block: tuple | None = None,
    missing_only: bool = False,
    title: str = "Data matrix",
    max_cells: int = 250_000,
    max_rows: int = 600,
    order_by_missing: bool = False,
):
    """
    Carpet plot of the data: one row per participant, sorted by batch, with a
    batch colour strip on the left; covariates first, then the features.

    Normal view: each column z-scored, missing values in black. Missing-only
    view: missing in black, present in light grey (as in the R package naniar).

    Large tables are drawn at reduced resolution, in blocks that never cross a
    batch or a modality boundary: in the normal view a block shows its middle
    participant and feature, in the missing-only view the share of missing
    values in the block, so no missing value is hidden.

    marked_block = (label, columns) places those feature columns last, under a
    labelled band (used to show the features excluded for missingness).

    order_by_missing: participants are ordered within each batch, and features
    within each group, so that similar missing-value patterns sit together
    (average-linkage clustering of the missing-value indicators).

    Returns (figure, note); note is "" or says how the matrix was reduced.
    """
    marked_label, marked_cols = marked_block if marked_block else (None, [])
    marked_cols = [c for c in marked_cols if c in df.columns]
    marked_set = set(marked_cols)
    main_feats = [c for c in feature_cols if c not in marked_set]

    # feature groups, in display order: modality groups, then any feature in
    # no group, then the marked block
    groups = []
    if modality_groups:
        placed = set()
        for g, gcols in modality_groups.items():
            cols_g = [c for c in gcols if c in main_feats and c not in placed]
            placed.update(cols_g)
            if cols_g:
                groups.append((g, cols_g))
        rest = [c for c in main_feats if c not in placed]
        if rest:
            groups.append(("Not in a group", rest))
    elif main_feats:
        groups.append((None, main_feats))
    if marked_cols:
        groups.append((marked_label, marked_cols))
    feats = [c for _, cols_g in groups for c in cols_g]
    cov = [c for c in covariate_cols if c in df.columns and c not in set(feats)]
    cols = cov + feats

    d = df[[site_col] + cols].copy()
    n_no_batch = int(d[site_col].isna().sum())
    d = d[d[site_col].notna()]
    for c in cols:  # categorical columns become numbers so they can be drawn
        if not pd.api.types.is_numeric_dtype(d[c]):
            uniq = sorted(d[c].dropna().astype(str).unique())
            d[c] = d[c].astype(str).map({v: i for i, v in enumerate(uniq)}).where(d[c].notna())
    _order = {s: i for i, s in enumerate(natural_sorted(d[site_col].astype(str).unique()))}
    d = d.iloc[d[site_col].astype(str).map(_order).argsort(kind="stable")].reset_index(drop=True)
    if order_by_missing:
        groups = [(g, _order_cols_by_missing(d, cols_g)) for g, cols_g in groups]
        feats = [c for _, cols_g in groups for c in cols_g]
        cols = cov + feats
        d = _order_rows_by_missing(d, site_col, feats)
    sites = d[site_col].astype(str).to_numpy()
    site_order = list(dict.fromkeys(sites))
    site_n = {s: int((sites == s).sum()) for s in site_order}
    mat = d[cols].to_numpy(dtype=float)
    miss = np.isnan(mat)

    # display blocks: rows within batches, columns within groups; tables up to
    # max_cells are shown at full resolution
    n_rows_show = len(d) if len(d) * len(cols) <= max_cells else min(len(d), max_rows)
    max_cols = max(len(cov) + 1, max_cells // max(n_rows_show, 1))
    row_blocks = [(a, b) for a, b, _ in _proportional_blocks([site_n[s] for s in site_order], n_rows_show)]
    feat_blocks = _proportional_blocks([len(c) for _, c in groups], max(1, max_cols - len(cov)))
    col_blocks = [(i, i + 1) for i in range(len(cov))] + [(len(cov) + a, len(cov) + b) for a, b, _ in feat_blocks]
    blocks_per_group = [sum(1 for *_, g in feat_blocks if g == gi) for gi in range(len(groups))]
    reduced = len(row_blocks) < len(d) or len(col_blocks) < len(cols)

    def _label(a, b):
        name = cols[a] if b - a == 1 else f"{cols[a]} to {cols[b - 1]} ({b - a} columns)"
        if missing_only:  # as in vis_miss: the percentage missing next to each column
            name += f" ({100 * miss[:, a:b].mean():.1f}%)" if miss.size else ""
        return name
    x_labels = [_label(a, b) for a, b in col_blocks]

    if missing_only:
        rs = np.array([a for a, _ in row_blocks])
        cs = np.array([a for a, _ in col_blocks])
        counts = np.add.reduceat(np.add.reduceat(miss.astype(float), rs, axis=0), cs, axis=1)
        sizes = np.outer([b - a for a, b in row_blocks], [b - a for a, b in col_blocks])
        zz = np.round(counts / sizes, 3)
        main = go.Heatmap(
            z=zz, x=x_labels, zmin=0, zmax=1,
            colorscale=[[0, "#DDDDDD"], [1, "black"]],
            showscale=reduced,
            colorbar=dict(title="missing", thickness=12, x=1.01),
            hovertemplate=("Column: %{x}<br>" + ("missing: %{z:.2f} of the block" if reduced else "missing: %{z:.0f}")
                           + "<extra></extra>"),
        )
    else:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-missing columns
            mu = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1
        z = (mat - mu) / sd
        rmid = [(a + b - 1) // 2 for a, b in row_blocks]
        cmid = [(a + b - 1) // 2 for a, b in col_blocks]
        zz = np.round(z[np.ix_(rmid, cmid)], 2)
        main = go.Heatmap(
            z=zz, x=x_labels, colorscale="RdBu_r", zmid=0, zmin=-3, zmax=3,
            colorbar=dict(title="z-score", thickness=12, x=1.01),
            hoverongaps=False,
            hovertemplate="Column: %{x}<br>z: %{z:.2f}<extra></extra>",
        )
        miss_layer = np.where(np.isnan(zz), 1.0, np.nan)

    # batch strip
    block_site = [sites[a] for a, _ in row_blocks]
    palette = px.colors.qualitative.Safe
    color_of = {s: palette[i % len(palette)] for i, s in enumerate(natural_sorted(site_order))}
    n_s = len(site_order)
    strip_cs = []
    for i, s in enumerate(site_order):
        strip_cs += [[round(i / n_s, 8), color_of[s]], [round(min((i + 1) / n_s, 1.0), 8), color_of[s]]]
    strip = go.Heatmap(
        z=[[site_order.index(s)] for s in block_site], colorscale=strip_cs,
        zmin=0, zmax=max(n_s - 1, 1), showscale=False,
        customdata=[[f"{s} (n = {site_n[s]})"] for s in block_site],
        hovertemplate="Batch: %{customdata}<extra></extra>",
    )

    fig = make_subplots(rows=1, cols=2, column_widths=[0.025, 0.975],
                        shared_yaxes=True, horizontal_spacing=0.003)
    fig.add_trace(strip, row=1, col=1)
    fig.add_trace(main, row=1, col=2)
    if not missing_only and np.isnan(miss_layer).sum() < miss_layer.size:
        fig.add_trace(go.Heatmap(
            z=miss_layer, x=x_labels, colorscale=[[0, "black"], [1, "black"]], showscale=False,
            hoverongaps=False, hovertemplate="Column: %{x}<br>missing<extra></extra>",
        ), row=1, col=2)

    # batch labels and boundaries
    ytv, ytt, start = [], [], 0
    for i in range(1, len(block_site) + 1):
        if i == len(block_site) or block_site[i] != block_site[i - 1]:
            ytv.append((start + i - 1) / 2)
            ytt.append(f"{block_site[start]} (n={site_n[block_site[start]]})")
            if i < len(block_site):
                # batch boundaries: never black like the missing values (dotted blue on
                # the missing-value map, white on the z-scored view)
                fig.add_hline(y=i - 0.5, line_color="white" if not missing_only else SEP_COLOR,
                              line_dash="solid" if not missing_only else "dot",
                              line_width=1.0 if not missing_only else 1.6, opacity=0.9, row=1, col=2)
            start = i

    # covariates | features separator, and group bands with labels
    if cov:
        fig.add_vline(x=len(cov) - 0.5, line_color=SEP_COLOR, line_dash="dot", line_width=2, row=1, col=2)
    show_bands = any(g is not None for g, _ in groups)
    if show_bands:
        band_palette = px.colors.qualitative.Bold
        x0, gi_color = len(cov), 0
        for (g, c), n_blk in zip(groups, blocks_per_group):
            x1 = x0 + n_blk
            if g is not None:
                if g == marked_label:
                    color = "#888888"
                else:
                    color = band_palette[gi_color % len(band_palette)]
                    gi_color += 1
                if x0 > len(cov):
                    fig.add_vline(x=x0 - 0.5, line_color=SEP_COLOR, line_dash="dot", line_width=1.6, row=1, col=2)
                fig.add_shape(type="rect", xref="x2", yref="paper", x0=x0 - 0.5, x1=x1 - 0.5,
                              y0=1.002, y1=1.028, fillcolor=color, line_width=0, layer="above")
                fig.add_annotation(xref="x2", yref="paper", x=(x0 + x1 - 1) / 2, y=1.035,
                                   text=f"{g} ({len(c)})", showarrow=False, textangle=-45,
                                   xanchor="left", yanchor="bottom", font=dict(size=13, color="#333333"))
            x0 = x1

    what = (f"missing values in black ({100 * miss.mean():.1f}% of cells), present in grey"
            if missing_only and miss.size else
            "missing values in black, present in grey" if missing_only else
            "z-scored per column, missing in black")
    longest = max((len(f"{g} ({len(c)})") for g, c in groups if g is not None), default=0)
    top = 75 + int(5.6 * longest) if show_bands else 60
    no_ticks = dict(showgrid=False, zeroline=False, showline=False, ticks="")
    fig.update_layout(
        title=dict(text=f"{title}: {len(d)} participants x {len(feats)} features"
                        f"{f' + {len(cov)} covariates' if cov else ''} (sorted by batch"
                        f"{'; participants and features grouped by similar missing values' if order_by_missing else ''}"
                        f"; {what})",
                   y=0.995, yanchor="top"),
        xaxis=dict(showticklabels=False, range=[-0.5, 0.5], **no_ticks),
        # fixed range: the group labels above the bands would otherwise widen the axis
        xaxis2=dict(showticklabels=len(col_blocks) <= 80, tickangle=45, tickfont=dict(size=10),
                    range=[-0.5, len(col_blocks) - 0.5], **no_ticks),
        yaxis=dict(tickmode="array", tickvals=ytv, ticktext=ytt, tickfont=dict(size=12),
                   range=[len(row_blocks) - 0.5, -0.5], **no_ticks),
        yaxis2=dict(showticklabels=False, range=[len(row_blocks) - 0.5, -0.5], **no_ticks),
        height=max(400, min(len(row_blocks) * 4, 900)) + (top - 60),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", family="Arial"),
        margin={"l": 130, "b": 90, "r": 60, "t": top},
    )

    notes = []
    if reduced:
        how = ("each cell shows the fraction of its block that is missing"
               if missing_only else "each cell shows the middle participant and feature of its block")
        notes.append(f"Shown at reduced resolution ({len(d)} participants in {len(row_blocks)} rows, "
                     f"{len(cols)} columns in {len(col_blocks)}); {how}.")
    if n_no_batch:
        notes.append(f"{n_no_batch} participant(s) with no batch value are not shown.")
    return fig, " ".join(notes)


# ── Missing-value assessment figures ─────────────────────────────────────────

def plot_missing_curve(feature_pct: pd.Series, measure_of: dict | None = None,
                       threshold: float | None = None, kind: str = "features") -> go.Figure:
    """
    For every share x from 0 to 100%, the features with more than x% of the
    participants missing: one line per modality group as defined by the user,
    or one line for all selected features when no groups are defined. The y-axis
    is the share of the group's features, so groups of different sizes compare;
    the hover gives the count.
    """
    # kind "participants": feature_pct holds each participant's share of the
    # selected features missing, and the line counts participants
    unit = "features" if kind == "features" else "participants"
    if measure_of and kind == "features":
        groups = list(dict.fromkeys(measure_of[c] for c in feature_pct.index))
        members = {g: [c for c in feature_pct.index if measure_of[c] == g] for g in groups}
    else:
        g0 = "All selected features" if kind == "features" else "All participants"
        groups, members = [g0], {g0: list(feature_pct.index)}
    top = float(feature_pct.max()) if len(feature_pct) else 0.0
    x_max = min(100.0, max(5.0, top * 1.1 + 1))
    xs = np.unique(np.concatenate([np.linspace(0, x_max, 401), feature_pct.values]))
    xs = xs[xs <= x_max]
    palette = px.colors.qualitative.Bold
    fig = go.Figure()
    for i, g in enumerate(groups):
        p = np.sort(feature_pct[members[g]].values)
        n_over = len(p) - np.searchsorted(p, xs, side="right")      # features with more than x% missing
        fig.add_trace(go.Scatter(
            x=xs, y=100 * n_over / len(p), mode="lines+markers", line_shape="hv", name=f"{g} (n={len(p)})",
            line=dict(color=palette[i % len(palette)] if kind == "features" else "#2A6EBB", width=2.2),
            marker=dict(size=7, opacity=0),          # invisible points, so a click on the line selects one
            customdata=np.stack([n_over, np.full(len(xs), len(p))], axis=1),
            hovertemplate=(f"{g}<br>more than %{{x:.1f}}% missing: %{{customdata[0]}} of "
                           f"%{{customdata[1]}} {unit} (%{{y:.1f}}%)<br>click to see them or exclude them<extra></extra>"),
        ))
    sizes = {len(members[g]) for g in groups}
    if len(sizes) == 1:
        n_all = sizes.pop()
        fig.add_trace(go.Scatter(x=[0, 0], y=[0, 0], yaxis="y2", mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.update_layout(yaxis2=dict(title=f"{unit.capitalize()} (n" + (" per group)" if len(groups) > 1 else ")"),
                                      overlaying="y", side="right", range=[0, n_all * 1.02], autorange=False,
                                      showgrid=False))
    if threshold is not None and threshold <= x_max:
        fig.add_vline(x=threshold, line_dash="dot", line_color=GREY, line_width=1.5,
                      annotation_text=f"{threshold:g}%", annotation_position="top right")
    if kind == "features":
        title, x_title = "Features with more than x% of participants missing", "Participants missing (x, %)"
        y_title = "Features (% of each modality group)" if len(groups) > 1 else "Features (%)"
    else:
        title, x_title, y_title = ("Participants missing more than x% of the selected features",
                                   "Features missing (x, %)", "Participants (%)")
    fig.update_layout(
        title=title, **WHITE_BG, height=430,
        margin=dict(l=10, r=60, t=60, b=120), legend=dict(BOTTOM_LEGEND),
        xaxis=dict(title=x_title, range=[0, x_max], autorange=False, showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title=y_title, range=[0, 102], autorange=False, showgrid=True, gridcolor="#EEEEEE"),
    )
    return fig


def plot_missing_by_batch_bars(batches: list, n: list, values_pct: list, over_pct: list,
                               threshold: float, all_values_pct: float, all_over_pct: float) -> go.Figure:
    """
    Per batch, two bars with their value printed: the feature values missing
    (% of the batch's values) and the participants missing more than the
    threshold (% of the batch); dotted lines give the same for all batches, so
    a batch that stands out is easy to see.
    """
    x = [f"{b}<br>(n={k})" for b, k in zip(batches, n)]
    fig = go.Figure()
    for name, y, color, ref in (
            ("Feature values missing (% of the batch's values)", values_pct, "#2A6EBB", all_values_pct),
            (f"Participants missing more than {threshold:g}% of the features (% of the batch)", over_pct,
             "#E34A33", all_over_pct)):
        fig.add_trace(go.Bar(x=x, y=y, name=f"{name}; all batches {ref:.1f}% (dotted line)",
                             marker_color=color, text=[f"{v:.1f}%" for v in y],
                             textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>" + name + ": %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=ref, line_dash="dot", line_color=color, line_width=1.5)
    top = max([*values_pct, *over_pct, all_values_pct, all_over_pct, 1.0])
    fig.update_layout(
        title="Missing values per batch", barmode="group", **WHITE_BG, height=470,
        margin=dict(l=10, r=30, t=60, b=140), legend=dict(BOTTOM_LEGEND),
        yaxis=dict(title="%", range=[0, top * 1.18], showgrid=True, gridcolor="#EEEEEE"),
        xaxis=dict(tickangle=0),
    )
    return fig


def plot_missing_by_batch_groups(batches: list, n: list, by_group: dict, all_by_group: dict) -> go.Figure:
    """Per batch, one bar per modality group: the feature values of that group missing
    (% of the batch's values for the group); the legend gives each group's value over all batches."""
    x = [f"{b}<br>(n={k})" for b, k in zip(batches, n)]
    palette = px.colors.qualitative.Bold
    fig = go.Figure()
    for i, (g, vals) in enumerate(by_group.items()):
        fig.add_trace(go.Bar(x=x, y=vals, name=f"{g}; all batches {all_by_group[g]:.1f}%",
                             marker_color=palette[i % len(palette)], text=[f"{v:.1f}%" for v in vals],
                             textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>" + g + ": %{y:.1f}% of the values missing<extra></extra>"))
    top = max([v for vals in by_group.values() for v in vals] + [1.0])
    fig.update_layout(
        title="Feature values missing per batch, by modality group", barmode="group", **WHITE_BG, height=470,
        margin=dict(l=10, r=30, t=60, b=140), legend=dict(BOTTOM_LEGEND),
        yaxis=dict(title="Feature values missing (%)", range=[0, top * 1.2], showgrid=True, gridcolor="#EEEEEE"),
        xaxis=dict(tickangle=0),
    )
    return fig
