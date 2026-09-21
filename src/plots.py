"""
plots.py
Plotly figures for all harmonization evaluation metrics.
All figures have white background and are returned as plotly Figure objects.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

BLUE   = "#2A6EBB"
ORANGE = "#EB6400"
GREEN  = "#2A9D8F"
GREY   = "#6C757D"

BEFORE_COLOR = ORANGE
AFTER_EBT    = BLUE
AFTER_EBF    = GREEN

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

    sites    = sorted(dev_before["site"].unique())
    site_pos = {s: i for i, s in enumerate(sites)}
    n_sites  = len(sites)
    use_numbers = n_sites > 12
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
            ticktext=([str(i + 1) for i in range(n_sites)] if use_numbers
                      else [_full_label(s) for s in sites]),
            tickangle=0 if use_numbers else 45,
            tickfont=dict(size=tick_size), row=1, col=col_idx,
        )

    n_key_lines = 0
    if use_numbers:
        parts = [f"Batch {i + 1} = {s}" + (f" (n={site_n[s]})" if site_n and s in site_n else "")
                 for i, s in enumerate(sites)]
        key_lines = ["      ".join(parts[k:k + 2]) for k in range(0, len(parts), 2)]
        n_key_lines = len(key_lines)
        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=-0.14, xanchor="left", yanchor="top",
            text="<b>Batch key</b><br>" + "<br>".join(key_lines),
            showarrow=False, align="left", font=dict(size=8, color="black"),
        )

    fig.update_yaxes(title_text="Batch mean z-score", row=1, col=1)
    fig.update_layout(
        title="Batch mean deviation from grand mean (z-scored per feature)",
        height=480 + (16 * n_key_lines if use_numbers else 0), **WHITE_BG,
        showlegend=False,
        margin=dict(b=(120 + 18 * n_key_lines) if use_numbers else 90),
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
        height=490, **WHITE_BG,
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5),
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

    y_bottom = 0.585 if has_ebf else 0.66
    fig.add_shape(type="rect", xref="paper", yref="paper",
                  x0=0.58, x1=0.995, y0=y_bottom, y1=0.995,
                  fillcolor="rgba(255,255,255,0.90)", line=dict(color="#cccccc", width=1),
                  layer="above")

    cx1, cx2 = 0.83, 0.945          # column x-positions (after p<0.05 / p>=0.05)
    r1, r2   = 0.84, 0.75           # row y-positions   (before p<0.05 / p>=0.05)

    # column header (after harmonization)
    ann((cx1 + cx2) / 2, 0.965, "<b>After harmonization</b>", size=9, color="#111")
    ann(cx1, 0.905, "p &lt; 0.05",   size=8, color="#444")
    ann(cx2, 0.905, "p &#8805; 0.05", size=8, color="#444")

    # row header (before harmonization)
    ann(0.60, (r1 + r2) / 2, "<b>Before<br>harmonization</b>", size=9,
        color="#111", xanchor="left")
    ann(0.775, r1, "p &lt; 0.05",   size=8, color="#444", xanchor="right")
    ann(0.775, r2, "p &#8805; 0.05", size=8, color="#444", xanchor="right")

    # cells: exactly the marker a feature of that type carries
    ann(cx1, r1, "&#9679;", size=18, color=red)     # filled red   = sig before & after
    ann(cx2, r1, "&#9679;", size=18, color=green)   # filled green = sig before, ns after
    ann(cx1, r2, "&#9675;", size=18, color=red)     # open red     = ns before, sig after
    ann(cx2, r2, "&#9675;", size=18, color=green)   # open green   = ns before & after

    if has_ebf:
        y_s = 0.64
        ann(0.63, y_s, "&#9679;", size=13, color="#333")
        ann(0.655, y_s, "EB=TRUE",  size=8, color="#444", xanchor="left")
        ann(0.80, y_s, "&#9670;", size=13, color="#333")
        ann(0.825, y_s, "EB=FALSE", size=8, color="#444", xanchor="left")


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

    sites    = sorted(set().union(*[set(d[0]["site"].unique()) for d in datasets if len(d[0]) > 0]))
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
                    f"<b>{feat}</b><br>ICC3: {icc:.3f}<br>Batch: {site}"
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
            tickangle=0 if use_numbers else 45,
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

    legend_y = -0.16 - 0.052 * n_key_lines
    bottom   = (140 + 22 * n_key_lines) if use_numbers else 150
    fig.update_yaxes(title_text="ICC3", range=[0, 1.08], row=1, col=1)
    fig.update_layout(
        title="Within-batch consistency by batch: ICC3 - Koo & Li (2016)",
        height=520 + (18 * n_key_lines if use_numbers else 0), **WHITE_BG,
        margin=dict(r=40, b=bottom),
        legend=dict(
            orientation="h",
            yanchor="top", y=legend_y,
            xanchor="center", x=0.5,
            title_text="ICC3 category",
        ),
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

    sites = sorted(set().union(*[set(d[0]["site"].unique()) for d in datasets if len(d[0]) > 0]))
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
        height=300 * n_rows + 140, **WHITE_BG,
        legend=dict(orientation="h", yanchor="top", y=-0.09 / max(n_rows, 1) - 0.03,
                    xanchor="center", x=0.5),
        margin=dict(b=90),
    )
    return fig


def plot_icc_ci_compare(icc_ebt, icc_ebf):
    """
    ICC(C,1) 95% confidence intervals per feature for EB=TRUE vs EB=FALSE, drawn
    as coloured vertical segments side by side so their overlap is visible.
    Features are sorted by the EB=TRUE ICC so the plot reads as two ordered
    bands; where the two coloured segments overlap in height, the confidence
    intervals overlap.
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
        hovertemplate="<b>%{text}</b><br>EB=TRUE ICC: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x + 0.2, y=m["icc3_f"],
        error_y=dict(type="data", symmetric=False,
                     array=m["icc3_upper_f"] - m["icc3_f"],
                     arrayminus=m["icc3_f"] - m["icc3_lower_f"],
                     color="rgba(42,157,143,0.35)", thickness=1.0, width=0),
        mode="markers", marker=dict(color=AFTER_EBF, size=3),
        name="EB=FALSE", text=m["feature"],
        hovertemplate="<b>%{text}</b><br>EB=FALSE ICC: %{y:.3f}<extra></extra>",
    ))
    for yval in [0.50, 0.75, 0.90]:
        fig.add_hline(y=yval, line_dash="dash", line_color=GREY, opacity=0.4, line_width=0.8)
    fig.update_layout(
        title="ICC(C,1) 95% confidence intervals: EB=TRUE vs EB=FALSE (per feature)",
        xaxis_title="feature (sorted by EB=TRUE ICC)",
        yaxis_title="ICC3",
        height=520, **WHITE_BG,
        xaxis=dict(showticklabels=False),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(b=60, r=40),
    )
    return fig
