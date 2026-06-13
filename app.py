"""
CHORD — Multisite Harmonization Assessment
Streamlit web application
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.harmonize import run_combat
from src.metrics   import (site_mean_deviation, spearman_raw_vs_harm,
                            compute_icc, age_correlations, ancova_site_effect,
                            demographic_summary,
                            compute_icc_by_site, compute_spearman_by_site)
from src.plots     import (plot_site_deviation, plot_spearman, plot_icc,
                           plot_age_correlations, plot_cohens_f,
                           plot_icc_by_site, plot_spearman_by_site)
from src.report    import generate_report, build_methods_paragraph

# ── Update this URL once the repo is live ─────────────────────────────────
GITHUB_URL = "https://github.com/adionicas/CHORD"

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CHORD", layout="wide", page_icon="🎵")

st.markdown("""
<div style="background:#2A6EBB;padding:28px 36px;border-radius:8px;margin-bottom:24px;position:relative;overflow:hidden">
  <div style="position:absolute;top:10px;right:24px;font-size:1.35rem;
              color:rgba(255,255,255,0.18);letter-spacing:0.25em;user-select:none">
    ♩ ♪ ♫ ♬ ♩ ♪ ♫
  </div>
  <h1 style="color:white;margin:0 0 4px 0;font-size:1.9rem">♩ CHORD</h1>
  <p style="color:rgba(255,255,255,0.70);margin:0 0 3px 0;font-size:0.82rem;letter-spacing:0.04em">
    Comprehensive Harmonization Open-platform with Reporting and Diagnostics
  </p>
  <p style="color:rgba(255,255,255,0.85);margin:0;font-size:0.95rem">
    Run ComBat harmonization on your multisite neuroimaging data — download the harmonized dataset and a publication-ready evaluation report
  </p>
  <div style="margin-top:14px;font-size:1.0rem;color:rgba(255,255,255,0.22);
              letter-spacing:0.30em;user-select:none">
    ♬ &nbsp; ♪ &nbsp; ♫ &nbsp; ♩ &nbsp; ♬ &nbsp; ♪ &nbsp; ♫ &nbsp; ♩
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.markdown("""
**CHORD** runs ComBat harmonization on your multisite neuroimaging data and evaluates its effectiveness.

Upload your data → get harmonized data + evaluation report.

**Metrics:**
- Site mean deviation (z-score)
- Site effect size (ANCOVA, Cohen's f)
- Within-site consistency: ICC3 overall and by site
- Within-site consistency: Spearman r overall and by site
- Age association preservation (Pearson r, FDR)

**ICC thresholds** — Koo & Li (2016):
- < 0.50: Poor
- 0.50–0.75: Moderate
- 0.75–0.90: Good
- ≥ 0.90: Excellent
    """)
    st.divider()
    if "YOUR_USERNAME" not in GITHUB_URL:
        st.markdown(f"[![GitHub](https://img.shields.io/badge/GitHub-CHORD-2A6EBB?logo=github)]({GITHUB_URL})")
        st.markdown(f"[View on GitHub]({GITHUB_URL})")
    else:
        st.info("GitHub link not set yet. Update `GITHUB_URL` at the top of `app.py` once the repository is created.")
    st.divider()
    st.caption("ComBat: Johnson et al. (2007) | neuroCombat: Fortin et al. (2017)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Upload your data")
st.caption("Upload a CSV or Excel file. Each row is one participant. Columns must include imaging features, a site/batch column, Age, and Sex.")

col_up, col_demo = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader(
        "Drag and drop your file here",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
with col_demo:
    st.markdown("**No data? Try the example dataset:**")
    if st.button("Load synthetic example data (215 participants, 6 sites, 20 FA features)"):
        st.session_state["use_demo"] = True

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.session_state["use_demo"] = False
    except Exception as e:
        st.error(f"Could not read file: {e}")
elif st.session_state.get("use_demo"):
    try:
        df = pd.read_csv("example_data.csv")
        st.info("Loaded synthetic example dataset (215 participants, 6 sites, 20 FA features — not real patient data)")
    except FileNotFoundError:
        st.error("Example data file not found.")

if df is None:
    st.stop()

st.success(f"Data loaded: **{df.shape[0]} participants** × **{df.shape[1]} columns**")
with st.expander("Preview data"):
    st.dataframe(df.head(8), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Configure
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Step 2 — Configure columns")

all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

def _guess(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    return cols[0] if cols else None

default_site = _guess(all_cols, ["site", "scanner", "batch", "center"])
default_age  = _guess(all_cols, ["age", "Age"])
default_sex  = _guess(all_cols, ["sex", "Sex", "gender"])

CUSTOM = "— type custom name below —"

def _col_selector(label, all_cols, default, key):
    """Selectbox + optional text input for manual override."""
    opts  = [CUSTOM] + all_cols
    d_idx = opts.index(default) if default in opts else 1
    sel   = st.selectbox(label, opts, index=d_idx, key=f"{key}_sel")
    if sel == CUSTOM:
        manual = st.text_input(
            f"Custom name for {label.split('/')[0].strip()}",
            key=f"{key}_txt",
            placeholder="Type exact column name from your file",
        )
        if manual and manual in all_cols:
            return manual
        elif manual:
            st.warning(f"'{manual}' not found in the uploaded data.")
            return None
        return None
    return sel

c1, c2, c3 = st.columns(3)
with c1:
    site_col = _col_selector("Site / Batch column", all_cols, default_site, "site")
with c2:
    age_col  = _col_selector("Age column",          all_cols, default_age,  "age")
with c3:
    sex_col  = _col_selector("Sex column",           all_cols, default_sex,  "sex")

if None in (site_col, age_col, sex_col):
    st.stop()

exclude_meta  = {site_col, age_col, sex_col}
auto_features = [c for c in num_cols if c not in exclude_meta]
all_df_cols   = df.columns.tolist()

import re as _re
def _prefix(col):
    m = _re.match(r'^([A-Za-z]+_)', col)
    return m.group(1) if m else None
prefix_groups = {}
for c in auto_features:
    p = _prefix(c)
    if p:
        prefix_groups.setdefault(p, []).append(c)

# Start empty — user must actively select
if "sel_features" not in st.session_state or \
        not set(st.session_state["sel_features"]).issubset(set(auto_features)):
    st.session_state["sel_features"] = []

st.markdown(f"**Step 2b — Select feature columns to harmonize** ({len(auto_features)} numeric columns available)")

# ── Option 1: Range by column name (primary) ──────────────────────────────
st.markdown("**Option 1 — Select a range**")
use_col_num = st.checkbox("Use column numbers instead of names", value=False,
                          help="Check this if column names are not informative enough to use as range boundaries")

if not use_col_num:
    na_c1, na_c2, na_c3 = st.columns([3, 3, 2])
    with na_c1:
        from_name = st.selectbox("From column", ["— select —"] + auto_features, key="range_from_name")
    with na_c2:
        to_name   = st.selectbox("To column",   ["— select —"] + auto_features, key="range_to_name")
    with na_c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Apply range", use_container_width=True, key="apply_name"):
            if from_name != "— select —" and to_name != "— select —":
                i1 = auto_features.index(from_name)
                i2 = auto_features.index(to_name)
                lo, hi = min(i1, i2), max(i1, i2)
                st.session_state["sel_features"] = auto_features[lo:hi + 1]
                st.rerun()
            else:
                st.warning("Select both a From and a To column.")
    if from_name != "— select —" and to_name != "— select —":
        i1, i2 = auto_features.index(from_name), auto_features.index(to_name)
        preview = auto_features[min(i1,i2):max(i1,i2) + 1]
        st.caption(f"Range covers {len(preview)} columns: "
                   f"{', '.join(preview[:5])}{'…' if len(preview) > 5 else ''}")
else:
    nb_c1, nb_c2, nb_c3 = st.columns([3, 3, 2])
    with nb_c1:
        from_num = st.number_input("From column №", min_value=1,
                                    max_value=len(all_df_cols), value=1, step=1)
    with nb_c2:
        to_num   = st.number_input("To column №",   min_value=1,
                                    max_value=len(all_df_cols), value=len(all_df_cols), step=1)
    with nb_c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Apply range", use_container_width=True, key="apply_num"):
            lo2, hi2 = min(from_num, to_num), max(from_num, to_num)
            range_cols = [c for c in all_df_cols[lo2-1:hi2] if c in auto_features]
            if range_cols:
                st.session_state["sel_features"] = range_cols
                st.rerun()
            else:
                st.warning("No numeric feature columns in that position range.")
    num_preview = all_df_cols[from_num-1:to_num]
    st.caption(f"Columns {from_num}–{to_num}: "
               f"{', '.join(num_preview[:5])}{'…' if len(num_preview) > 5 else ''}")

# ── Option 3: Quick-select buttons ────────────────────────────────────────
st.markdown("**Option 3 — Quick-select**")
qs_c1, qs_c2, _ = st.columns([1, 1, 6])
if qs_c1.button("✔ Select all", use_container_width=True):
    st.session_state["sel_features"] = auto_features
    st.rerun()
if qs_c2.button("✖ Clear all", use_container_width=True):
    st.session_state["sel_features"] = []
    st.rerun()

# ── Option 4: Exclude by keyword ──────────────────────────────────────────
st.markdown("**Option 4 — Exclude columns whose name contains a keyword**")
ex_c1, ex_c2 = st.columns([4, 2])
with ex_c1:
    excl_input = st.text_input(
        "Exclude keyword(s)",
        placeholder='e.g.  unassigned   or   unassigned, reward, LIM',
        label_visibility="collapsed",
    )
with ex_c2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Exclude matching", use_container_width=True):
        if excl_input.strip():
            keywords = [k.strip().lower() for k in excl_input.split(",") if k.strip()]
            before   = st.session_state.get("sel_features", [])
            kept     = [c for c in before
                        if not any(kw in c.lower() for kw in keywords)]
            removed  = len(before) - len(kept)
            st.session_state["sel_features"] = kept
            st.rerun()
if excl_input.strip():
    keywords = [k.strip().lower() for k in excl_input.split(",") if k.strip()]
    would_remove = [c for c in st.session_state.get("sel_features", [])
                    if any(kw in c.lower() for kw in keywords)]
    if would_remove:
        st.caption(f"Would remove {len(would_remove)} columns: "
                   f"{', '.join(would_remove[:6])}{'…' if len(would_remove) > 6 else ''}")
    else:
        st.caption("No currently selected columns match that keyword.")

# ── Final review multiselect ───────────────────────────────────────────────
n_sel = len(st.session_state.get("sel_features", []))
st.markdown(f"**Currently selected: {n_sel} features** — review or edit below")
feature_cols = st.multiselect(
    "Selected features",
    options=auto_features,
    key="sel_features",
    label_visibility="collapsed",
)

if not feature_cols:
    st.info("No features selected yet. Use one of the options above to select features.")
    st.stop()

# ── Full name preview ──────────────────────────────────────────────────────
with st.expander(f"View all {len(feature_cols)} selected feature names in full"):
    # show as a compact numbered list in 3 columns
    cols_per_row = 3
    rows = [feature_cols[i:i+cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]
    for i, row in enumerate(rows):
        grid = st.columns(cols_per_row)
        for j, name in enumerate(row):
            idx = i * cols_per_row + j + 1
            grid[j].markdown(f"`{idx}.` {name}")

sites = df[site_col].dropna().unique()
st.info(f"Sites: **{', '.join(sorted(sites.astype(str)))}** ({len(sites)} sites) | Features: **{len(feature_cols)}**")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Data preview (optional)")
st.caption("Visual overview of the selected features before harmonization. "
           "Values are z-scored per feature; missing data shown in grey. "
           "Participants sorted by site so site effects are visible.")

if st.button("Generate data matrix preview", use_container_width=False):
    import plotly.graph_objects as go
    import numpy as np

    # sort participants by site
    plot_df = df[[site_col] + feature_cols].copy()
    plot_df = plot_df.sort_values(site_col).reset_index(drop=True)

    # z-score each feature
    mat = plot_df[feature_cols].values.astype(float)
    col_means = np.nanmean(mat, axis=0)
    col_sds   = np.nanstd(mat, axis=0)
    col_sds[col_sds == 0] = 1
    z = (mat - col_means) / col_sds

    # site boundary lines
    sites_ordered = plot_df[site_col].values
    boundary_y = []
    for i in range(1, len(sites_ordered)):
        if sites_ordered[i] != sites_ordered[i-1]:
            boundary_y.append(i - 0.5)

    # y-axis labels: show site name at midpoint of each block
    ytick_vals, ytick_text = [], []
    prev, start = sites_ordered[0], 0
    for i, s in enumerate(sites_ordered):
        if s != prev or i == len(sites_ordered) - 1:
            end = i if s != prev else i + 1
            mid = (start + end - 1) / 2
            ytick_vals.append(mid)
            ytick_text.append(str(prev))
            prev, start = s, i
    # last block
    if sites_ordered[-1] == prev:
        pass  # already captured above

    fig_carpet = go.Figure(go.Heatmap(
        z=z,
        x=feature_cols,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-3, zmax=3,
        colorbar=dict(title="z-score", thickness=14),
        hoverongaps=False,
        hovertemplate="Feature: %{x}<br>Participant: %{y}<br>z: %{z:.2f}<extra></extra>",
    ))

    # site boundary lines
    for by in boundary_y:
        fig_carpet.add_hline(y=by, line_color="black", line_width=1.2, opacity=0.7)

    n_feat = len(feature_cols)
    height = max(400, min(len(plot_df) * 4, 900))
    fig_carpet.update_layout(
        title=f"Raw data matrix — {len(plot_df)} participants × {n_feat} features (sorted by {site_col})",
        xaxis=dict(showticklabels=n_feat <= 60,
                   tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickmode="array", tickvals=ytick_vals, ticktext=ytick_text,
                   tickfont=dict(size=9)),
        height=height,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", family="Arial"),
        margin=dict(l=80, b=100),
    )
    # grey for NaN — add a trace for missing cells
    nan_rows, nan_cols_idx = np.where(np.isnan(z))
    if len(nan_rows) > 0:
        fig_carpet.add_trace(go.Scatter(
            x=[feature_cols[c] for c in nan_cols_idx],
            y=nan_rows.tolist(),
            mode="markers",
            marker=dict(color="lightgrey", size=3, symbol="square"),
            name="Missing",
            hovertemplate="Missing: %{x}, participant %{y}<extra></extra>",
        ))
    pct_missing = 100 * np.isnan(mat).sum() / mat.size
    st.plotly_chart(fig_carpet, use_container_width=True)
    st.caption(f"Missing values: {np.isnan(mat).sum():,} cells ({pct_missing:.1f}% of matrix) shown in grey. "
               f"Horizontal lines = site boundaries.")

st.divider()
st.subheader("Step 3 — ComBat configuration and run")

eb_options = {
    "EB=TRUE  (Empirical Bayes, recommended)":    ("ebt_only",  True,  False),
    "EB=FALSE  (feature-wise, no EB shrinkage)":  ("ebf_only",  False, True),
    "Compare EB=TRUE vs EB=FALSE":                ("compare",   True,  True),
}
eb_label = st.radio("ComBat configuration", list(eb_options.keys()),
                    horizontal=True, index=0)
eb_mode, run_ebt, run_ebf = eb_options[eb_label]

if st.button("▶  Run Harmonization", type="primary", use_container_width=True):

    progress = st.progress(0, "Starting...")
    try:
        # ── Harmonize ──────────────────────────────────────────────────────
        harm_ebt, harm_ebf = None, None

        if run_ebt:
            progress.progress(5, "Running ComBat (EB=TRUE)...")
            harm_ebt = run_combat(df, feature_cols, site_col, age_col, sex_col, eb=True)

        if run_ebf:
            progress.progress(20, "Running ComBat (EB=FALSE)...")
            harm_ebf = run_combat(df, feature_cols, site_col, age_col, sex_col, eb=False)

        # primary result for site deviation (before panel always uses raw)
        harm_primary = harm_ebt if harm_ebt is not None else harm_ebf

        # ── Metrics ────────────────────────────────────────────────────────
        progress.progress(33, "Computing site deviation...")
        dev_before = site_mean_deviation(df,          feature_cols, site_col)
        dev_ebt    = site_mean_deviation(harm_ebt,    feature_cols, site_col) if harm_ebt is not None else None
        dev_ebf    = site_mean_deviation(harm_ebf,    feature_cols, site_col) if harm_ebf is not None else None

        spm_ebt, spm_ebf = None, None  # Spearman removed; ICC3 is sufficient for consistency

        progress.progress(50, "Computing ICC (overall)...")
        icc_parts = []
        if harm_ebt is not None:
            ic = compute_icc(df, harm_ebt, feature_cols); ic["harmonization"] = "EB=TRUE";  icc_parts.append(ic)
        if harm_ebf is not None:
            ic = compute_icc(df, harm_ebf, feature_cols); ic["harmonization"] = "EB=FALSE"; icc_parts.append(ic)
        icc_all = pd.concat(icc_parts, ignore_index=True) if icc_parts else pd.DataFrame()
        icc_ebt = icc_parts[0] if run_ebt and icc_parts else None
        icc_ebf = icc_parts[-1] if run_ebf and len(icc_parts) > (1 if run_ebt else 0) else (icc_parts[0] if not run_ebt and icc_parts else None)

        progress.progress(67, "Computing ICC (by site)...")
        icc_site_ebt = compute_icc_by_site(df, harm_ebt, feature_cols, site_col) if harm_ebt is not None else None
        icc_site_ebf = compute_icc_by_site(df, harm_ebf, feature_cols, site_col) if harm_ebf is not None else None

        progress.progress(74, "Computing age correlations...")
        age_parts = [age_correlations(df, feature_cols, age_col, "Before harmonization")]
        if harm_ebt is not None: age_parts.append(age_correlations(harm_ebt, feature_cols, age_col, "After (EB=TRUE)"))
        if harm_ebf is not None: age_parts.append(age_correlations(harm_ebf, feature_cols, age_col, "After (EB=FALSE)"))
        age_all = pd.concat(age_parts, ignore_index=True)

        progress.progress(80, "Computing ANCOVA site effects...")
        anc_before = ancova_site_effect(df, feature_cols, site_col, age_col, sex_col, "Before")
        anc_ebt    = ancova_site_effect(harm_ebt, feature_cols, site_col, age_col, sex_col, "EB=TRUE")  if harm_ebt is not None else None
        anc_ebf    = ancova_site_effect(harm_ebf, feature_cols, site_col, age_col, sex_col, "EB=FALSE") if harm_ebf is not None else None

        # ── Figures ────────────────────────────────────────────────────────
        progress.progress(87, "Generating figures...")
        fig_site  = plot_site_deviation(dev_before,
                                        dev_ebt if dev_ebt is not None else pd.DataFrame(),
                                        dev_ebf)
        fig_anc   = plot_cohens_f(anc_before,
                                   anc_ebt if anc_ebt is not None else pd.DataFrame(),
                                   anc_ebf)
        fig_icc   = plot_icc(icc_all) if len(icc_all) > 0 else None
        fig_age   = plot_age_correlations(age_all)

        site_n = df.groupby(site_col)[site_col].count().to_dict()
        site_n = {str(k): int(v) for k, v in site_n.items()}

        fig_icc_site = plot_icc_by_site(
            icc_site_ebt if icc_site_ebt is not None else pd.DataFrame(),
            icc_site_ebf,
            site_n=site_n,
        ) if (icc_site_ebt is not None and len(icc_site_ebt) > 0) else None

        # ── Report ─────────────────────────────────────────────────────────
        progress.progress(94, "Building report...")
        demo_df    = demographic_summary(df, site_col, age_col, sex_col)
        n_retained = len(harm_primary) if harm_primary is not None else 0
        html_report = generate_report(
            df_raw=df, site_col=site_col, age_col=age_col, sex_col=sex_col,
            feature_cols=feature_cols, n_retained=n_retained,
            run_ebf=(run_ebf and harm_ebf is not None),
            demo_df=demo_df,
            dev_before=dev_before, dev_ebt=dev_ebt, dev_ebf=dev_ebf,
            icc_ebt=icc_ebt, icc_ebf=icc_ebf,
            spm_ebt=spm_ebt, spm_ebf=spm_ebf,
            anc_before=anc_before, anc_ebt=anc_ebt, anc_ebf=anc_ebf,
            age_df=age_all,
            fig_site_dev=fig_site, fig_icc=fig_icc, fig_icc_site=fig_icc_site,
            fig_spearman=None, fig_age=fig_age, fig_cohens_f=fig_anc,
        )

        methods_para = build_methods_paragraph(
            site_col=site_col, age_col=age_col, sex_col=sex_col,
            run_ebf=(run_ebf and harm_ebf is not None),
            github_url=GITHUB_URL,
        )

        progress.progress(100, "Done.")

        # build harmonized CSVs for download
        harm_csv_ebt = harm_ebt.to_csv(index=False).encode() if harm_ebt is not None else None
        harm_csv_ebf = harm_ebf.to_csv(index=False).encode() if harm_ebf is not None else None

        st.session_state.update(dict(
            results_ready=True,
            methods_para=methods_para,
            fig_site=fig_site, fig_icc=fig_icc,
            fig_age=fig_age,   fig_anc=fig_anc,
            fig_icc_site=fig_icc_site,
            html_report=html_report,
            icc_ebt=icc_ebt, icc_ebf=icc_ebf,
            run_ebt=run_ebt, run_ebf=run_ebf,
            harm_csv_ebt=harm_csv_ebt,
            harm_csv_ebf=harm_csv_ebf,
        ))

    except Exception as e:
        progress.empty()
        st.error(f"An error occurred: {e}")
        st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("results_ready"):

    st.divider()
    st.subheader("Results")

    icc_ebt = st.session_state.get("icc_ebt")
    icc_ebf = st.session_state.get("icc_ebf")

    # Summary metric boxes
    metric_cols = []
    if icc_ebt is not None and len(icc_ebt) > 0:
        metric_cols += [("Median ICC3 (EB=TRUE)",  f"{icc_ebt['icc3'].median():.3f}")]
    if icc_ebf is not None and len(icc_ebf) > 0:
        metric_cols += [("Median ICC3 (EB=FALSE)", f"{icc_ebf['icc3'].median():.3f}")]

    if metric_cols:
        cols = st.columns(len(metric_cols))
        for col, (label, val) in zip(cols, metric_cols):
            col.metric(label, val)

    # Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Site Deviation",
        "Site Effect Size (Cohen's f)",
        "Within-Site Consistency — Overall",
        "Within-Site Consistency — By Site",
        "Age Associations",
        "Methods paragraph",
    ])

    with t1:
        st.plotly_chart(st.session_state["fig_site"], use_container_width=True)
        st.caption("Each point = one (site, feature) pair. Site means should cluster near zero after harmonization.")

    with t2:
        st.plotly_chart(st.session_state["fig_anc"], use_container_width=True)
        st.caption("ANCOVA Type II (Age + Sex as covariates). Points below the diagonal = reduced site effect. Each point = one feature.")

    with t3:
        if st.session_state.get("fig_icc"):
            st.plotly_chart(st.session_state["fig_icc"], use_container_width=True)
            st.caption("Distribution of ICC3 values across all features. Colored bands: Poor < 0.50 | Moderate 0.50–0.75 | Good 0.75–0.90 | Excellent ≥ 0.90 (Koo & Li 2016). Dashed lines = median per condition.")

    with t4:
        if st.session_state.get("fig_icc_site"):
            st.plotly_chart(st.session_state["fig_icc_site"], use_container_width=True)
            st.caption("Each box = distribution of ICC3 across features for that site. Sites with smaller sample sizes may show lower consistency, particularly without EB. Colored bands: Poor/Moderate/Good/Excellent (Koo & Li 2016).")
        else:
            st.info("By-site ICC not available (need at least 6 participants per site).")

    with t5:
        if st.session_state.get("fig_age"):
            st.plotly_chart(st.session_state["fig_age"], use_container_width=True)
            st.caption("Each point = one feature. Grey circle = not significant either condition | Filled circle = FDR significant after only | Orange diamond = FDR significant before only | Purple square = FDR significant in both. Pearson r computed independently per condition; no formal test of difference applied.")

    with t6:
        st.markdown("#### Methods paragraph for main manuscript")
        st.caption("Copy the text below and paste it directly into your manuscript Methods section. Adapt bracketed placeholders and citation numbers to match your reference style.")
        para = st.session_state.get("methods_para", "")
        st.text_area(
            label="Methods paragraph",
            value=para,
            height=260,
            label_visibility="collapsed",
        )
        st.caption("This paragraph is also included at the end of the downloaded HTML report.")

    st.divider()
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            label="⬇  Download Report (HTML)",
            data=st.session_state["html_report"],
            file_name="harmonization_report.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
        )
        st.caption("Supplementary material report")
    with dl2:
        if st.session_state.get("harm_csv_ebt"):
            st.download_button(
                label="⬇  Harmonized data — EB=TRUE",
                data=st.session_state["harm_csv_ebt"],
                file_name="harmonized_EBT.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with dl3:
        if st.session_state.get("harm_csv_ebf"):
            st.download_button(
                label="⬇  Harmonized data — EB=FALSE",
                data=st.session_state["harm_csv_ebf"],
                file_name="harmonized_EBF.csv",
                mime="text/csv",
                use_container_width=True,
            )
