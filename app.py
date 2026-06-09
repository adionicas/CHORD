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
    Transparent and standardized assessment of multisite ComBat harmonization in neuroimaging
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
**CHORD** evaluates ComBat harmonization effectiveness in multisite neuroimaging datasets.

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

c1, c2, c3 = st.columns(3)
with c1:
    site_col = st.selectbox("Site / Batch column", all_cols,
                             index=all_cols.index(default_site) if default_site in all_cols else 0)
with c2:
    age_col  = st.selectbox("Age column", all_cols,
                             index=all_cols.index(default_age) if default_age in all_cols else 0)
with c3:
    sex_col  = st.selectbox("Sex column", all_cols,
                             index=all_cols.index(default_sex) if default_sex in all_cols else 0)

exclude_meta  = {site_col, age_col, sex_col}
auto_features = [c for c in num_cols if c not in exclude_meta]

# ── Feature selection with quick-select controls ──────────────────────────
import re as _re

# Detect modality prefixes (e.g. FA_, MD_, fMRI_)
def _prefix(col):
    m = _re.match(r'^([A-Za-z]+_)', col)
    return m.group(1) if m else None

prefix_groups = {}
for c in auto_features:
    p = _prefix(c)
    if p:
        prefix_groups.setdefault(p, []).append(c)

# Initialise session state for feature selection
if "sel_features" not in st.session_state or set(st.session_state["sel_features"]) - set(auto_features):
    st.session_state["sel_features"] = auto_features

st.markdown(f"**Feature columns** — {len(auto_features)} numeric columns detected")

# Quick-action buttons
n_prefixes = len(prefix_groups)
btn_labels  = ["Select all", "Clear all"] + [f"{p.rstrip('_')}" for p in prefix_groups]
btn_cols    = st.columns(len(btn_labels))

if btn_cols[0].button("✔ Select all"):
    st.session_state["sel_features"] = auto_features
    st.rerun()
if btn_cols[1].button("✖ Clear all"):
    st.session_state["sel_features"] = []
    st.rerun()
for i, (prefix, cols) in enumerate(prefix_groups.items(), start=2):
    if btn_cols[i].button(prefix.rstrip("_"), help=f"Toggle all {prefix.rstrip('_')} features"):
        current = set(st.session_state["sel_features"])
        if cols[0] in current:           # if any already selected → remove group
            st.session_state["sel_features"] = [c for c in st.session_state["sel_features"] if c not in cols]
        else:                            # otherwise add group
            existing = [c for c in st.session_state["sel_features"] if c not in cols]
            st.session_state["sel_features"] = existing + cols
        st.rerun()

feature_cols = st.multiselect(
    "Selected features (edit manually or use buttons above)",
    options=auto_features,
    key="sel_features",
    label_visibility="collapsed",
)

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

sites = df[site_col].dropna().unique()
st.info(f"Sites: **{', '.join(sorted(sites.astype(str)))}** ({len(sites)} sites) | Features: **{len(feature_cols)}**")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run
# ─────────────────────────────────────────────────────────────────────────────
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

        progress.progress(44, "Computing Spearman r (overall)...")
        spm_parts = []
        if harm_ebt is not None:
            s = spearman_raw_vs_harm(df, harm_ebt, feature_cols); s["harmonization"] = "EB=TRUE";  spm_parts.append(s)
        if harm_ebf is not None:
            s = spearman_raw_vs_harm(df, harm_ebf, feature_cols); s["harmonization"] = "EB=FALSE"; spm_parts.append(s)
        spm_all = pd.concat(spm_parts, ignore_index=True) if spm_parts else pd.DataFrame()
        spm_ebt = spm_parts[0] if run_ebt and spm_parts else None
        spm_ebf = spm_parts[-1] if run_ebf and len(spm_parts) > (1 if run_ebt else 0) else (spm_parts[0] if not run_ebt and spm_parts else None)

        progress.progress(53, "Computing Spearman r (by site)...")
        spm_site_ebt = compute_spearman_by_site(df, harm_ebt, feature_cols, site_col) if harm_ebt is not None else None
        spm_site_ebf = compute_spearman_by_site(df, harm_ebf, feature_cols, site_col) if harm_ebf is not None else None

        progress.progress(60, "Computing ICC (overall)...")
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
        fig_spm   = plot_spearman(spm_all) if len(spm_all) > 0 else None
        fig_age   = plot_age_correlations(age_all)

        fig_icc_site = plot_icc_by_site(
            icc_site_ebt if icc_site_ebt is not None else pd.DataFrame(),
            icc_site_ebf,
        ) if (icc_site_ebt is not None and len(icc_site_ebt) > 0) else None

        fig_spm_site = plot_spearman_by_site(
            spm_site_ebt if spm_site_ebt is not None else pd.DataFrame(),
            spm_site_ebf,
        ) if (spm_site_ebt is not None and len(spm_site_ebt) > 0) else None

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
            fig_spearman=fig_spm, fig_age=fig_age, fig_cohens_f=fig_anc,
        )

        methods_para = build_methods_paragraph(
            site_col=site_col, age_col=age_col, sex_col=sex_col,
            run_ebf=(run_ebf and harm_ebf is not None),
            github_url=GITHUB_URL,
        )

        progress.progress(100, "Done.")

        st.session_state.update(dict(
            results_ready=True,
            methods_para=methods_para,
            fig_site=fig_site, fig_spm=fig_spm, fig_icc=fig_icc,
            fig_age=fig_age,   fig_anc=fig_anc,
            fig_icc_site=fig_icc_site,
            fig_spm_site=fig_spm_site,
            html_report=html_report,
            icc_ebt=icc_ebt, icc_ebf=icc_ebf,
            spm_ebt=spm_ebt, spm_ebf=spm_ebf,
            run_ebt=run_ebt, run_ebf=run_ebf,
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
    spm_ebt = st.session_state.get("spm_ebt")
    spm_ebf = st.session_state.get("spm_ebf")

    # Summary metric boxes
    metric_cols = []
    if icc_ebt is not None and len(icc_ebt) > 0:
        metric_cols += [("Median ICC (EB=TRUE)",      f"{icc_ebt['icc3'].median():.3f}")]
    if spm_ebt is not None and len(spm_ebt) > 0:
        metric_cols += [("Median Spearman r (EB=TRUE)", f"{spm_ebt['spearman_r'].median():.3f}")]
    if icc_ebf is not None and len(icc_ebf) > 0:
        metric_cols += [("Median ICC (EB=FALSE)",      f"{icc_ebf['icc3'].median():.3f}")]
    if spm_ebf is not None and len(spm_ebf) > 0:
        metric_cols += [("Median Spearman r (EB=FALSE)", f"{spm_ebf['spearman_r'].median():.3f}")]

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
        st.markdown("**ICC(C,1) — overall distribution across all features**")
        if st.session_state.get("fig_icc"):
            st.plotly_chart(st.session_state["fig_icc"], use_container_width=True)
            st.caption("ICC(C,1) between raw and harmonized values. Colored bands: Poor < 0.50 | Moderate 0.50–0.75 | Good 0.75–0.90 | Excellent ≥ 0.90 (Koo & Li 2016). Each point = one imaging feature.")
        st.markdown("**Spearman r — overall distribution across all features**")
        if st.session_state.get("fig_spm"):
            st.plotly_chart(st.session_state["fig_spm"], use_container_width=True)
            st.caption("Spearman rank correlation between raw and harmonized values across participants. Each point = one imaging feature.")

    with t4:
        st.markdown("**ICC(C,1) per site** — distribution of values across features, computed within each site")
        if st.session_state.get("fig_icc_site"):
            st.plotly_chart(st.session_state["fig_icc_site"], use_container_width=True)
            st.caption("Each box = distribution of ICC(C,1) across features for that site. Sites with smaller sample sizes may show lower consistency, particularly without EB. Colored bands as in panel above.")
        else:
            st.info("By-site ICC not available (need at least 6 participants per site).")

        st.markdown("**Spearman r per site** — distribution of values across features, computed within each site")
        if st.session_state.get("fig_spm_site"):
            st.plotly_chart(st.session_state["fig_spm_site"], use_container_width=True)
            st.caption("Each box = distribution of Spearman r across features for that site.")
        else:
            st.info("By-site Spearman r not available.")

    with t5:
        if st.session_state.get("fig_age"):
            st.plotly_chart(st.session_state["fig_age"], use_container_width=True)
            st.caption("Each point = one imaging feature. X-axis: Pearson r with Age before harmonization. Y-axis: Pearson r after. Points above the diagonal = stronger age association after harmonization. Color = FDR significance (Benjamini–Hochberg).")

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
    st.download_button(
        label="⬇  Download Full Report (HTML)",
        data=st.session_state["html_report"],
        file_name="harmonization_report.html",
        mime="text/html",
        type="primary",
        use_container_width=True,
    )
    st.caption("Self-contained HTML file suitable for supplementary material. Open in any browser.")
