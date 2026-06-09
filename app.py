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

# ── Row 1: Select all / Clear all ─────────────────────────────────────────
r1c1, r1c2, _ = st.columns([1, 1, 6])
if r1c1.button("✔ Select all", use_container_width=True):
    st.session_state["sel_features"] = auto_features
    st.rerun()
if r1c2.button("✖ Clear all", use_container_width=True):
    st.session_state["sel_features"] = []
    st.rerun()

# ── Row 2+: One button per detected prefix, max 6 per row ─────────────────
CHUNK = 6
prefix_list = list(prefix_groups.items())
if prefix_list:
    st.caption("Toggle modality groups:")
    for chunk_start in range(0, len(prefix_list), CHUNK):
        chunk = prefix_list[chunk_start:chunk_start + CHUNK]
        btn_cols = st.columns(len(chunk))
        for j, (prefix, cols) in enumerate(chunk):
            label = prefix.rstrip("_")
            # truncate long names to keep buttons readable
            display = label if len(label) <= 14 else label[:13] + "…"
            if btn_cols[j].button(display, key=f"btn_{prefix}",
                                  help=f"Toggle all {label} features ({len(cols)} columns)",
                                  use_container_width=True):
                current = set(st.session_state["sel_features"])
                if cols[0] in current:
                    st.session_state["sel_features"] = [c for c in st.session_state["sel_features"] if c not in cols]
                else:
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

        st.session_state.update(dict(
            results_ready=True,
            methods_para=methods_para,
            fig_site=fig_site, fig_icc=fig_icc,
            fig_age=fig_age,   fig_anc=fig_anc,
            fig_icc_site=fig_icc_site,
            html_report=html_report,
            icc_ebt=icc_ebt, icc_ebf=icc_ebf,
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
    st.download_button(
        label="⬇  Download Full Report (HTML)",
        data=st.session_state["html_report"],
        file_name="harmonization_report.html",
        mime="text/html",
        type="primary",
        use_container_width=True,
    )
    st.caption("Self-contained HTML file suitable for supplementary material. Open in any browser.")
