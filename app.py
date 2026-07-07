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
                            compute_icc_by_site, compute_spearman_by_site,
                            compute_extra_associations)
from src.plots     import (plot_site_deviation, plot_spearman, plot_icc,
                           plot_age_correlations, plot_cohens_f,
                           plot_icc_by_site, plot_spearman_by_site,
                           plot_extra_associations)
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
    ComBat harmonization of multisite neuroimaging data, returning the harmonized dataset and a publication-ready evaluation report
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
**CHORD** performs ComBat harmonization of multisite neuroimaging data and produces a standardized evaluation report.

Input: a table of imaging features with site, age, and sex columns. Output: the harmonized table and an evaluation report.

**Recommended metrics (default ON):**
- Site mean deviation (z-score) — always included
- Within-site consistency by site (ICC3) — primary recommended metric
- Site effect size (ANCOVA, Cohen's f)

**Optional metrics:**
- Age associations (Pearson r, FDR-corrected). Optional: can be confounded by head motion, and are uninformative when the age range is narrow
- Additional variable associations (OLS) — e.g. injury severity, time since injury

**Why ICC by site is the primary metric:**
Within-site ICC measures whether harmonization preserved the internal variability of each site's data. A high ICC means the rank ordering of participants is intact after batch correction — i.e., changing site means and scales did not distort within-site biological variability.

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

# ── Covariates to preserve in the ComBat model ────────────────────────────
st.markdown("**Covariates to preserve in the ComBat model**")
st.caption(
    "ComBat harmonization can preserve the biological variability of key variables "
    "of interest — for example Age and Sex — while removing site-related variance. "
    "Only the Site/Batch column is strictly required. Everything else is optional."
)

other_candidates = [c for c in all_cols if c not in {site_col, age_col, sex_col}]

cov_c1, cov_c2 = st.columns(2)
with cov_c1:
    st.markdown("**Continuous covariates** *(numeric values — e.g. Age, days since injury)*")
    inc_age = st.checkbox(
        f"Include **{age_col}** (auto-detected)",
        value=True, key="inc_age",
    )
    extra_cont_add = st.multiselect(
        "Add more continuous covariates",
        options=[c for c in other_candidates if c in num_cols],
        key="extra_cont",
        placeholder="e.g. days since injury, TSI …",
    )
with cov_c2:
    st.markdown("**Categorical covariates** *(group labels — e.g. Sex, diagnosis, group)*")
    inc_sex = st.checkbox(
        f"Include **{sex_col}** (auto-detected)",
        value=True, key="inc_sex",
    )
    extra_cat_add = st.multiselect(
        "Add more categorical covariates",
        options=other_candidates,
        key="extra_cat",
        placeholder="e.g. Group, Diagnosis, Handedness …",
    )

# Final covariate lists passed to ComBat
continuous_covariates = ([age_col] if inc_age else []) + extra_cont_add
categorical_covariates = ([sex_col] if inc_sex else []) + extra_cat_add

# Also expose for downstream use (diagnostics always use age/sex if present)
extra_continuous  = extra_cont_add
extra_categorical = extra_cat_add

# Summary box
all_covars_display = (
    [f"{age_col} (continuous)" for _ in [1] if inc_age]
    + [f"{c} (continuous)" for c in extra_cont_add]
    + [f"{sex_col} (categorical)" for _ in [1] if inc_sex]
    + [f"{c} (categorical)" for c in extra_cat_add]
)
if all_covars_display:
    st.info(
        f"ComBat model: batch = **{site_col}**  |  covariates = "
        + ", ".join(f"**{c}**" for c in all_covars_display)
    )
else:
    st.warning(
        f"ComBat model: batch = **{site_col}** only — no covariates. "
        "Site effects will be removed but no biological variability is explicitly preserved."
    )

exclude_meta  = {site_col, age_col, sex_col} | set(extra_cont_add) | set(extra_cat_add)
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
    na_c1, na_c2 = st.columns(2)
    with na_c1:
        from_name = st.selectbox("From column", ["— select —"] + auto_features, key="range_from_name")
    with na_c2:
        to_name   = st.selectbox("To column",   ["— select —"] + auto_features, key="range_to_name")
    if from_name != "— select —" and to_name != "— select —":
        i1, i2  = auto_features.index(from_name), auto_features.index(to_name)
        preview = auto_features[min(i1, i2):max(i1, i2) + 1]
        # Apply automatically whenever the two boundaries change. The change
        # check keeps later manual edits (or keyword exclusions) from being
        # overwritten on every rerun.
        if st.session_state.get("_last_range_name") != (from_name, to_name):
            st.session_state["_last_range_name"] = (from_name, to_name)
            st.session_state["sel_features"] = preview
            st.rerun()
        st.caption(f"Range covers {len(preview)} columns: "
                   f"{', '.join(preview[:5])}{'…' if len(preview) > 5 else ''}")
else:
    nb_c1, nb_c2 = st.columns(2)
    with nb_c1:
        from_num = st.number_input("From column №", min_value=1,
                                    max_value=len(all_df_cols), value=1, step=1)
    with nb_c2:
        to_num   = st.number_input("To column №",   min_value=1,
                                    max_value=len(all_df_cols), value=len(all_df_cols), step=1)
    range_cols = [c for c in all_df_cols[min(from_num, to_num) - 1:max(from_num, to_num)]
                  if c in auto_features]
    # Seed on first render so the default full range does not auto-select;
    # apply automatically only when the user changes a boundary.
    if "_last_range_num" not in st.session_state:
        st.session_state["_last_range_num"] = (from_num, to_num)
    elif st.session_state["_last_range_num"] != (from_num, to_num):
        st.session_state["_last_range_num"] = (from_num, to_num)
        if range_cols:
            st.session_state["sel_features"] = range_cols
            st.rerun()
    num_preview = all_df_cols[from_num-1:to_num]
    st.caption(f"Columns {from_num}–{to_num}: "
               f"{', '.join(num_preview[:5])}{'…' if len(num_preview) > 5 else ''}")

# ── Option 2: Exclude by keyword ──────────────────────────────────────────
st.markdown("**Option 2 — Exclude columns whose name contains a keyword**")
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
    import re as _re2
    keywords     = [k.strip().lower() for k in excl_input.split(",") if k.strip()]
    would_remove = [c for c in st.session_state.get("sel_features", [])
                    if any(kw in c.lower() for kw in keywords)]

    def _highlight(name, kws):
        """Wrap each matching keyword in the column name with a yellow highlight."""
        result = name
        for kw in kws:
            pattern = _re2.compile(_re2.escape(kw), _re2.IGNORECASE)
            result  = pattern.sub(
                lambda m: f'<mark style="background:#FFD700;padding:0 2px;'
                          f'border-radius:2px;font-weight:bold">{m.group()}</mark>',
                result,
            )
        return result

    if would_remove:
        st.warning(f"**{len(would_remove)} column{'s' if len(would_remove) > 1 else ''} "
                   f"match{'es' if len(would_remove) == 1 else ''} and would be removed:**")
        rm_rows = [would_remove[i:i+3] for i in range(0, len(would_remove), 3)]
        for row in rm_rows:
            grid = st.columns(3)
            for j, name in enumerate(row):
                highlighted = _highlight(name, keywords)
                grid[j].markdown(
                    f'<span style="font-family:monospace;font-size:0.88em">{highlighted}</span>',
                    unsafe_allow_html=True,
                )
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
  try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import numpy as np

    # columns to display: covariates first, then imaging features
    # categorical variables are label-encoded so they can be z-scored
    covariate_cols = [c for c in [age_col, sex_col] + extra_continuous + extra_categorical
                      if c in df.columns]
    all_plot_cols  = covariate_cols + feature_cols

    plot_df = df[[site_col] + all_plot_cols].copy()

    # encode any categorical / string columns to numeric
    for col in all_plot_cols:
        if not pd.api.types.is_numeric_dtype(plot_df[col]):
            uniq = sorted(plot_df[col].dropna().astype(str).unique())
            mapping = {v: i for i, v in enumerate(uniq)}
            plot_df[col] = plot_df[col].astype(str).map(mapping)

    plot_df = plot_df.sort_values(site_col).reset_index(drop=True)

    # z-score every column
    mat      = plot_df[all_plot_cols].values.astype(float)
    col_means = np.nanmean(mat, axis=0)
    col_sds   = np.nanstd(mat, axis=0)
    col_sds[col_sds == 0] = 1
    z = (mat - col_means) / col_sds

    # site color mapping — same palette as ICC by-site plot
    sites_ordered  = np.array(plot_df[site_col].astype(str))
    sites_unique   = sorted(set(sites_ordered))
    palette        = px.colors.qualitative.Safe
    site_color_map = {s: palette[i % len(palette)] for i, s in enumerate(sites_unique)}

    # site strip: numeric index per participant row
    site_idx = np.array([[sites_unique.index(s)] for s in sites_ordered], dtype=float)

    # site boundaries for horizontal lines
    boundary_y = [i - 0.5 for i in range(1, len(sites_ordered))
                  if sites_ordered[i] != sites_ordered[i-1]]

    # sample size per site
    site_n_map = plot_df.groupby(site_col).size().to_dict()

    # y-tick labels: site name + n at midpoint of each block
    ytick_vals, ytick_text, prev, start = [], [], sites_ordered[0], 0
    for i, s in enumerate(sites_ordered):
        if s != prev:
            ytick_vals.append((start + i - 1) / 2)
            ytick_text.append(f"{prev} (n={site_n_map.get(prev, '?')})")
            prev, start = s, i
    ytick_vals.append((start + len(sites_ordered) - 1) / 2)
    ytick_text.append(f"{prev} (n={site_n_map.get(prev, '?')})")

    # discrete colorscale for site strip — each site occupies 1/n_sites of [0,1]
    n_sites = len(sites_unique)
    site_cs = []
    for i, s in enumerate(sites_unique):
        lo = round(i / n_sites, 8)
        hi = round(min((i + 1) / n_sites, 1.0), 8)
        site_cs.append([lo, site_color_map[s]])
        site_cs.append([hi, site_color_map[s]])

    # subplots: thin site strip | main heatmap
    fig_carpet = make_subplots(
        rows=1, cols=2,
        column_widths=[0.025, 0.975],
        shared_yaxes=True,
        horizontal_spacing=0.003,
    )

    # ── Site color strip (left) ──
    site_n_col = np.array([site_n_map.get(s, '?') for s in sites_ordered])
    fig_carpet.add_trace(go.Heatmap(
        z=site_idx,
        colorscale=site_cs,
        zmin=0, zmax=n_sites - 1,
        showscale=False,
        hovertemplate="Site: %{customdata[0]}<br>n = %{customdata[1]}<extra></extra>",
        customdata=np.column_stack([sites_ordered, site_n_col]),
    ), row=1, col=1)

    # ── Main data heatmap (right) ──
    fig_carpet.add_trace(go.Heatmap(
        z=z,
        x=all_plot_cols,
        colorscale="RdBu_r",
        zmid=0, zmin=-3, zmax=3,
        colorbar=dict(title="z-score", thickness=12, x=1.01),
        hoverongaps=False,
        hovertemplate="Column: %{x}<br>z: %{z:.2f}<extra></extra>",
    ), row=1, col=2)

    # vertical separator between covariate columns and imaging features
    if covariate_cols:
        sep_x = len(covariate_cols) - 0.5
        fig_carpet.add_vline(x=sep_x, line_color="black",
                             line_width=2, opacity=0.7, row=1, col=2)

    # grey squares for missing values
    nan_rows, nan_cols_idx = np.where(np.isnan(z))
    if len(nan_rows) > 0:
        fig_carpet.add_trace(go.Scatter(
            x=[all_plot_cols[c] for c in nan_cols_idx],
            y=nan_rows.tolist(),
            mode="markers",
            marker=dict(color="lightgrey", size=2, symbol="square"),
            name="Missing",
            showlegend=True,
            xaxis="x2", yaxis="y",
        ), row=1, col=2)

    # site boundary lines on both panels
    for by in boundary_y:
        for col_i in [1, 2]:
            fig_carpet.add_hline(y=by, line_color="black",
                                  line_width=1.0, opacity=0.6, row=1, col=col_i)

    n_feat  = len(all_plot_cols)
    height  = max(400, min(len(plot_df) * 4, 900))
    covar_label = f" | first {len(covariate_cols)} columns = covariates (Age, Sex, ...)" if covariate_cols else ""
    fig_carpet.update_layout(
        title=f"Raw data matrix — {len(plot_df)} participants × {n_feat} columns "
              f"(sorted by {site_col}; z-scored){covar_label}",
        xaxis=dict(showticklabels=False, showgrid=False),
        xaxis2=dict(showticklabels=n_feat <= 80, tickangle=45,
                    tickfont=dict(size=7)),
        yaxis=dict(tickmode="array", tickvals=ytick_vals, ticktext=ytick_text,
                   tickfont=dict(size=9), autorange="reversed"),
        height=height,
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", family="Arial"),
        margin=dict(l=110, b=90, r=60),
    )

    pct_missing = 100 * np.isnan(mat).sum() / mat.size
    st.plotly_chart(fig_carpet, use_container_width=True)
    st.caption(f"Left strip = site identity (colored). "
               f"Main panel = z-scored feature values (red = high, blue = low, grey = missing). "
               f"Missing: {np.isnan(mat).sum():,} cells ({pct_missing:.1f}%). "
               f"Horizontal lines = site boundaries.")
  except Exception as _preview_err:
    st.error(f"Preview could not be generated: {_preview_err}")

# ── Data summary table ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Sample summary table (optional)**")
st.caption(
    "Builds a descriptive table of your sample broken down by site, so you can "
    "check how participants are distributed across sites before harmonizing "
    "(for example, whether age or sex is balanced across sites). It also serves "
    "as a sample-characteristics table for a manuscript. "
    "There is one row per site, plus an Overall row, and a participant count (N) per row. "
    "Choose which variables to describe below: numeric variables are summarized as "
    "mean (standard deviation) and text or category variables as counts and percentages. "
    "Any column from your uploaded file can be added, for example age, sex, days since injury, or scanner."
)

# Variable candidates: everything that is not a selected imaging feature
summary_candidates = [c for c in df.columns if c not in set(feature_cols)]
default_summary_vars = [c for c in [age_col, sex_col] if c in summary_candidates]

sum_c1, sum_c2 = st.columns([4, 1])
with sum_c1:
    summary_vars = st.multiselect(
        "Variables to describe by site (each becomes a column in the table)",
        options=summary_candidates,
        default=default_summary_vars,
        key="summary_vars",
    )
with sum_c2:
    st.markdown("<br>", unsafe_allow_html=True)
    gen_summary = st.button("Generate summary", use_container_width=True)

if gen_summary and summary_vars:
    try:
        sites_list = sorted(df[site_col].dropna().unique().astype(str))
        rows = []
        for grp_label in sites_list + ["Overall"]:
            sub = df if grp_label == "Overall" else df[df[site_col].astype(str) == grp_label]
            row = {"Site": grp_label, "N": len(sub)}
            for var in summary_vars:
                col_data = sub[var].dropna()
                if len(col_data) == 0:
                    row[var] = "—"
                elif pd.api.types.is_numeric_dtype(df[var]):
                    row[var] = f"{col_data.mean():.2f} ({col_data.std():.2f})"
                else:
                    vc = col_data.astype(str).value_counts()
                    total = len(col_data)
                    parts = [f"{v}: {n} ({100*n/total:.0f}%)"
                             for v, n in vc.head(6).items()]
                    row[var] = "  |  ".join(parts)
            rows.append(row)

        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.caption("Continuous: mean (SD).  Categorical: n (%) per category.")
    except Exception as _sum_err:
        st.error(f"Summary could not be generated: {_sum_err}")

st.divider()
st.subheader("Site exclusion (optional)")
st.caption(
    "Exclude sites before harmonization — for example sites that are too small "
    "or whose distribution differs substantially from the rest."
)
all_sites_n = df[site_col].value_counts().sort_index()
site_options = [f"{s}  (n={all_sites_n[s]})" for s in all_sites_n.index]
site_label_to_name = {f"{s}  (n={all_sites_n[s]})": s for s in all_sites_n.index}
excluded_labels = st.multiselect(
    "Sites to exclude",
    options=site_options,
    default=[],
    key="excluded_sites",
    placeholder="None — all sites included",
)
excluded_sites = [site_label_to_name[l] for l in excluded_labels]
if excluded_sites:
    st.warning(
        f"Excluding {len(excluded_sites)} site(s): {', '.join(str(s) for s in excluded_sites)}. "
        f"Remaining participants: {int((~df[site_col].isin(excluded_sites)).sum())}."
    )

st.divider()
st.subheader("Evaluation and report options")
st.caption(
    "Select which metrics to compute and include in the supplementary report. "
    "Within-site consistency by site (ICC3) is the primary recommended metric: it directly measures "
    "whether harmonization preserved the internal variability of each site's data (i.e., that changing "
    "site means and scales did not distort within-site biological signal). "
    "Age correlations are optional because they may be confounded by motion in pediatric fMRI samples, "
    "or may be unstable in restricted age ranges where developmental trajectories are non-linear."
)

eval_c1, eval_c2 = st.columns(2)
with eval_c1:
    include_icc_by_site = st.checkbox(
        "Within-site consistency by site (ICC3) — recommended",
        value=True, key="inc_icc_site",
        help=(
            "ICC(C,1) between raw and harmonized values computed within each site. "
            "This is the primary metric for assessing whether harmonization preserved within-site "
            "biological variability. Minimum 3 participants per site required (pingouin constraint)."
        ),
    )
    include_cohens_f = st.checkbox(
        "Site effect size (ANOVA Cohen's f) — recommended",
        value=True, key="inc_cohens_f",
        help=(
            "ANCOVA with site as grouping factor and age + sex as covariates (Type II sums of squares). "
            "Quantifies the magnitude of residual site-related variance before and after harmonization."
        ),
    )
with eval_c2:
    include_age_corr = st.checkbox(
        "Age correlations (Pearson r, FDR corrected)",
        value=False, key="inc_age_corr",
        help=(
            "Pearson correlation between the age variable and each imaging feature, before and after "
            "harmonization. Optional: not all datasets have a meaningful age variable, and age "
            "correlations can be confounded by motion or unstable in restricted age ranges."
        ),
    )
    include_extra_assoc = st.checkbox(
        "Additional variable associations",
        value=False, key="inc_extra_assoc",
        help=(
            "Evaluate associations between additional clinical or demographic variables and imaging "
            "features before and after harmonization. Uses OLS regression controlling for the "
            "same covariates passed to ComBat. Continuous variables: Pearson r + regression "
            "t-statistic. Categorical variables: partial eta-squared and Cohen's f."
        ),
    )

assoc_cont_vars, assoc_cat_vars = [], []
if include_extra_assoc:
    ea_c1, ea_c2 = st.columns(2)
    with ea_c1:
        assoc_cont_vars = st.multiselect(
            "Continuous variables to evaluate",
            options=[c for c in num_cols if c not in set(feature_cols) | {site_col}],
            key="assoc_cont",
            placeholder="e.g. days since injury, TSI, GCS score ...",
        )
    with ea_c2:
        assoc_cat_vars = st.multiselect(
            "Categorical variables to evaluate",
            options=[c for c in all_cols if c not in set(feature_cols) | {site_col}],
            key="assoc_cat",
            placeholder="e.g. Group, Diagnosis, Injury severity ...",
        )

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
        # ── Apply site exclusion ────────────────────────────────────────────
        if excluded_sites:
            df_harm = df[~df[site_col].isin(excluded_sites)].copy().reset_index(drop=True)
        else:
            df_harm = df

        # ── Harmonize ──────────────────────────────────────────────────────
        harm_ebt, harm_ebf = None, None

        if run_ebt:
            progress.progress(5, "Running ComBat (EB=TRUE)...")
            harm_ebt = run_combat(df_harm, feature_cols, site_col,
                                   continuous_covariates=continuous_covariates,
                                   categorical_covariates=categorical_covariates,
                                   eb=True)

        if run_ebf:
            progress.progress(20, "Running ComBat (EB=FALSE)...")
            harm_ebf = run_combat(df_harm, feature_cols, site_col,
                                   continuous_covariates=continuous_covariates,
                                   categorical_covariates=categorical_covariates,
                                   eb=False)

        # primary result for site deviation (before panel always uses raw)
        harm_primary = harm_ebt if harm_ebt is not None else harm_ebf

        # ── Metrics ────────────────────────────────────────────────────────
        progress.progress(33, "Computing site deviation...")
        dev_before = site_mean_deviation(df_harm,     feature_cols, site_col)
        dev_ebt    = site_mean_deviation(harm_ebt,    feature_cols, site_col) if harm_ebt is not None else None
        dev_ebf    = site_mean_deviation(harm_ebf,    feature_cols, site_col) if harm_ebf is not None else None

        spm_ebt, spm_ebf = None, None  # Spearman removed; ICC3 is sufficient for consistency

        progress.progress(50, "Computing ICC (overall)...")
        icc_parts = []
        if harm_ebt is not None:
            ic = compute_icc(df_harm, harm_ebt, feature_cols); ic["harmonization"] = "EB=TRUE";  icc_parts.append(ic)
        if harm_ebf is not None:
            ic = compute_icc(df_harm, harm_ebf, feature_cols); ic["harmonization"] = "EB=FALSE"; icc_parts.append(ic)
        icc_all = pd.concat(icc_parts, ignore_index=True) if icc_parts else pd.DataFrame()
        icc_ebt = icc_parts[0] if run_ebt and icc_parts else None
        icc_ebf = icc_parts[-1] if run_ebf and len(icc_parts) > (1 if run_ebt else 0) else (icc_parts[0] if not run_ebt and icc_parts else None)

        icc_site_ebt = icc_site_ebf = None
        if include_icc_by_site:
            progress.progress(67, "Computing ICC (by site)...")
            icc_site_ebt = compute_icc_by_site(df_harm, harm_ebt, feature_cols, site_col) if harm_ebt is not None else None
            icc_site_ebf = compute_icc_by_site(df_harm, harm_ebf, feature_cols, site_col) if harm_ebf is not None else None

        age_all = pd.DataFrame()
        if include_age_corr:
            progress.progress(74, "Computing age correlations...")
            age_parts = [age_correlations(df_harm, feature_cols, age_col, "Before harmonization")]
            if harm_ebt is not None: age_parts.append(age_correlations(harm_ebt, feature_cols, age_col, "After (EB=TRUE)"))
            if harm_ebf is not None: age_parts.append(age_correlations(harm_ebf, feature_cols, age_col, "After (EB=FALSE)"))
            age_all = pd.concat(age_parts, ignore_index=True)

        anc_before = anc_ebt = anc_ebf = None
        if include_cohens_f:
            progress.progress(80, "Computing ANCOVA site effects...")
            anc_before = ancova_site_effect(df_harm, feature_cols, site_col, age_col, sex_col, "Before")
            anc_ebt    = ancova_site_effect(harm_ebt, feature_cols, site_col, age_col, sex_col, "EB=TRUE")  if harm_ebt is not None else None
            anc_ebf    = ancova_site_effect(harm_ebf, feature_cols, site_col, age_col, sex_col, "EB=FALSE") if harm_ebf is not None else None

        extra_assoc_df = pd.DataFrame()
        if include_extra_assoc and (assoc_cont_vars or assoc_cat_vars):
            progress.progress(82, "Computing additional variable associations...")
            assoc_parts = [compute_extra_associations(
                df_harm, feature_cols, assoc_cont_vars, assoc_cat_vars,
                continuous_covariates, categorical_covariates, "Before harmonization")]
            if harm_ebt is not None:
                assoc_parts.append(compute_extra_associations(
                    harm_ebt, feature_cols, assoc_cont_vars, assoc_cat_vars,
                    continuous_covariates, categorical_covariates, "After (EB=TRUE)"))
            if harm_ebf is not None:
                assoc_parts.append(compute_extra_associations(
                    harm_ebf, feature_cols, assoc_cont_vars, assoc_cat_vars,
                    continuous_covariates, categorical_covariates, "After (EB=FALSE)"))
            non_empty = [p for p in assoc_parts if len(p) > 0]
            extra_assoc_df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

        # ── Figures ────────────────────────────────────────────────────────
        progress.progress(87, "Generating figures...")
        site_n = df_harm[site_col].value_counts().to_dict()
        fig_site = plot_site_deviation(dev_before,
                                       dev_ebt if dev_ebt is not None else pd.DataFrame(),
                                       dev_ebf,
                                       site_n=site_n)

        fig_anc = None
        if include_cohens_f and anc_before is not None:
            fig_anc = plot_cohens_f(anc_before,
                                    anc_ebt if anc_ebt is not None else pd.DataFrame(),
                                    anc_ebf)

        fig_icc = plot_icc(icc_all) if len(icc_all) > 0 else None

        fig_age = None
        if include_age_corr and len(age_all) > 0:
            fig_age = plot_age_correlations(age_all)

        site_n = df.groupby(site_col)[site_col].count().to_dict()
        site_n = {str(k): int(v) for k, v in site_n.items()}

        _have_site_icc = ((icc_site_ebt is not None and len(icc_site_ebt) > 0)
                          or (icc_site_ebf is not None and len(icc_site_ebf) > 0))
        fig_icc_site = plot_icc_by_site(
            icc_site_ebt if icc_site_ebt is not None else pd.DataFrame(),
            icc_site_ebf,
            site_n=site_n,
        ) if _have_site_icc else None

        fig_extra_assoc = None
        if include_extra_assoc and len(extra_assoc_df) > 0:
            fig_extra_assoc = plot_extra_associations(extra_assoc_df)

        # ── Report ─────────────────────────────────────────────────────────
        progress.progress(94, "Building report...")
        demo_df    = demographic_summary(df_harm, site_col, age_col, sex_col)
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
            extra_continuous=continuous_covariates,
            extra_categorical=categorical_covariates,
            include_age=include_age_corr,
            include_cohens_f=include_cohens_f,
            include_icc_by_site=include_icc_by_site,
            include_extra_assoc=(include_extra_assoc and len(extra_assoc_df) > 0),
            extra_assoc_df=extra_assoc_df if len(extra_assoc_df) > 0 else None,
            fig_extra_assoc=fig_extra_assoc,
            assoc_cont_vars=assoc_cont_vars,
            assoc_cat_vars=assoc_cat_vars,
        )

        methods_para = build_methods_paragraph(
            site_col=site_col, age_col=age_col, sex_col=sex_col,
            run_ebf=(run_ebf and harm_ebf is not None),
            github_url=GITHUB_URL,
            extra_continuous=continuous_covariates,
            extra_categorical=categorical_covariates,
            include_age=include_age_corr,
            include_cohens_f=include_cohens_f,
            include_icc_by_site=include_icc_by_site,
            include_extra_assoc=(include_extra_assoc and len(extra_assoc_df) > 0),
            assoc_cont_vars=assoc_cont_vars,
            assoc_cat_vars=assoc_cat_vars,
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
            fig_extra_assoc=fig_extra_assoc,
            html_report=html_report,
            icc_ebt=icc_ebt, icc_ebf=icc_ebf,
            run_ebt=run_ebt, run_ebf=run_ebf,
            harm_csv_ebt=harm_csv_ebt,
            harm_csv_ebf=harm_csv_ebf,
            include_cohens_f=include_cohens_f,
            include_icc_by_site=include_icc_by_site,
            include_age_corr=include_age_corr,
            include_extra_assoc=include_extra_assoc,
            extra_assoc_df=extra_assoc_df,
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
    _inc_cohens_f   = st.session_state.get("include_cohens_f",   True)
    _inc_icc_site   = st.session_state.get("include_icc_by_site", True)
    _inc_age        = st.session_state.get("include_age_corr",    False)
    _inc_extra      = st.session_state.get("include_extra_assoc", False)
    _extra_assoc_df = st.session_state.get("extra_assoc_df", pd.DataFrame())

    # Build tab list dynamically based on selected metrics
    tab_names = ["Site Deviation"]
    if _inc_cohens_f:
        tab_names.append("Site Effect Size (Cohen's f)")
    if _inc_icc_site:
        tab_names.append("Within-Site Consistency — By Site")
    if _inc_age:
        tab_names.append("Age Associations")
    if _inc_extra and isinstance(_extra_assoc_df, pd.DataFrame) and len(_extra_assoc_df) > 0:
        tab_names.append("Variable Associations")
    tab_names.append("Methods paragraph")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    with tabs[tab_idx]:
        st.plotly_chart(st.session_state["fig_site"], use_container_width=True)
        st.caption("Each point = one (site, feature) pair. Site means should cluster near zero after harmonization.")
    tab_idx += 1

    if _inc_cohens_f:
        with tabs[tab_idx]:
            if st.session_state.get("fig_anc"):
                st.plotly_chart(st.session_state["fig_anc"], use_container_width=True)
                st.caption("Each point = one feature. Site effect from ANCOVA Type II (Age + Sex as covariates); significance is the uncorrected p-value (p < 0.05), not corrected for multiple comparisons. Colour encodes the after-harmonization status (red = p < 0.05, green = p >= 0.05) and fill encodes the before-harmonization status (filled = p < 0.05, open = p >= 0.05), as summarised in the 2 x 2 legend. Points below the diagonal = reduced site effect size.")
            else:
                st.info("Cohen's f not computed for this run.")
        tab_idx += 1

    if _inc_icc_site:
        with tabs[tab_idx]:
            if st.session_state.get("fig_icc_site"):
                st.plotly_chart(st.session_state["fig_icc_site"], use_container_width=True)
                st.caption(
                    "Each box = distribution of ICC3 across features for that site. "
                    "Within-site ICC is the primary recommended metric: it measures whether harmonization "
                    "preserved the rank ordering of participants within each site, indicating that "
                    "within-site biological variability was not distorted by batch correction. "
                    "Sites with fewer participants may show lower consistency, particularly without EB. "
                    "Minimum 3 participants per site required. "
                    "Colored bands: Poor / Moderate / Good / Excellent (Koo & Li, 2016)."
                )
            else:
                st.info("By-site ICC not available (need at least 3 participants per site).")
        tab_idx += 1

    if _inc_age:
        with tabs[tab_idx]:
            if st.session_state.get("fig_age"):
                st.plotly_chart(st.session_state["fig_age"], use_container_width=True)
                st.caption("Each point = one feature. Grey circle = not significant either condition | Filled circle = FDR significant after only | Orange diamond = FDR significant before only | Purple square = FDR significant in both. Pearson r computed independently per condition; no formal test of difference applied.")
            else:
                st.info("Age correlations not available.")
        tab_idx += 1

    if _inc_extra and isinstance(_extra_assoc_df, pd.DataFrame) and len(_extra_assoc_df) > 0:
        with tabs[tab_idx]:
            if st.session_state.get("fig_extra_assoc"):
                st.plotly_chart(st.session_state["fig_extra_assoc"], use_container_width=True)
                st.caption(
                    "Each point = one feature. Effect size before (x-axis) vs after (y-axis) harmonization. "
                    "Continuous variables: Pearson r. Categorical variables: Cohen's f (from OLS ANOVA Type II). "
                    "All models control for the same covariates passed to ComBat (excluding the variable itself). "
                    "FDR correction (Benjamini-Hochberg) applied per variable across features."
                )
            else:
                st.info("No extra variable associations computed.")
        tab_idx += 1

    with tabs[tab_idx]:
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
            label="⬇  Report (HTML)",
            data=st.session_state["html_report"],
            file_name="harmonization_report.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
        )
        st.caption("Interactive figures")
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
