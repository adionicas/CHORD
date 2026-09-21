"""
CHORD — Multisite Harmonization Assessment
Streamlit web application
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.harmonize import (run_combat, run_combat_grouped, run_combat_per_feature,
                           run_combat_by_missing_pattern, run_combat_with_imputation)
from src.metrics   import (site_mean_deviation, spearman_raw_vs_harm,
                            compute_icc, age_correlations, ancova_site_effect,
                            demographic_summary,
                            compute_icc_by_site, compute_spearman_by_site,
                            compute_extra_associations)
from src.plots     import (plot_site_deviation, plot_spearman, plot_icc,
                           plot_age_correlations, plot_cohens_f,
                           plot_icc_by_site, plot_spearman_by_site,
                           plot_extra_associations, plot_single_association,
                           plot_icc_ci_compare)
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

Input: a table of imaging features with batch, age, and sex columns. Output: the harmonized table and an evaluation report.

**Recommended metrics (default ON):**
- Batch mean deviation (z-score), always included
- Within-batch consistency by batch (ICC3), primary recommended metric
- Batch effect size (ANCOVA, Cohen's f)

**Optional metrics:**
- Age associations (Pearson r, FDR-corrected). Optional: can be confounded by head motion, and are uninformative when the age range is narrow
- Additional variable associations (OLS) — e.g. injury severity, time since injury

**Why ICC by batch is the primary metric:**
Within-batch ICC measures whether harmonization preserved the internal variability of each batch's data. A high ICC means the rank ordering of participants is intact after batch correction, that is, changing batch means and scales did not distort within-batch biological variability.

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
    if st.button("Load synthetic example data (215 participants, 6 batches, 20 FA features)"):
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
        st.info("Loaded synthetic example dataset (215 participants, 6 batches, 20 FA features, not real patient data)")
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
    "of interest, for example Age and Sex, while removing batch-related variance. "
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
        f"ComBat model: batch = **{site_col}** only, no covariates. "
        "Batch effects will be removed but no biological variability is explicitly preserved."
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
st.info(f"Batches: **{', '.join(sorted(sites.astype(str)))}** ({len(sites)} batches) | Features: **{len(feature_cols)}**")

st.divider()
st.subheader("Step 2c. Multiple imaging modalities or measure types (optional)")

st.caption(
    "Different imaging modalities or different measures (for example cortical "
    "thickness, surface area, mean diffusivity, sulcal depth) do not necessarily "
    "share the same distribution or numeric scale. ComBat's empirical Bayes step "
    "pools information across features to estimate each batch's location and scale "
    "parameters, which assumes the pooled features are comparably distributed. "
    "When the selected features span measures on different scales, estimating "
    "those parameters separately for each modality or measure keeps the pooled "
    "features exchangeable and avoids distorting the empirical Bayes priors "
    "(Johnson et al., 2007; Fortin et al., 2017, 2018). If enabled below, each "
    "group is harmonized in its own ComBat run, with the same batch variable and "
    "the same covariates, and the harmonized blocks are recombined per participant."
)


def _suggest_modality_tokens(cols):
    """Propose candidate grouping tokens from the selected feature names.

    First tries the text after the last '.' (common for surface / measure
    naming such as ``L_S_central.meandepth_native``); if that partitions every
    column, those suffixes are suggested. Otherwise falls back to the prefix
    before the first '_'.
    """
    suff = {}
    for c in cols:
        if "." in c:
            key = c.rsplit(".", 1)[1]
            suff[key] = suff.get(key, 0) + 1
    if len(suff) >= 2 and sum(suff.values()) == len(cols):
        return sorted(suff)
    pref = {}
    for c in cols:
        if "_" in c:
            key = c.split("_", 1)[0] + "_"
            pref[key] = pref.get(key, 0) + 1
    if len(pref) >= 2:
        return sorted(pref)
    return sorted(suff)


modality_groups = None
use_modality = st.checkbox(
    "My selected features include multiple modalities or measure types. "
    "Harmonize each group separately",
    value=False, key="use_modality_groups",
)

if use_modality:
    st.markdown(
        "Define one group per modality or measure type using a text token that "
        "appears in that group's column names (for example a suffix such as "
        "`meandepth_native`, or a prefix such as `FA_`). Enter one token per line. "
        "Each selected feature is assigned to the first group whose token it "
        "contains."
    )
    if st.button("Suggest groups from column names"):
        st.session_state["modality_tokens_text"] = "\n".join(
            _suggest_modality_tokens(feature_cols)
        )
        st.rerun()
    tokens_text = st.text_area(
        "One token per line",
        key="modality_tokens_text",
        placeholder="hull_junction_length_native\nmeandepth_native\nsurface_native\nopening",
        height=130,
    )
    tokens = [t.strip() for t in tokens_text.splitlines() if t.strip()]
    if tokens:
        groups, assigned = {}, set()
        for t in tokens:
            matched = [c for c in feature_cols
                       if t.lower() in c.lower() and c not in assigned]
            if matched:
                groups[t] = matched
                assigned.update(matched)
        unassigned = [c for c in feature_cols if c not in assigned]

        if groups:
            preview = pd.DataFrame([
                {"Group (token)": g, "Columns matched": len(cols)}
                for g, cols in groups.items()
            ])
            st.dataframe(preview, hide_index=True, use_container_width=True)
            with st.expander("View all columns in each group"):
                for g, cols in groups.items():
                    st.markdown(f"**{g}** ({len(cols)} columns)")
                    st.write(", ".join(cols))

        if unassigned:
            st.warning(
                f"{len(unassigned)} selected feature(s) match no token and would "
                "not be harmonized. Add a token that covers them, or remove them "
                "from the feature selection above."
            )
            with st.expander(f"View {len(unassigned)} unassigned column(s)"):
                st.write(unassigned)

        if len(groups) >= 2 and not unassigned:
            modality_groups = groups
            st.success(
                f"{len(groups)} groups defined, covering all {len(feature_cols)} "
                "selected features. Each group will be harmonized in its own "
                "ComBat run."
            )
        elif len(groups) < 2:
            st.info(
                "Define at least two groups to harmonize by modality. With fewer "
                "than two groups, harmonization runs over all features together."
            )
        else:
            st.info(
                "Resolve the unassigned columns above to enable grouped "
                "harmonization."
            )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Data preview (optional)")
st.caption("Visual overview of the selected features before harmonization. "
           "Values are z-scored per feature; missing data shown in black. "
           "Participants sorted by batch so batch effects are visible.")

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
    # when modality grouping is used, order features so each modality is contiguous
    if modality_groups:
        ordered_feats = [c for g, cols in modality_groups.items() for c in cols]
    else:
        ordered_feats = list(feature_cols)
    all_plot_cols  = covariate_cols + ordered_feats

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
        hovertemplate="Batch: %{customdata[0]}<br>n = %{customdata[1]}<extra></extra>",
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

    # ── Modality color strip, separators, and labels (when grouping is used) ──
    if modality_groups:
        mod_palette = px.colors.qualitative.Bold
        mod_names   = list(modality_groups.keys())
        mod_color   = {g: mod_palette[i % len(mod_palette)] for i, g in enumerate(mod_names)}
        x0 = len(covariate_cols)
        for gi, gname in enumerate(mod_names):
            x1 = x0 + len(modality_groups[gname])
            if gi > 0:  # separator between modality blocks
                fig_carpet.add_vline(x=x0 - 0.5, line_color="black",
                                     line_width=1.5, opacity=0.85, row=1, col=2)
            fig_carpet.add_shape(  # colored band above the block
                type="rect", xref="x2", yref="paper",
                x0=x0 - 0.5, x1=x1 - 0.5, y0=1.002, y1=1.028,
                fillcolor=mod_color[gname], line_width=0, layer="above",
            )
            fig_carpet.add_annotation(  # modality label above the block
                xref="x2", yref="paper", x=(x0 + x1 - 1) / 2, y=1.035,
                text=gname, showarrow=False, xanchor="center", yanchor="bottom",
                font=dict(size=9, color=mod_color[gname]),
            )
            x0 = x1

    # black cells for missing values — fills each missing cell so missingness
    # is clearly visible against the red/blue data
    miss_layer = np.where(np.isnan(z), 1.0, np.nan)
    if np.isnan(z).any():
        fig_carpet.add_trace(go.Heatmap(
            z=miss_layer,
            x=all_plot_cols,
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False,
            hoverongaps=False,
            hovertemplate="Column: %{x}<br>missing<extra></extra>",
        ), row=1, col=2)

    # site boundary lines on both panels
    for by in boundary_y:
        for col_i in [1, 2]:
            fig_carpet.add_hline(y=by, line_color="black",
                                  line_width=1.0, opacity=0.6, row=1, col=col_i)

    n_feat  = len(all_plot_cols)
    height  = max(400, min(len(plot_df) * 4, 900))
    covar_label = f" | first {len(covariate_cols)} columns = covariates (Age, Sex, ...)" if covariate_cols else ""
    _title_text = (f"Raw data matrix — {len(plot_df)} participants × {n_feat} columns "
                   f"(sorted by {site_col}; z-scored){covar_label}")
    _title = (dict(text=_title_text, y=0.995, yanchor="top")
              if modality_groups else _title_text)
    fig_carpet.update_layout(
        title=_title,
        xaxis=dict(showticklabels=False, showgrid=False),
        xaxis2=dict(showticklabels=n_feat <= 80, tickangle=45,
                    tickfont=dict(size=7)),
        yaxis=dict(tickmode="array", tickvals=ytick_vals, ticktext=ytick_text,
                   tickfont=dict(size=9), autorange="reversed"),
        height=height,
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", family="Arial"),
        margin={"l": 110, "b": 90, "r": 60, **({"t": 70} if modality_groups else {})},
    )

    pct_missing = 100 * np.isnan(mat).sum() / mat.size
    mod_note = ("Top strip = modality group (colored, separated by vertical lines). "
                if modality_groups else "")
    st.plotly_chart(fig_carpet, use_container_width=True)
    st.caption(f"Left strip = batch identity (colored). {mod_note}"
               f"Main panel = z-scored feature values (red = high, blue = low, black = missing). "
               f"Missing: {np.isnan(mat).sum():,} cells ({pct_missing:.1f}%). "
               f"Horizontal lines = batch boundaries.")
  except Exception as _preview_err:
    st.error(f"Preview could not be generated: {_preview_err}")

# ── Missing value assessment ────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Missing value assessment (optional)**")
st.caption(
    "Counts missing values per variable among the selected features and the "
    "covariates. neuroCombat requires complete data within each ComBat run, so a "
    "participant missing a value is excluded from that variable's run. Use this "
    "to see which variables drive the missingness."
)

# ── Model-variable (demographic) missingness — always shown ──
# A participant missing the site, age, sex, or any added covariate is dropped
# from EVERY ComBat run, whatever the feature-missing method chosen at Step 3.
_model_vars = [c for c in [site_col, age_col, sex_col] + extra_continuous + extra_categorical
               if c in df.columns]
_model_miss = df[_model_vars].isna().sum()
_n_model_incomplete = int(df[_model_vars].isna().any(axis=1).sum())
if _n_model_incomplete > 0:
    st.warning(
        f"{_n_model_incomplete} of {len(df)} participants are missing at least one "
        f"model variable (batch, age, sex, or an added covariate) and are dropped "
        f"from every ComBat run, regardless of the feature-missing method below."
    )
    _mm = pd.DataFrame({
        "Model variable": _model_vars,
        "Missing (n)":    _model_miss.values,
        "Missing (%)":    (100 * _model_miss / len(df)).round(1).values,
    })
    _mm = _mm[_mm["Missing (n)"] > 0].sort_values("Missing (n)", ascending=False)
    st.dataframe(_mm, use_container_width=True, hide_index=True)
else:
    st.caption("All participants have complete batch, age, sex, and added covariates.")

if st.button("Assess missing values"):
    try:
        assess_cols = [c for c in [age_col, sex_col] + extra_continuous + extra_categorical
                       if c in df.columns] + list(feature_cols)
        n_total = len(df)
        miss_n = df[assess_cols].isna().sum()
        miss_pct = (100 * miss_n / n_total) if n_total else miss_n * 0
        n_complete_feats = int(df[list(feature_cols)].notna().all(axis=1).sum())
        total_missing_cells = int(df[list(feature_cols)].isna().sum().sum())
        n_cols_with_missing = int((df[list(feature_cols)].isna().sum() > 0).sum())

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Participants", f"{n_total}")
        mc2.metric("Complete on all features", f"{n_complete_feats}")
        mc3.metric("Feature columns with missing", f"{n_cols_with_missing} / {len(feature_cols)}")
        mc4.metric("Missing feature cells", f"{total_missing_cells:,}")

        _cov_set = set([age_col, sex_col] + extra_continuous + extra_categorical)
        def _grp_of(c):
            if c in _cov_set:
                return "(covariate)"
            if modality_groups:
                for g, gcols in modality_groups.items():
                    if c in gcols:
                        return g
            if "." in c:
                return c.rsplit(".", 1)[1]
            if "_" in c:
                return c.split("_", 1)[0] + "_"
            return "(feature)"

        miss_df = pd.DataFrame({
            "Variable":      assess_cols,
            "Measure group": [_grp_of(c) for c in assess_cols],
            "Missing (n)":   miss_n.values,
            "Missing (%)":   miss_pct.round(1).values,
            "Present (n)":   (n_total - miss_n.values),
        })
        miss_df = miss_df[miss_df["Missing (n)"] > 0].sort_values(
            "Missing (n)", ascending=False).reset_index(drop=True)

        if len(miss_df) == 0:
            st.success("No missing values in the selected features or covariates.")
        else:
            st.caption(
                f"{len(miss_df)} of {len(assess_cols)} variables have at least one "
                f"missing value; the remaining {len(assess_cols) - len(miss_df)} are "
                "complete. Sorted by count, most missing first."
            )
            st.dataframe(miss_df, use_container_width=True, hide_index=True)
    except Exception as _miss_err:
        st.error(f"Missing-value assessment could not be generated: {_miss_err}")

# ── Exclude features by missing percentage ───────────────────────────────────
st.markdown("**Exclude features by missing percentage (optional)**")
st.caption(
    "Optionally set a threshold to list the selected features whose missing "
    "percentage exceeds it, then remove them from the selection in one step. The "
    "box is empty by default, so nothing is filtered until you enter a value. "
    "Missing percentage is computed per feature across all uploaded participants. "
    "When multiple modalities are used, the exceedances are broken down by modality."
)

def _exclude_high_missing_features(cols_to_drop, threshold, miss_pcts_dict=None):
    drop = set(cols_to_drop)
    st.session_state["sel_features"] = [
        c for c in st.session_state.get("sel_features", []) if c not in drop
    ]
    prev = st.session_state.get("_excl_info", {"n": 0, "threshold": None})
    st.session_state["_excl_info"] = {"n": prev["n"] + len(drop), "threshold": threshold}
    prev_list = st.session_state.get("_excl_feat_list", [])
    for feat in cols_to_drop:
        pct_val = miss_pcts_dict.get(feat, float("nan")) if miss_pcts_dict else float("nan")
        prev_list.append({"Feature": feat, "Missing (%)": round(float(pct_val), 1)})
    st.session_state["_excl_feat_list"] = prev_list

def _undo_all_exclusions():
    restored = [d.get("Feature") for d in st.session_state.get("_excl_feat_list", []) if d.get("Feature")]
    cur = st.session_state.get("sel_features", [])
    st.session_state["sel_features"] = cur + [f for f in restored if f not in cur]
    st.session_state["_excl_feat_list"] = []
    st.session_state["_excl_info"] = {"n": 0, "threshold": None}

# Persistent record of what has been excluded so far — stays visible after
# pressing Exclude (the over-threshold list below clears because those features
# are gone, but this record does not).
_excluded_so_far = st.session_state.get("_excl_feat_list", [])
if _excluded_so_far:
    def _grp_excl(c):
        if modality_groups:
            for g, gcols in modality_groups.items():
                if c in gcols:
                    return g
        if "." in c:
            return c.rsplit(".", 1)[1]
        if "_" in c:
            return c.split("_", 1)[0] + "_"
        return "(feature)"
    _ex_df = pd.DataFrame(_excluded_so_far)
    _ex_df["Measure group"] = _ex_df["Feature"].map(_grp_excl)
    _ex_df = _ex_df[["Feature", "Measure group", "Missing (%)"]]
    st.markdown(f"**Excluded so far: {len(_ex_df)} feature(s)** (removed from the selection)")
    st.dataframe(_ex_df, use_container_width=True, hide_index=True)
    st.button("Undo all exclusions (restore to selection)", on_click=_undo_all_exclusions)

thr_col = st.columns([1, 2])[0]
with thr_col:
    miss_thr = st.number_input(
        "Missing % threshold", min_value=0.0, max_value=100.0,
        value=None, step=5.0, key="miss_excl_thr", placeholder="e.g. 25",
    )

if miss_thr is None:
    st.caption("Enter a percentage above to list the features that exceed it.")
else:
    feat_miss_pct = df[list(feature_cols)].isna().mean() * 100
    over = feat_miss_pct[feat_miss_pct > miss_thr].sort_values(ascending=False)
    if len(over) == 0:
        st.caption(f"No selected features exceed {miss_thr:.0f}% missing.")
    else:
        # measure / modality label per feature: use the defined groups when
        # present, otherwise derive it from the column name
        if modality_groups:
            col_to_measure = {c: g for g, cols in modality_groups.items() for c in cols}
        else:
            def _measure_of(c):
                if "." in c:
                    return c.rsplit(".", 1)[1]
                if "_" in c:
                    return c.split("_", 1)[0] + "_"
                return "(unknown)"
            col_to_measure = {c: _measure_of(c) for c in feature_cols}

        measures = {}
        for c in feature_cols:
            measures.setdefault(col_to_measure[c], []).append(c)

        st.warning(
            f"{len(over)} of {len(feature_cols)} selected features exceed "
            f"{miss_thr:.0f}% missing. Breakdown by measure below."
        )
        # per-measure summary: how many exceed the threshold in each measure
        summ = pd.DataFrame([
            {
                "Measure / modality":      meas,
                "Features over threshold": int((feat_miss_pct.reindex(mcols) > miss_thr).sum()),
                "Total features":          len(mcols),
                "Mean missing (%)":        round(float(feat_miss_pct.reindex(mcols).mean()), 1),
            }
            for meas, mcols in measures.items()
        ]).sort_values("Features over threshold", ascending=False).reset_index(drop=True)
        st.dataframe(summ, use_container_width=True, hide_index=True)
        # detailed list of the features over threshold, with their measure
        with st.expander(f"View the {len(over)} features over threshold and their measure"):
            over_df = pd.DataFrame({
                "Variable":           list(over.index),
                "Measure / modality": [col_to_measure[c] for c in over.index],
                "Missing (%)":        over.values.round(1),
            })
            st.dataframe(over_df, use_container_width=True, hide_index=True)

        st.button(
            f"Exclude these {len(over)} feature(s) from the selection",
            on_click=_exclude_high_missing_features, args=(list(over.index), miss_thr, over.to_dict()),
        )

# ── Data summary table ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Sample summary table (optional)**")
st.caption(
    "Builds a descriptive table of your sample broken down by batch, so you can "
    "check how participants are distributed across batches before harmonizing "
    "(for example, whether age or sex is balanced across batches). It also serves "
    "as a sample-characteristics table for a manuscript. "
    "There is one row per batch, plus an Overall row, and a participant count (N) per row. "
    "Choose which variables to describe below: numeric variables are summarized as "
    "mean (standard deviation) and text or category variables as counts and percentages. "
    "Any column from your uploaded file can be added, for example age, sex, days since injury, or scanner."
)

# Variable candidates: everything that is not a selected imaging feature
summary_candidates = [c for c in df.columns if c not in set(feature_cols)]
default_summary_vars = [c for c in [age_col, sex_col] + extra_continuous + extra_categorical
                        if c in summary_candidates]

sum_c1, sum_c2 = st.columns([4, 1])
with sum_c1:
    summary_vars = st.multiselect(
        "Variables to describe by batch (each becomes a column in the table)",
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
            row = {"Batch": grp_label, "N": len(sub)}
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
st.subheader("Batch exclusion (optional)")
st.caption(
    "Exclude batches before harmonization, for example batches that are too small "
    "or whose distribution differs substantially from the rest. The table below "
    "summarizes each batch so you can decide, showing the sample size and the "
    "model variables (age, sex, and any covariates you added at Step 2)."
)

# By-site summary table (auto-shown) to inform the exclusion decision.
_site_sum_vars = [c for c in [age_col, sex_col] + extra_continuous + extra_categorical
                  if c in df.columns]
_site_rows = []
for _s in sorted(df[site_col].dropna().unique().astype(str)):
    _sub = df[df[site_col].astype(str) == _s]
    _r = {"Batch": _s, "N": len(_sub)}
    for _v in _site_sum_vars:
        _cd = _sub[_v].dropna()
        if len(_cd) == 0:
            _r[_v] = "—"
        elif pd.api.types.is_numeric_dtype(df[_v]):
            _r[_v] = f"{_cd.mean():.1f} ({_cd.std():.1f})"
        else:
            _vc = _cd.astype(str).value_counts()
            _r[_v] = ", ".join(f"{k}: {v}" for k, v in _vc.head(3).items())
    _site_rows.append(_r)
st.dataframe(pd.DataFrame(_site_rows), use_container_width=True, hide_index=True)
st.caption("N = participants per batch. Continuous variables: mean (SD). "
           "Categorical variables: top categories with counts.")

all_sites_n = df[site_col].value_counts().sort_index()
site_options = [f"{s}  (n={all_sites_n[s]})" for s in all_sites_n.index]
site_label_to_name = {f"{s}  (n={all_sites_n[s]})": s for s in all_sites_n.index}
excluded_labels = st.multiselect(
    "Batches to exclude",
    options=site_options,
    default=[],
    key="excluded_sites",
    placeholder="None, all batches included",
)
excluded_sites = [site_label_to_name[l] for l in excluded_labels]
if excluded_sites:
    st.warning(
        f"Excluding {len(excluded_sites)} batch(es): {', '.join(str(s) for s in excluded_sites)}. "
        f"Remaining participants: {int((~df[site_col].isin(excluded_sites)).sum())}."
    )

st.divider()
st.subheader("Evaluation and report options")
st.caption(
    "Select which metrics to compute and include in the supplementary report. "
    "Within-batch consistency by batch (ICC3) is the primary recommended metric; it directly measures "
    "whether harmonization preserved the internal variability of each batch's data (that is, that changing "
    "batch means and scales did not distort within-batch biological signal). "
    "Age correlations are optional because they may be confounded by motion in pediatric fMRI samples, "
    "or may be unstable in restricted age ranges where developmental trajectories are non-linear."
)

eval_c1, eval_c2 = st.columns(2)
with eval_c1:
    include_icc_by_site = st.checkbox(
        "Within-batch consistency by batch (ICC3), recommended",
        value=True, key="inc_icc_site",
        help=(
            "ICC(C,1) between raw and harmonized values computed within each batch. "
            "This is the primary metric for assessing whether harmonization preserved within-batch "
            "biological variability. Minimum 3 participants per batch required (pingouin constraint)."
        ),
    )
    include_cohens_f = st.checkbox(
        "Batch effect size (ANOVA Cohen's f), recommended",
        value=True, key="inc_cohens_f",
        help=(
            "ANCOVA with batch as grouping factor and age + sex as covariates (Type II sums of squares). "
            "Quantifies the magnitude of residual batch-related variance before and after harmonization."
        ),
    )
with eval_c2:
    include_age_corr = st.checkbox(
        "Preserved-covariate associations (Age, Sex)",
        value=False, key="inc_age_corr",
        help=(
            "Shows each preserved covariate (Age, and Sex when it is included in the model) as its own "
            "results tab, with its association to every feature before and after harmonization. "
            "Continuous covariates use Pearson r; categorical covariates use Cohen's f. Each model "
            "controls for the other preserved covariates, excluding the covariate being evaluated. "
            "Age associations can be confounded by motion or unstable in restricted age ranges."
        ),
    )
    include_extra_assoc = st.checkbox(
        "Additional variable associations",
        value=False, key="inc_extra_assoc",
        help=(
            "Adds a results tab for each additional clinical or demographic variable you select below, "
            "with its association to every feature before and after harmonization. Continuous variables "
            "use Pearson r; categorical variables use Cohen's f. Each model controls for the same "
            "covariates passed to ComBat, excluding the variable being evaluated."
        ),
    )

assoc_cont_vars, assoc_cat_vars = [], []
if include_extra_assoc:
    ea_c1, ea_c2 = st.columns(2)
    with ea_c1:
        _cont_opts = [c for c in num_cols if c not in set(feature_cols) | {site_col}]
        assoc_cont_vars = st.multiselect(
            "Continuous variables to evaluate",
            options=_cont_opts,
            default=[c for c in extra_continuous if c in _cont_opts],
            key="assoc_cont",
            placeholder="e.g. days since injury, TSI, GCS score ...",
        )
    with ea_c2:
        _cat_opts = [c for c in all_cols if c not in set(feature_cols) | {site_col}]
        assoc_cat_vars = st.multiselect(
            "Categorical variables to evaluate",
            options=_cat_opts,
            default=[c for c in extra_categorical if c in _cat_opts],
            key="assoc_cat",
            placeholder="e.g. Group, Diagnosis, Injury severity ...",
        )

st.divider()
st.subheader("Step 3 — ComBat configuration and run")

# ── Missing-value handling ──────────────────────────────────────────────────
_feat_na = df[list(feature_cols)].isna()
if bool(_feat_na.any().any()):
    _cov_ok = df[[site_col] + continuous_covariates + categorical_covariates].notna().all(axis=1)
    _cc_all = int((~_feat_na.any(axis=1) & _cov_ok).sum())
    st.warning(
        f"Your selected features contain missing values, and the pattern looks "
        f"scattered. With strict complete-case harmonization a participant must be "
        f"complete across every feature in a run, which would keep only {_cc_all} "
        f"of {len(df)} participants. Choose how to handle missing values below. "
        f"Median imputation and pattern grouping keep Empirical Bayes; the "
        f"per-feature option runs without it."
    )

_MISS_OPTS = {
    "Complete case (drop participants missing any feature in a run)": "complete",
    "Impute missing with median, keep Empirical Bayes":              "impute",
    "Group features by shared missing pattern, keep Empirical Bayes": "pattern",
    "Per-feature, feature-wise (no Empirical Bayes)":                "per_feature",
}
missing_mode_label = st.selectbox(
    "Missing-value handling", list(_MISS_OPTS.keys()), index=0, key="missing_mode",
    help="Complete case drops any participant missing a feature in the run. "
         "Median imputation fills missing values with each feature's median, "
         "harmonizes with Empirical Bayes, then restores the missing cells to "
         "blank. Pattern grouping harmonizes together the features that are "
         "missing in the same participants, keeping Empirical Bayes within each "
         "group. Per-feature harmonizes each feature on its own complete cases, "
         "without Empirical Bayes.",
)
missing_mode = _MISS_OPTS[missing_mode_label]
per_feature = (missing_mode == "per_feature")

# How many participants remain under the chosen handling (after any site exclusion)
_work = df[~df[site_col].isin(excluded_sites)] if excluded_sites else df
_cov_cols = [site_col] + continuous_covariates + categorical_covariates
_cov_ok = _work[_cov_cols].notna().all(axis=1)
_feat_ok_all = _work[list(feature_cols)].notna().all(axis=1)
_feat_ok_any = _work[list(feature_cols)].notna().any(axis=1)
if missing_mode == "complete":
    _retained = int((_cov_ok & _feat_ok_all).sum())
elif missing_mode == "impute":
    _retained = int(_cov_ok.sum())
else:  # pattern / per_feature
    _retained = int((_cov_ok & _feat_ok_any).sum())
_dropped = len(_work) - _retained
st.info(
    f"Participants with this method: **{len(_work)}** in the data "
    f"(after batch exclusion) → **{_retained}** retained, **{_dropped}** dropped. "
    f"Of those dropped, {int((~_cov_ok).sum())} are missing a model variable "
    f"(batch/age/sex/covariate) and the rest are dropped by the feature-missing rule."
)

eb_options = {
    "EB=TRUE  (Empirical Bayes, recommended)":    ("ebt_only",  True,  False),
    "EB=FALSE  (feature-wise, no EB shrinkage)":  ("ebf_only",  False, True),
    "Compare EB=TRUE vs EB=FALSE":                ("compare",   True,  True),
}
eb_label = st.radio("ComBat configuration", list(eb_options.keys()),
                    horizontal=True, index=0, disabled=per_feature)
eb_mode, run_ebt, run_ebf = eb_options[eb_label]
if per_feature:
    st.caption("Per-feature mode runs feature-wise, so the Empirical Bayes "
               "setting above does not apply.")
    run_ebt, run_ebf = False, True
elif run_ebf and missing_mode in ("impute", "pattern"):
    st.caption("This missing-value handling matters for the EB=TRUE estimates, "
               "because Empirical Bayes pools across features. Under EB=FALSE "
               "(feature-wise) there is no pooling, so it has little effect on "
               "those estimates.")

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

        def _harmonize(eb):
            if missing_mode == "impute":
                return run_combat_with_imputation(
                    df_harm, feature_cols, site_col,
                    continuous_covariates=continuous_covariates,
                    categorical_covariates=categorical_covariates,
                    eb=eb, groups=modality_groups)
            if missing_mode == "pattern":
                return run_combat_by_missing_pattern(
                    df_harm, feature_cols, site_col,
                    continuous_covariates=continuous_covariates,
                    categorical_covariates=categorical_covariates, eb=eb)
            if missing_mode == "per_feature":
                return run_combat_per_feature(
                    df_harm, feature_cols, site_col,
                    continuous_covariates=continuous_covariates,
                    categorical_covariates=categorical_covariates)
            if modality_groups:
                return run_combat_grouped(
                    df_harm, modality_groups, site_col,
                    continuous_covariates=continuous_covariates,
                    categorical_covariates=categorical_covariates, eb=eb)
            return run_combat(
                df_harm, feature_cols, site_col,
                continuous_covariates=continuous_covariates,
                categorical_covariates=categorical_covariates, eb=eb)

        if run_ebt:
            progress.progress(5, "Running ComBat (EB=TRUE)...")
            harm_ebt = _harmonize(True)

        if run_ebf:
            progress.progress(20, "Running per-feature ComBat..." if per_feature
                              else "Running ComBat (EB=FALSE)...")
            harm_ebf = _harmonize(False)

        # primary result for site deviation (before panel always uses raw)
        harm_primary = harm_ebt if harm_ebt is not None else harm_ebf

        # ── Metrics ────────────────────────────────────────────────────────
        progress.progress(33, "Computing batch deviation...")
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
            progress.progress(67, "Computing ICC (by batch)...")
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
            progress.progress(80, "Computing ANCOVA batch effects...")
            anc_before = ancova_site_effect(df_harm, feature_cols, site_col, age_col, sex_col, "Before")
            anc_ebt    = ancova_site_effect(harm_ebt, feature_cols, site_col, age_col, sex_col, "EB=TRUE")  if harm_ebt is not None else None
            anc_ebf    = ancova_site_effect(harm_ebf, feature_cols, site_col, age_col, sex_col, "EB=FALSE") if harm_ebf is not None else None

        # Per-variable associations for the results tabs. The evaluated set is
        # every covariate preserved in the ComBat model (Age, Sex, and any extra
        # covariates) plus any additional variables selected, so age and sex are
        # shown in the same grammar as every other variable. Each variable's model
        # controls for the remaining preserved covariates (excluding itself).
        assoc_unified_df = pd.DataFrame()
        _feat_set  = set(feature_cols)
        eval_cont  = [c for c in dict.fromkeys(continuous_covariates + assoc_cont_vars)
                      if c in df.columns and c not in _feat_set]
        eval_cat   = [c for c in dict.fromkeys(categorical_covariates + assoc_cat_vars)
                      if c in df.columns and c not in _feat_set]
        if (include_age_corr or include_extra_assoc) and (eval_cont or eval_cat):
            progress.progress(82, "Computing per-variable associations...")
            u_parts = [compute_extra_associations(
                df_harm, feature_cols, eval_cont, eval_cat,
                continuous_covariates, categorical_covariates, "Before harmonization")]
            if harm_ebt is not None:
                u_parts.append(compute_extra_associations(
                    harm_ebt, feature_cols, eval_cont, eval_cat,
                    continuous_covariates, categorical_covariates, "After (EB=TRUE)"))
            if harm_ebf is not None:
                u_parts.append(compute_extra_associations(
                    harm_ebf, feature_cols, eval_cont, eval_cat,
                    continuous_covariates, categorical_covariates, "After (EB=FALSE)"))
            u_ne = [p for p in u_parts if len(p) > 0]
            assoc_unified_df = pd.concat(u_ne, ignore_index=True) if u_ne else pd.DataFrame()

        # The report keeps its existing additional-variable figure (user-added
        # variables only). It is derived from the unified table so the same
        # models are never computed twice.
        extra_assoc_df = pd.DataFrame()
        if include_extra_assoc and len(assoc_unified_df) > 0 and (assoc_cont_vars or assoc_cat_vars):
            _user_vars   = set(assoc_cont_vars) | set(assoc_cat_vars)
            extra_assoc_df = assoc_unified_df[
                assoc_unified_df["variable"].isin(_user_vars)
            ].reset_index(drop=True)

        # ── Figures ────────────────────────────────────────────────────────
        progress.progress(87, "Generating figures...")
        site_n = df_harm[site_col].value_counts().to_dict()
        # per-site counts AFTER complete-case filtering (what ICC-by-site sees)
        site_n_complete = (harm_primary[site_col].astype(str).value_counts().to_dict()
                           if harm_primary is not None else {})
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

        fig_icc_ci = None
        if run_ebf and icc_ebt is not None and icc_ebf is not None:
            fig_icc_ci = plot_icc_ci_compare(icc_ebt, icc_ebf)

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
        # missing-data documentation for the report
        _fm   = df_harm[feature_cols].isna()
        _fpct = _fm.mean() * 100
        _k    = int((_fpct > 0).sum())
        missing_summary = {
            "k":     _k,
            "min":   round(float(_fpct[_fpct > 0].min()), 1) if _k else 0.0,
            "max":   round(float(_fpct.max()), 1),
            "total": int(_fm.sum().sum()),
        } if _k else None
        excl_info = st.session_state.get("_excl_info")
        excl_feat_list = st.session_state.get("_excl_feat_list", [])
        html_report = generate_report(
            df_raw=df, site_col=site_col, age_col=age_col, sex_col=sex_col,
            feature_cols=feature_cols, n_retained=n_retained,
            modality_groups=modality_groups,
            missing_handling=missing_mode,
            missing_summary=missing_summary,
            excl_info=excl_info,
            excl_feat_list=excl_feat_list,
            fig_icc_ci=fig_icc_ci,
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
            assoc_unified_df=assoc_unified_df,
            site_n_complete=site_n_complete,
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
    _assoc_uni      = st.session_state.get("assoc_unified_df", pd.DataFrame())
    _assoc_vars     = (list(dict.fromkeys(_assoc_uni["variable"].tolist()))
                       if isinstance(_assoc_uni, pd.DataFrame) and len(_assoc_uni) > 0 else [])

    # Build tab list dynamically based on selected metrics. Age, Sex, and every
    # additional variable each get their own association tab, in the same style.
    tab_names = ["Batch Deviation"]
    if _inc_cohens_f:
        tab_names.append("Batch Effect Size (Cohen's f)")
    if _inc_icc_site:
        tab_names.append("Within-Batch Consistency by Batch")
    for _v in _assoc_vars:
        tab_names.append(str(_v))
    tab_names.append("Methods paragraph")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    with tabs[tab_idx]:
        st.plotly_chart(st.session_state["fig_site"], use_container_width=True)
        st.caption("Each point = one (batch, feature) pair. Batch means should cluster near zero after harmonization.")
    tab_idx += 1

    if _inc_cohens_f:
        with tabs[tab_idx]:
            if st.session_state.get("fig_anc"):
                st.plotly_chart(st.session_state["fig_anc"], use_container_width=True)
                st.caption("Each point = one feature. Batch effect from ANCOVA Type II (Age + Sex as covariates); significance is the uncorrected p-value (p < 0.05), not corrected for multiple comparisons. Colour encodes the after-harmonization status (red = p < 0.05, green = p >= 0.05) and fill encodes the before-harmonization status (filled = p < 0.05, open = p >= 0.05), as summarised in the 2 x 2 legend. Points below the diagonal = reduced batch effect size.")
            else:
                st.info("Cohen's f not computed for this run.")
        tab_idx += 1

    if _inc_icc_site:
        with tabs[tab_idx]:
            if st.session_state.get("fig_icc_site"):
                st.plotly_chart(st.session_state["fig_icc_site"], use_container_width=True)
                st.caption(
                    "Each box = distribution of ICC3 across features for that batch. "
                    "Within-batch ICC is the primary recommended metric; it measures whether harmonization "
                    "preserved the rank ordering of participants within each batch, indicating that "
                    "within-batch biological variability was not distorted by batch correction. "
                    "Batches with fewer participants may show lower consistency, particularly without EB. "
                    "Minimum 3 participants per batch required. "
                    "Colored bands: Poor / Moderate / Good / Excellent (Koo & Li, 2016)."
                )
            else:
                sc    = st.session_state.get("site_n_complete", {})
                small = {s: n for s, n in sc.items() if n < 3}
                if sc and small:
                    st.info(
                        "By-batch ICC needs at least 3 participants per batch with complete data. "
                        "Rows with a missing value in any selected feature, or in batch, age, or sex, "
                        "are excluded before harmonization, which can reduce per-batch counts. "
                        "Batches below 3 after that filtering: "
                        + ", ".join(f"{s} (n={n})" for s, n in sorted(small.items()))
                        + ". If you expected more, check the feature columns for missing values."
                    )
                elif sc:
                    st.info(
                        "By-batch ICC could not be computed even though each batch has at least 3 "
                        "participants with complete data. Per-batch counts: "
                        + ", ".join(f"{s} (n={n})" for s, n in sorted(sc.items()))
                        + ". Please report this dataset shape so it can be investigated."
                    )
                else:
                    st.info("By-batch ICC not available (need at least 3 participants per batch with complete data).")
        tab_idx += 1

    # One tab per evaluated variable (Age, Sex, and any additional variables),
    # each in the same before-versus-after grammar.
    for _v in _assoc_vars:
        with tabs[tab_idx]:
            _fig_v = plot_single_association(_assoc_uni, _v)
            if _fig_v is not None:
                st.plotly_chart(_fig_v, use_container_width=True)
                _sub_v = _assoc_uni[_assoc_uni["variable"] == _v]
                _vt_v  = _sub_v["var_type"].iloc[0] if len(_sub_v) else "continuous"
                _eff_v = "Pearson r" if _vt_v == "continuous" else "Cohen's f"
                st.caption(
                    f"Each point = one feature. Effect size for {_v} before (x-axis) versus after "
                    f"(y-axis) harmonization, shown as {_eff_v}. Significance is the FDR-corrected "
                    f"(Benjamini-Hochberg, per variable) p-value from a linear model that controls for "
                    f"the other preserved covariates (excluding {_v}). Grey circle = not significant in "
                    f"either condition. Filled circle = FDR significant after only. Orange diamond = FDR "
                    f"significant before only. Purple square = FDR significant in both. Dashed diagonal = no change."
                )
            else:
                st.info(f"No association could be computed for {_v}.")
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
