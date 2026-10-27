"""
CHORD — Multisite Harmonization Assessment
Streamlit web application
"""

import re

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.naming    import measure_keys, natural_sorted
from src.harmonize import (run_combat, run_combat_grouped, run_combat_per_feature,
                           run_combat_by_missing_pattern, run_combat_with_imputation,
                           missing_pattern_groups)
from src.metrics   import (site_mean_deviation, spearman_raw_vs_harm,
                            compute_icc, age_correlations, ancova_site_effect,
                            demographic_summary,
                            compute_icc_by_site, compute_icc_within_summary,
                            compute_spearman_by_site,
                            compute_extra_associations)
from src.plots     import (plot_site_deviation, plot_spearman, plot_icc, plot_data_matrix,
                           plot_age_correlations, plot_cohens_f,
                           plot_icc_by_site, plot_spearman_by_site,
                           plot_extra_associations, plot_single_association,
                           plot_icc_ci_compare, plot_missing_by_measure,
                           plot_missing_by_batch, plot_missing_curve,
                           plot_missing_by_batch_bars, plot_missing_batch_by_group,
                           plot_missing_present_by_batch)
from src.report    import generate_report, build_methods_paragraph, data_prep_sentence

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

Input: a table in which each row is a participant, containing a batch column and one or more imaging feature columns; biological covariates to be preserved during harmonization, typically age and sex, may be included but are not required. Output: the harmonized feature table and an evaluation report.

**Recommended metrics (default ON):**
- Batch mean deviation (z-score), always included
- Within-batch consistency by batch (ICC(3,1)), primary recommended metric
- Batch effect size (ANCOVA, Cohen's f)

**Optional metrics:**
- Age associations (Pearson r, FDR-corrected). Optional: can be confounded by head motion, and are uninformative when the age range is narrow
- Additional variable associations (OLS) — e.g. injury severity, time since injury
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
        data_key = f"upload:{uploaded.name}:{uploaded.size}"
        data_stem = uploaded.name.rsplit(".", 1)[0]   # downloads are named after the loaded table
        st.session_state["use_demo"] = False
    except Exception as e:
        st.error(f"Could not read file: {e}")
elif st.session_state.get("use_demo"):
    try:
        df = pd.read_csv("example_data.csv")
        data_key = "example_data.csv"
        data_stem = "example_data"
        st.info("Loaded synthetic example dataset (215 participants, 6 batches, 20 FA features, not real patient data)")
    except FileNotFoundError:
        st.error("Example data file not found.")

if df is None:
    st.stop()

st.success(f"Data loaded: **{df.shape[0]} participants** × **{df.shape[1]} columns**")
with st.expander("Preview data", expanded=True):
    _whole = st.toggle("Show the whole table", key="preview_whole_table")
    if not _whole:
        st.dataframe(df.head(8), use_container_width=True)
        st.caption(f"First 8 of {len(df)} rows. Switch on \"Show the whole table\" to scroll through all of it.")
    else:
        _col_filter = st.text_input(
            "Only the columns whose name contains (optional)", key="preview_col_filter",
            placeholder="for example FA_ or Left-Amygdala")
        _cols = ([c for c in df.columns if _col_filter.lower() in str(c).lower()]
                 if _col_filter else list(df.columns))
        # keep the page fast: at most about 300,000 cells at a time
        _k = max(1, 300_000 // max(len(df), 1))
        _shown = _cols[:_k]
        st.dataframe(df[_shown], use_container_width=True, height=520)
        if len(_shown) < len(_cols):
            st.caption(f"{len(df)} rows; showing the first {len(_shown)} of {len(_cols)} columns, "
                       "because the table is large. Type part of a column name above to see other columns.")
        else:
            st.caption(f"{len(df)} rows x {len(_shown)} columns.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Configure
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Step 2 — Configure the ComBat model")

all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

def _guess(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    return cols[0] if cols else None

def _guess_optional(cols, keywords):
    """
    Find an optional column (age, sex) by name. A keyword must begin a part of
    the name (parts split at non-letters and non-digits), so "Age_at_scan" and
    "sex" match but "AverageFA" does not; returns None when nothing matches,
    instead of falling back to the first column.
    """
    for kw in keywords:
        for c in cols:
            parts = re.split(r"[^a-z0-9]+", str(c).lower())
            if any(p.startswith(kw) for p in parts if p):
                return c
    return None

default_site = _guess(all_cols, ["site", "scanner", "batch", "center"])
default_age  = _guess_optional(all_cols, ["age"])
default_sex  = _guess_optional(all_cols, ["sex", "gender"])

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

def _record_feature_exclusion(cols_to_drop, threshold, pcts, groups):
    """Remove features from the selection and record them with the threshold used."""
    drop = set(cols_to_drop)
    st.session_state["sel_features"] = [c for c in st.session_state.get("sel_features", []) if c not in drop]
    prev = st.session_state.get("_excl_info", {"n": 0, "threshold": None})
    st.session_state["_excl_info"] = {"n": prev["n"] + len(drop), "threshold": threshold}
    grp = {c: g for g, gcols in (groups or {}).items() for c in gcols}
    lst = st.session_state.get("_excl_feat_list", [])
    for f in cols_to_drop:
        lst.append({"Feature": f, "Modality group": grp.get(f),
                    "Participants missing (%)": round(float(pcts.get(f, float("nan"))), 1)})
    st.session_state["_excl_feat_list"] = lst


def _record_participant_exclusion(entries, threshold, key_of_data):
    """Record participants as excluded, with the threshold used."""
    prev = [d for d in st.session_state.get("_excl_part_list", []) if d.get("data_key") == key_of_data]
    prev += entries
    st.session_state["_excl_part_list"] = prev
    st.session_state["_excl_part_info"] = {"n": len(prev), "threshold": threshold}


@st.experimental_dialog("Missing values above this percentage", width="large")
def _missing_at_share(x, feat_pct, part_pct, measure_of, kind="features", part_entries=None,
                      groups=None, key_of_data=None):
    """What a click on a missing-value curve shows, with a button to exclude at
    that share: the features with more than x% of the participants missing
    (kind "features"), or the participants missing more than x% of the selected
    features (kind "participants")."""
    if kind == "features":
        over = feat_pct[feat_pct > x].sort_values(ascending=False)
        st.markdown(f"**Features with more than {x:.1f}% of the participants missing: "
                    f"{len(over)} of {len(feat_pct)}**")
        if measure_of:
            _g = pd.Series(measure_of)
            st.dataframe(pd.DataFrame([
                {"Modality group": g, "Features": int((_g == g).sum()),
                 f"More than {x:.1f}% missing": int(_g.reindex(over.index).eq(g).sum())}
                for g in dict.fromkeys(_g.values)]), hide_index=True, use_container_width=True)
        if len(over):
            _t = pd.DataFrame({"Feature": over.index})
            if measure_of:
                _t["Modality group"] = [measure_of.get(c, "") for c in over.index]
            _t["Participants missing (%)"] = over.values.round(1)
            st.dataframe(_t, hide_index=True, use_container_width=True, height=min(38 + 35 * len(_t), 300))
            if st.button(f"Exclude these {len(over)} feature(s) (more than {x:.1f}% of the participants missing)",
                         type="primary", on_click=_record_feature_exclusion,
                         args=(list(over.index), round(float(x), 1), over.to_dict(), groups)):
                st.rerun()
    else:
        over = part_pct[part_pct > x].sort_values(ascending=False)
        st.markdown(f"**Participants missing more than {x:.1f}% of the selected features: "
                    f"{len(over)} of {len(part_pct)}**")
        if len(over):
            _e = [e for e in (part_entries or []) if e["row"] in set(over.index)]
            st.dataframe(pd.DataFrame(_e)[["Participant", "Batch", "Features missing (%)"]]
                         .sort_values("Features missing (%)", ascending=False),
                         hide_index=True, use_container_width=True, height=min(38 + 35 * len(_e), 300))
            if st.button(f"Exclude these {len(over)} participant(s) (missing more than {x:.1f}% of the features)",
                         type="primary", on_click=_record_participant_exclusion,
                         args=(_e, round(float(x), 1), key_of_data)):
                st.rerun()
    st.caption("Excluding records this percentage as the threshold; the report and the Methods paragraph state it. "
               "Exclusions can be undone in the exclusion steps below.")
    if st.button(f"Only set {x:.1f}% as the threshold under the curve"):
        st.session_state["miss_assess_thr" if kind == "features" else "miss_assess_part_thr"] = round(float(x), 1)
        st.rerun()


# Participant ID column, guessed from common names (participant_id, subID,
# subject, RecordID, ...) among columns whose values are all different.
_ID_NAMES = ["participantid", "participant", "subjectid", "subject", "subid", "subjid", "sub",
             "srcsubjectid", "recordid", "scanid", "outputid", "id", "eid", "guid"]


def _guess_id_col(data, exclude=()):
    import re as _re_id
    cands = [c for c in data.columns if c not in exclude]
    norm = {c: _re_id.sub(r"[^a-z0-9]", "", str(c).lower()) for c in cands}

    def all_different(c):
        v = data[c].dropna()
        return len(v) > 0 and v.nunique() == len(v)
    for name in _ID_NAMES:
        for c in cands:
            if norm[c] == name and all_different(c):
                return c
    for c in cands:
        if norm[c].endswith("id") and all_different(c):
            return c
    return None


site_col = _col_selector("Site / Batch column", all_cols, default_site, "site")
# A batch column of whole numbers with empty cells is read as decimals (1.0, 2.0, ...);
# its labels are kept as the whole numbers they are.
if site_col and pd.api.types.is_float_dtype(df[site_col]):
    _b = df[site_col].dropna()
    if len(_b) and (_b == _b.round()).all():
        df[site_col] = df[site_col].map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
st.caption(
    "Use your study's batch column. If every site used a single scanner with no hardware or "
    "software changes, the site column can serve as the batch. If a site used more than one "
    "scanner, or scanners were upgraded during data collection, define the batch as the "
    "combination of site and scanner. If known hardware or software version changes occurred "
    "during data collection and were traced, this can also be reflected in the batch variable, "
    "meaning the batch variable would reflect a combination of site, scanner and "
    "hardware/software version upgrades."
)

if site_col is None:
    st.stop()

# Age and sex are optional covariates, not required columns. They are detected
# automatically here and offered, pre-selected, in the covariate section below.
age_col = default_age if (default_age and default_age in all_cols) else None
sex_col = default_sex if (default_sex and default_sex in all_cols) else None

# ── Covariates to preserve in the ComBat model ────────────────────────────
st.markdown("**Covariates to preserve in the ComBat model**")
st.caption(
    "ComBat harmonization can preserve the biological variability of key variables "
    "of interest, for example age and sex, while removing batch-related variance. "
    "Only the batch column is strictly required; every covariate is optional. Age and sex "
    "are detected automatically and pre-selected when present, and can be unchecked, for "
    "example in a study with a single sex or no age variability."
)

other_candidates = [c for c in all_cols if c not in {site_col, age_col, sex_col}]

cov_c1, cov_c2 = st.columns(2)
with cov_c1:
    st.markdown("**Continuous covariates** *(numeric values, for example age, days since injury)*")
    if age_col is not None:
        inc_age = st.checkbox(
            f"Include **{age_col}** (auto-detected age)",
            value=True, key="inc_age",
        )
    else:
        inc_age = False
        st.caption("No age column was auto-detected. Add one with \"Add more covariates\" "
                   "if your study has age.")
with cov_c2:
    st.markdown("**Categorical covariates** *(group labels, for example sex, diagnosis, group)*")
    if sex_col is not None:
        inc_sex = st.checkbox(
            f"Include **{sex_col}** (auto-detected sex)",
            value=True, key="inc_sex",
        )
    else:
        inc_sex = False
        st.caption("No sex column was auto-detected. Add one with \"Add more covariates\" "
                   "if your study has sex.")

# Further covariates only when asked for
extra_cont_add, extra_cat_add = [], []
if st.checkbox("Add more covariates", key="add_more_covariates"):
    add_c1, add_c2 = st.columns(2)
    with add_c1:
        extra_cont_add = st.multiselect(
            "More continuous covariates",
            options=[c for c in other_candidates if c in num_cols],
            key="extra_cont",
            placeholder="e.g. days since injury, TSI …",
        )
    with add_c2:
        extra_cat_add = st.multiselect(
            "More categorical covariates",
            options=other_candidates,
            key="extra_cat",
            placeholder="e.g. Group, Diagnosis, Handedness …",
        )

# Final covariate lists passed to ComBat
continuous_covariates = ([age_col] if (inc_age and age_col) else []) + extra_cont_add
categorical_covariates = ([sex_col] if (inc_sex and sex_col) else []) + extra_cat_add

# Age and sex as actually used (None when not detected or unchecked); the
# ANCOVA batch-effect model uses the same ones as ComBat
age_used = age_col if (inc_age and age_col) else None
sex_used = sex_col if (inc_sex and sex_col) else None

# Also expose for downstream use
extra_continuous  = extra_cont_add
extra_categorical = extra_cat_add

# Summary box
all_covars_display = (
    ([f"{age_col} (continuous)"] if (inc_age and age_col) else [])
    + [f"{c} (continuous)" for c in extra_cont_add]
    + ([f"{sex_col} (categorical)"] if (inc_sex and sex_col) else [])
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

def _fmt_num(v):
    """A number with at most two decimals, without trailing zeros (8.39, 18.22, 9)."""
    return f"{v:.2f}".rstrip("0").rstrip(".") if pd.notna(v) else "—"


def _fmt_val(v):
    """A category label: whole numbers without decimals (1, not 1.0)."""
    return f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)


def _by_batch_table(data, batch_col, variables, categorical=None):
    """
    One row per batch plus an Overall row (participants with a batch value):
    N, then for each variable mean (SD) and range when it is continuous, or n (%)
    per category when it is in ``categorical`` (as the user set it; by default
    the categorical covariates) or is not a number, and always the missing
    values, n (%).
    """
    categorical = set(categorical_covariates if categorical is None else categorical)
    data = data[data[batch_col].notna()]
    rows = []
    for label in natural_sorted(data[batch_col].unique().astype(str)) + ["Overall"]:
        sub = data if label == "Overall" else data[data[batch_col].astype(str) == label]
        row = {"Batch": label, "N": len(sub)}
        for var in variables:
            col = sub[var]
            vals = col.dropna()
            n_miss = int(col.isna().sum())
            miss_txt = f"{n_miss} ({100 * n_miss / len(sub):.0f}%)" if len(sub) else "0"
            if var not in categorical and pd.api.types.is_numeric_dtype(data[var]):
                head = f"{var} [mean (SD); range; missing n (%)]"
                row[head] = (f"{vals.mean():.2f} ({vals.std():.2f}); {_fmt_num(vals.min())} to "
                             f"{_fmt_num(vals.max())}; {miss_txt}") if len(vals) else f"—; —; {miss_txt}"
            else:
                head = f"{var} [n (%) per value; missing n (%)]"
                vc = vals.map(_fmt_val).value_counts()
                order = natural_sorted(vc.index)
                parts = [f"{v}: {vc[v]} ({100 * vc[v] / len(sub):.0f}%)" for v in order[:8]]
                if len(order) > 8:
                    parts.append(f"+ {len(order) - 8} more")
                row[head] = ", ".join(parts) + f"; missing {miss_txt}"
        rows.append(row)
    return pd.DataFrame(rows)


# ── Check the demographics (before choosing the features) ──────────────────
st.markdown("**Check the demographics**")
st.caption("Look at the batch and the covariates before harmonizing, to see whether anything in the "
           "table needs fixing first: typos or misspelled categories, codes that are not real values "
           "(such as -999, 999 or \"Unknown\"), unexpected ranges, or missing values.")
if st.toggle("Check the batch and the covariates", value=True, key="show_demo_table"):
    st.caption("Your covariates are listed to start with; add any other column to the list of its kind, so "
               "that CHORD describes it the right way (a mean for continuous variables, counts for categories, "
               "also when the categories are coded as numbers, such as sex coded 1 and 2).")
    _dc1, _dc2 = st.columns(2)
    with _dc1:
        _demo_cont = st.multiselect(
            "Continuous variables to check (numbers, such as age or time since injury)",
            options=[c for c in df.columns if c != site_col and pd.api.types.is_numeric_dtype(df[c])],
            default=[c for c in continuous_covariates if c in df.columns],
            key="demo_vars_cont",
        )
    with _dc2:
        _demo_cat = st.multiselect(
            "Categorical variables to check (groups, such as sex or diagnosis)",
            options=[c for c in df.columns if c != site_col],
            default=[c for c in categorical_covariates if c in df.columns],
            key="demo_vars_cat",
        )
    _demo_vars = _demo_cont + [c for c in _demo_cat if c not in _demo_cont]
    _demo_cat_set = set(_demo_cat)
    if set(_demo_cont) & _demo_cat_set:
        st.caption("A variable in both lists is described as a category: "
                   + ", ".join(sorted(set(_demo_cont) & _demo_cat_set)) + ".")
    st.markdown("By batch")
    st.dataframe(_by_batch_table(df, site_col, _demo_vars, categorical=_demo_cat_set),
                 use_container_width=True, hide_index=True)
    _no_batch = int(df[site_col].isna().sum())
    st.caption("Each column says what it shows: for continuous variables mean (SD); range; missing n (%), "
               "for categories n (%) per value; missing n (%). Missing is shown also when it is 0."
               + (f" {_no_batch} participant(s) without a batch value are not in the table." if _no_batch else ""))

    st.markdown("The values of each variable")
    _by_b = st.toggle("Show by batch", key="demo_by_batch",
                      help="Categories: one bar per batch split by value. Continuous variables: one box per batch.")
    _roles = [(site_col, "batch")] + [
        (_c, "continuous covariate" if _c in continuous_covariates and _c not in _demo_cat_set else
             "categorical covariate" if _c in categorical_covariates else
             "added for checking, categorical" if _c in _demo_cat_set else "added for checking, continuous")
        for _c in _demo_vars]
    _bdf = df[df[site_col].notna()].copy()
    _bdf["_batch"] = _bdf[site_col].astype(str)
    _border = natural_sorted(_bdf["_batch"].unique())
    _pal = px.colors.qualitative.Safe
    _plot_layout = dict(margin=dict(l=10, r=10, t=10, b=40), plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color="black"))
    for _tab_ui, (_c, _role) in zip(st.tabs([_c for _c, _ in _roles]), _roles):
        _s = df[_c]
        _n_miss = int(_s.isna().sum())
        _miss_txt = f"missing: {_n_miss} ({100 * _n_miss / max(len(_s), 1):.1f}%)"
        _cat = _c == site_col or _c in _demo_cat_set or not pd.api.types.is_numeric_dtype(_s)
        with _tab_ui:
            if _cat:
                _vc = _s.dropna().map(_fmt_val).value_counts()
                _vals = natural_sorted(_vc.index)
                _all = pd.DataFrame({"Value": _vals + ["(missing)"],
                                     "Participants": [int(_vc[v]) for v in _vals] + [_n_miss]})
                _all["Participants (%)"] = (100 * _all["Participants"] / max(len(_s), 1)).round(1)
                st.markdown(f"**{_c}** ({_role}): {_s.nunique()} distinct values; {_miss_txt}")
                _p1, _p2 = st.columns([3, 2])
                with _p1:
                    if _by_b and _c != site_col:
                        # one 100% bar per batch, split by value
                        _lab = _bdf[_c].map(lambda v: "(missing)" if pd.isna(v) else _fmt_val(v))
                        _ct = pd.crosstab(_bdf["_batch"], _lab).reindex(index=_border, fill_value=0)
                        _ct = _ct[[v for v in _vals + ["(missing)"] if v in _ct.columns]]
                        _pct = 100 * _ct.div(_ct.sum(axis=1), axis=0)
                        _f = go.Figure()
                        for _i, _v in enumerate(_ct.columns):
                            _f.add_trace(go.Bar(x=_border, y=_pct[_v], name=_v,
                                                marker_color="#BBBBBB" if _v == "(missing)" else _pal[_i % len(_pal)],
                                                customdata=_ct[_v], hovertemplate=f"Batch %{{x}}<br>{_v}: %{{customdata}} "
                                                f"(%{{y:.0f}}%)<extra></extra>"))
                        _f.update_layout(barmode="stack", height=300, legend=dict(orientation="h", y=-0.25),
                                         yaxis_title="Participants of the batch (%)", xaxis_title="Batch",
                                         xaxis=dict(type="category"), **_plot_layout)
                    else:
                        _f = go.Figure(go.Bar(
                            x=_all["Value"], y=_all["Participants"],
                            marker_color=["#BBBBBB" if v == "(missing)" else "#2A6EBB" for v in _all["Value"]],
                            text=[f"{n} ({p:.0f}%)" for n, p in zip(_all["Participants"], _all["Participants (%)"])],
                            textposition="outside", cliponaxis=False,
                            hovertemplate="%{x}: %{y} participants<extra></extra>"))
                        _f.update_layout(height=300, yaxis_title="Participants", xaxis=dict(type="category"),
                                         **_plot_layout)
                    st.plotly_chart(_f, use_container_width=True)
                with _p2:
                    st.dataframe(_all, hide_index=True, use_container_width=True,
                                 height=min(38 + 35 * len(_all), 300))
            elif _s.notna().any():
                # continuous: a summary, the distribution, and the lowest and highest values,
                # where codes such as -999 or 999 show up
                _v = _s.dropna()
                st.markdown(f"**{_c}** ({_role}): {_v.nunique()} distinct values; "
                            f"{_fmt_num(_v.min())} to {_fmt_num(_v.max())}; mean {_v.mean():.2f} "
                            f"(SD {_v.std():.2f}); {_miss_txt}")
                _h1, _h2 = st.columns([3, 2])
                with _h1:
                    if _by_b:
                        _f = go.Figure()
                        for _bb in _border:
                            _yy = _bdf.loc[_bdf["_batch"] == _bb, _c].dropna()
                            _f.add_trace(go.Box(y=_yy, name=str(_bb), boxpoints="outliers", marker_color="#2A6EBB",
                                                line_color="#2A6EBB", showlegend=False))
                        _f.update_layout(height=300, yaxis_title=_c, xaxis_title="Batch",
                                         xaxis=dict(type="category"), **_plot_layout)
                    else:
                        _f = go.Figure(go.Histogram(x=_v, nbinsx=40, marker_color="#2A6EBB"))
                        _f.update_layout(height=300, bargap=0.05, xaxis_title=_c, yaxis_title="Participants",
                                         **_plot_layout)
                    st.plotly_chart(_f, use_container_width=True)
                with _h2:
                    _srt = np.sort(_v.unique())
                    st.dataframe(pd.DataFrame({
                        "Lowest values": [_fmt_num(x) for x in _srt[:5]] + [""] * (5 - len(_srt[:5])),
                        "Highest values": [_fmt_num(x) for x in _srt[::-1][:5]] + [""] * (5 - len(_srt[:5]))}),
                        hide_index=True, use_container_width=True)
            else:
                st.markdown(f"**{_c}** ({_role}): no values; {_miss_txt}")

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
else:
    # Re-assigned on every run so the selection is kept also on runs where its
    # box is not drawn (large selections are shown compactly, see below).
    st.session_state["sel_features"] = list(st.session_state["sel_features"])

st.divider()
st.subheader("Step 3 — Select the features to harmonize")
st.caption(f"{len(auto_features)} numeric columns are available.")

# ── Range of feature columns (the main way to select) ──────────────────────
st.markdown("**Choose the first and last feature column**")
st.caption(
    "Three ways to choose the features:\n"
    "- **By name:** pick the first and the last feature column; every column between them is selected. "
    "Tick \"To the last column\" to go to the end of the table, and type part of a name in a box to find it.\n"
    "- **By number:** switch on \"Use column numbers instead of names\", for example from column 5 to the last "
    "column. \"Show all column names with their numbers\" lists every column with its number, to see where "
    "your features start and end.\n"
    "- **By keyword:** further down, select the columns whose name contains a keyword, or exclude them from "
    "the columns you chose.\n\n"
    "The name lists hold only the columns that can be features (numeric, and not the batch or a covariate "
    "chosen in Step 2). You can review the selected features below. If the selected features include more "
    "than one modality or measure type (for example FA and MD, or thickness and volume), tick the option "
    "under \"Multiple imaging modalities or measure types\" right after choosing them, so that each is "
    "harmonized on its own."
)
n_cols   = len(all_df_cols)
last_col = all_df_cols[-1]

# Every column of the table with its number: with thousands of columns the
# From/To lists are too long to scroll through.
if st.toggle("Show all column names with their numbers", key="show_col_numbers"):
    lc1, lc2 = st.columns([2, 1])
    with lc2:
        find = st.text_input("Find a column", key="col_number_find", label_visibility="collapsed",
                             placeholder="Find a column, e.g. node100")
    num_table = pd.DataFrame({"№": range(1, n_cols + 1), "Column": all_df_cols})
    if find.strip():
        num_table = num_table[num_table["Column"].str.contains(find.strip(), case=False, regex=False)]
    lc1.caption(f"{len(num_table)} of {n_cols} columns shown. Click the № header to sort, for example to "
                "show the end of the table first.")
    st.dataframe(num_table, hide_index=True, use_container_width=True, height=300,
                 column_config={"№": st.column_config.NumberColumn(width="small")})

use_col_num = st.toggle("Use column numbers instead of names", value=False, key="range_use_col_num",
                        help="Switch on if column names are not informative enough to use as range boundaries")


def _to_last_tick():
    """The "To the last column" tick under the To box (same widget in both modes)."""
    ticked = st.checkbox("To the last column", key="range_to_last")
    if ticked:
        note = f"Last column: {last_col} (column {n_cols})"
        if auto_features and last_col not in auto_features:
            note += f"; it cannot be a feature, so the range ends at {auto_features[-1]}"
        st.caption(note)
    return ticked


if not use_col_num:
    na_c1, na_c2 = st.columns(2)
    with na_c1:
        from_name = st.selectbox("From column", ["— select —"] + auto_features, key="range_from_name")
    with na_c2:
        if st.session_state.get("range_to_last", False):
            to_name = None
            st.selectbox("To column", [last_col], disabled=True, key="range_to_last_shown")
        else:
            to_name = st.selectbox("To column", ["— select —"] + auto_features, key="range_to_name")
        to_last = _to_last_tick()
    if from_name != "— select —" and (to_last or to_name != "— select —"):
        i1 = auto_features.index(from_name)
        i2 = len(auto_features) - 1 if to_last else auto_features.index(to_name)
        preview = auto_features[min(i1, i2):max(i1, i2) + 1]
        bounds  = (from_name, "<last column>" if to_last else to_name)
        # Apply automatically whenever the two boundaries change. The change
        # check keeps later manual edits (or keyword exclusions) from being
        # overwritten on every rerun.
        if st.session_state.get("_last_range_name") != bounds:
            st.session_state["_last_range_name"] = bounds
            st.session_state["sel_features"] = preview
            st.rerun()
        st.caption(f"Range covers {len(preview)} columns: "
                   f"{', '.join(preview[:5])}{'…' if len(preview) > 5 else ''}")
else:
    nb_c1, nb_c2 = st.columns(2)
    with nb_c1:
        from_num = st.number_input("From column №", min_value=1,
                                    max_value=n_cols, value=1, step=1)
        st.caption(f"Column {from_num}: {all_df_cols[from_num - 1]}")
    with nb_c2:
        if st.session_state.get("range_to_last", False):
            to_num = n_cols
            st.number_input("To column №", min_value=1, max_value=n_cols, value=n_cols, step=1,
                            disabled=True, key="range_to_last_num_shown")
        else:
            to_num = st.number_input("To column №", min_value=1,
                                     max_value=n_cols, value=n_cols, step=1)
            st.caption(f"Column {to_num}: {all_df_cols[to_num - 1]}")
        to_last = _to_last_tick()
    range_cols = [c for c in all_df_cols[min(from_num, to_num) - 1:max(from_num, to_num)]
                  if c in auto_features]
    # Seed on first render so the default full range does not auto-select;
    # apply automatically only when the user changes a boundary.
    bounds = (from_num, to_num, to_last)
    if "_last_range_num" not in st.session_state:
        st.session_state["_last_range_num"] = bounds
    elif st.session_state["_last_range_num"] != bounds:
        st.session_state["_last_range_num"] = bounds
        if range_cols:
            st.session_state["sel_features"] = range_cols
            st.rerun()
    num_preview = all_df_cols[from_num-1:to_num]
    st.caption(f"Columns {from_num}–{to_num}: "
               f"{', '.join(num_preview[:5])}{'…' if len(num_preview) > 5 else ''}")

# ── Final review multiselect ───────────────────────────────────────────────
n_sel = len(st.session_state.get("sel_features", []))
MAX_CHIPS = 300   # above this the browser slows down badly when every feature is drawn as a chip
if n_sel <= MAX_CHIPS:
    st.markdown(f"**Currently selected: {n_sel} features** — review or edit below")
    feature_cols = st.multiselect(
        "Selected features",
        options=auto_features,
        key="sel_features",
        label_visibility="collapsed",
    )
else:
    feature_cols = list(st.session_state["sel_features"])
    st.markdown(f"**Currently selected: {n_sel} features** — review or edit below")
    st.caption(f"From {feature_cols[0]} to {feature_cols[-1]}. Too many to show one by one (more than "
               f"{MAX_CHIPS}): open \"View all {n_sel} selected feature names in full\" below to see them; "
               "change the selection with the range above or by keyword below.")



# ── Full name preview ──────────────────────────────────────────────────────
if feature_cols:
    with st.expander(f"View all {len(feature_cols)} selected feature names in full"):
        if len(feature_cols) > MAX_CHIPS:
            # many features: one scrollable numbered table instead of one line per name
            st.dataframe(pd.DataFrame({"№": range(1, len(feature_cols) + 1), "Feature": feature_cols}),
                         hide_index=True, use_container_width=True, height=300,
                         column_config={"№": st.column_config.NumberColumn(width="small")})
        else:
            # show as a compact numbered list in 3 columns
            cols_per_row = 3
            rows = [feature_cols[i:i+cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]
            for i, row in enumerate(rows):
                grid = st.columns(cols_per_row)
                for j, name in enumerate(row):
                    idx = i * cols_per_row + j + 1
                    grid[j].markdown(f"`{idx}.` {name}")

# ── Optional: select or exclude by keyword ─────────────────────────────────
if st.checkbox("Select or exclude columns by keyword", key="use_keyword_exclusion"):
    st.caption("\"Select matching\" selects every available column whose name contains a keyword (it "
               "replaces the selection); \"Exclude matching\" removes those columns from the selection. "
               "Separate several keywords with commas.")
    ex_c1, ex_c2, ex_c3 = st.columns([4, 1, 1])
    with ex_c1:
        excl_input = st.text_input(
            "Keyword(s)", key="excl_keywords",
            placeholder='e.g.  FA_   or   unassigned, reward, LIM',
            label_visibility="collapsed",
        )

    def _keywords():
        text = st.session_state.get("excl_keywords", "")
        return [k.strip().lower() for k in text.split(",") if k.strip()]

    def _select_keyword_matches():
        if _keywords():
            st.session_state["sel_features"] = [c for c in auto_features
                                                if any(kw in c.lower() for kw in _keywords())]

    def _exclude_keyword_matches():
        if _keywords():
            st.session_state["sel_features"] = [
                c for c in st.session_state.get("sel_features", [])
                if not any(kw in c.lower() for kw in _keywords())]
    with ex_c2:
        st.button("Select matching", use_container_width=True, on_click=_select_keyword_matches)
    with ex_c3:
        st.button("Exclude matching", use_container_width=True, on_click=_exclude_keyword_matches)
    if excl_input.strip():
        _n_match_all = sum(any(kw in c.lower() for kw in _keywords()) for c in auto_features)
        st.caption(f"{_n_match_all} of the {len(auto_features)} available columns contain it.")
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
            shown_rm = would_remove[:60]
            rm_rows = [shown_rm[i:i+3] for i in range(0, len(shown_rm), 3)]
            for row in rm_rows:
                grid = st.columns(3)
                for j, name in enumerate(row):
                    highlighted = _highlight(name, keywords)
                    grid[j].markdown(
                        f'<span style="font-family:monospace;font-size:0.88em">{highlighted}</span>',
                        unsafe_allow_html=True,
                    )
            if len(would_remove) > 60:
                st.caption(f"… and {len(would_remove) - 60} more.")
        else:
            st.caption("No currently selected columns match that keyword.")

if not feature_cols:
    st.info("No features selected yet. Choose the first and last feature column above, pick features "
            "in the list, or select them by keyword.")
    st.stop()

# Features ComBat cannot harmonize: no value at all, or one value for everyone
_empty_feats = [c for c in feature_cols if df[c].notna().sum() == 0]
_const_feats = [c for c in feature_cols if c not in set(_empty_feats) and df[c].nunique(dropna=True) <= 1]
if _empty_feats or _const_feats:
    st.warning(
        "Some selected features cannot be harmonized, so ComBat leaves them as they are; consider "
        "excluding them (for example with \"Exclude matching\" above)."
        + (f"\n\n- No value for any participant ({len(_empty_feats)}): " + ", ".join(_empty_feats[:10])
           + (" …" if len(_empty_feats) > 10 else "") if _empty_feats else "")
        + (f"\n\n- The same value for every participant ({len(_const_feats)}): " + ", ".join(_const_feats[:10])
           + (" …" if len(_const_feats) > 10 else "") if _const_feats else ""))

sites = df[site_col].dropna().unique()
st.info(f"Batches (n = {len(sites)}): **{', '.join(natural_sorted(sites.astype(str)))}**")

st.markdown("**Multiple imaging modalities or measure types (optional)**")

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
    """Suggested grouping tokens, one per guessed measure, in column order
    (see src/naming.py); the user checks and edits them."""
    return list(dict.fromkeys(measure_keys(cols).values()))


def _token_matches(token, cols):
    """Columns a grouping token selects: a token ending in "_" (a prefix such
    as FA_) selects the names that start with it, when there are any; any
    other token selects the names that contain it."""
    t = token.lower()
    if t.endswith("_"):
        starts = [c for c in cols if c.lower().startswith(t)]
        if starts:
            return starts
    return [c for c in cols if t in c.lower()]


modality_groups = None
use_modality = st.checkbox(
    "My selected features include multiple modalities or measure types. "
    "Harmonize each group separately",
    value=False, key="use_modality_groups",
)

if use_modality:
    st.markdown(
        "Define one group per modality or measure type using a text token that "
        "appears in that group's column names, for example a prefix such as `FA_`, "
        "`MD_` or `fMRI_`, which selects the names that start with it; any other "
        "text selects the names that contain it. Enter one token per line. "
        "Each selected feature is assigned to the first group whose token it "
        "matches. A group should be one measure, not a brain region."
    )
    if st.button("Suggest groups from column names"):
        st.session_state["modality_tokens_text"] = "\n".join(
            _suggest_modality_tokens(feature_cols)
        )
        st.rerun()
    tokens_text = st.text_area(
        "One token per line",
        key="modality_tokens_text",
        placeholder="FA_\nMD_\nfMRI_",
        height=130,
    )
    tokens = [t.strip() for t in tokens_text.splitlines() if t.strip()]
    if tokens:
        groups, assigned = {}, set()
        for t in tokens:
            matched = [c for c in _token_matches(t, feature_cols) if c not in assigned]
            if matched:
                groups[t] = matched
                assigned.update(matched)
        unassigned = [c for c in feature_cols if c not in assigned]

        st.caption(
            "Suggested groups are guesses from the column names, and naming differs "
            "between datasets: check that each group is one measure (for example FA, "
            "cortical thickness or a T1w/T2w ratio) and edit the list if not."
        )
        _small = {g: c for g, c in groups.items() if len(c) <= 2}
        if _small:
            st.warning(
                f"{len(_small)} group(s) have only one or two columns: "
                + "; ".join(f"{g} ({', '.join(c)})" for g, c in _small.items())
                + ". Such a small group is often a brain region or a single summary "
                "value rather than a measure. Check it; a one-column group is "
                "harmonized without Empirical Bayes, which needs at least two features."
            )

        if groups:
            st.markdown(f"**{len(groups)} groups** (click a group to show its columns)")
            for g, cols in groups.items():
                with st.expander(f"{g}: {len(cols)} column{'s' if len(cols) != 1 else ''} (show)"):
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
# STEP 4 — Run
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Data preview")
st.caption("Overview of the selected features, one row per participant sorted by batch, "
           "with covariates first. Values are z-scored per column and missing values are black. "
           "This first view is before any exclusion; once features or batches are excluded below, "
           "a second view after the exclusions appears at the end of the batch exclusion section.")


@st.cache_data(show_spinner=False, max_entries=8)
def _matrix_figure(data_key, rows_key, site_col, covariate_cols, feats, groups, marked,
                   missing_only, title, _rows_df, order_by_missing=False):
    """Cached data-matrix figure; the table itself (_rows_df) is identified by the keys."""
    return plot_data_matrix(
        _rows_df, site_col, list(covariate_cols), list(feats),
        modality_groups={g: list(c) for g, c in groups} if groups else None,
        marked_block=(marked[0], list(marked[1])) if marked else None,
        missing_only=missing_only, title=title, order_by_missing=order_by_missing)


def _show_matrix_preview(rows_df, rows_key, feats, title, marked=None, missing_only=None):
    covs = [c for c in [age_col, sex_col] + extra_continuous + extra_categorical
            if c and c in rows_df.columns]
    groups = tuple((g, tuple(c)) for g, c in modality_groups.items()) if modality_groups else None
    try:
        fig, note = _matrix_figure(
            data_key, rows_key, site_col, tuple(covs), tuple(feats), groups,
            (marked[0], tuple(marked[1])) if marked else None,
            bool(st.session_state.get("preview_missing_only", False)) if missing_only is None
            else missing_only, title, rows_df,
            bool(st.session_state.get("preview_order_missing", False)))
        st.plotly_chart(fig, use_container_width=True)
        if note:
            st.caption(note)
    except Exception as _preview_err:
        st.error(f"Preview could not be generated: {_preview_err}")


_excluded_before = [d.get("Feature") for d in st.session_state.get("_excl_feat_list", [])
                    if d.get("Feature") in df.columns]
# participants excluded by missing percentage (row labels of the table loaded now)
_excl_part_rows = [d["row"] for d in st.session_state.get("_excl_part_list", [])
                   if d.get("data_key") == data_key and d.get("row") in df.index]
_model_cols_now = [c for c in [site_col, age_col, sex_col] + extra_continuous + extra_categorical
                   if c and c in df.columns]
_nothing_missing = not df[list(feature_cols) + _excluded_before + _model_cols_now].isna().any().any()

# always offered, also when nothing is missing (an all-grey view confirms it)
_pv1, _pv2 = st.columns(2)
with _pv1:
    st.toggle("Show missing values only", key="preview_missing_only",
              help="Shows only where values are missing (black) and present (grey), "
                   "like the vis_miss plot of the R package naniar. Applies to both previews.")
with _pv2:
    st.toggle("Group participants and features with similar missing values", key="preview_order_missing",
              help="Orders the participants within each batch, and the features within each modality "
                   "group, so that similar missing-value patterns sit together (clustering of the missing "
                   "values). Batches and groups keep their order. Applies to both previews and to the map "
                   "of missing values.")
_preview_excluded = [d.get("Feature") for d in st.session_state.get("_excl_feat_list", [])
                     if d.get("Feature") in df.columns]
_show_matrix_preview(
    df, "all", list(feature_cols) + [f for f in _preview_excluded if f not in feature_cols],
    "Before exclusions",
    marked=("Excluded for missingness", _preview_excluded) if _preview_excluded else None)

# ── Missing value assessment ────────────────────────────────────────────────
st.markdown("---")
if _nothing_missing:
    st.markdown("**Missing values**")
    st.success(f"No missing values in the selected features or in the model variables "
               f"(batch and covariates), so all {len(df)} participants are used and there is "
               f"nothing to assess or exclude.")
    _show_miss_tools = st.toggle("Show the missing-value tools anyway", key="show_missing_tools")
else:
    _show_miss_tools = True

if _show_miss_tools:
    st.markdown("**Missing value assessment (optional)**")
    st.caption(
        "This section is also useful for cleaning the data in general: it shows which participants "
        "and features have missing values, so you can spot problems in the table (for example a "
        "processing step that failed, a measure missing for a whole site, or participants with almost "
        "no data) and decide what to fix or exclude before harmonizing. For harmonization itself, "
        "two kinds of missingness matter, and this section shows them "
        "separately. neuroCombat requires a complete feature matrix when Empirical Bayes is "
        "used, because Empirical Bayes pools information across features, so a participant with "
        "any missing feature is dropped from that run. When Empirical Bayes is not used, features "
        "can be harmonized one at a time and this completeness requirement is relaxed."
    )

    st.markdown("**Missingness in the model variables (for example, batch, age, sex, and added covariates)**")

    # ── Model-variable (demographic) missingness — always shown ──
    # A participant missing the site, age, sex, or any added covariate is dropped
    # from EVERY ComBat run, whatever the feature-missing method chosen at Step 4.
    _model_vars = [c for c in [site_col, age_col, sex_col] + extra_continuous + extra_categorical
                   if c in df.columns]
    _model_miss = df[_model_vars].isna().sum()
    _n_model_incomplete = int(df[_model_vars].isna().any(axis=1).sum())
    if _n_model_incomplete > 0:
        st.warning(
            f"{_n_model_incomplete} of {len(df)} participants are missing at least one "
            f"model variable (batch, age, sex, or an added covariate). They are dropped "
            f"from every ComBat run, whatever the feature-missing method below, and are "
            f"not in the harmonized table."
        )
        _mm = pd.DataFrame({
            "Model variable": _model_vars,
            "Participants missing (n)": _model_miss.values,
            f"Participants missing (% of {len(df)})": (100 * _model_miss / len(df)).round(1).values,
        })
        _mm = _mm[_mm["Participants missing (n)"] > 0].sort_values("Participants missing (n)", ascending=False)
        st.dataframe(_mm, use_container_width=True, hide_index=True)

        # Which participants these are, on request; the ID column is guessed from its name.
        if st.toggle("Show participants with missing model variables/covariates", key="show_missing_ids"):
            _id_guess = _guess_id_col(df, exclude=set(_model_vars) | set(feature_cols))
            _id_opts = [c for c in df.columns if c not in set(feature_cols)]
            _id_col = st.selectbox(
                "Participant ID column", _id_opts,
                index=_id_opts.index(_id_guess) if _id_guess in _id_opts else 0, key="id_col",
                help="Guessed from the column names; pick another column if this is not the participant ID.")
            _inc = df[df[_model_vars].isna().any(axis=1)]
            _ids = pd.DataFrame({
                "Participant": _inc[_id_col].astype(str).values,
                "Missing":     [", ".join(v for v in _model_vars if pd.isna(r[v]))
                                for _, r in _inc[_model_vars].iterrows()],
            })
            st.dataframe(_ids, use_container_width=True, hide_index=True,
                         height=min(38 + 35 * len(_ids), 300))
    else:
        st.caption("All participants have complete batch, age, sex, and added covariates.")

    st.markdown("**Missingness in the imaging features**")
    st.caption("\"Assess missing values\" shows how many selected features and participants have "
               "missing values: per modality group (when groups are defined above), per feature, per "
               "participant with a figure by batch, and a map of every missing value. It only shows "
               "them; nothing is changed or excluded.")
    if st.button("Assess missing values"):
        st.session_state["_miss_assess_on"] = True   # stays open until hidden
    if st.session_state.get("_miss_assess_on"):
        st.button("Hide the missing-value assessment",
                  on_click=lambda: st.session_state.update(_miss_assess_on=False))
        try:
            assess_cols = [c for c in [age_col, sex_col] + extra_continuous + extra_categorical
                           if c in df.columns] + list(feature_cols)
            n_total = len(df)
            miss_n = df[assess_cols].isna().sum()
            miss_pct = (100 * miss_n / n_total) if n_total else miss_n * 0
            feats_l = list(feature_cols)
            _cov_set = set([age_col, sex_col] + extra_continuous + extra_categorical)
            # Groups only as defined under "Multiple imaging modalities or measure
            # types"; nothing is guessed here.
            _col_group = {c: g for g, gcols in (modality_groups or {}).items() for c in gcols}

            def _grp_of(c):
                if c in _cov_set:
                    return "(covariate)"
                return _col_group.get(c, "Not in a group")
            measure_of = {c: _grp_of(c) for c in feats_l} if modality_groups else {}
            measures = list(dict.fromkeys(measure_of.values()))

            feat_isna = df[feats_l].isna()
            feat_pct = 100 * feat_isna.mean()
            part_pct = 100 * feat_isna.mean(axis=1)
            # ── features by share missing: the curve, then the threshold under it ──
            st.markdown("**How many features and participants are above each percentage missing**")
            st.caption("The first curve gives, for every percentage x, the features with more than x% of the "
                       "participants missing" + (", one line per modality group as defined above"
                                                 if modality_groups else
                       " (define the groups under \"Multiple imaging modalities or measure types\" above "
                       "to get one line per modality or measure type)")
                       + "; the second, the participants missing more than x% of the selected features. "
                       "Set a threshold in the box under each curve, or click a point on a line to see who or "
                       "what is above it; both can exclude them with that threshold.")
            _f_thr_now = float(st.session_state.get("miss_assess_thr", 10.0) or 0.0)
            _p_thr_now = float(st.session_state.get("miss_assess_part_thr", 10.0) or 0.0)
            _ck = st.session_state.get("_miss_curve_n", 0)
            _pid_c = st.session_state.get("id_col") or _guess_id_col(df, exclude=set(_model_vars) | set(feats_l))
            _excl_rows_set = set(_excl_part_rows)

            def _part_entries(rows):
                return [{"data_key": data_key, "row": r,
                         "Participant": (str(df.at[r, _pid_c]) if _pid_c in df.columns
                                         else f"Row {df.index.get_loc(r) + 1}"),
                         "Batch": "—" if pd.isna(df.at[r, site_col]) else str(df.at[r, site_col]),
                         "Features missing (%)": round(float(part_pct[r]), 1)}
                        for r in rows if r not in _excl_rows_set]

            _clicked = None
            # features
            _ev_f = st.plotly_chart(plot_missing_curve(feat_pct, measure_of or None, _f_thr_now),
                                    use_container_width=True, on_select="rerun", selection_mode="points",
                                    key=f"miss_curve_f_{_ck}")
            _pts = (_ev_f or {}).get("selection", {}).get("points", []) if _ev_f else []
            if _pts:
                _clicked = ("features", float(_pts[0]["x"]))
            fc1, fc2 = st.columns([1, 3])
            with fc1:
                a_thr = st.number_input("Threshold: participants missing (%)", min_value=0.0, max_value=100.0,
                                        value=10.0, step=1.0, key="miss_assess_thr")
            _f_over = feat_pct[feat_pct > a_thr].sort_values(ascending=False)
            with fc2:
                st.markdown(
                    f"<br>**Features:** {int((feat_pct > 0).sum())} of {len(feats_l)} "
                    f"({100 * (feat_pct > 0).mean():.1f}%) have at least one missing value; "
                    f"**{len(_f_over)} ({100 * len(_f_over) / max(len(feats_l), 1):.1f}%)** have more than "
                    f"{a_thr:g}% of the participants missing.", unsafe_allow_html=True)
                if len(_f_over):
                    st.button(f"Exclude these {len(_f_over)} feature(s)", key="curve_excl_feats",
                              on_click=_record_feature_exclusion,
                              args=(list(_f_over.index), a_thr, _f_over.to_dict(), modality_groups))
            # participants
            _ev_p = st.plotly_chart(plot_missing_curve(part_pct, None, _p_thr_now, kind="participants"),
                                    use_container_width=True, on_select="rerun", selection_mode="points",
                                    key=f"miss_curve_p_{_ck}")
            _pts = (_ev_p or {}).get("selection", {}).get("points", []) if _ev_p else []
            if _pts:
                _clicked = ("participants", float(_pts[0]["x"]))
            pc1, pc2 = st.columns([1, 3])
            with pc1:
                p_thr = st.number_input("Threshold: features missing (%)", min_value=0.0, max_value=100.0,
                                        value=10.0, step=1.0, key="miss_assess_part_thr")
            _p_over = part_pct[(part_pct > p_thr) & ~part_pct.index.isin(_excl_part_rows)]
            with pc2:
                st.markdown(
                    f"<br>**Participants:** **{int((part_pct > p_thr).sum())} of {n_total} "
                    f"({100 * (part_pct > p_thr).mean():.1f}%)** miss more than {p_thr:g}% of the features"
                    + (f" ({int((part_pct > p_thr).sum()) - len(_p_over)} of them already excluded)"
                       if int((part_pct > p_thr).sum()) != len(_p_over) else "") + ".",
                    unsafe_allow_html=True)
                if len(_p_over):
                    st.button(f"Exclude these {len(_p_over)} participant(s)", key="curve_excl_parts",
                              on_click=_record_participant_exclusion,
                              args=(_part_entries(_p_over.index), p_thr, data_key))
            if _clicked:
                # new chart keys clear the click, so the window opens once per click
                st.session_state["_miss_curve_n"] = _ck + 1
                _entries = (_part_entries(part_pct[part_pct > _clicked[1]].index)
                            if _clicked[0] == "participants" else None)
                _missing_at_share(_clicked[1], feat_pct, part_pct, measure_of, kind=_clicked[0],
                                  part_entries=_entries, groups=modality_groups, key_of_data=data_key)

            # ── per modality group (only when the groups are defined) ──
            if modality_groups:
                st.markdown("**Missing values per modality group**")
                st.caption("Each bar is one modality group as defined above; its features are counted by "
                           "how many of the participants miss them (%).")
                st.plotly_chart(plot_missing_by_measure(feat_pct, measure_of), use_container_width=True)
            _per_meas = []
            for m in (measures if modality_groups else []):
                cols_m = [c for c in feats_l if measure_of[c] == m]
                p_m, na_m = feat_pct[cols_m], feat_isna[cols_m]
                _per_meas.append({
                    "Modality group": m, "Features": len(cols_m),
                    "With any missing": int((p_m > 0).sum()),
                    f"Over {a_thr:g}% missing": int((p_m > a_thr).sum()),
                    "Median % missing": round(float(p_m.median()), 1),
                    "Highest % missing": round(float(p_m.max()), 1),
                    "Participants missing some": int(na_m.any(axis=1).sum()),
                    "Participants missing all": int(na_m.all(axis=1).sum()),
                })
            if _per_meas:
                st.dataframe(pd.DataFrame(_per_meas), use_container_width=True, hide_index=True)

            # ── per feature ──
            st.markdown("**Missing values per feature**")
            miss_df = pd.DataFrame({
                "Variable":      assess_cols,
                "Modality group": [_grp_of(c) for c in assess_cols],
                "Participants missing (n)": miss_n.values,
                f"Participants missing (% of {n_total})": miss_pct.round(1).values,
                "Participants with a value (n)": (n_total - miss_n.values),
            })
            miss_df = miss_df[miss_df["Participants missing (n)"] > 0].sort_values(
                "Participants missing (n)", ascending=False).reset_index(drop=True)
            if not modality_groups:
                miss_df = miss_df.drop(columns="Modality group")

            if len(miss_df) == 0:
                st.success("No missing values in the selected features or covariates.")
            else:
                st.caption(
                    f"{len(miss_df)} of {len(assess_cols)} variables have at least one "
                    f"missing value; the remaining {len(assess_cols) - len(miss_df)} are "
                    "complete. Sorted by count, most missing first."
                )
                st.dataframe(miss_df, use_container_width=True, hide_index=True)

            # ── per batch ──
            st.markdown("**Missing values per batch**")
            _has_batch = df[site_col].notna()
            _pid = st.session_state.get("id_col") or _guess_id_col(df, exclude=set(_model_vars) | set(feats_l))
            _bt = []
            for b in natural_sorted(df.loc[_has_batch, site_col].astype(str).unique()):
                _in = _has_batch & (df[site_col].astype(str) == b)
                _na_b, _pp_b = feat_isna[_in], part_pct[_in]
                _row = {"Batch": b, "Participants (n)": int(_in.sum()),
                        f"Participants missing more than {p_thr:g}% of the features (n)": int((_pp_b > p_thr).sum()),
                        f"Participants missing more than {p_thr:g}% of the features (% of the batch)":
                            round(100 * float((_pp_b > p_thr).mean()), 1),
                        "Feature values missing (% of the batch's values)": round(100 * float(_na_b.values.mean()), 2),
                        "Features with any missing value in the batch (n)": int(_na_b.any().sum())}
                for m in (measures if modality_groups else []):
                    cols_m = [c for c in feats_l if measure_of[c] == m]
                    _row[f"{m}: values missing (%)"] = round(100 * float(_na_b[cols_m].values.mean()), 2)
                _bt.append(_row)
            st.dataframe(pd.DataFrame(_bt), use_container_width=True, hide_index=True)
            if (~_has_batch).any():
                st.caption(f"{int((~_has_batch).sum())} participant(s) without a batch value are not in this table.")
            _btab = pd.DataFrame(_bt)
            st.plotly_chart(plot_missing_by_batch_bars(
                list(_btab["Batch"]), list(_btab["Participants (n)"]),
                list(_btab["Feature values missing (% of the batch's values)"]),
                list(_btab[f"Participants missing more than {p_thr:g}% of the features (% of the batch)"]),
                p_thr, 100 * float(feat_isna[_has_batch].values.mean()),
                100 * float((part_pct[_has_batch] > p_thr).mean())), use_container_width=True)
            st.plotly_chart(plot_missing_present_by_batch(
                list(_btab["Batch"]), list(_btab["Participants (n)"]),
                list(_btab["Feature values missing (% of the batch's values)"])), use_container_width=True)
            if modality_groups:
                _grid = _btab.set_index("Batch")[[f"{m}: values missing (%)" for m in measures]]
                _grid.columns = measures
                st.plotly_chart(plot_missing_batch_by_group(_grid), use_container_width=True)

            # ── per participant ──
            st.markdown("**Missing values per participant**")
            _pid = st.session_state.get("id_col") or _guess_id_col(df, exclude=set(_model_vars) | set(feats_l))
            _rows = part_pct > 0
            _first = (df.loc[_rows, _pid].astype(str).values if _pid in df.columns
                      else np.flatnonzero(_rows.values) + 1)
            ptab = pd.DataFrame({
                (_pid if _pid in df.columns else "Row in the table"): _first,
                "Batch": df.loc[_rows, site_col].astype(str).where(_has_batch[_rows], "—").values,
                "Features missing (n)": feat_isna[_rows].sum(axis=1).values,
                f"Features missing (% of the {len(feats_l)} selected)": part_pct[_rows].round(1).values,
            })
            if 1 < len(measures) <= 20:
                for m in measures:
                    cols_m = [c for c in feats_l if measure_of[c] == m]
                    ptab[f"{m}: features missing (% of {len(cols_m)})"] = (
                        100 * feat_isna.loc[_rows, cols_m].mean(axis=1)).round(1).values
            ptab = ptab.sort_values("Features missing (n)", ascending=False, kind="stable").reset_index(drop=True)
            if len(ptab) == 0:
                st.caption("No participant misses a selected feature.")
            else:
                st.caption(f"{len(ptab)} of {n_total} participants miss at least one selected feature; "
                           "most missing first." + ("" if 1 < len(measures) <= 20 or len(measures) == 1
                                                    else " Columns per modality group are shown for up to 20 groups."))
                st.dataframe(ptab, use_container_width=True, hide_index=True,
                             height=min(38 + 35 * len(ptab), 400))

            # ── map ──
            st.markdown("**Map of missing values**")
            # missing values at a glance, as vis_miss in the R package naniar
            _show_matrix_preview(df, "all", feature_cols, "Missing values", missing_only=True)
        except Exception as _miss_err:
            st.error(f"Missing-value assessment could not be generated: {_miss_err}")

    # ── Exclude features by missing percentage ───────────────────────────────────
    st.markdown("**Exclude features by missing percentage (optional)**")
    st.caption(
        "Optionally set a threshold to list the selected features whose missing "
        "percentage exceeds it, then remove them from the selection in one step. The "
        "box is empty by default, so nothing is filtered until you enter a value. "
        "Missing percentage is computed per feature across all uploaded participants. "
        "When modality groups are defined above, the exceedances are broken down by group."
    )

    def _exclude_high_missing_features(cols_to_drop, threshold, miss_pcts_dict=None):
        _record_feature_exclusion(cols_to_drop, threshold, miss_pcts_dict or {}, modality_groups)

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
        _ex_df = pd.DataFrame(_excluded_so_far)
        _ex_cols = ["Feature"] + (["Modality group"] if "Modality group" in _ex_df.columns
                                  and _ex_df["Modality group"].notna().any() else []) + ["Participants missing (%)"]
        _ex_df = _ex_df[_ex_cols]
        st.markdown(f"**Excluded: {len(_ex_df)} feature(s)** (removed from the selection)")
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
            # modality group per feature: only the groups defined above, never guessed
            col_to_measure = {c: g for g, cols in (modality_groups or {}).items() for c in cols}
            st.warning(
                f"{len(over)} of {len(feature_cols)} selected features exceed {miss_thr:.0f}% missing."
                + (" Breakdown by modality group below." if modality_groups else "")
            )
            if modality_groups:
                measures = {}
                for c in feature_cols:
                    measures.setdefault(col_to_measure.get(c, "Not in a group"), []).append(c)
                # per-group summary: how many exceed the threshold in each group
                summ = pd.DataFrame([
                    {
                        "Modality group":          meas,
                        "Features over threshold": int((feat_miss_pct.reindex(mcols) > miss_thr).sum()),
                        "Total features":          len(mcols),
                        "Mean missing (%)":        round(float(feat_miss_pct.reindex(mcols).mean()), 1),
                    }
                    for meas, mcols in measures.items()
                ]).sort_values("Features over threshold", ascending=False).reset_index(drop=True)
                st.dataframe(summ, use_container_width=True, hide_index=True)
            # detailed list of the features over threshold
            with st.expander(f"View the {len(over)} features over missingness threshold"):
                over_df = pd.DataFrame({"Variable": list(over.index)})
                if modality_groups:
                    over_df["Modality group"] = [col_to_measure.get(c, "Not in a group") for c in over.index]
                over_df["Participants missing (%)"] = over.values.round(1)
                st.dataframe(over_df, use_container_width=True, hide_index=True)

            st.button(
                f"Exclude these {len(over)} feature(s) from the selection",
                on_click=_exclude_high_missing_features, args=(list(over.index), miss_thr, over.to_dict()),
            )

    # ── Exclude participants by missing percentage ──────────────────────────────
    st.markdown("**Exclude participants by missing percentage (optional)**")
    st.caption(
        "Optionally set a threshold to list the participants missing more than that share of the "
        "selected features, then exclude them in one step. The box is empty by default, so nobody is "
        "excluded until you enter a value. The share is computed over the features selected now, so "
        "exclude features first if you want to. Excluded participants are left out of harmonization "
        "and of the harmonized table, and the report lists them."
    )
    _pid_x = st.session_state.get("id_col") or _guess_id_col(df, exclude=set(_model_vars) | set(feature_cols))

    def _part_label(r):
        if _pid_x in df.columns and pd.notna(df.at[r, _pid_x]):
            return str(df.at[r, _pid_x])
        return f"Row {df.index.get_loc(r) + 1}"

    def _exclude_participants(rows, threshold, pcts):
        _record_participant_exclusion(
            [{"data_key": data_key, "row": r, "Participant": _part_label(r),
              "Batch": "—" if pd.isna(df.at[r, site_col]) else str(df.at[r, site_col]),
              "Features missing (%)": round(float(pcts[r]), 1)} for r in rows],
            threshold, data_key)

    def _undo_participant_exclusions():
        st.session_state["_excl_part_list"] = []
        st.session_state["_excl_part_info"] = {"n": 0, "threshold": None}

    _excl_parts_now = [d for d in st.session_state.get("_excl_part_list", []) if d.get("data_key") == data_key]
    if _excl_parts_now:
        st.markdown(f"**Excluded: {len(_excl_parts_now)} participant(s)** (left out of harmonization)")
        _xp = pd.DataFrame(_excl_parts_now)[["Participant", "Batch", "Features missing (%)"]]
        st.dataframe(_xp, use_container_width=True, hide_index=True, height=min(38 + 35 * len(_xp), 300))
        st.button("Undo all participant exclusions", on_click=_undo_participant_exclusions)

    pthr_col = st.columns([1, 2])[0]
    with pthr_col:
        part_thr = st.number_input(
            "Missing % threshold for participants", min_value=0.0, max_value=100.0,
            value=None, step=5.0, key="miss_part_thr", placeholder="e.g. 50",
        )
    if part_thr is None:
        st.caption("Enter a percentage above to list the participants who exceed it.")
    else:
        _pp = df[list(feature_cols)].isna().mean(axis=1) * 100
        _pp = _pp.drop(index=_excl_part_rows)
        _over_p = _pp[_pp > part_thr].sort_values(ascending=False, kind="stable")
        if len(_over_p) == 0:
            st.caption(f"No participant misses more than {part_thr:g}% of the selected features.")
        else:
            st.warning(f"{len(_over_p)} of {len(_pp)} participants miss more than {part_thr:g}% "
                       "of the selected features.")
            with st.expander(f"View the {len(_over_p)} participants over the threshold"):
                st.dataframe(pd.DataFrame({
                    "Participant": [_part_label(r) for r in _over_p.index],
                    "Batch": ["—" if pd.isna(df.at[r, site_col]) else str(df.at[r, site_col]) for r in _over_p.index],
                    "Features missing (%)": _over_p.values.round(1),
                }), use_container_width=True, hide_index=True)
            st.button(f"Exclude these {len(_over_p)} participant(s)", on_click=_exclude_participants,
                      args=(list(_over_p.index), part_thr, _over_p.to_dict()))

    # ── How missing feature values are handled ──────────────────────────────────
    st.markdown("**How remaining missing values are handled**")
    _feat_na = df[list(feature_cols)].isna()
    if bool(_feat_na.any().any()):
        _cov_ok = df[[site_col] + continuous_covariates + categorical_covariates].notna().all(axis=1)
        _cc_all = int((~_feat_na.any(axis=1) & _cov_ok).sum())
        st.warning(
            f"Your selected features contain missing values, and the pattern looks "
            f"scattered. With strict complete-case harmonization a participant must be "
            f"complete across every feature in a run, which would keep only {_cc_all} "
            f"of {len(df)} participants. Choose how to handle missing values below. "
            f"For scattered missing data, pattern grouping keeps every observed value "
            f"and keeps Empirical Bayes; median imputation also keeps Empirical Bayes "
            f"but suits data with few missing values; the per-feature option runs "
            f"without Empirical Bayes."
        )

    st.caption(
        "How the remaining missing values are handled depends on Empirical Bayes. If Empirical Bayes is not used, "
        "missing values are less of a problem, because each feature is harmonized on its own available data, "
        "so no imputation or pattern grouping is needed (this is the per-feature option). If Empirical Bayes "
        "is used, which is the default, the tool imputes missing feature values with each feature's median "
        "for harmonization only, and the imputed values are never reported. This median default suits data "
        "with few missing values; for scattered missing data, grouping features by shared missing pattern "
        "keeps every observed value. Complete cases only is also available."
    )

    _MISS_OPTS = {
        "Complete case (drop participants missing any feature in a run)": "complete",
        "Impute missing with median, keep Empirical Bayes":              "impute",
        "Group features by shared missing pattern, keep Empirical Bayes": "pattern",
        "Do not use Empirical Bayes (each feature on its own)":          "per_feature",
    }
    missing_mode_label = st.selectbox(
        "Missing-value handling", list(_MISS_OPTS.keys()), index=1, key="missing_mode",
        help="Complete case drops any participant missing a feature in the run. "
             "Median imputation fills missing values with each feature's median, "
             "harmonizes with Empirical Bayes, then restores the missing cells to "
             "blank; it suits data with few missing values, because many filled-in "
             "medians pull the batches toward each other and can under-correct the "
             "batch effects. Pattern grouping harmonizes together the features that "
             "are missing in the same participants, keeping Empirical Bayes within "
             "each group; it suits scattered missing data. A feature whose missing "
             "pattern no other feature shares is harmonized on its own, without "
             "Empirical Bayes, which needs at least two features. Not using Empirical "
             "Bayes harmonizes each feature on its own complete cases; Step 4 then "
             "runs EB=FALSE only.",
    )
    missing_mode = _MISS_OPTS[missing_mode_label]
    per_feature = (missing_mode == "per_feature")

    st.markdown("**What each method does**")
    _methods = pd.DataFrame([
        {"Method": "Complete case",
         "What it does": "Drops any participant missing any feature in the run",
         "Empirical Bayes": "Yes",
         "Participants used": "Only participants complete on all features"},
        {"Method": "Median imputation (default)",
         "What it does": "Fills each missing value with that feature's median, harmonizes, then restores the originally missing cells to blank; suits data with few missing values",
         "Empirical Bayes": "Yes",
         "Participants used": "All participants"},
        {"Method": "Missing-pattern grouping",
         "What it does": "Harmonizes together the features missing in the same participants, one Empirical Bayes run per pattern; suits scattered missing data",
         "Empirical Bayes": "Yes, except for a feature whose missing pattern no other feature shares",
         "Participants used": "Each participant, for the features they have"},
        {"Method": "Do not use Empirical Bayes",
         "What it does": "Harmonizes each feature on its own complete cases, one feature at a time; Step 4 then runs EB=FALSE only",
         "Empirical Bayes": "No",
         "Participants used": "Each participant, for the features they have"},
    ])
    # a plain table without row numbers, with the text wrapped
    st.markdown("| " + " | ".join(_methods.columns) + " |\n|" + "---|" * len(_methods.columns) + "\n"
                + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in _methods.itertuples(index=False)))
    # How many participants remain under the chosen handling (after the participants
    # excluded by missing percentage)
    _work = df.drop(index=_excl_part_rows)
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
    _cov_drop = int((~_cov_ok).sum())
    _feat_drop = max(_dropped - _cov_drop, 0)
    _drop_parts = []
    if _cov_drop:
        _drop_parts.append(f"{_cov_drop} are missing a model variable (batch, age, sex, or a covariate)")
    if _feat_drop:
        _drop_parts.append(f"{_feat_drop} have missing feature values under this method")
    if _dropped == 0:
        st.info(f"All **{len(_work)}** participants are used.")
    else:
        _drop_expl = " and ".join(_drop_parts)
        st.info(
            f"Participants with this method: **{len(_work)}** in the data"
            + (f" (after the {len(_excl_part_rows)} excluded by missing percentage)" if _excl_part_rows else "")
            + " "
            f"→ **{_retained}** retained, **{_dropped}** dropped. "
            f"Of the {_dropped} dropped, {_drop_expl}."
        )
else:
    missing_mode = "impute"   # the default; with no missing values all methods give the same result
    per_feature = False

# ── Sample summary and batch exclusion ────────────────────────────────────
st.markdown("---")
st.subheader("Batch exclusion (optional)")
st.caption(
    "Check how many participants each batch has and how they compare, and exclude a batch here "
    "if it should not be harmonized, for example one that is too small or very different from "
    "the rest. Excluded batches are left out of the harmonization and the report. The table shows "
    "the model variables; for other variables, use \"Show a demographic table by batch\" in Step 2."
)
st.dataframe(
    _by_batch_table(df, site_col, [c for c in [age_used, sex_used] + extra_continuous + extra_categorical
                                   if c and c in df.columns]),
    use_container_width=True, hide_index=True)
st.caption("N = participants per batch. Numeric variables: mean (SD); range. Categories: n (%).")

all_sites_n = df[site_col].value_counts()
all_sites_n = all_sites_n[natural_sorted(all_sites_n.index)]
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

if _preview_excluded or excluded_sites or _excl_part_rows:
    st.markdown("**Data preview after exclusions**")
    st.caption("The same view as the data preview above, without the excluded features, participants "
               "and batches. The \"Show missing values only\" switch above applies here too.")
    _after = df.drop(index=_excl_part_rows)
    _show_matrix_preview(
        _after[~_after[site_col].isin(excluded_sites)] if excluded_sites else _after,
        ("without", tuple(sorted(map(str, excluded_sites))), tuple(map(str, _excl_part_rows))),
        feature_cols, "After exclusions")

st.divider()
st.subheader("Evaluation and report options")
st.caption(
    "Select which metrics to compute and include in the supplementary report. "
    "Within-batch consistency by batch (ICC(3,1)) is the primary recommended metric; it directly measures "
    "whether harmonization preserved the internal variability of each batch's data (that is, that changing "
    "batch means and scales did not distort within-batch biological signal). "
    "Age correlations are optional because they may be confounded by motion in pediatric fMRI samples, "
    "or may be unstable in restricted age ranges where developmental trajectories are non-linear."
)

eval_c1, eval_c2 = st.columns(2)
with eval_c1:
    include_icc_by_site = st.checkbox(
        "Within-batch consistency by batch (ICC(3,1)), recommended",
        value=True, key="inc_icc_site",
        help=(
            "ICC(3,1), the consistency form of the intraclass correlation coefficient, between raw and "
            "harmonized values computed within each batch. The consistency form is used because "
            "harmonization intentionally shifts batch means and scales; an absolute-agreement ICC would "
            "penalize that intended correction, whereas the consistency form measures whether the relative "
            "ordering of participants is preserved. This is the primary metric for assessing whether "
            "harmonization preserved within-batch biological variability. Minimum 3 participants per batch "
            "required (pingouin constraint)."
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

use_fdr = st.checkbox(
    "Apply FDR correction (Benjamini-Hochberg) to the association p-values",
    value=False, key="use_fdr",
    help=(
        "When on, age and additional-variable associations are corrected for multiple comparisons "
        "with the Benjamini-Hochberg false discovery rate; when multiple imaging modalities are "
        "defined, the correction is applied within each modality. When off, the uncorrected p-value "
        "is used. The report states the correction method and the number of hypotheses per correction group."
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
st.subheader("Step 4 — ComBat configuration and run")
st.caption("EB=TRUE runs ComBat with Empirical Bayes, which pools information across features to "
           "stabilize each batch's estimates (recommended). EB=FALSE adjusts each feature on its own. "
           "Compare runs both, and the report puts them side by side.")

eb_options = {
    "EB=TRUE  (Empirical Bayes, recommended)":    ("ebt_only",  True,  False),
    "EB=FALSE  (feature-wise, no EB shrinkage)":  ("ebf_only",  False, True),
    "Compare EB=TRUE vs EB=FALSE":                ("compare",   True,  True),
}
eb_label = st.radio("ComBat configuration", list(eb_options.keys()),
                    horizontal=True, index=1 if per_feature else 0, disabled=per_feature)
eb_mode, run_ebt, run_ebf = eb_options[eb_label]
if per_feature:
    st.caption("You chose \"Do not use Empirical Bayes\" for the remaining missing values, so ComBat "
               "runs as EB=FALSE only, each feature on its own; the report shows EB=FALSE results and "
               "the download is the EB=FALSE table.")
    eb_mode, run_ebt, run_ebf = "ebf_only", False, True

if st.button("▶  Run Harmonization", type="primary", use_container_width=True):

    progress = st.progress(0, "Starting...")
    try:
        # ── Apply site exclusion ────────────────────────────────────────────
        _site_excl_mask = (df[site_col].isin(excluded_sites) if excluded_sites
                           else pd.Series(False, index=df.index))
        _part_excl_mask = pd.Series(df.index.isin(_excl_part_rows), index=df.index)
        if excluded_sites or _excl_part_rows:
            df_harm = df[~_site_excl_mask & ~_part_excl_mask].copy().reset_index(drop=True)
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
                    categorical_covariates=categorical_covariates, eb=eb, groups=modality_groups)
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

        # Features harmonized in a run of their own although EB=TRUE was asked:
        # Empirical Bayes needs at least two features, so run_combat runs them
        # feature-wise. Features left blank in a harmonized table are reported too.
        no_eb_features = []
        if harm_ebt is not None:
            if missing_mode == "pattern":
                _runs = missing_pattern_groups(df_harm, feature_cols).values()
            elif modality_groups:
                _runs = modality_groups.values()
            else:
                _runs = [feature_cols]
            no_eb_features = [cols[0] for cols in _runs if len(cols) == 1]
        blank_features = [f for f in feature_cols
                          if any(h is not None and h[f].isna().all() for h in (harm_ebt, harm_ebf))]

        # ── Metrics ────────────────────────────────────────────────────────
        progress.progress(33, "Computing batch deviation...")
        dev_before = site_mean_deviation(df_harm,     feature_cols, site_col)
        dev_ebt    = site_mean_deviation(harm_ebt,    feature_cols, site_col) if harm_ebt is not None else None
        dev_ebf    = site_mean_deviation(harm_ebf,    feature_cols, site_col) if harm_ebf is not None else None

        spm_ebt, spm_ebf = None, None  # Spearman removed; ICC3 is sufficient for consistency

        progress.progress(55, "Computing within-batch ICC(3,1)...")
        # ICC(3,1) is computed within each batch, per feature per batch. The
        # overall, pooled ICC is intentionally not computed. The per-batch values
        # feed the by-batch distribution figure and are summarized per feature as
        # median [IQR] across batches for the feature-level tables.
        icc_site_ebt = (compute_icc_by_site(df_harm, harm_ebt, feature_cols, site_col)
                        if harm_ebt is not None else None)
        icc_site_ebf = (compute_icc_by_site(df_harm, harm_ebf, feature_cols, site_col)
                        if harm_ebf is not None else None)
        icc_ebt = (compute_icc_within_summary(df_harm, harm_ebt, feature_cols, site_col, by=icc_site_ebt)
                   if harm_ebt is not None else None)
        icc_ebf = (compute_icc_within_summary(df_harm, harm_ebf, feature_cols, site_col, by=icc_site_ebf)
                   if harm_ebf is not None else None)
        icc_all = pd.DataFrame()  # overall ICC removed; ICC is within-batch only

        # When multiple modalities are defined, FDR is applied within each modality.
        _feature_group = ({f: g for g, feats in modality_groups.items() for f in feats}
                          if modality_groups else None)

        age_all = pd.DataFrame()
        if include_age_corr and age_col:
            progress.progress(74, "Computing age correlations...")
            age_parts = [age_correlations(df_harm, feature_cols, age_col, "Before harmonization", use_fdr=use_fdr, feature_group=_feature_group)]
            if harm_ebt is not None: age_parts.append(age_correlations(harm_ebt, feature_cols, age_col, "After (EB=TRUE)", use_fdr=use_fdr, feature_group=_feature_group))
            if harm_ebf is not None: age_parts.append(age_correlations(harm_ebf, feature_cols, age_col, "After (EB=FALSE)", use_fdr=use_fdr, feature_group=_feature_group))
            age_all = pd.concat(age_parts, ignore_index=True)

        anc_before = anc_ebt = anc_ebf = None
        if include_cohens_f:
            progress.progress(80, "Computing ANCOVA batch effects...")
            anc_before = ancova_site_effect(df_harm, feature_cols, site_col, age_used, sex_used, "Before")
            anc_ebt    = ancova_site_effect(harm_ebt, feature_cols, site_col, age_used, sex_used, "EB=TRUE")  if harm_ebt is not None else None
            anc_ebf    = ancova_site_effect(harm_ebf, feature_cols, site_col, age_used, sex_used, "EB=FALSE") if harm_ebf is not None else None

        # Per-variable associations for the results tabs. Each checkbox controls
        # only its own tabs: the preserved-covariate toggle contributes Age, Sex,
        # and any extra covariates; the additional-variable toggle contributes the
        # variables selected below. Each variable's model controls for the
        # remaining preserved covariates (excluding itself). A column is evaluated
        # under a single type: if it appears in both lists, the continuous
        # assignment wins, so a variable is never plotted twice.
        assoc_unified_df = pd.DataFrame()
        _feat_set  = set(feature_cols)
        _cont_src  = ((continuous_covariates if include_age_corr else [])
                      + (assoc_cont_vars if include_extra_assoc else []))
        _cat_src   = ((categorical_covariates if include_age_corr else [])
                      + (assoc_cat_vars if include_extra_assoc else []))
        eval_cont  = [c for c in dict.fromkeys(_cont_src)
                      if c in df.columns and c not in _feat_set]
        eval_cat   = [c for c in dict.fromkeys(_cat_src)
                      if c in df.columns and c not in _feat_set and c not in set(eval_cont)]
        if eval_cont or eval_cat:
            progress.progress(82, "Computing per-variable associations...")
            u_parts = [compute_extra_associations(
                df_harm, feature_cols, eval_cont, eval_cat,
                continuous_covariates, categorical_covariates, "Before harmonization",
                use_fdr=use_fdr, feature_group=_feature_group)]
            if harm_ebt is not None:
                u_parts.append(compute_extra_associations(
                    harm_ebt, feature_cols, eval_cont, eval_cat,
                    continuous_covariates, categorical_covariates, "After (EB=TRUE)",
                    use_fdr=use_fdr, feature_group=_feature_group))
            if harm_ebf is not None:
                u_parts.append(compute_extra_associations(
                    harm_ebf, feature_cols, eval_cont, eval_cat,
                    continuous_covariates, categorical_covariates, "After (EB=FALSE)",
                    use_fdr=use_fdr, feature_group=_feature_group))
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

        fig_icc = None  # overall ICC bar chart removed; ICC is within-batch only

        fig_icc_ci = None  # per-feature ICC median/IQR figure left out of the report (Oni, 26 October 2026)

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
        ) if (_have_site_icc and include_icc_by_site) else None

        fig_extra_assoc = None
        extra_assoc_figs = []
        if include_extra_assoc and len(extra_assoc_df) > 0:
            fig_extra_assoc = plot_extra_associations(extra_assoc_df)
            # one figure per additional variable for the report (S4.4.1, S4.4.2, ...)
            for _rv in list(dict.fromkeys(extra_assoc_df["variable"].tolist())):
                _rf = plot_single_association(extra_assoc_df, _rv)
                if _rf is not None:
                    extra_assoc_figs.append({"variable": _rv, "fig": _rf})

        # ── Report ─────────────────────────────────────────────────────────
        progress.progress(94, "Building report...")
        n_retained = len(harm_primary) if harm_primary is not None else 0
        # Table S1 describes the participants retained for harmonization
        demo_df    = demographic_summary(
            df_harm.loc[harm_primary.index] if harm_primary is not None else df_harm,
            site_col, age_col, sex_col)
        # why participants were not harmonized (the counts add up to the total)
        _cov_cols  = continuous_covariates + categorical_covariates
        _no_batch  = df_harm[site_col].isna()
        _no_cov    = (~_no_batch & df_harm[_cov_cols].isna().any(axis=1)) if _cov_cols \
                     else pd.Series(False, index=df_harm.index)
        excl_counts = {
            "batch_excluded":       int(_site_excl_mask.sum()),
            "missing_pct_excluded": int((_part_excl_mask & ~_site_excl_mask).sum()),
            "batch_missing":        int(_no_batch.sum()),
            "cov_missing":          int(_no_cov.sum()),
        }
        excl_counts["features"] = (len(df) - n_retained - excl_counts["batch_excluded"]
                                   - excl_counts["missing_pct_excluded"]
                                   - excl_counts["batch_missing"] - excl_counts["cov_missing"])
        # missing values before harmonization, for the report's data preparation tables:
        # the selected features (with those later excluded) in all uploaded participants
        _thr_a = float(st.session_state.get("miss_assess_thr") or 10.0)
        _thr_p = float(st.session_state.get("miss_assess_part_thr") or 10.0)
        _feats_all = list(feature_cols) + [f for f in _excluded_before if f not in feature_cols]
        _na_all = df[_feats_all].isna()
        _fp_all, _pp_all = 100 * _na_all.mean(), 100 * _na_all.mean(axis=1)
        _xi = st.session_state.get("_excl_info") or {"n": 0, "threshold": None}
        _xpi = st.session_state.get("_excl_part_info") or {"n": 0, "threshold": None}
        missing_overview = {
            "threshold": _thr_a, "n_participants": len(df), "n_features": len(_feats_all),
            "feat_any": int((_fp_all > 0).sum()), "feat_over": int((_fp_all > _thr_a).sum()),
            "part_any": int((_pp_all > 0).sum()), "part_over": int((_pp_all > _thr_p).sum()),
            "part_threshold": _thr_p,
            "feat_excl": int(_xi.get("n") or 0), "feat_excl_thr": _xi.get("threshold"),
            "part_excl": len(_excl_part_rows), "part_excl_thr": _xpi.get("threshold"),
            "by_group": [
                {"Modality group": g, "Features": len(gc),
                 "With any missing": int((_fp_all[gc] > 0).sum()),
                 f"More than {_thr_a:g}% missing": int((_fp_all[gc] > _thr_a).sum()),
                 "Median % missing": round(float(_fp_all[gc].median()), 1)}
                for g, gc in (modality_groups or {}).items() if gc
            ],
        }
        excl_part_list = [{k: d[k] for k in ("Participant", "Batch", "Features missing (%)")}
                          for d in st.session_state.get("_excl_part_list", [])
                          if d.get("data_key") == data_key and d.get("row") in df.index]
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
        if not use_fdr:
            fdr_note = ("No multiple-comparison correction was applied to the association p-values; "
                        "uncorrected p-values are reported.")
        elif modality_groups:
            _grp_sizes = "; ".join(f"{g}, {len(feats)} features" for g, feats in modality_groups.items())
            fdr_note = ("Benjamini-Hochberg false discovery rate correction was applied within each modality, "
                        f"across the features within that modality ({_grp_sizes}).")
        else:
            fdr_note = ("Benjamini-Hochberg false discovery rate correction was applied across all "
                        f"{len(feature_cols)} features.")
        html_report = generate_report(
            df_raw=df, site_col=site_col, age_col=age_col, sex_col=sex_col,
            feature_cols=feature_cols, n_retained=n_retained,
            modality_groups=modality_groups,
            missing_handling=missing_mode,
            missing_summary=missing_summary,
            excl_info=excl_info,
            excl_feat_list=excl_feat_list,
            excl_counts=excl_counts,
            missing_overview=missing_overview,
            excl_part_list=excl_part_list,
            excl_part_info=st.session_state.get("_excl_part_info"),
            no_eb_features=no_eb_features,
            blank_features=blank_features,
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
            extra_continuous=extra_continuous,
            extra_categorical=extra_categorical,
            covariates_cont=continuous_covariates,
            covariates_cat=categorical_covariates,
            ancova_covars=[c for c in (age_used, sex_used) if c],
            include_age=(include_age_corr and bool(age_col)),
            include_cohens_f=include_cohens_f,
            include_icc_by_site=include_icc_by_site,
            include_extra_assoc=(include_extra_assoc and len(extra_assoc_df) > 0),
            extra_assoc_df=extra_assoc_df if len(extra_assoc_df) > 0 else None,
            fig_extra_assoc=fig_extra_assoc,
            extra_assoc_figs=extra_assoc_figs,
            assoc_cont_vars=assoc_cont_vars,
            assoc_cat_vars=assoc_cat_vars,
            fdr_note=fdr_note,
        )

        methods_para = build_methods_paragraph(
            site_col=site_col, age_col=age_col, sex_col=sex_col,
            run_ebf=(run_ebf and harm_ebf is not None),
            github_url=GITHUB_URL,
            extra_continuous=extra_continuous,
            extra_categorical=extra_categorical,
            covariates_cont=continuous_covariates,
            covariates_cat=categorical_covariates,
            ancova_covars=[c for c in (age_used, sex_used) if c],
            include_age=(include_age_corr and bool(age_col)),
            include_cohens_f=include_cohens_f,
            include_icc_by_site=include_icc_by_site,
            include_extra_assoc=(include_extra_assoc and len(extra_assoc_df) > 0),
            assoc_cont_vars=assoc_cont_vars,
            assoc_cat_vars=assoc_cat_vars,
            data_prep_text=data_prep_sentence(
                missing_overview, st.session_state.get("_excl_info"),
                st.session_state.get("_excl_part_info") if _excl_part_rows else None, missing_mode),
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
            no_eb_features=no_eb_features,
            blank_features=blank_features,
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

    _blank = st.session_state.get("blank_features") or []
    if _blank:
        st.warning(
            f"{len(_blank)} feature(s) could not be harmonized and are blank in the "
            f"harmonized table: {', '.join(_blank)}. The report lists them too."
        )
    _no_eb = st.session_state.get("no_eb_features") or []
    if _no_eb:
        st.info(
            f"{len(_no_eb)} feature(s) were harmonized in a run of their own, so "
            f"without Empirical Bayes, which needs at least two features: "
            f"{', '.join(_no_eb)}. The report lists them too."
        )

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
        st.caption(
            "Each point = one (batch, feature) pair. Batch means should cluster near zero after harmonization. "
            "Each feature is standardized to a z-score for display only, so that features measured on different "
            "scales, for example when several measures are harmonized together, can be shown on one axis. This "
            "standardization affects only the figure; ComBat harmonizes the raw feature values."
        )
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
                    "Each box = distribution of ICC(3,1) across features for that batch. "
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
            file_name=f"{data_stem}_harmonization_report.html",
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
                file_name=f"{data_stem}_harmonized_EBT.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with dl3:
        if st.session_state.get("harm_csv_ebf"):
            st.download_button(
                label="⬇  Harmonized data — EB=FALSE",
                data=st.session_state["harm_csv_ebf"],
                file_name=f"{data_stem}_harmonized_EBF.csv",
                mime="text/csv",
                use_container_width=True,
            )
