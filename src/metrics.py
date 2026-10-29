"""
metrics.py
All harmonization evaluation metrics.

Functions return tidy DataFrames with one row per feature (and optionally per site).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pingouin as pg
import warnings
from src.naming import natural_sorted

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 0. Demographic summary — per-site and overall
# ---------------------------------------------------------------------------
def demographic_summary(
    df: pd.DataFrame,
    site_col: str,
    age_col: str,
    sex_col: str,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per site plus a Total row.
    Columns: Site, N, Age mean (SD), Age range, Female n (%)
    """
    rows = []
    female_tokens = {"female", "f", "woman", "1", "1.0"}

    def _n_female(sex_series):
        s = sex_series.dropna().astype(str).str.strip().str.lower()
        return int(s.isin(female_tokens).sum())

    # Age and sex are optional. When their column is absent, the corresponding
    # cell reports "N/A" rather than raising.
    def _age_stats(grp):
        if not age_col or age_col not in grp.columns:
            return "N/A", "N/A"
        av = grp[age_col].dropna()
        if len(av) == 0:
            return "N/A", "N/A"
        return f"{av.mean():.2f} ({av.std():.2f})", f"{av.min():.1f}–{av.max():.1f}"

    def _female_str(grp, n):
        if not sex_col or sex_col not in grp.columns or n == 0:
            return "N/A"
        nf = _n_female(grp[sex_col])
        return f"{nf} ({100 * nf / n:.1f}%)"

    for site in natural_sorted(df[site_col].dropna().unique().astype(str)):
        grp    = df[df[site_col].astype(str) == site]
        am, ar = _age_stats(grp)
        rows.append({
            "Site":          site,
            "N":             len(grp),
            "Age mean (SD)": am,
            "Age range":     ar,
            "Female n (%)":  _female_str(grp, len(grp)),
        })

    am, ar = _age_stats(df)
    rows.append({
        "Site":          "Total",
        "N":             len(df),
        "Age mean (SD)": am,
        "Age range":     ar,
        "Female n (%)":  _female_str(df, len(df)),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Site mean z-score deviation
#    For each feature, z-score values grand-mean; compute site mean z-score.
#    After harmonization these should cluster near zero.
# ---------------------------------------------------------------------------
def site_mean_deviation(df: pd.DataFrame, features: list, site_col: str) -> pd.DataFrame:
    rows = []
    for f in features:
        d = df[[site_col, f]].dropna()
        if len(d) < 5 or d[site_col].nunique() < 2:
            continue
        grand_mean = d[f].mean()
        grand_sd   = d[f].std()
        if grand_sd == 0:
            continue
        d = d.copy()
        d["z"] = (d[f] - grand_mean) / grand_sd
        for site, grp in d.groupby(site_col):
            rows.append({
                "feature":      f,
                "site":         site,
                "site_mean_z":  grp["z"].mean(),
                "n":            len(grp),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Spearman r — raw vs harmonized, per feature
#    Measures how much the rank ordering of participants is preserved.
# ---------------------------------------------------------------------------
def spearman_raw_vs_harm(
    df_raw:  pd.DataFrame,
    df_harm: pd.DataFrame,
    features: list,
    id_col:  str | None = None,
) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in df_raw.columns or f not in df_harm.columns:
            continue
        if id_col and id_col in df_raw.columns and id_col in df_harm.columns:
            merged = df_raw[[id_col, f]].merge(
                df_harm[[id_col, f]], on=id_col, suffixes=("_raw", "_harm")
            ).dropna()
            x = merged[f + "_raw"].values
            y = merged[f + "_harm"].values
        else:
            # align by index
            idx = df_raw[f].dropna().index.intersection(df_harm[f].dropna().index)
            if len(idx) < 10:
                continue
            x = df_raw.loc[idx, f].values
            y = df_harm.loc[idx, f].values

        if len(x) < 10:
            continue
        r, p = spearmanr(x, y)
        rows.append({"feature": f, "spearman_r": float(r), "p_value": float(p), "n": len(x)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. ICC3 — two-way mixed effects, consistency (raw vs harmonized)
#    Koo & Li (2016) thresholds: <0.50 poor, 0.50-0.75 moderate,
#    0.75-0.90 good, >0.90 excellent
# ---------------------------------------------------------------------------
def compute_icc(
    df_raw:  pd.DataFrame,
    df_harm: pd.DataFrame,
    features: list,
    id_col:  str | None = None,
) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in df_raw.columns or f not in df_harm.columns:
            continue
        if id_col and id_col in df_raw.columns and id_col in df_harm.columns:
            merged = df_raw[[id_col, f]].merge(
                df_harm[[id_col, f]], on=id_col, suffixes=("_raw", "_harm")
            ).dropna()
            raw_vals  = merged[f + "_raw"].values
            harm_vals = merged[f + "_harm"].values
            subj_ids  = merged[id_col].values
        else:
            idx = df_raw[f].dropna().index.intersection(df_harm[f].dropna().index)
            if len(idx) < 3:
                continue
            raw_vals  = df_raw.loc[idx, f].values
            harm_vals = df_harm.loc[idx, f].values
            subj_ids  = idx.values

        if len(raw_vals) < 3:
            continue

        long = pd.DataFrame({
            "subject": list(subj_ids) * 2,
            "rater":   ["raw"] * len(raw_vals) + ["harmonized"] * len(harm_vals),
            "value":   list(raw_vals) + list(harm_vals),
        })

        try:
            icc_res = pg.intraclass_corr(
                data=long, targets="subject", raters="rater", ratings="value"
            )
            row3 = icc_res[icc_res["Type"] == "ICC(C,1)"].iloc[0]
            rows.append({
                "feature":    f,
                "icc3":       float(row3["ICC"]),
                "icc3_lower": float(row3["CI95"][0]),
                "icc3_upper": float(row3["CI95"][1]),
                "n":          len(raw_vals),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Age correlation — Pearson r per feature, FDR corrected
# ---------------------------------------------------------------------------
def _apply_significance(res, use_fdr=True, feature_group=None):
    """
    Add significance columns to a per-feature association table from its
    'p_value' column. When use_fdr is True, Benjamini-Hochberg correction is
    applied within each modality group given by feature_group (a mapping from
    feature name to group name); when feature_group is None the correction is
    applied across all features together. When use_fdr is False the uncorrected
    p-value is used. Adds: p_fdr (NaN when uncorrected), sig_fdr (the significance
    flag actually used), correction (label), and n_hyp (the number of hypotheses
    in the correction group used for that feature's decision).
    """
    res = res.copy()
    if len(res) == 0:
        return res
    res["p_fdr"]   = np.nan
    res["sig_fdr"] = False
    res["n_hyp"]   = 0
    if use_fdr:
        res["correction"] = "FDR (Benjamini-Hochberg)"
        if feature_group:
            grp = res["feature"].map(feature_group).fillna("(ungrouped)")
        else:
            grp = pd.Series("all", index=res.index)
        for _, idx in res.groupby(grp).groups.items():
            sub_p = res.loc[idx, "p_value"]
            if len(sub_p) == 0:
                continue
            _, p_fdr, _, _ = multipletests(sub_p, method="fdr_bh")
            res.loc[idx, "p_fdr"]   = p_fdr
            res.loc[idx, "sig_fdr"] = p_fdr < 0.05
            res.loc[idx, "n_hyp"]   = len(sub_p)
    else:
        res["correction"] = "none (uncorrected)"
        res["sig_fdr"]    = res["p_value"] < 0.05
        res["n_hyp"]      = len(res)
    return res


def age_correlations(
    df: pd.DataFrame,
    features: list,
    age_col: str,
    label: str = "",
    use_fdr: bool = True,
    feature_group: dict | None = None,
) -> pd.DataFrame:
    rows = []
    for f in features:
        d = df[[age_col, f]].dropna()
        if len(d) < 10:
            continue
        r, p = pearsonr(d[age_col].values, d[f].values)
        rows.append({"feature": f, "r": float(r), "p_value": float(p),
                     "n": len(d), "harmonization": label})
    res = pd.DataFrame(rows)
    return _apply_significance(res, use_fdr=use_fdr, feature_group=feature_group)


# ---------------------------------------------------------------------------
# 5. ANCOVA — Cohen's f and partial eta-squared for site effect
#    Controls for Age and Sex. Type II sums of squares.
# ---------------------------------------------------------------------------
def ancova_site_effect(
    df: pd.DataFrame,
    features: list,
    site_col: str,
    age_col:  str,
    sex_col:  str,
    label: str = "",
) -> pd.DataFrame:
    rows = []
    for f in features:
        cols_needed = [f, site_col] + ([age_col] if age_col else []) + ([sex_col] if sex_col else [])
        d = df[cols_needed].dropna()
        if len(d) < 10 or d[site_col].nunique() < 2:
            continue
        try:
            safe_f = f.replace(" ", "_").replace("-", "_").replace(".", "_")
            d = d.rename(columns={f: safe_f})
            terms = [f"C({site_col})"]
            if age_col:
                terms.append(f"{age_col}")
            if sex_col:
                terms.append(f"C({sex_col})")
            formula = f"{safe_f} ~ " + " + ".join(terms)
            model   = smf.ols(formula, data=d).fit()
            aov     = sm.stats.anova_lm(model, typ=2)
            site_key = f"C({site_col})"
            if site_key not in aov.index:
                continue
            ss_site  = aov.loc[site_key, "sum_sq"]
            ss_total = aov["sum_sq"].sum()
            eta_sq   = ss_site / ss_total
            cohens_f = float(np.sqrt(eta_sq / max(1 - eta_sq, 1e-10)))
            p_val    = float(aov.loc[site_key, "PR(>F)"])
            rows.append({
                "feature":       f,
                "eta_sq":        float(eta_sq),
                "cohens_f":      cohens_f,
                "p_value":       p_val,
                "harmonization": label,
                "n":             len(d),
            })
        except Exception:
            continue
    res = pd.DataFrame(rows)
    if len(res) > 0 and "p_value" in res.columns:
        _, p_fdr, _, _ = multipletests(res["p_value"], method="fdr_bh")
        res["p_fdr"]   = p_fdr
        res["sig_fdr"] = p_fdr < 0.05
    return res


# ---------------------------------------------------------------------------
# 6. ICC by site — ICC(C,1) per site, distribution across features
# ---------------------------------------------------------------------------
def _icc_c1_two(raw, harm):
    """
    ICC(3,1) = ICC(C,1), two-way, consistency, single measurement, for two "raters"
    (raw and harmonized) over the same n participants (McGraw and Wong 1996; Shrout and
    Fleiss 1979): (MSR - MSE) / (MSR + MSE). The same value as pingouin's ICC(C,1).
    """
    y = np.column_stack([raw, harm]).astype(float)
    n, k = y.shape
    m = y.mean()
    ss_r = k * ((y.mean(axis=1) - m) ** 2).sum()
    ss_c = n * ((y.mean(axis=0) - m) ** 2).sum()
    ss_e = ((y - m) ** 2).sum() - ss_r - ss_c
    ms_r, ms_e = ss_r / (n - 1), ss_e / ((n - 1) * (k - 1))
    den = ms_r + (k - 1) * ms_e
    return float((ms_r - ms_e) / den) if den != 0 else float("nan")


def compute_icc_by_site(df_raw, df_harm, features, site_col, progress=None):
    """
    Within-batch ICC(3,1) between raw and harmonized values, per feature per batch,
    on the participants with both values (at least 3). ``progress(done, total)`` is
    called after each batch, so a long run reports how far it is.
    """
    if site_col not in df_harm.columns:
        return pd.DataFrame()
    rows = []
    sites = natural_sorted(df_harm[site_col].dropna().unique().astype(str))
    for k_site, site in enumerate(sites):
        raw_site  = df_raw[df_raw[site_col].astype(str) == site] if site_col in df_raw.columns else df_raw
        harm_site = df_harm[df_harm[site_col].astype(str) == site]
        common_idx = raw_site.index.intersection(harm_site.index)
        if len(common_idx) >= 3:
            for f in features:
                if f not in raw_site.columns or f not in harm_site.columns:
                    continue
                rv = raw_site.loc[common_idx, f].to_numpy(dtype=float)
                hv = harm_site.loc[common_idx, f].to_numpy(dtype=float)
                ok = ~(np.isnan(rv) | np.isnan(hv))
                if ok.sum() < 3:
                    continue
                rows.append({"site": site, "feature": f, "icc3": _icc_c1_two(rv[ok], hv[ok]), "n": int(ok.sum())})
        if progress is not None:
            progress(k_site + 1, len(sites))
    return pd.DataFrame(rows)


def compute_icc_within_summary(df_raw, df_harm, features, site_col, by=None):
    """
    Within-batch ICC(3,1) summarized per feature.

    The consistency-form ICC(3,1) between raw and harmonized values is computed
    inside each batch (compute_icc_by_site) and then summarized across batches for
    each feature as the median and the interquartile range. The overall, pooled
    ICC is intentionally not used, because pooling across batches conflates the
    batch effects that harmonization removes.

    ``by`` may be a precomputed compute_icc_by_site result, to avoid recomputing it.

    Returns one row per feature. Column names match compute_icc so downstream
    tables and figures reuse the same structure, with icc3 = across-batch median
    and icc3_lower / icc3_upper = the first and third quartiles (the interquartile
    range), and n = the number of batches contributing.
    """
    cols = ["feature", "icc3", "icc3_lower", "icc3_upper", "n"]
    if by is None:
        by = compute_icc_by_site(df_raw, df_harm, features, site_col)
    if by is None or len(by) == 0:
        return pd.DataFrame(columns=cols)
    rows = []
    for f, sub in by.groupby("feature"):
        vals = sub["icc3"].dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "feature":    f,
            "icc3":       float(vals.median()),
            "icc3_lower": float(vals.quantile(0.25)),
            "icc3_upper": float(vals.quantile(0.75)),
            "n":          int(len(vals)),
        })
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# 7. Spearman by site
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 8. Extra variable associations — OLS per feature
#    For each extra variable (continuous or categorical), fits:
#      feature ~ var + [other covariates from ComBat model]
#    Continuous: Pearson r + regression t / p.
#    Categorical: partial eta-squared, Cohen's f, F p-value.
#    FDR (Benjamini-Hochberg) across features per variable.
# ---------------------------------------------------------------------------
def compute_extra_associations(
    df: pd.DataFrame,
    features: list,
    extra_continuous: list,
    extra_categorical: list,
    continuous_covariates: list,
    categorical_covariates: list,
    label: str = "",
    use_fdr: bool = True,
    feature_group: dict | None = None,
) -> pd.DataFrame:
    def _safe(s):
        return s.replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")

    rows = []

    for var in (extra_continuous or []):
        if var not in df.columns:
            continue
        cont_covs = [c for c in (continuous_covariates or []) if c != var and c in df.columns]
        cat_covs  = [c for c in (categorical_covariates or []) if c != var and c in df.columns]
        for f in features:
            needed = list({f, var} | set(cont_covs) | set(cat_covs))
            d = df[needed].dropna()
            if len(d) < 10:
                continue
            try:
                rename = {col: _safe(col) for col in needed}
                d2 = d.rename(columns=rename)
                sf, sv = _safe(f), _safe(var)
                rhs = [sv] + [_safe(c) for c in cont_covs] + [f"C({_safe(c)})" for c in cat_covs]
                model = smf.ols(f"{sf} ~ " + " + ".join(rhs), data=d2).fit()
                if sv not in model.params.index:
                    continue
                r, _ = pearsonr(d[var].values, d[f].values)
                rows.append({
                    "feature":       f,
                    "variable":      var,
                    "var_type":      "continuous",
                    "effect_size":   float(r),
                    "pearson_r":     float(r),
                    "beta":          float(model.params[sv]),
                    "t_stat":        float(model.tvalues[sv]),
                    "p_value":       float(model.pvalues[sv]),
                    "n":             len(d),
                    "harmonization": label,
                })
            except Exception:
                continue

    for var in (extra_categorical or []):
        if var not in df.columns:
            continue
        cont_covs = [c for c in (continuous_covariates or []) if c != var and c in df.columns]
        cat_covs  = [c for c in (categorical_covariates or []) if c != var and c in df.columns]
        for f in features:
            needed = list({f, var} | set(cont_covs) | set(cat_covs))
            d = df[needed].dropna()
            if len(d) < 10 or d[var].nunique() < 2:
                continue
            try:
                rename = {col: _safe(col) for col in needed}
                d2 = d.rename(columns=rename)
                sf, sv = _safe(f), _safe(var)
                rhs = [f"C({sv})"] + [_safe(c) for c in cont_covs] + [f"C({_safe(c)})" for c in cat_covs]
                model = smf.ols(f"{sf} ~ " + " + ".join(rhs), data=d2).fit()
                aov = sm.stats.anova_lm(model, typ=2)
                key = f"C({sv})"
                if key not in aov.index:
                    continue
                ss_var   = aov.loc[key, "sum_sq"]
                ss_total = aov["sum_sq"].sum()
                eta_sq   = float(ss_var / ss_total)
                cf       = float(np.sqrt(eta_sq / max(1 - eta_sq, 1e-10)))
                rows.append({
                    "feature":       f,
                    "variable":      var,
                    "var_type":      "categorical",
                    "effect_size":   cf,
                    "eta_sq":        eta_sq,
                    "cohens_f":      cf,
                    "p_value":       float(aov.loc[key, "PR(>F)"]),
                    "n":             len(d),
                    "harmonization": label,
                })
            except Exception:
                continue

    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res

    parts = []
    for var in res["variable"].unique():
        sub = res[res["variable"] == var].copy()
        parts.append(_apply_significance(sub, use_fdr=use_fdr, feature_group=feature_group))
    return pd.concat(parts, ignore_index=True)


def compute_spearman_by_site(df_raw, df_harm, features, site_col):
    if site_col not in df_harm.columns:
        return pd.DataFrame()
    rows = []
    for site in natural_sorted(df_harm[site_col].dropna().unique().astype(str)):
        raw_site  = df_raw[df_raw[site_col].astype(str) == site] if site_col in df_raw.columns else df_raw
        harm_site = df_harm[df_harm[site_col].astype(str) == site]
        common_idx = raw_site.index.intersection(harm_site.index)
        if len(common_idx) < 3:
            continue
        for f in features:
            if f not in raw_site.columns or f not in harm_site.columns:
                continue
            rv = raw_site.loc[common_idx, f].dropna()
            hv = harm_site.loc[common_idx, f].dropna()
            idx2 = rv.index.intersection(hv.index)
            if len(idx2) < 3:
                continue
            r, _ = spearmanr(rv[idx2].values, hv[idx2].values)
            rows.append({"site": site, "feature": f, "spearman_r": float(r), "n": len(idx2)})
    return pd.DataFrame(rows)
