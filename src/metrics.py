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

    for site in sorted(df[site_col].dropna().unique().astype(str)):
        grp      = df[df[site_col].astype(str) == site]
        n        = len(grp)
        age_vals = grp[age_col].dropna()
        nf       = _n_female(grp[sex_col])
        rows.append({
            "Site":            site,
            "N":               n,
            "Age mean (SD)":   f"{age_vals.mean():.2f} ({age_vals.std():.2f})" if len(age_vals) > 0 else "N/A",
            "Age range":       f"{age_vals.min():.1f}–{age_vals.max():.1f}" if len(age_vals) > 0 else "N/A",
            "Female n (%)":    f"{nf} ({100 * nf / n:.1f}%)",
        })

    # Overall
    age_all = df[age_col].dropna()
    nf_all  = _n_female(df[sex_col])
    n_all   = len(df)
    rows.append({
        "Site":          "Total",
        "N":             n_all,
        "Age mean (SD)": f"{age_all.mean():.2f} ({age_all.std():.2f})" if len(age_all) > 0 else "N/A",
        "Age range":     f"{age_all.min():.1f}–{age_all.max():.1f}" if len(age_all) > 0 else "N/A",
        "Female n (%)":  f"{nf_all} ({100 * nf_all / n_all:.1f}%)",
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
def age_correlations(
    df: pd.DataFrame,
    features: list,
    age_col: str,
    label: str = "",
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
    if len(res) > 0:
        _, p_fdr, _, _ = multipletests(res["p_value"], method="fdr_bh")
        res["p_fdr"]   = p_fdr
        res["sig_fdr"] = p_fdr < 0.05
    return res


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
        cols_needed = [f, site_col, age_col, sex_col]
        d = df[cols_needed].dropna()
        if len(d) < 10 or d[site_col].nunique() < 2:
            continue
        try:
            safe_f = f.replace(" ", "_").replace("-", "_").replace(".", "_")
            d = d.rename(columns={f: safe_f})
            formula = f"{safe_f} ~ C({site_col}) + {age_col} + C({sex_col})"
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
def compute_icc_by_site(df_raw, df_harm, features, site_col):
    if site_col not in df_harm.columns:
        return pd.DataFrame()
    rows = []
    for site in sorted(df_harm[site_col].dropna().unique().astype(str)):
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
            long = pd.DataFrame({
                "subject": list(idx2) * 2,
                "rater":   ["raw"] * len(idx2) + ["harmonized"] * len(idx2),
                "value":   list(rv[idx2]) + list(hv[idx2]),
            })
            try:
                icc_res = pg.intraclass_corr(data=long, targets="subject", raters="rater", ratings="value")
                row3 = icc_res[icc_res["Type"] == "ICC(C,1)"].iloc[0]
                rows.append({"site": site, "feature": f, "icc3": float(row3["ICC"]), "n": len(idx2)})
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Spearman by site
# ---------------------------------------------------------------------------
def compute_spearman_by_site(df_raw, df_harm, features, site_col):
    if site_col not in df_harm.columns:
        return pd.DataFrame()
    rows = []
    for site in sorted(df_harm[site_col].dropna().unique().astype(str)):
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
