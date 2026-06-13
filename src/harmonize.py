"""
harmonize.py
ComBat harmonization wrapper using the Python neuroCombat package.
"""

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat


def encode_sex(series: pd.Series) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(s.dropna().unique())
        if set(vals).issubset({0, 1}):
            return s
        mapping = {v: i for i, v in enumerate(vals)}
        return s.map(mapping)
    s_lower = s.str.strip().str.lower()
    mapping = {"male": 0, "m": 0, "man": 0,
               "female": 1, "f": 1, "woman": 1}
    recoded = s_lower.map(mapping)
    if recoded.isna().any() and not series.isna().any():
        uniques = sorted(s_lower.dropna().unique())
        mapping2 = {v: i for i, v in enumerate(uniques)}
        recoded = s_lower.map(mapping2)
    return recoded


def encode_categorical(series: pd.Series) -> pd.Series:
    """Label-encode any categorical column to integers."""
    if pd.api.types.is_numeric_dtype(series):
        return series.copy()
    uniques = sorted(series.dropna().astype(str).unique())
    mapping = {v: i for i, v in enumerate(uniques)}
    return series.astype(str).map(mapping)


def run_combat(
    df: pd.DataFrame,
    feature_cols: list,
    site_col: str,
    age_col: str,
    sex_col: str,
    eb: bool = True,
    extra_continuous: list | None = None,
    extra_categorical: list | None = None,
) -> pd.DataFrame:
    """
    Run neuroCombat on df for the given feature columns.

    Parameters
    ----------
    df                : input DataFrame (participants x columns)
    feature_cols      : list of numeric feature column names to harmonize
    site_col          : batch/site column name
    age_col           : age column name (continuous covariate)
    sex_col           : sex column name (will be binarized)
    eb                : True = Empirical Bayes; False = feature-wise estimation
    extra_continuous  : additional continuous covariates to preserve
    extra_categorical : additional categorical covariates to preserve (label-encoded)

    Returns
    -------
    DataFrame with harmonized feature columns plus site/age/sex columns.
    """
    extra_cont = extra_continuous or []
    extra_cat  = extra_categorical or []

    required = [site_col, age_col, sex_col] + extra_cont + extra_cat + feature_cols
    required = list(dict.fromkeys(required))  # deduplicate, preserve order
    subset = df[required].copy()
    subset["_sex_bin"] = encode_sex(subset[sex_col])

    covar_check_cols = [site_col, age_col, "_sex_bin"] + extra_cont + extra_cat
    complete_mask = subset[covar_check_cols + feature_cols].notna().all(axis=1)
    subset = subset[complete_mask].copy().reset_index(drop=False)

    if subset.empty:
        raise ValueError("No rows with complete data across covariates and all features.")

    data_matrix = subset[feature_cols].values.T  # (p, n)

    # Build covars DataFrame
    covars = pd.DataFrame({
        site_col: subset[site_col].values,
        "Age":    subset[age_col].values.astype(float),
        "Sex":    subset["_sex_bin"].values.astype(float),
    })
    continuous_cols = ["Age"]
    categorical_cols = ["Sex"]

    for col in extra_cont:
        safe = f"_econt_{col}"
        covars[safe] = subset[col].values.astype(float)
        continuous_cols.append(safe)

    for col in extra_cat:
        safe = f"_ecat_{col}"
        covars[safe] = encode_categorical(subset[col]).values.astype(float)
        categorical_cols.append(safe)

    result = neuroCombat(
        dat=data_matrix,
        covars=covars,
        batch_col=site_col,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        eb=eb,
        mean_only=False,
    )

    harm_matrix = result["data"].T  # (n, p)

    preserve_cols = [c for c in [site_col, age_col, sex_col] + extra_cont + extra_cat
                     if c in subset.columns]
    out = subset[["index"] + preserve_cols].copy()
    out = out.rename(columns={"index": "_orig_index"})
    harm_df = pd.DataFrame(harm_matrix, columns=feature_cols, index=subset.index)
    out = pd.concat([out.reset_index(drop=True), harm_df.reset_index(drop=True)], axis=1)
    out = out.set_index("_orig_index")
    out.index.name = None

    return out
