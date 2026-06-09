"""
harmonize.py
ComBat harmonization wrapper using the Python neuroCombat package.
Runs both EB=True and EB=False and returns both harmonized DataFrames.
"""

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat


def encode_sex(series: pd.Series) -> pd.Series:
    """
    Convert Sex column to binary numeric (0/1).
    Handles strings like Male/Female, M/F, 0/1, 1/2.
    """
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(s.dropna().unique())
        if set(vals).issubset({0, 1}):
            return s
        # recode to 0/1
        mapping = {v: i for i, v in enumerate(vals)}
        return s.map(mapping)
    s_lower = s.str.strip().str.lower()
    mapping = {"male": 0, "m": 0, "man": 0,
               "female": 1, "f": 1, "woman": 1}
    recoded = s_lower.map(mapping)
    if recoded.isna().any() and not series.isna().any():
        # unknown strings: use label encoding
        uniques = sorted(s_lower.dropna().unique())
        mapping2 = {v: i for i, v in enumerate(uniques)}
        recoded = s_lower.map(mapping2)
    return recoded


def run_combat(
    df: pd.DataFrame,
    feature_cols: list,
    site_col: str,
    age_col: str,
    sex_col: str,
    eb: bool = True,
) -> pd.DataFrame:
    """
    Run neuroCombat on df for the given feature columns.

    Parameters
    ----------
    df           : input DataFrame (participants x columns)
    feature_cols : list of numeric feature column names to harmonize
    site_col     : batch/site column name
    age_col      : age column name (continuous covariate)
    sex_col      : sex column name (will be binarized)
    eb           : True = Empirical Bayes; False = feature-wise estimation

    Returns
    -------
    DataFrame with same index as df, harmonized feature columns,
    plus site/age/sex columns preserved.
    """
    # Work on complete cases only (no NA in site, age, sex, or any feature)
    required = [site_col, age_col, sex_col] + feature_cols
    subset = df[required].copy()
    subset["_sex_bin"] = encode_sex(subset[sex_col])

    complete_mask = subset[[site_col, age_col, "_sex_bin"] + feature_cols].notna().all(axis=1)
    subset = subset[complete_mask].copy()
    subset = subset.reset_index(drop=False)  # keep original index

    if subset.empty:
        raise ValueError("No rows with complete data across site, age, sex, and all features.")

    # neuroCombat expects (features x subjects)
    data_matrix = subset[feature_cols].values.T  # (p, n)

    covars = pd.DataFrame({
        site_col: subset[site_col].values,
        "Age":    subset[age_col].values.astype(float),
        "Sex":    subset["_sex_bin"].values.astype(float),
    })

    result = neuroCombat(
        dat=data_matrix,
        covars=covars,
        batch_col=site_col,
        continuous_cols=["Age"],
        categorical_cols=["Sex"],
        eb=eb,
        mean_only=False,
    )

    harm_matrix = result["data"].T  # (n, p)

    out = subset[["index", site_col, age_col, sex_col]].copy()
    out = out.rename(columns={"index": "_orig_index"})
    harm_df = pd.DataFrame(harm_matrix, columns=feature_cols, index=subset.index)
    out = pd.concat([out.reset_index(drop=True), harm_df.reset_index(drop=True)], axis=1)
    out = out.set_index("_orig_index")
    out.index.name = None

    return out
