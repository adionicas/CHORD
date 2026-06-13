"""
harmonize.py
ComBat harmonization wrapper using the Python neuroCombat package.
All covariates are passed explicitly — nothing is hardcoded.
"""

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat


def _encode_categorical(series: pd.Series) -> pd.Series:
    """Label-encode a column to integers (0, 1, 2, …)."""
    if pd.api.types.is_numeric_dtype(series):
        # if already numeric and only 2 values, keep as-is
        return series.copy().astype(float)
    s = series.astype(str).str.strip().str.lower()
    # common sex encodings → 0/1
    sex_map = {"male": 0, "m": 0, "man": 0, "female": 1, "f": 1, "woman": 1}
    recoded = s.map(sex_map)
    if recoded.isna().any():
        # generic label encoding
        uniques = sorted(s.dropna().unique())
        recoded = s.map({v: i for i, v in enumerate(uniques)})
    return recoded.astype(float)


def run_combat(
    df: pd.DataFrame,
    feature_cols: list,
    site_col: str,
    continuous_covariates: list | None = None,
    categorical_covariates: list | None = None,
    eb: bool = True,
) -> pd.DataFrame:
    """
    Run neuroCombat on df.

    Parameters
    ----------
    df                     : input DataFrame (participants × columns)
    feature_cols           : imaging feature column names to harmonize
    site_col               : batch/site column name
    continuous_covariates  : list of column names for continuous covariates
                             (e.g. ['Age', 'TSI'])
    categorical_covariates : list of column names for categorical covariates
                             (e.g. ['Sex', 'Group'])
    eb                     : True = Empirical Bayes; False = feature-wise

    Returns
    -------
    DataFrame with harmonized features plus site and covariate columns.
    """
    cont_cols = continuous_covariates or []
    cat_cols  = categorical_covariates or []

    if not cont_cols and not cat_cols:
        import warnings
        warnings.warn(
            "No covariates specified. ComBat will remove site effects only, "
            "with no biological variability explicitly preserved.",
            stacklevel=2,
        )

    required = [site_col] + cont_cols + cat_cols + feature_cols
    required = list(dict.fromkeys(required))  # deduplicate
    subset   = df[required].copy()

    # Drop rows with missing site, covariates, or any feature
    check_cols    = [site_col] + cont_cols + cat_cols + feature_cols
    complete_mask = subset[check_cols].notna().all(axis=1)
    subset = subset[complete_mask].copy().reset_index(drop=False)

    if subset.empty:
        raise ValueError(
            "No rows with complete data across site, all covariates, and all features."
        )

    data_matrix = subset[feature_cols].values.T  # (p, n)

    # Build covars DataFrame
    covars = pd.DataFrame({site_col: subset[site_col].values})
    comb_cont_names = []
    comb_cat_names  = []

    for col in cont_cols:
        safe = f"_cont_{col}"
        covars[safe] = subset[col].values.astype(float)
        comb_cont_names.append(safe)

    for col in cat_cols:
        safe = f"_cat_{col}"
        covars[safe] = _encode_categorical(subset[col]).values
        comb_cat_names.append(safe)

    result = neuroCombat(
        dat=data_matrix,
        covars=covars,
        batch_col=site_col,
        continuous_cols=comb_cont_names if comb_cont_names else None,
        categorical_cols=comb_cat_names if comb_cat_names else None,
        eb=eb,
        mean_only=False,
    )

    harm_matrix = result["data"].T  # (n, p)

    preserve = [c for c in [site_col] + cont_cols + cat_cols if c in subset.columns]
    out      = subset[["index"] + preserve].copy().rename(columns={"index": "_orig_index"})
    harm_df  = pd.DataFrame(harm_matrix, columns=feature_cols, index=subset.index)
    out      = pd.concat([out.reset_index(drop=True), harm_df.reset_index(drop=True)], axis=1)
    out      = out.set_index("_orig_index")
    out.index.name = None
    return out
