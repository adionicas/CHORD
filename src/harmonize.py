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

    required = list(dict.fromkeys([site_col] + cont_cols + cat_cols + feature_cols))
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in the uploaded data: {missing}")

    # Rows must be complete on the columns ComBat actually uses (site,
    # covariates, and features). ALL other columns, including subject IDs and
    # any additional metadata, are carried through unchanged so the harmonized
    # output remains a complete, usable table.
    complete_mask = df[required].notna().all(axis=1)
    work = df[complete_mask].copy()

    if work.empty:
        raise ValueError(
            "No rows with complete data across site, all covariates, and all features."
        )

    data_matrix = work[feature_cols].values.T  # (p, n)

    # Build covars DataFrame
    covars = pd.DataFrame({site_col: work[site_col].values})
    comb_cont_names = []
    comb_cat_names  = []

    for col in cont_cols:
        safe = f"_cont_{col}"
        covars[safe] = work[col].values.astype(float)
        comb_cont_names.append(safe)

    for col in cat_cols:
        safe = f"_cat_{col}"
        covars[safe] = _encode_categorical(work[col]).values
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

    # Replace only the feature columns with their harmonized values; every
    # other column (subject IDs, extra metadata, covariates, site) and the
    # original row index are preserved so downstream metrics can still pair
    # raw and harmonized rows by index.
    out = work.copy()
    out[feature_cols] = harm_matrix
    return out


def run_combat_grouped(
    df: pd.DataFrame,
    groups: dict,
    site_col: str,
    continuous_covariates: list | None = None,
    categorical_covariates: list | None = None,
    eb: bool = True,
) -> pd.DataFrame:
    """
    Run ComBat separately for each modality / measure-type group and recombine.

    Each group is harmonized in its own neuroCombat run, with its own empirical
    Bayes location and scale priors, using the same batch column and the same
    covariates for every group. This is appropriate when the feature columns
    span imaging modalities or measures that are not on a comparable scale,
    because the empirical Bayes step pools information across the features within
    a single run and assumes those pooled features are comparably distributed.

    Parameters
    ----------
    df                     : input DataFrame (participants × columns)
    groups                 : dict mapping group name → list of feature columns.
                             Groups must be disjoint; every column is harmonized
                             in exactly one run.
    site_col               : batch/site column name
    continuous_covariates  : continuous covariate column names (same for every group)
    categorical_covariates : categorical covariate column names (same for every group)
    eb                     : True = Empirical Bayes; False = feature-wise

    Returns
    -------
    A DataFrame over the union of participants retained by any group's run
    (per-group complete cases). Each group's feature columns hold harmonized
    values for the participants that group retained and NaN for participants
    excluded from that group's run. All non-feature columns are carried through
    unchanged from df, so the output pairs with the raw table by row index for
    the downstream metrics.
    """
    if not groups:
        raise ValueError("No modality groups provided.")

    all_features = [f for feats in groups.values() for f in feats]

    # A column must belong to exactly one group.
    seen, dups = set(), set()
    for f in all_features:
        if f in seen:
            dups.add(f)
        seen.add(f)
    if dups:
        raise ValueError(
            f"Columns assigned to more than one modality group: {sorted(dups)}"
        )

    per_group = {}
    used_index = pd.Index([])
    for gname, feats in groups.items():
        if not feats:
            continue
        out_g = run_combat(
            df, feats, site_col,
            continuous_covariates=continuous_covariates,
            categorical_covariates=categorical_covariates,
            eb=eb,
        )
        per_group[gname] = out_g
        used_index = used_index.union(out_g.index)

    if len(used_index) == 0:
        raise ValueError(
            "No participants had complete data in any modality group."
        )

    # Recombine on the original row index. Feature columns start as NaN so raw
    # values never leak in as harmonized; each group's harmonized block is then
    # filled in for the participants that group retained.
    combined = df.loc[used_index].copy()
    for f in all_features:
        combined[f] = np.nan
    for gname, out_g in per_group.items():
        feats = groups[gname]
        combined.loc[out_g.index, feats] = out_g[feats]
    return combined


def run_combat_per_feature(
    df: pd.DataFrame,
    feature_cols: list,
    site_col: str,
    continuous_covariates: list | None = None,
    categorical_covariates: list | None = None,
) -> pd.DataFrame:
    """
    Harmonize each feature independently on its own complete cases.

    Every feature is run through neuroCombat on its own, using only the
    participants for whom that feature (and the batch and covariates) is
    observed. Scattered missingness in other features therefore does not remove
    a participant from a feature they do have. Because a run contains a single
    feature, there is no empirical Bayes pooling across features; this is
    equivalent to feature-wise (EB=FALSE) ComBat applied one feature at a time.
    It is the appropriate choice when features have scattered missing data, so
    that requiring completeness across a whole block would remove most
    participants.

    Returns a frame over the union of participants retained by any feature, with
    each feature harmonized for its own complete cases and NaN elsewhere. All
    non-feature columns are carried through unchanged from df, so the output
    pairs with the raw table by row index for the downstream metrics.
    """
    used_index = pd.Index([])
    per_feat = {}
    for f in feature_cols:
        try:
            out_f = run_combat(
                df, [f], site_col,
                continuous_covariates=continuous_covariates,
                categorical_covariates=categorical_covariates,
                eb=False,
            )
        except (ValueError, KeyError):
            # a feature with no usable rows or fewer than two sites is skipped
            continue
        per_feat[f] = out_f
        used_index = used_index.union(out_f.index)

    if len(used_index) == 0:
        raise ValueError("No participants had complete data for any feature.")

    combined = df.loc[used_index].copy()
    for f in feature_cols:
        combined[f] = np.nan
    for f, out_f in per_feat.items():
        combined.loc[out_f.index, [f]] = out_f[[f]]
    return combined


def run_combat_by_missing_pattern(
    df: pd.DataFrame,
    feature_cols: list,
    site_col: str,
    continuous_covariates: list | None = None,
    categorical_covariates: list | None = None,
    eb: bool = True,
) -> pd.DataFrame:
    """
    Harmonize features grouped by their shared missing-value pattern, keeping EB.

    Features that are missing in exactly the same participants are complete on
    the same participants, so they can be harmonized together in one neuroCombat
    run with Empirical Bayes pooling across those features, while a different
    missing pattern runs separately. Features with no missing values form a
    single group harmonized on all participants. This keeps Empirical Bayes
    wherever a pattern is shared by two or more features, and retains each
    participant for every feature they actually have. Features whose pattern is
    unique run on their own (Empirical Bayes then has nothing to pool).

    Returns a frame over the union of participants retained by any group, with
    each feature harmonized for its own complete cases and NaN elsewhere.
    """
    groups = {}
    for f in feature_cols:
        key = tuple(np.flatnonzero(df[f].isna().to_numpy()).tolist())
        groups.setdefault(key, []).append(f)
    named = {f"pattern_{i + 1}": cols for i, (k, cols) in enumerate(groups.items())}
    return run_combat_grouped(
        df, named, site_col,
        continuous_covariates=continuous_covariates,
        categorical_covariates=categorical_covariates,
        eb=eb,
    )


def run_combat_with_imputation(
    df: pd.DataFrame,
    feature_cols: list,
    site_col: str,
    continuous_covariates: list | None = None,
    categorical_covariates: list | None = None,
    eb: bool = True,
    groups: dict | None = None,
) -> pd.DataFrame:
    """
    Fill missing feature values with each feature's median, harmonize with EB,
    then restore the originally-missing cells to NaN in the output.

    Imputation lets every participant enter the run, so Empirical Bayes pools
    across all features, and masking the imputed cells afterward keeps them out
    of the harmonized result. When ``groups`` (a modality mapping) is given,
    harmonization runs per group on the imputed data, so each measure keeps its
    own Empirical Bayes estimates.
    """
    work = df.copy()
    na_mask = work[feature_cols].isna()
    medians = work[feature_cols].median(numeric_only=True)
    work[feature_cols] = work[feature_cols].fillna(medians)

    if groups:
        out = run_combat_grouped(
            work, groups, site_col,
            continuous_covariates=continuous_covariates,
            categorical_covariates=categorical_covariates,
            eb=eb,
        )
    else:
        out = run_combat(
            work, feature_cols, site_col,
            continuous_covariates=continuous_covariates,
            categorical_covariates=categorical_covariates,
            eb=eb,
        )

    # restore originally-missing cells to NaN so imputed values are not reported
    for f in feature_cols:
        miss_idx = na_mask.index[na_mask[f].to_numpy()]
        idx = out.index.intersection(miss_idx)
        if len(idx):
            out.loc[idx, f] = np.nan
    return out
