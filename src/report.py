"""
report.py
Generates a self-contained HTML report formatted as a research paper
supplementary material section.
"""

import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime
from jinja2 import Template


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute summary statistics for the prose sections
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(val, decimals=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def _summarise_cohens_f(anc_df):
    if anc_df is None or len(anc_df) == 0:
        return {}
    f = anc_df["cohens_f"].dropna()
    return {
        "mean":   _fmt(f.mean()),
        "sd":     _fmt(f.std()),
        "median": _fmt(f.median()),
        "min":    _fmt(f.min()),
        "max":    _fmt(f.max()),
        "n_small":  int((f >= 0.10).sum()),
        "n_medium": int((f >= 0.25).sum()),
        "n_large":  int((f >= 0.40).sum()),
        "n_total":  len(f),
    }


def _summarise_icc(icc_df):
    if icc_df is None or len(icc_df) == 0:
        return {}
    v = icc_df["icc3"].dropna()
    return {
        "mean":       _fmt(v.mean()),
        "sd":         _fmt(v.std()),
        "median":     _fmt(v.median()),
        "min":        _fmt(v.min()),
        "max":        _fmt(v.max()),
        "n_poor":     int((v < 0.50).sum()),
        "n_moderate": int(((v >= 0.50) & (v < 0.75)).sum()),
        "n_good":     int(((v >= 0.75) & (v < 0.90)).sum()),
        "n_excellent":int((v >= 0.90).sum()),
        "n_total":    len(v),
    }


def _summarise_spearman(spm_df):
    if spm_df is None or len(spm_df) == 0:
        return {}
    v = spm_df["spearman_r"].dropna()
    return {
        "median": _fmt(v.median()),
        "min":    _fmt(v.min()),
        "max":    _fmt(v.max()),
        "n_total": len(v),
    }


def _summarise_age(age_df, label):
    if age_df is None or len(age_df) == 0:
        return {}
    sub = age_df[age_df["harmonization"] == label]
    if len(sub) == 0:
        return {}
    n_sig = int(sub["sig_fdr"].sum()) if "sig_fdr" in sub.columns else 0
    n_tot = len(sub)
    return {
        "n_sig":   n_sig,
        "n_total": n_tot,
        "pct_sig": f"{100 * n_sig / n_tot:.1f}" if n_tot > 0 else "N/A",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature list formatter
# ─────────────────────────────────────────────────────────────────────────────
def _format_feature_list(features):
    """Render all features as a compact pill grid."""
    pills = "".join(
        f'<span style="display:inline-block;background:#e8ecf0;border-radius:3px;'
        f'padding:1px 6px;margin:2px 3px 2px 0;font-family:monospace;font-size:8.5pt;'
        f'color:#2A6EBB">{f}</span>'
        for f in features
    )
    return f'<div style="line-height:2">{pills}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Plotly → HTML snippet
# ─────────────────────────────────────────────────────────────────────────────
def _fig_html(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"responsive": True})


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame → HTML table
# ─────────────────────────────────────────────────────────────────────────────
def _df_table(df, max_rows=300, float_fmt=4):
    if df is None or len(df) == 0:
        return "<p><em>No data available.</em></p>"
    d = df.head(max_rows).copy()
    for col in d.select_dtypes(include="float").columns:
        d[col] = d[col].map(lambda x: f"{x:.{float_fmt}f}" if pd.notna(x) else "")
    return d.to_html(index=False, border=0, classes="data-table",
                     justify="left", na_rep="")


def _demo_table(df):
    return _df_table(df, float_fmt=2)


def _summary_table(icc_df, spm_df, anc_before, anc_after, label):
    frames = []
    if icc_df is not None and len(icc_df) > 0:
        frames.append(icc_df[["feature", "icc3", "icc3_lower", "icc3_upper", "n"]]
                      .rename(columns={"icc3": "ICC3", "icc3_lower": "ICC3 lower 95% CI",
                                       "icc3_upper": "ICC3 upper 95% CI"})
                      .set_index("feature"))
    if spm_df is not None and len(spm_df) > 0:
        frames.append(spm_df[["feature", "spearman_r"]]
                      .rename(columns={"spearman_r": "Spearman r (raw vs harmonized)"})
                      .set_index("feature"))
    if anc_after is not None and len(anc_after) > 0:
        frames.append(anc_after[["feature", "cohens_f", "eta_sq"]]
                      .rename(columns={"cohens_f": "Cohen's f (after)", "eta_sq": "η²p (after)"})
                      .set_index("feature"))
    if anc_before is not None and len(anc_before) > 0:
        frames.append(anc_before[["feature", "cohens_f"]]
                      .rename(columns={"cohens_f": "Cohen's f (before)"})
                      .set_index("feature"))
    if not frames:
        return "<p><em>No data.</em></p>"
    result = frames[0]
    for t in frames[1:]:
        result = result.join(t, how="outer")
    return _df_table(result.reset_index().rename(columns={"feature": "Feature"}))


# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Harmonization Assessment — Supplementary Material</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  /* ── Typography ── */
  body { font-family: "Times New Roman", Times, serif; font-size: 12pt;
         line-height: 1.65; color: #111; background: #fff;
         margin: 0; padding: 0; }
  .page { max-width: 960px; margin: 0 auto; padding: 48px 56px; }
  h1 { font-size: 15pt; font-weight: bold; text-align: center;
       margin: 0 0 4px 0; }
  .authors { text-align: center; font-size: 10pt; color: #444;
             margin-bottom: 32px; }
  h2 { font-size: 13pt; font-weight: bold; margin: 32px 0 8px 0;
       border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  h3 { font-size: 12pt; font-weight: bold; margin: 20px 0 6px 0; }
  p  { margin: 0 0 10px 0; text-align: justify; }
  /* ── Tables ── */
  table.data-table { border-collapse: collapse; width: 100%;
                     font-size: 9.5pt; font-family: Arial, sans-serif;
                     margin: 12px 0 20px 0; }
  table.data-table th { background: #e8ecf0; padding: 6px 10px;
                        text-align: left; font-weight: bold;
                        border: 1px solid #bbb; }
  table.data-table td { padding: 4px 10px; border: 1px solid #d0d0d0; }
  table.data-table tr:nth-child(even) td { background: #f7f9fb; }
  table.data-table tr:last-child td { font-weight: bold;
                                       background: #eef2f6; }
  .table-note { font-size: 9pt; font-family: Arial, sans-serif;
                color: #555; margin-top: -12px; margin-bottom: 20px; }
  .table-title { font-size: 10pt; font-weight: bold;
                 font-family: Arial, sans-serif; margin: 16px 0 4px 0; }
  /* ── Figures ── */
  .fig-wrap  { margin: 16px 0 8px 0; }
  .fig-caption { font-size: 9.5pt; font-family: Arial, sans-serif;
                 color: #333; margin-bottom: 24px; }
  .fig-caption b { font-weight: bold; }
  /* ── Summary boxes ── */
  .summary-row { display: flex; gap: 12px; flex-wrap: wrap;
                 margin: 16px 0; }
  .sbox { background: #f0f4f8; border-radius: 5px;
          padding: 10px 16px; min-width: 120px; font-family: Arial; }
  .sbox .val { font-size: 1.4rem; font-weight: bold; color: #2A6EBB; }
  .sbox .lbl { font-size: 8pt; color: #666; margin-top: 2px; }
  /* ── References ── */
  .ref-list { font-size: 9.5pt; font-family: Arial; padding-left: 20px; }
  .ref-list li { margin-bottom: 5px; }
  /* ── Header ── */
  .supp-label { text-align: center; font-size: 10pt; font-weight: bold;
                letter-spacing: 0.05em; color: #555;
                margin-bottom: 6px; }
  hr.divider { border: none; border-top: 2px solid #2A6EBB; margin: 20px 0; }
  .footer { text-align: center; font-size: 8.5pt; font-family: Arial;
            color: #aaa; margin-top: 48px; padding-top: 12px;
            border-top: 1px solid #ddd; }
</style>
</head>
<body>
<div class="page">

<p class="supp-label">SUPPLEMENTARY MATERIAL</p>
<hr class="divider">
<h1>Multisite ComBat Harmonization Assessment</h1>
<p class="authors">
  Generated by CHORD (Comprehensive Harmonization Open-platform with Reporting and Diagnostics)
  &nbsp;|&nbsp; {{ date }}
</p>

<!-- ─────────────────────────────────────── S1 ──────────────────────────── -->
<h2>S1. Dataset Overview</h2>

<p>
  The dataset comprised <b>{{ n_participants }}</b> participants acquired across
  <b>{{ n_sites }}</b> sites ({{ site_list }}).
  A total of <b>{{ n_features }}</b> imaging features were submitted for harmonization.
  {% if n_retained != n_participants %}
  After excluding participants with missing values in the batch variable or any covariate,
  <b>{{ n_retained }}</b> participants were retained for harmonization.
  {% endif %}
  The batch variable was <em>{{ site_col }}</em>; covariates preserved during
  harmonization were <em>{{ age_col }}</em> (continuous) and <em>{{ sex_col }}</em> (categorical){% if extra_continuous %}, {{ extra_continuous | join(", ") }} (continuous){% endif %}{% if extra_categorical %}, {{ extra_categorical | join(", ") }} (categorical){% endif %}.
</p>

<p class="table-title">Table S1. Sample characteristics by site.</p>
{{ demo_table }}
<p class="table-note">
  <em>Note.</em> Age is reported as mean (standard deviation).
  Sex distribution reports the number and percentage of female participants.
</p>

<h3>S1.1 Imaging features</h3>
<p>
  The following {{ n_features }} features were included in the harmonization analysis:
  {{ feature_list | safe }}
</p>

<!-- ─────────────────────────────────────── S2 ──────────────────────────── -->
<h2>S2. Harmonization Procedure</h2>

<p>
  Scanner-related batch effects were removed using ComBat harmonization
  [<a href="#ref1">1</a>, <a href="#ref2">2</a>], implemented via the
  <em>neuroCombat</em> Python package (v0.2.12). ComBat models site-related
  variability through a location and scale (L/S) adjustment model, estimating
  additive and multiplicative batch effects for each feature. When Empirical
  Bayes (EB) estimation is enabled, ComBat pools information across all features
  to refine the estimation of site-specific location and scale parameters,
  which can stabilize estimates particularly in batches with small sample sizes
  or low within-site variability [<a href="#ref1">1</a>]. When EB is disabled
  (EB=FALSE), ComBat performs feature-wise L/S adjustments independently,
  without borrowing strength across features [<a href="#ref3">3</a>].
</p>

<p>
  In the present analysis, <em>{{ site_col }}</em> was used as the batch
  variable. <em>{{ age_col }}</em> (continuous) and <em>{{ sex_col }}</em>
  (categorical) were included as biological covariates to preserve their
  associated variability following harmonization.
  {% if run_ebf %}
  Both ComBat configurations were applied: with Empirical Bayes estimation
  (EB=TRUE) and without (EB=FALSE), enabling a direct comparison of their
  respective effects on site-effect reduction and within-site consistency
  preservation.
  {% else %}
  ComBat was applied with Empirical Bayes estimation (EB=TRUE).
  {% endif %}
</p>

<p>
  Prior to harmonization, features were examined for missing data.
  Only participants with complete observations in the batch variable,
  all covariates, and at least one feature within a given modality group
  were included in the harmonization for that modality.
  Features were harmonized independently per modality group using
  participants with non-missing values for that group.
</p>

<!-- ─────────────────────────────────────── S3 ──────────────────────────── -->
<h2>S3. Harmonization Evaluation Framework</h2>

<p>
  Harmonization effectiveness was evaluated using four complementary approaches
  that together assess (a) the magnitude of site-related variability before and
  after harmonization, (b) the extent to which original within-site variability
  was preserved, and (c) the degree to which biologically motivated associations
  were maintained. All analyses were conducted using Python (scipy, statsmodels,
  pingouin).
</p>

<h3>S3.1 Site Effect Quantification</h3>
<p>
  The magnitude of site-related effects was quantified using two complementary approaches.
  First, for each imaging feature a one-way analysis of covariance (ANCOVA) model
  was fitted with site as the grouping factor and age and sex as covariates,
  using Type II sums of squares. The proportion of variance attributable to site
  was expressed as partial eta-squared (&#951;<sup>2</sup><sub>p</sub>), and
  converted to Cohen's <em>f</em> using the formula
  <em>f</em> = &#8730;(&#951;<sup>2</sup><sub>p</sub> / (1 &#8722; &#951;<sup>2</sup><sub>p</sub>)).
  Effect size benchmarks follow Cohen (1988) [<a href="#ref4">4</a>]:
  small &#8805; 0.10, medium &#8805; 0.25, large &#8805; 0.40.
  Second, for each feature values were standardized to a <em>z</em>-score
  (subtracting the grand mean and dividing by the grand standard deviation
  across all participants) and the mean <em>z</em>-score was computed per site.
  Successful harmonization should reduce site mean <em>z</em>-scores toward zero.
</p>

<h3>S3.2 Within-Site Consistency</h3>
<p>
  Within-site consistency was evaluated as the degree to which the rank ordering
  of participants was preserved following harmonization, using two approaches.
  First, intraclass correlation coefficients (ICC) were computed between
  pre-harmonization (raw) and post-harmonization values for each feature,
  modeled as a two-way mixed-effects, consistency model (ICC3; equivalently denoted ICC(C,1) in McGraw &amp; Wong, 1996) [<a href="#ref5">5</a>]. ICC values were interpreted according to Koo and
  Li (2016) [<a href="#ref5">5</a>]: &lt; 0.50 = poor consistency,
  0.50&#8211;0.75 = moderate, 0.75&#8211;0.90 = good, &#8805; 0.90 = excellent.
  High ICC values indicate that the relative standing of participants is
  well preserved after harmonization, which is an important criterion for
  ensuring that harmonization does not inadvertently remove meaningful
  within-site biological variability [<a href="#ref3">3</a>].
</p>

<h3>S3.3 Preservation of Biological Associations</h3>
<p>
  To evaluate whether ComBat harmonization preserves biologically meaningful
  variability, Pearson correlation coefficients were computed between
  <em>{{ age_col }}</em> and each imaging feature, before and after each
  harmonization approach. Multiple comparisons were controlled using the
  Benjamini&#8211;Hochberg false discovery rate (FDR) correction
  [<a href="#ref6">6</a>], applied separately within each harmonization
  condition. A shift in the distribution of age&#8211;feature correlations
  following harmonization, or changes in the number of features significantly
  associated with age, was used as an indicator of changes in biologically
  relevant signal [<a href="#ref2">2</a>, <a href="#ref3">3</a>].
</p>

<!-- ─────────────────────────────────────── S4 ──────────────────────────── -->
<h2>S4. Results</h2>

<h3>S4.1 Site Effects Before and After Harmonization</h3>
<p>
  Before harmonization, the mean (SD) Cohen's <em>f</em> across
  {{ anc_before.n_total }} features was
  {{ anc_before.mean }} ({{ anc_before.sd }}),
  with {{ anc_before.n_large }} features ({{ pct(anc_before.n_large, anc_before.n_total) }}%)
  showing large site effects (Cohen's <em>f</em> &#8805; 0.40),
  {{ anc_before.n_medium - anc_before.n_large }} features showing medium effects
  (0.25 &#8804; <em>f</em> &lt; 0.40), and
  {{ anc_before.n_small - anc_before.n_medium }} features showing small effects
  (0.10 &#8804; <em>f</em> &lt; 0.25).
  Following harmonization with EB=TRUE, the mean (SD) Cohen's <em>f</em>
  was {{ anc_ebt.mean }} ({{ anc_ebt.sd }})
  (median = {{ anc_ebt.median }}, range {{ anc_ebt.min }}&#8211;{{ anc_ebt.max }}).
  {% if run_ebf %}
  Following harmonization with EB=FALSE, the mean (SD) Cohen's <em>f</em>
  was {{ anc_ebf.mean }} ({{ anc_ebf.sd }})
  (median = {{ anc_ebf.median }}, range {{ anc_ebf.min }}&#8211;{{ anc_ebf.max }}).
  {% endif %}
  Site mean <em>z</em>-score deviations from the grand mean are shown in Figure S1;
  deviations should approach zero after effective harmonization.
</p>

<div class="fig-wrap">{{ fig_site_dev }}</div>
<p class="fig-caption">
  <b>Figure S1.</b> Site mean deviation from the grand mean before and after harmonization.
  For each imaging feature, values were standardized to a <em>z</em>-score
  (grand mean = 0, grand SD = 1) and the site mean was computed.
  Each point represents one (site, feature) pair. After effective harmonization,
  site means should cluster near zero. Diamond marker = mean across all features
  for each site.
</p>

<div class="fig-wrap">{{ fig_cohens_f }}</div>
<p class="fig-caption">
  <b>Figure S2.</b> Site effect size (Cohen's <em>f</em>) before versus after harmonization.
  Each point represents one imaging feature. Points below the diagonal indicate
  a reduction in site effect following harmonization. Reference lines on both axes
  denote conventional effect size benchmarks: small &#8805; 0.10, medium &#8805; 0.25,
  large &#8805; 0.40 (Cohen, 1988), allowing direct reading of effect size category
  both before (x-axis) and after (y-axis) harmonization.
  ANCOVA models included age and sex as covariates (Type II sums of squares).
  Axis range is determined by the maximum Cohen's <em>f</em> value observed in the data (no ceiling imposed);
  large site effects may produce values substantially above 1.0.
</p>

<h3>S4.2 Within-Site Consistency</h3>
<p>
  ICC3 was computed between raw and harmonized values for
  {{ icc_ebt.n_total }} features. Following harmonization with EB=TRUE,
  the mean (SD) ICC was {{ icc_ebt.mean }} ({{ icc_ebt.sd }})
  (median = {{ icc_ebt.median }}, range {{ icc_ebt.min }}&#8211;{{ icc_ebt.max }}).
  {{ icc_ebt.n_excellent }} features ({{ pct(icc_ebt.n_excellent, icc_ebt.n_total) }}%)
  showed excellent consistency (&#8805; 0.90),
  {{ icc_ebt.n_good }} features ({{ pct(icc_ebt.n_good, icc_ebt.n_total) }}%)
  showed good consistency (0.75&#8211;0.90),
  {{ icc_ebt.n_moderate }} features ({{ pct(icc_ebt.n_moderate, icc_ebt.n_total) }}%)
  showed moderate consistency (0.50&#8211;0.75), and
  {{ icc_ebt.n_poor }} features ({{ pct(icc_ebt.n_poor, icc_ebt.n_total) }}%)
  showed poor consistency (&lt; 0.50).
  {% if run_ebf and icc_ebf.n_total %}
  Following harmonization with EB=FALSE, the mean (SD) ICC was
  {{ icc_ebf.mean }} ({{ icc_ebf.sd }})
  (median = {{ icc_ebf.median }}, range {{ icc_ebf.min }}&#8211;{{ icc_ebf.max }}).
  {{ icc_ebf.n_excellent }} features ({{ pct(icc_ebf.n_excellent, icc_ebf.n_total) }}%)
  showed excellent consistency, {{ icc_ebf.n_good }} good,
  {{ icc_ebf.n_moderate }} moderate, and {{ icc_ebf.n_poor }} poor.
  {% endif %}
  The overall ICC3 distribution (Figure S3) and the per-site breakdown (Figure S4)
  are shown below.
</p>

<div class="fig-wrap">{{ fig_icc }}</div>
<p class="fig-caption">
  <b>Figure S3.</b> Number of features falling in each ICC3 consistency category (Koo &amp; Li, 2016 [<a href="#ref5">5</a>]).
  Categories: poor (&lt; 0.50), moderate (0.50&#8211;0.75), good (0.75&#8211;0.90), excellent (&#8805; 0.90).
  Each bar shows the count of features per category; counts are shown above each bar.
  One bar group per harmonization condition.
</p>

{% if fig_icc_site %}
<div class="fig-wrap">{{ fig_icc_site }}</div>
<p class="fig-caption">
  <b>Figure S4.</b> Within-site consistency by site: ICC3 (two-way mixed effects, consistency) between raw and harmonized values,
  computed separately for each site. Site labels include participant count (n).
  Each box shows the distribution of ICC3 values across all features within that site.
  Sites with smaller sample sizes may show lower consistency, particularly when
  harmonization is performed without Empirical Bayes estimation [<a href="#ref3">3</a>, <a href="#ref8">8</a>].
  Colored bands denote Koo and Li (2016) [<a href="#ref5">5</a>] thresholds (as in Figure S3).
</p>
{% endif %}

<h3>S4.3 Biological Signal Preservation (Age Associations)</h3>
<p>
  Before harmonization, {{ age_before.n_sig }} of {{ age_before.n_total }} features
  ({{ age_before.pct_sig }}%) showed a statistically significant association with
  {{ age_col }} at FDR-corrected &#945; &lt; .05.
  Following harmonization with EB=TRUE, {{ age_ebt.n_sig }} features
  ({{ age_ebt.pct_sig }}%) showed significant age associations.
  {% if run_ebf and age_ebf.n_total %}
  Following harmonization with EB=FALSE, {{ age_ebf.n_sig }} features
  ({{ age_ebf.pct_sig }}%) showed significant age associations.
  {% endif %}
  The relationship between age associations before and after harmonization is
  shown in Figure S5. Each point represents one imaging feature; its position
  on the x-axis indicates its Pearson <em>r</em> with {{ age_col }} before
  harmonization and on the y-axis after harmonization. Points above the diagonal
  indicate features whose association with age strengthened after harmonization;
  points below indicate weakening. Points are colored according to FDR significance
  after harmonization.
</p>

<div class="fig-wrap">{{ fig_age }}</div>
<p class="fig-caption">
  <b>Figure S5.</b> Age associations before versus after harmonization.
  Each point represents one imaging feature. The x-axis shows Pearson <em>r</em>
  between {{ age_col }} and the feature before harmonization; the y-axis shows
  the same correlation after harmonization. Pearson <em>r</em> was computed
  independently for each condition; no formal test of the difference between
  correlations was applied. Points above the diagonal indicate a strengthened age
  association after harmonization; points below indicate weakening.
  Symbol and color encode FDR significance category (Benjamini&#8211;Hochberg;
  applied separately within each condition):
  grey circle = not significant in either condition;
  filled circle = FDR significant after harmonization only (new association);
  orange diamond = FDR significant before harmonization only (lost association);
  purple square = FDR significant in both conditions (preserved association).
  Dashed diagonal = no change; dotted lines at <em>r</em> = 0.
</p>

<!-- ─────────────────────────────────────── S5 ──────────────────────────── -->
<h2>S5. Feature-Level Results</h2>

<p class="table-title">Table S2. Feature-level harmonization metrics — EB=TRUE.</p>
{{ table_ebt }}
<p class="table-note">
  <em>Note.</em>
  ICC3 = intraclass correlation coefficient (two-way mixed effects, consistency; Shrout &amp; Fleiss type 3)
  between raw and harmonized values (Koo &amp; Li, 2016):
  &lt; 0.50 poor, 0.50&#8211;0.75 moderate, 0.75&#8211;0.90 good, &#8805; 0.90 excellent.
  Cohen's <em>f</em> = site effect size from ANCOVA (Age + Sex as covariates, Type II SS).
  Cohen's <em>f</em> before = pre-harmonization site effect; Cohen's <em>f</em> after = post-harmonization site effect.
</p>

{% if run_ebf %}
<p class="table-title">Table S3. Feature-level harmonization metrics — EB=FALSE.</p>
{{ table_ebf }}
<p class="table-note">
  <em>Note.</em> See Table S2 note for metric descriptions.
</p>
{% endif %}

<!-- ─────────────────────────────────────── Refs ────────────────────────── -->
<h2>References</h2>
<ol class="ref-list">
  <li id="ref1">Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. <em>Biostatistics</em>. 2007;8(1):118&#8211;127.</li>
  <li id="ref2">Fortin J-P, Parker D, Tun&#231; B, et al. Harmonization of multi-site diffusion tensor imaging data. <em>NeuroImage</em>. 2017;161:149&#8211;170.</li>
  <li id="ref3">Onicas AI, Ware AL, Harris AD, et al. Multisite harmonization of structural DTI networks in children: an A-CAP study. <em>Frontiers in Neurology</em>. 2022. doi:10.3389/fneur.2022.850642</li>
  <li id="ref8">Onicas AI, Keleher F, Bickart KC, MacDonald CL, Brown A, Cook L, Rivara FP, Gioia GA, Giza CC, Dennis EL. ComBat harmonization with and without empirical Bayes estimation for resting-state functional connectivity in pediatric mild traumatic brain injury: a CARE4Kids study. <em>ResearchSquare</em> [Preprint]. 2026. doi:10.21203/rs.3.rs-9418750</li>
  <li id="ref4">Cohen J. <em>Statistical Power Analysis for the Behavioral Sciences</em>. 2nd ed. Erlbaum; 1988.</li>
  <li id="ref5">Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. <em>Journal of Chiropractic Medicine</em>. 2016;15(2):155&#8211;163.</li>
  <li id="ref6">Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. <em>Journal of the Royal Statistical Society Series B</em>. 1995;57(1):289&#8211;300.</li>
  <li id="ref7">Fortin J-P, Cullen N, Sheline YI, et al. Harmonization of cortical thickness measurements across scanners and sites. <em>NeuroImage</em>. 2018;167:104&#8211;120.</li>
</ol>

<!-- ── Methods paragraph for main manuscript ── -->
<h2>Methods Paragraph for Main Manuscript</h2>
<p style="font-size:9pt;font-family:Arial,sans-serif;color:#666;margin-bottom:10px;">
  The paragraph below can be copied directly into the Methods section of your manuscript.
  Adapt bracketed placeholders and reference numbers to match your citation style.
</p>
<div style="background:#f0f4f8;border-left:4px solid #2A6EBB;border-radius:4px;
            padding:18px 22px;margin:0 0 30px 0;">
  <p style="font-size:11pt;font-family:'Times New Roman',serif;line-height:1.75;
            margin:0;text-align:justify;color:#111;">
    Scanner-related batch effects were corrected and evaluated using CHORD
    (Comprehensive Harmonization Open-platform with Reporting and Diagnostics;
    [GitHub URL]).
    ComBat harmonization [1, 2] was applied using the <em>neuroCombat</em> Python package
    (v0.2.12), with {{ site_col }} as the batch variable and {{ age_col }} (continuous)
    and {{ sex_col }} (categorical) included as biological covariates to preserve their
    associated variability.
    {% if run_ebf %}
    Two ComBat configurations were compared: with Empirical Bayes (EB) estimation,
    which pools information across features to stabilize batch effect parameter estimates [1],
    and without (EB=FALSE), which applies independent feature-wise location and scale adjustments [3, <a href="#ref8">8</a>].
    {% else %}
    ComBat was applied with Empirical Bayes estimation, which pools information
    across features to stabilize batch effect parameter estimates [1].
    {% endif %}
    Harmonization effectiveness was evaluated using five complementary metrics:
    (1) site mean <em>z</em>-score deviation from the grand mean before and after harmonization,
    assessing residual site-related variability;
    (2) analysis of covariance (ANCOVA, Type II sums of squares; covariates: {{ age_col }}, {{ sex_col }})
    with site as the grouping factor, quantifying site effect size as Cohen's <em>f</em> [4];
    (3) intraclass correlation coefficient (ICC3, two-way mixed effects, consistency) between
    pre- and post-harmonization values for each feature, interpreted according to Koo and Li [5]:
    poor (&lt; 0.50), moderate (0.50&#8211;0.75), good (0.75&#8211;0.90), excellent (&#8805; 0.90); and
    (4) Pearson correlation between {{ age_col }} and each imaging feature before and after
    harmonization, with false discovery rate correction (Benjamini&#8211;Hochberg) [6],
    to assess preservation of biologically relevant age-related variability.
    All metrics are reported in the Supplementary Material.
    [1] Johnson et al., <em>Biostatistics</em>, 2007.
    [2] Fortin et al., <em>NeuroImage</em>, 2017.
    [3] Onicas et al., <em>Frontiers in Neurology</em>, 2022. [8] Onicas et al., <em>ResearchSquare</em> [Preprint], 2026.
    [4] Cohen, 1988.
    [5] Koo &amp; Li, <em>J Chiropr Med</em>, 2016.
    [6] Benjamini &amp; Hochberg, <em>J R Stat Soc B</em>, 1995.
  </p>
</div>

<div class="footer">
  CHORD &nbsp;|&nbsp; Generated {{ date }} &nbsp;|&nbsp;
  {{ n_participants }} participants &nbsp;|&nbsp;
  {{ n_features }} features &nbsp;|&nbsp; {{ n_sites }} sites
</div>

</div><!-- .page -->
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Methods paragraph — plain text for the Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
def build_methods_paragraph(
    site_col: str,
    age_col: str,
    sex_col: str,
    run_ebf: bool,
    github_url: str = "[GitHub URL]",
    extra_continuous: list | None = None,
    extra_categorical: list | None = None,
) -> str:
    """Return a plain-text methods paragraph for copy-paste into a manuscript."""
    if run_ebf:
        eb_sentence = (
            f"Two ComBat configurations were compared: with Empirical Bayes (EB) estimation, "
            f"which pools information across features to stabilize batch effect parameter estimates (Johnson et al., 2007), "
            f"and without (EB=FALSE), which applies independent feature-wise location and scale adjustments (Onicas et al., 2022; Onicas et al., 2026 [Preprint])."
        )
    else:
        eb_sentence = (
            "ComBat was applied with Empirical Bayes estimation, which pools information "
            "across features to stabilize batch effect parameter estimates (Johnson et al., 2007)."
        )

    return (
        f"Scanner-related batch effects were corrected and evaluated using CHORD "
        f"(Comprehensive Harmonization Open-platform with Reporting and Diagnostics; {github_url}). "
        f"ComBat harmonization (Johnson et al., 2007; Fortin et al., 2017) was applied using the "
        f"neuroCombat Python package (v0.2.12), with {site_col} as the batch variable and "
        f"{age_col} (continuous) and {sex_col} (categorical)"
        + (f", {', '.join(extra_continuous)} (continuous)" if extra_continuous else "")
        + (f", and {', '.join(extra_categorical)} (categorical)" if extra_categorical else "")
        + f" included as biological covariates to preserve their associated variability. "
        f"{eb_sentence} "
        f"Harmonization effectiveness was evaluated using five complementary metrics: "
        f"(1) site mean z-score deviation from the grand mean before and after harmonization, "
        f"assessing residual site-related variability; "
        f"(2) analysis of covariance (ANCOVA, Type II sums of squares; covariates: {age_col}, {sex_col}) "
        f"with site as the grouping factor, quantifying site effect size as Cohen's f (Cohen, 1988); "
        f"(3) intraclass correlation coefficient (ICC3, two-way mixed effects, consistency) between "
        f"pre- and post-harmonization values for each feature, interpreted according to Koo and Li (2016): "
        f"poor (< 0.50), moderate (0.50-0.75), good (0.75-0.90), excellent (>= 0.90); "
        f"(4) Pearson correlation between {age_col} and each imaging feature before and after "
        f"harmonization, with false discovery rate correction (Benjamini & Hochberg, 1995), "
        f"to assess preservation of biologically relevant age-related variability. "
        f"All metrics are reported in the Supplementary Material."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(
    df_raw,
    site_col, age_col, sex_col,
    feature_cols,
    n_retained,
    run_ebf,
    demo_df,
    dev_before, dev_ebt, dev_ebf,
    icc_ebt, icc_ebf,
    spm_ebt, spm_ebf,
    anc_before, anc_ebt, anc_ebf,
    age_df,
    fig_site_dev, fig_icc, fig_icc_site, fig_spearman, fig_age, fig_cohens_f,
    extra_continuous=None,
    extra_categorical=None,
) -> str:

    sites = sorted(df_raw[site_col].dropna().unique().astype(str))

    anc_b  = _summarise_cohens_f(anc_before)
    anc_t  = _summarise_cohens_f(anc_ebt)
    anc_f  = _summarise_cohens_f(anc_ebf)
    icc_t  = _summarise_icc(icc_ebt)
    icc_f  = _summarise_icc(icc_ebf)
    spm_t  = _summarise_spearman(spm_ebt)
    spm_f  = _summarise_spearman(spm_ebf)
    age_b  = _summarise_age(age_df, "Before harmonization")
    age_t  = _summarise_age(age_df, "After (EB=TRUE)")
    age_f  = _summarise_age(age_df, "After (EB=FALSE)")

    def pct(num, den):
        if not den:
            return "N/A"
        return f"{100 * num / den:.1f}"

    tmpl = Template(TEMPLATE)
    html = tmpl.render(
        date            = datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_participants  = len(df_raw),
        n_sites         = len(sites),
        site_list       = ", ".join(sites),
        n_features      = len(feature_cols),
        n_retained      = n_retained,
        site_col           = site_col,
        age_col            = age_col,
        sex_col            = sex_col,
        run_ebf            = run_ebf,
        extra_continuous   = extra_continuous or [],
        extra_categorical  = extra_categorical or [],
        feature_list    = _format_feature_list(feature_cols),
        demo_table      = _demo_table(demo_df),
        anc_before      = anc_b,
        anc_ebt         = anc_t,
        anc_ebf         = anc_f,
        icc_ebt         = icc_t,
        icc_ebf         = icc_f,
        spm_ebt         = spm_t,
        spm_ebf         = spm_f,
        age_before      = age_b,
        age_ebt         = age_t,
        age_ebf         = age_f,
        fig_site_dev    = _fig_html(fig_site_dev),
        fig_cohens_f    = _fig_html(fig_cohens_f),
        fig_icc         = _fig_html(fig_icc) if fig_icc is not None else "",
        fig_icc_site    = _fig_html(fig_icc_site) if fig_icc_site is not None else None,
        fig_spearman    = _fig_html(fig_spearman) if fig_spearman is not None else "",
        fig_age         = _fig_html(fig_age),
        table_ebt       = _summary_table(icc_ebt, spm_ebt, anc_before, anc_ebt, "EB=TRUE"),
        table_ebf       = _summary_table(icc_ebf, spm_ebf, anc_before, anc_ebf, "EB=FALSE") if run_ebf else "",
        pct             = pct,
    )
    return html
