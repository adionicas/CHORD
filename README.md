# 🎵 CHORD
### Comprehensive Harmonization Open-platform with Reporting and Diagnostics

```
  ♩  ♪  ♫  ♬  ♩  ♪  ♫
   C  H  O  R  D
  ────────────────────
  Multisite · ComBat · Neuroimaging
```

CHORD is an open-source, browser-based tool for evaluating multisite ComBat harmonization
in neuroimaging datasets. Upload a table of imaging features, configure a few columns,
click Run — and receive a fully formatted supplementary report ready for manuscript submission.

---

## What CHORD reports

| Metric | What it measures |
|---|---|
| Site mean z-score deviation | Residual site-related variability before and after harmonization |
| ANCOVA Cohen's f | Site effect size controlling for age and sex (Type II SS) |
| ICC3 (overall + by site) | Within-site consistency: how well participant rank ordering is preserved |
| Spearman r (overall + by site) | Non-parametric rank-order preservation per site |
| Age associations | Whether biologically motivated age–feature correlations are maintained |

CHORD compares **EB=TRUE** (Empirical Bayes, default) and **EB=FALSE** (feature-wise)
configurations of ComBat side by side.

---

## Requirements

| Option | What you need |
|---|---|
| **Docker (recommended)** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) — free, works on Mac/Windows/Linux |
| **Local Python** | Python 3.11+, pip |

---

## Quick start — Docker (recommended)

Docker keeps everything contained. No Python installation needed.

**Step 1 — Install Docker Desktop** (one time only)

Download from https://www.docker.com/products/docker-desktop/ and install it.

**Step 2 — Download CHORD**

```bash
git clone https://github.com/adionicas/CHORD.git
cd chord
```

Or download the ZIP from GitHub and unzip it.

**Step 3 — Launch CHORD**

```bash
docker compose up --build
```

The first launch downloads dependencies (~2 min). Subsequent launches take ~10 seconds.

**Step 4 — Open in your browser**

```
http://localhost:8501
```

That is it. CHORD runs entirely on your machine — no data is uploaded anywhere.

**To stop CHORD:**

```bash
docker compose down
```

---

## Quick start — Local Python (no Docker)

If you have Python 3.11+ installed:

```bash
git clone https://github.com/adionicas/CHORD.git
cd chord
pip install -r requirements.txt
python3 -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## How to use CHORD

### Step 1 — Prepare your data

Your input file must be a CSV or Excel file (.csv, .xlsx) with:

- One row per participant
- A **site/batch column** (e.g. `Site`, `Scanner`, `Batch`)
- An **age column** (numeric, continuous)
- A **sex column** (numeric or text: Male/Female, 0/1, M/F)
- Any number of **numeric imaging feature columns**

Example layout:

| Site | Age | Sex | FA_CC | FA_CST | MD_CC | ... |
|------|-----|-----|-------|--------|-------|-----|
| Site_A | 14.2 | Female | 0.512 | 0.634 | 0.0009 | ... |
| Site_B | 16.7 | Male | 0.489 | 0.601 | 0.0011 | ... |

No other preprocessing is required. Missing values are handled automatically
(participants with missing site, age, or sex are excluded from harmonization for that modality).

### Step 2 — Upload and configure

1. Drag your file onto the upload area, or click to browse
2. Select which column is the **site/batch** variable
3. Select the **age** column
4. Select the **sex** column
5. CHORD auto-detects all remaining numeric columns as features
   - Use the **Select all** / **Clear all** buttons for bulk selection
   - Use the modality prefix buttons (e.g. **FA**, **MD**) to toggle entire groups
   - Or edit the multiselect directly for fine-grained control

### Step 3 — Choose ComBat configuration and run

Select one of three options:

| Option | When to use |
|---|---|
| **EB=TRUE** (default) | Standard ComBat; recommended when number of features > sample size, or when sites have small and variable sample sizes |
| **EB=FALSE** | Feature-wise estimation; no pooling across features |
| **Compare EB=TRUE vs EB=FALSE** | Side-by-side evaluation of both; useful when choosing between configurations |

Click **Run Harmonization**.

### Step 4 — Review results

Results appear in five tabs:

1. **Site Deviation** — Site mean z-scores before and after; should approach zero after harmonization
2. **Site Effect Size (Cohen's f)** — Scatter of effect size before vs after; points below the diagonal = reduced site effect
3. **Within-Site Consistency (Overall)** — ICC3 and Spearman r across all features; colored bands show Poor/Moderate/Good/Excellent zones
4. **Within-Site Consistency (By Site)** — Same metrics broken down per site; reveals which sites show lower consistency
5. **Age Associations** — Scatter of Pearson r before vs after; assesses whether biological age-related signal is preserved

### Step 5 — Download the report

Click **Download Full Report (HTML)** at the bottom.

The report is a self-contained HTML file that:
- Opens in any browser, no internet required
- Contains all figures (interactive), all metric tables, and a full methods section
- Includes a pre-written **methods paragraph** formatted for direct inclusion in a manuscript
- Is suitable for submission as supplementary material

---

## Input format details

| Column type | Required | Format |
|---|---|---|
| Site / Batch | Yes | Any string or integer (e.g. `Site_A`, `ROCH1`, `1`) |
| Age | Yes | Numeric (years) |
| Sex | Yes | `Male`/`Female`, `M`/`F`, `0`/`1`, or `1`/`2` — CHORD auto-encodes |
| Features | Yes (at least 1) | Numeric. No prefix convention required |
| Participant ID | No | If present, ignored unless named `output_id` |

---

## Minimum sample size

CHORD requires at least:
- **2 sites** in the batch variable
- **6 participants per site** for by-site ICC and Spearman calculations
- **10 participants per feature** for Spearman and ICC overall

---

## Citing CHORD

If you use CHORD in your research, please cite:

> Onicas AI, et al. CHORD: Comprehensive Harmonization Open-platform with Reporting
> and Diagnostics. *[Journal]*, *[Year]*. GitHub: https://github.com/adionicas/CHORD

ComBat references to include:

> Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data
> using empirical Bayes methods. *Biostatistics*. 2007;8(1):118–127.

> Fortin JP, Parker D, Tunç B, et al. Harmonization of multi-site diffusion tensor imaging data.
> *NeuroImage*. 2017;161:149–170.

> Onicas AI, Ware AL, Harris AD, et al. Impact of ComBat harmonization on structural connectivity
> in pediatric concussion. *NeuroImage*. 2022.

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

For questions, bug reports, or feature requests, open an issue on GitHub.
