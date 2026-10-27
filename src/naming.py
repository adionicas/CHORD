"""
naming.py
Guess the measure (modality) of each feature column from its name, for the
suggested modality groups and for the measure labels in the missing-value
tables. These are suggestions for the user to check, never applied silently.
"""

import re as _re


def natural_sorted(labels):
    """Batch labels in natural order: numbers by value, so 2 comes before 11
    (also inside names: Site2 before Site11); otherwise alphabetical."""
    def key(v):
        return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", str(v))]
    return sorted(labels, key=key)


# a name part that names a brain region rather than a measure: FreeSurfer
# style labels with hyphens (ctx-lh-cuneus, Left-Amygdala, Brain-Stem) or a
# hemisphere word
_HEMISPHERE_WORDS = {"left", "right", "lh", "rh", "l", "r", "bilateral"}


def _is_region_part(part: str) -> bool:
    p = part.lower()
    return "-" in p or p in _HEMISPHERE_WORDS or p.startswith("ctx")


def measure_keys(cols) -> dict:
    """
    Map each column name to a guess of its measure.

    - Names with a "." (for example L_S_central.meandepth_native): the text
      after the last ".", when every column has one.
    - Names that start with a measure (FA_CC, fMRI_DefaultMode): the first
      part followed by "_", for example "FA_".
    - Names that start with a brain region (ctx-lh-cuneus_t1w_t2w_mean,
      lh_bankssts_thickness): the ending that names the measure. Of the
      possible endings, the one is taken under which the same region also
      occurs with another measure (so t1w_t2w_mean and t2w_t1w_mean stay
      apart instead of both becoming "mean"); among those, the one shared by
      the most columns; if none, the last part.
    - A name with no "_" or "." is its own group.
    """
    cols = [str(c) for c in cols]
    if len(cols) >= 2 and all("." in c for c in cols):
        dotted = {c: c.rsplit(".", 1)[1] for c in cols}
        if len(set(dotted.values())) >= 2:
            return dotted

    keys, region_first = {}, []
    for c in cols:
        parts = c.split("_")
        if len(parts) < 2:
            keys[c] = c
        elif _is_region_part(parts[0]):
            region_first.append(c)
        else:
            keys[c] = parts[0] + "_"

    # for region-first names: which endings each leading part occurs with
    endings_of_start, n_with_ending = {}, {}
    for c in region_first:
        parts = c.split("_")
        for i in range(1, len(parts)):
            start, end = "_".join(parts[:i]), "_".join(parts[i:])
            endings_of_start.setdefault(start, set()).add(end)
            n_with_ending[end] = n_with_ending.get(end, 0) + 1
    for c in region_first:
        parts = c.split("_")
        shared = [("_".join(parts[i:]), n_with_ending["_".join(parts[i:])])
                  for i in range(1, len(parts))
                  if len(endings_of_start["_".join(parts[:i])]) >= 2]
        keys[c] = max(shared, key=lambda kv: kv[1])[0] if shared else parts[-1]
    return {c: keys[c] for c in cols}
