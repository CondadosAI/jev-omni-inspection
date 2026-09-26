"""Scoring, as pre-registered in DESIGN.md.

Primary endpoint: macro image AUROC over the 12 categories. The headline k* is the smallest k at
which PatchCore's macro AUROC beats a VLM's with the lower end of a 95% paired bootstrap interval
above zero. Everything else is descriptive.

Bootstrap: resample normals and anomalies separately inside each category (class balance per
category is fixed by the split), the same indices for every system in a replicate; for PatchCore a
replicate also draws one of the three sampling seeds.

Triage: auto-accept below t_lo, auto-reject above t_hi, a human in between. t_lo lets at most 5% of
the fold's training anomalies through; t_hi rejects at most 5% of its training normals. Fitted by
5-fold cross-fitting so every test image is scored by thresholds that never saw it. Per-category
thresholds are the like-for-like comparison (PatchCore scores are not on a common scale across
products, and a line fits one threshold per product anyway); a single global threshold pair is
reported for the VLMs as well, whose log-odds are on one scale.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata

from . import visa

VLM_SYSTEMS = {"A0": ("jev", "single"), "A1": ("jev", "ref"), "B0": ("base", "single"), "B1": ("base", "ref")}
ESCAPE, OVERKILL, PREVALENCE = 0.05, 0.05, 0.01


# --- loading ---------------------------------------------------------------------------------
def load_vlm(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    """One column per VLM system: log-odds averaged over the two option orders."""
    rows = [json.loads(line) for p in paths if p.exists() for line in p.open()]
    df = pd.DataFrame(rows)
    cols, bias = {}, {}
    for name, (model, config) in VLM_SYSTEMS.items():
        sub = df[(df.model == model) & (df.config == config)]
        if sub.empty:
            continue
        wide = sub.pivot_table(index="image", columns="order", values="logodds_defective")
        if wide.isna().any().any() or set(wide.columns) != {0, 1}:
            raise ValueError(f"{name}: some images lack one of the two option orders")
        cols[name] = wide.mean(axis=1)
        bias[name] = float((wide[0] - wide[1]).mean())  # >0: "defective" favoured when listed second
    return pd.DataFrame(cols), bias


def load_patchcore(path: Path) -> dict[str, pd.DataFrame]:
    """{k: DataFrame(index=image, columns=seed)}."""
    df = pd.DataFrame([json.loads(line) for line in path.open()])
    return {str(k): g.pivot_table(index="image", columns="seed", values="score") for k, g in df.groupby("k")}


# --- AUROC -----------------------------------------------------------------------------------
def auroc_rows(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC for each row of s (replicates x images) against labels y (same for every row, or per row).
    Mann-Whitney via average ranks, so ties count one half."""
    s = np.atleast_2d(s)
    y = np.broadcast_to(y, s.shape)
    r = rankdata(s, axis=1)
    na = y.sum(1)
    nn = y.shape[1] - na
    return ((r * y).sum(1) - na * (na + 1) / 2) / (na * nn)


class Table:
    """Test images in a fixed order, with the per-category index sets the bootstrap resamples."""

    def __init__(self, root: Path, categories=visa.CATEGORIES):
        items = [i for i in visa.load(root) if i.split == "test" and i.obj in categories]
        self.images = [i.image for i in items]
        self.obj = np.array([i.obj for i in items])
        self.y = np.array([i.is_anomaly for i in items], dtype=int)
        self.types = [i.types for i in items]
        self.categories = [c for c in categories if c in set(self.obj)]
        self.idx = {c: (np.where((self.obj == c) & (self.y == 0))[0], np.where((self.obj == c) & (self.y == 1))[0])
                    for c in self.categories}

    def resample(self, B: int, rng) -> dict[str, np.ndarray]:
        """Per category: (B, n_c) indices, normals first then anomalies, drawn with replacement."""
        out = {}
        for c, (n, a) in self.idx.items():
            out[c] = np.hstack([rng.choice(n, (B, len(n))), rng.choice(a, (B, len(a)))])
        return out

    def labels(self, c):
        n, a = self.idx[c]
        return np.r_[np.zeros(len(n), int), np.ones(len(a), int)]


def macro_boot(tab: Table, draws: dict, scores: np.ndarray, seed_pick=None) -> np.ndarray:
    """(B,) macro AUROC. scores: (n_images,) or (n_images, n_seeds) with seed_pick (B,) choosing one."""
    per_cat = []
    for c in tab.categories:
        ix = draws[c]
        s = scores[ix] if scores.ndim == 1 else scores[ix, seed_pick[:, None]]
        per_cat.append(auroc_rows(s, tab.labels(c)))
    return np.mean(per_cat, 0)


def per_category(tab: Table, s: np.ndarray) -> dict[str, float]:
    out = {}
    for c in tab.categories:
        n, a = tab.idx[c]
        ix = np.r_[n, a]
        out[c] = float(auroc_rows(s[ix], tab.labels(c))[0])
    return out


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=p.get)
    m, adj, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return adj


# --- triage ----------------------------------------------------------------------------------
def _thresholds(s, y):
    a, n = s[y == 1], s[y == 0]
    t_lo = np.quantile(a, ESCAPE, method="lower")  # accept strictly below: <= 5% of anomalies
    t_hi = np.quantile(n, 1 - OVERKILL, method="higher")  # reject strictly above: <= 5% of normals
    if t_lo > t_hi:  # the two classes separate at these rates: one cut decides everything
        t_lo = t_hi = (t_lo + t_hi) / 2
    return t_lo, t_hi


def _folds(groups: np.ndarray, n_folds: int, rng) -> np.ndarray:
    """Stratified fold id per image, stratified on the given group labels."""
    fold = np.empty(len(groups), int)
    for g in np.unique(groups):
        ix = np.where(groups == g)[0]
        fold[rng.permutation(ix)] = np.arange(len(ix)) % n_folds
    return fold


def triage(tab: Table, s: np.ndarray, per_category_thresholds: bool, n_folds=5, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    strata = np.char.add(tab.obj.astype(str), tab.y.astype(str))
    fold = _folds(strata, n_folds, rng)
    decision = np.empty(len(s), dtype="<U6")
    scopes = tab.categories if per_category_thresholds else [None]
    for c in scopes:
        in_scope = np.ones(len(s), bool) if c is None else tab.obj == c
        for f in range(n_folds):
            fit_ix, app_ix = in_scope & (fold != f), in_scope & (fold == f)
            t_lo, t_hi = _thresholds(s[fit_ix], tab.y[fit_ix])
            d = np.where(s[app_ix] < t_lo, "accept", np.where(s[app_ix] > t_hi, "reject", "human"))
            decision[app_ix] = d
    a, n = tab.y == 1, tab.y == 0
    escapes = int((decision[a] == "accept").sum())
    n_a = int(a.sum())  # Clopper-Pearson interval on the escape count
    lo = beta.ppf(0.025, escapes, n_a - escapes + 1) if escapes else 0.0
    hi = beta.ppf(0.975, escapes + 1, n_a - escapes) if escapes < n_a else 1.0
    auto_def = float((decision[a] != "human").mean())
    auto_good = float((decision[n] != "human").mean())
    return {"escape": escapes / a.sum(), "escape_ci95": [float(lo), float(hi)], "escapes": escapes,
            "overkill": float((decision[n] == "reject").mean()),
            "auto_given_defective": auto_def, "auto_given_good": auto_good,
            f"coverage_at_prevalence_{PREVALENCE}": PREVALENCE * auto_def + (1 - PREVALENCE) * auto_good}


# --- main ------------------------------------------------------------------------------------
def summarise(root: Path, vlm_paths: list[Path], patchcore_path: Path | None, B=2000, seed=0,
              categories=visa.CATEGORIES) -> dict:
    tab = Table(root, categories)
    vlm, bias = load_vlm(vlm_paths)
    systems = {name: vlm[name].reindex(tab.images).to_numpy() for name in vlm.columns}
    for name, s in systems.items():
        if np.isnan(s).any():
            raise ValueError(f"{name}: {np.isnan(s).sum()} test images without a score")
    pc = {}
    if patchcore_path is not None and patchcore_path.exists():
        for k, wide in load_patchcore(patchcore_path).items():
            m = wide.reindex(tab.images).to_numpy()
            if np.isnan(m).any():
                raise ValueError(f"PatchCore k={k}: missing scores")
            pc[k] = m
    ks = [k for k in map(str, (1, 2, 4, 8, 16, 64, "all")) if k in pc]

    rng = np.random.default_rng(seed)
    draws = tab.resample(B, rng)
    seed_pick = {k: rng.integers(0, pc[k].shape[1], B) for k in ks}
    boot = {name: macro_boot(tab, draws, s) for name, s in systems.items()}
    boot.update({f"C{k}": macro_boot(tab, draws, pc[k], seed_pick[k]) for k in ks})

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    point = {name: float(np.mean(list(per_category(tab, s).values()))) for name, s in systems.items()}
    point.update({f"C{k}": float(np.mean([np.mean(list(per_category(tab, pc[k][:, j]).values()))
                                          for j in range(pc[k].shape[1])])) for k in ks})
    out = {"n_test": len(tab.images), "bootstrap": B, "positional_bias_logodds": bias,
           "macro_auroc": {n: {"point": point[n], "ci95": ci(boot[n])} for n in point}}

    out["k_star"] = {}
    for v in systems:
        rows, k_star = [], None
        for k in ks:
            d = boot[f"C{k}"] - boot[v]
            rows.append({"k": k, "diff": point[f"C{k}"] - point[v], "ci95": ci(d)})
            if k_star is None and np.percentile(d, 2.5) > 0:
                k_star = k
        out["k_star"][v] = {"k_star": k_star, "curve": rows}

    out["per_category_auroc"] = {n: per_category(tab, s) for n, s in systems.items()}
    out["per_category_auroc"].update({f"C{k}": {c: float(np.mean([per_category(tab, pc[k][:, j])[c]
                                                                  for j in range(pc[k].shape[1])]))
                                                for c in tab.categories} for k in ks})
    out["per_group_auroc"] = {n: {g: float(np.mean([pcs[c] for c in cats if c in pcs]))
                                  for g, cats in visa.GROUPS.items()}
                              for n, pcs in out["per_category_auroc"].items()}

    if {"A0", "B0"} <= systems.keys():  # does the head help at all on inspection? (descriptive)
        pvals, diffs = {}, {}
        for c in tab.categories:
            ix = draws[c]
            d = auroc_rows(systems["A0"][ix], tab.labels(c)) - auroc_rows(systems["B0"][ix], tab.labels(c))
            diffs[c] = float(np.mean(d))
            pvals[c] = max(1 / B, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out["A0_vs_B0"] = {"macro_diff_ci95": ci(boot["A0"] - boot["B0"]),
                           "per_category_diff": diffs, "per_category_p_holm": holm(pvals)}

    out["triage"] = {}
    for n, s in systems.items():
        out["triage"][n] = {"per_category": triage(tab, s, True), "global": triage(tab, s, False)}
    for k in ks:
        per_seed = [triage(tab, pc[k][:, j], True) for j in range(pc[k].shape[1])]
        out["triage"][f"C{k}"] = {"per_category": {m: (float(np.mean([r[m] for r in per_seed]))
                                                       if not isinstance(per_seed[0][m], list) else None)
                                                   for m in per_seed[0]}}

    by_type = defaultdict(dict)  # descriptive: which defect types each system misses
    type_systems = {n: s[:, None] for n, s in systems.items()} | {f"C{k}": pc[k] for k in ks if k in ("16", "all")}
    for n, m in type_systems.items():  # (images, seeds): AUROC per seed, then the mean, as elsewhere
        for c in tab.categories:
            nrm, anom = tab.idx[c]
            for t in sorted({t for i in anom for t in tab.types[i]}):
                sel = np.array([i for i in anom if t in tab.types[i]])
                if len(sel) >= 5:
                    ix = np.r_[nrm, sel]
                    y = np.r_[np.zeros(len(nrm), int), np.ones(len(sel), int)]
                    auc = np.mean([auroc_rows(m[ix, j], y)[0] for j in range(m.shape[1])])
                    by_type[n][f"{c}/{t}"] = {"auroc": float(auc), "n": int(len(sel))}
    out["per_defect_type_auroc"] = by_type
    return out
