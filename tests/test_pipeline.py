import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from jev_inspection import score, visa

needs_visa = pytest.mark.skipif(not (visa.DEFAULT_ROOT / "split_csv/1cls.csv").exists(), reason="VisA not downloaded")


def test_prompt_is_the_frozen_one():
    p = visa.prompt("capsules", "ref", 1)
    assert p.state == ("Production-line inspection photos of green gel capsules. The first image is a "
                       "known-good example. The second image is the one under inspection.")
    assert p.question == "Is everything in the photo good, or is at least one part defective?"
    assert p.options == ("defective", "all good") and p.defective_index == 0
    assert visa.prompt("cashew", "single", 0).options[visa.prompt("cashew", "single", 0).defective_index] == "defective"


def test_draws_are_nested_and_exclude_the_reference():
    fr = visa.frozen()
    for obj in visa.CATEGORIES:
        ref = fr["reference"][obj]
        for seed, by_k in fr["patchcore_subsets"][obj].items():
            prev = []
            for k in map(str, fr["ks"]):
                cur = by_k[k]
                assert len(cur) == int(k) and cur[:len(prev)] == prev and ref not in cur
                prev = cur


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 200).astype(float)  # heavy ties on purpose
    y = rng.integers(0, 2, 200)
    assert score.auroc_rows(s, y)[0] == pytest.approx(roc_auc_score(y, s))


def test_holm_is_monotone_and_capped():
    adj = score.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}


@needs_visa
def test_summary_on_synthetic_scores(tmp_path):
    """Fake VLMs at known separability and a PatchCore that improves with k: k* must land where the
    curve crosses, and the full summary must build."""
    rng = np.random.default_rng(1)
    tests = [i for i in visa.load() if i.split == "test"]
    with open(tmp_path / "jev_visa.jsonl", "w") as fj, open(tmp_path / "base_visa.jsonl", "w") as fb:
        for it in tests:
            for model, fh, sep in (("jev", fj, 1.0), ("base", fb, 0.8)):
                for config in visa.CONFIGS:
                    for order in visa.ORDERS:
                        lo = rng.normal(sep * it.is_anomaly, 1.0)
                        fh.write(json.dumps({"model": model, "image": it.image, "config": config, "order": order,
                                             "logodds_defective": lo}) + "\n")
    with open(tmp_path / "pc.jsonl", "w") as fh:
        for it in tests:
            for k, sep in zip((1, 4, 16, "all"), (0.3, 1.0, 2.0, 3.0)):
                for seed in (0, 1, 2):
                    fh.write(json.dumps({"obj": it.obj, "k": k, "seed": seed, "image": it.image,
                                         "score": rng.normal(sep * it.is_anomaly, 1.0)}) + "\n")
    s = score.summarise(visa.DEFAULT_ROOT, [tmp_path / "jev_visa.jsonl", tmp_path / "base_visa.jsonl"],
                        tmp_path / "pc.jsonl", B=200)
    assert s["n_test"] == 2162
    assert s["k_star"]["A0"]["k_star"] == "16"  # separation 2.0 clearly beats 1.0; 1.0 vs 1.0 does not
    # two option orders averaged: noise sd 1/sqrt(2), so AUROC = Phi(1) = 0.841
    assert 0.81 < s["macro_auroc"]["A0"]["point"] < 0.87
    assert abs(s["positional_bias_logodds"]["A0"]) < 0.2
    t = s["triage"]["Call"]["per_category"]
    assert t["escape"] <= 0.12 and 0 <= t["coverage_at_prevalence_0.01"] <= 1
