"""Clef's input path end to end on CPU, with a tiny random backbone and head built from the real
release's config, processor and code (a few MB; the 19 GB of weights are never fetched).

Checks what the pod run relies on: the record encodes with real VisA photos in both configs, Clef's
option order is the same whatever order the frozen prompt lists, the log-odds are read by option id,
and our path reproduces the release's own systemone().
"""
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from jev_inspection import visa, vlm

needs_visa = pytest.mark.skipif(not (visa.DEFAULT_ROOT / "split_csv/1cls.csv").exists(), reason="VisA not downloaded")
SMALL_FILES = [f for f in vlm.CLEF_FILES if f not in ("joint_head.safetensors", "model.safetensors.index.json")]


@pytest.fixture(scope="module")
def tiny_clef(tmp_path_factory):
    from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    try:
        path = Path(vlm._fetch(*vlm.CLEF["clef-flash"], SMALL_FILES))
    except Exception as e:  # offline
        pytest.skip(f"cannot fetch the Clef-Flash config and code: {e}")
    cfg = json.loads((path / "config.json").read_text())
    t, v = cfg["text_config"], cfg["vision_config"]
    t.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
             num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
             linear_key_head_dim=16, linear_value_head_dim=16,
             layer_types=["linear_attention"] * 3 + ["full_attention"])
    # patch_size, temporal_patch_size, in_channels and spatial_merge_size stay as released
    v.update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    torch.manual_seed(0)
    backbone = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**cfg)).eval()
    sys.path.insert(0, str(path))
    import joint_schema_model as jsm
    head = jsm.JointSchemaHead(hidden_size=64, width=32, routing_layers=1, layers=1, heads=2, feedforward=64).eval()
    with torch.no_grad():  # the release zero-initialises the gates; open them so the head is not constant
        head.residual_gate.fill_(1.0)
        head.prior_logit_scale.fill_(1.0)
    backbone, head = backbone.to(torch.bfloat16), head.to(torch.bfloat16)  # as load_release_model does
    clf = object.__new__(vlm.Clef)
    clf.jsm, clf.model, clf.device = jsm, jsm.ClefModel(backbone, head).eval(), "cpu"
    clf.processor = AutoProcessor.from_pretrained(path)
    return clf


def _small(img, side=448):
    """The tiny model runs on CPU; the token-count test below uses full size."""
    img = img.copy()
    img.thumbnail((side, side))
    return img


@needs_visa
def test_option_order_is_fixed_by_clef_and_logodds_read_by_id(tiny_clef):
    img = _small(Image.open(visa.DEFAULT_ROOT / visa.reference("cashew")).convert("RGB"))
    out = {}
    for order in visa.ORDERS:
        p = visa.prompt("cashew", "single", order)
        z, ids, _ = vlm.clef_logits(tiny_clef, vlm.clef_record(p, [img]))
        assert ids == [visa.GOOD, visa.DEFECTIVE]  # sorted, whatever p.options says
        out[order] = (z[ids.index(visa.DEFECTIVE)] - z[ids.index(visa.GOOD)]).item()
    assert out[0] == pytest.approx(out[1], abs=1e-9)  # same input, same answer
    assert out[0] != 0.0


@needs_visa
def test_ref_config_takes_two_images(tiny_clef):
    p = visa.prompt("pcb1", "ref", 0)
    imgs = [_small(Image.open(visa.DEFAULT_ROOT / f).convert("RGB")) for f in
            (visa.reference("pcb1"), visa.reference("cashew"))]
    _, _, n_two = vlm.clef_logits(tiny_clef, vlm.clef_record(p, imgs))
    _, _, n_one = vlm.clef_logits(tiny_clef, vlm.clef_record(visa.prompt("pcb1", "single", 0), imgs[1:]))
    assert n_two > n_one


@needs_visa
def test_our_path_matches_systemone(tiny_clef):
    res = vlm.check_clef_equivalence(tiny_clef, visa.DEFAULT_ROOT)  # full-size photo; the tiny model copes
    assert res["max_dp_vs_systemone"] <= 5e-5  # systemone() rounds probabilities to 4 decimals


@needs_visa
def test_token_count_at_full_resolution(tiny_clef):
    """Not an assertion on the model: records how many tokens a full-size VisA photo costs, so the
    pod's memory and the pre-registered resolution are set from a number."""
    counts = {}
    for obj in visa.CATEGORIES:
        img = Image.open(visa.DEFAULT_ROOT / visa.reference(obj)).convert("RGB")
        enc = tiny_clef.jsm.encode_record(tiny_clef.processor.tokenizer,
                                          vlm.clef_record(visa.prompt(obj, "ref", 0), [img, img]),
                                          processor=tiny_clef.processor)
        counts[obj] = (img.size, len(enc.input_ids))
    print(json.dumps(counts))
    assert all(n < 16384 for _, n in counts.values())
