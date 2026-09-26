from __future__ import annotations

import json
from pathlib import Path

import click

from . import visa


@click.group()
def main():
    """VisA inspection: PatchCore k-sweep against zero-shot VLMs."""


@main.command()
@click.option("--model", type=click.Choice(["jev", "base"]), required=True)
@click.option("--config", "configs", multiple=True, default=visa.CONFIGS, type=click.Choice(visa.CONFIGS))
@click.option("--root", type=Path, default=visa.DEFAULT_ROOT)
@click.option("--out", type=Path, default=Path("results"))
@click.option("--category", "categories", multiple=True, type=click.Choice(visa.CATEGORIES))
@click.option("--limit", type=int, help="first N normals and N anomalies per category (smoke test)")
@click.option("--device", default="cuda")
def vlm(model, configs, root, out, categories, limit, device):
    """Score the test set with Jev-Omni or Gemma 4 zero-shot (GPU pod)."""
    from .vlm import run
    run(model, configs, visa.ORDERS, root, out, device, limit, categories)


@main.command()
@click.option("--root", type=Path, default=visa.DEFAULT_ROOT)
@click.option("--out", type=Path, default=Path("results"))
@click.option("--n", default=50)
@click.option("--device", default="cuda")
def probe(root, out, n, device):
    """Contamination probe: does Gemma 4 name the dataset?"""
    from .vlm import contamination_probe
    out.mkdir(parents=True, exist_ok=True)
    contamination_probe(root, out, n, device)


@main.command()
@click.option("--root", type=Path, default=visa.DEFAULT_ROOT)
@click.option("--out", type=Path, default=Path("results"))
@click.option("--category", "categories", multiple=True, type=click.Choice(visa.CATEGORIES))
@click.option("--k", "ks", multiple=True, help="default: 1 2 4 8 16 64 all")
@click.option("--seed", "seeds", multiple=True, type=int, default=(0, 1, 2))
@click.option("--backbone", default="wide_resnet50_2")
@click.option("--maps-for", nargs=2, help="save anomaly maps for this k and seed, e.g. --maps-for 16 0")
@click.option("--device", default="cuda")
def patchcore(root, out, categories, ks, seeds, backbone, maps_for, device):
    """PatchCore fitted on k good parts per category."""
    from .patchcore import KS, sweep
    ks = [k if k == "all" else int(k) for k in ks] or list(KS)
    maps = (maps_for[0], int(maps_for[1])) if maps_for else None
    sweep(root, out, categories or visa.CATEGORIES, ks, seeds, backbone, device, maps)


@main.command()
@click.option("--root", type=Path, default=visa.DEFAULT_ROOT)
@click.option("--results", type=Path, default=Path("results"))
@click.option("--patchcore-file", default="patchcore_wide_resnet50_2.jsonl")
@click.option("--category", "categories", multiple=True, type=click.Choice(visa.CATEGORIES))
@click.option("--bootstrap", default=2000)
def score(root, results, patchcore_file, categories, bootstrap):
    """Primary endpoint, k*, triage and breakdowns -> results/summary.json."""
    from .score import summarise
    s = summarise(root, [results / "jev_visa.jsonl", results / "base_visa.jsonl"], results / patchcore_file,
                  B=bootstrap, categories=categories or visa.CATEGORIES)
    (results / "summary.json").write_text(json.dumps(s, indent=1))
    click.echo(json.dumps({k: s[k] for k in ("macro_auroc", "k_star")}, indent=1))


@main.command()
@click.option("--root", type=Path, default=visa.DEFAULT_ROOT)
@click.option("--spotdiff", type=Path, required=True, help="checkout of amazon-science/spot-diff")
@click.option("--out", type=Path, default=Path("data"))
def step0(root, spotdiff, out):
    """Counts, defect vocabulary and the frozen draws (already committed in data/)."""
    import subprocess
    import sys
    subprocess.run([sys.executable, str(Path(__file__).with_name("step0.py")), str(root), str(spotdiff), str(out)],
                   check=True)
