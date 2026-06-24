"""Export trained artifacts to the web app and to Hugging Face."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

WEB_DATA_DIR = Path("web/public/data")

# Generation checkpoints live in their own folder on the Hub so they do not
# clutter the repo root (which keeps the model card, ladder.json, handler, etc.).
CKPT_DIR = "checkpoints"


# --- Hub persistence for the continuous-training ladder --------------------

def push_ladder_to_hub(repo_id: str, run_dir: str, *, keep_last: int = 12) -> str:
    """Upload ladder.json + recent generation checkpoints to a HF model repo.

    The Hub is the durable store, so a free Space with ephemeral disk loses nothing
    on restart. Only the most recent ``keep_last`` checkpoints are kept hot, and
    they are uploaded under ``checkpoints/`` to keep the repo root tidy.
    """
    from huggingface_hub import HfApi, create_repo

    run = Path(run_dir)
    create_repo(repo_id, exist_ok=True, repo_type="model")
    api = HfApi()
    ladder = run / "ladder.json"
    if ladder.exists():
        api.upload_file(path_or_fileobj=str(ladder), path_in_repo="ladder.json", repo_id=repo_id)
    ckpts = sorted(run.glob("gen-*.pt"))[-keep_last:]
    for ck in ckpts:
        api.upload_file(
            path_or_fileobj=str(ck), path_in_repo=f"{CKPT_DIR}/{ck.name}", repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def pull_ladder_from_hub(repo_id: str, run_dir: str) -> bool:
    """Download ladder.json + its checkpoints from a HF model repo into ``run_dir``.

    Checkpoints are read from ``checkpoints/`` (falling back to the repo root for
    older layouts) and flattened into ``run_dir`` so the server finds them by name.
    Returns True if a ladder was found. Best-effort; returns False on any failure.
    """
    from huggingface_hub import hf_hub_download

    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    try:
        ladder_path = hf_hub_download(repo_id, "ladder.json", local_dir=str(run))
    except Exception:
        return False
    data = json.loads(Path(ladder_path).read_text())
    for gen in data.get("generations", []):
        name = f"{gen['name']}.pt"
        for repo_path in (f"{CKPT_DIR}/{name}", name):  # new layout, then legacy
            try:
                fp = Path(hf_hub_download(repo_id, repo_path, local_dir=str(run)))
            except Exception:
                continue
            target = run / name
            if fp != target:  # flatten checkpoints/<name>.pt -> <name>.pt
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fp), str(target))
            break
    return True


def write_web_data(payloads: dict[str, dict], *, out_dir: str | Path = WEB_DATA_DIR) -> list[str]:
    """Write ``{name: json-serialisable}`` payloads into the web public dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in payloads.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(str(path))
    return written


HF_ENDPOINT_DIR = Path(__file__).resolve().parent.parent / "deploy" / "hf_endpoint"


def push_model_to_hf(repo_id: str, model_path: str, *, card: str | None = None,
                     private: bool = False, with_handler: bool = True) -> str:
    """Upload a checkpoint (+ card + Inference Endpoint handler) to the HF Hub.

    Requires ``huggingface_hub`` and a logged-in token (``huggingface-cli login``
    or ``HF_TOKEN``). When ``with_handler`` is set, also uploads ``handler.py``
    and ``requirements.txt`` from ``deploy/hf_endpoint/`` so the repo can be
    served as a custom Inference Endpoint. Returns the repo URL.
    """
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, exist_ok=True, private=private)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=Path(model_path).name,
        repo_id=repo_id,
    )
    if card:
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
    if with_handler:
        for name in ("handler.py", "requirements.txt"):
            src = HF_ENDPOINT_DIR / name
            if src.exists():
                api.upload_file(
                    path_or_fileobj=str(src), path_in_repo=name, repo_id=repo_id
                )
    return f"https://huggingface.co/{repo_id}"


def model_card(metrics: dict, *, repo_id: str) -> str:
    """A minimal Hugging Face model card for the prediction model."""
    return f"""---
license: apache-2.0
library_name: pytorch
tags: [chess, reinforcement-learning, information-theory]
---

# Shannon's Gambit - Chess Prediction Model (`{repo_id}`)

Multi-head residual network trained on real Lichess games. Heads: policy
(next move), value + win/draw/loss (outcome), and player rating (Elo).

Trained as part of [Shannon's Gambit](https://github.com/). See the repo for
the full MDP/Bellman, deep-RL, and information-theory pipeline.

## Final training metrics

```json
{json.dumps(metrics, indent=2)}
```

## Input/Output

* Input: 18x8x8 board planes (see `shannons_gambit/data/encode.py`).
* Output: policy logits over 4672 moves, scalar value in [-1, 1], WDL logits,
  standardised rating.
"""
