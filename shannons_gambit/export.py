"""Export trained artifacts to the web app and to Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path

WEB_DATA_DIR = Path("web/public/data")


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
