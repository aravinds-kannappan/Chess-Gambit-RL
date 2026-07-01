"""Export trained artifacts to the web app and to Hugging Face."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

WEB_DATA_DIR = Path("web/public/data")

# The HF model repo mirrors the training pipeline as folders:
#   pretrain/  -> supervised behavioural-cloning net (the served base, model.pt)
#   posttrain/ -> self-play RL refinement (the versioned generation checkpoints)
# keeping the repo root for the model card, ladder.json, handler, requirements.
PRETRAIN_DIR = "pretrain"
CKPT_DIR = "posttrain"
_LEGACY_CKPT_DIRS = ("checkpoints",)  # earlier layouts, still readable


# --- Hub persistence for the continuous-training ladder --------------------

def push_ladder_to_hub(repo_id: str, run_dir: str, *, keep_last: int = 12) -> str:
    """Upload ladder.json + recent generation checkpoints to a HF model repo.

    The Hub is the durable store, so a free Space with ephemeral disk loses nothing
    on restart. Only the most recent ``keep_last`` checkpoints are kept hot, and
    they are uploaded under ``posttrain/`` (the RL-refinement stage).
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


def pull_ladder_from_hub(repo_id: str, run_dir: str, *, max_checkpoints: int = 8) -> bool:
    """Download ladder.json + the *served* checkpoints from a HF model repo.

    Only the checkpoints that can actually be served are fetched: the gated
    champion, the highest-Elo entry, and the most recent ``max_checkpoints``
    generations. Pulling every historical generation (the ladder can hold
    thousands) is what made the Space hang in startup and restart-loop. Files are
    flattened into ``run_dir`` so the server finds them by name. Best-effort.
    """
    from huggingface_hub import hf_hub_download

    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    try:
        ladder_path = hf_hub_download(repo_id, "ladder.json", local_dir=str(run))
    except Exception:
        return False
    data = json.loads(Path(ladder_path).read_text())
    gens = data.get("generations", [])
    wanted = {g["gen"] for g in gens[-max_checkpoints:]}
    if data.get("champion_gen") is not None:
        wanted.add(data["champion_gen"])
    if gens:  # the entry serve.predict()/best() would resolve to
        wanted.add(max(gens, key=lambda g: g.get("elo", 0))["gen"])
    for gen in gens:
        if gen["gen"] not in wanted:
            continue
        name = f"{gen['name']}.pt"
        # current layout, then legacy folders, then bare repo root
        repo_paths = [f"{CKPT_DIR}/{name}", *[f"{d}/{name}" for d in _LEGACY_CKPT_DIRS], name]
        for repo_path in repo_paths:
            try:
                fp = Path(hf_hub_download(repo_id, repo_path, local_dir=str(run)))
            except Exception:
                continue
            target = run / name
            if fp != target:  # flatten posttrain/<name>.pt -> <name>.pt
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fp), str(target))
            break
    return True


def push_personal(repo_id: str, session_id: str, ckpt_path: str) -> str:
    """Persist a per-session personal checkpoint to the Hub.

    Free Spaces have ephemeral disks, so an adapted personal net vanishes on
    every restart unless it lives in the model repo (under ``personal/``).
    """
    from huggingface_hub import HfApi

    HfApi().upload_file(path_or_fileobj=ckpt_path,
                        path_in_repo=f"personal/{session_id}.pt", repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def pull_personal(repo_id: str, session_id: str, run_dir: str) -> str | None:
    """Fetch a session's personal checkpoint from the Hub (None if absent)."""
    from huggingface_hub import hf_hub_download

    target = Path(run_dir) / "personal" / f"{session_id}.pt"
    if target.exists():
        return str(target)
    try:
        fp = Path(hf_hub_download(repo_id, f"personal/{session_id}.pt", local_dir=run_dir))
    except Exception:
        return None
    if fp != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fp), str(target))
    return str(target)


def push_opening_book(repo_id: str, book_path: str) -> str:
    """Upload the learned opening book to the model repo (served by the Space)."""
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, exist_ok=True, repo_type="model")
    HfApi().upload_file(path_or_fileobj=book_path, path_in_repo="opening_book.json",
                        repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def pull_opening_book(repo_id: str, run_dir: str) -> str | None:
    """Download ``opening_book.json`` into ``run_dir`` (best-effort)."""
    from huggingface_hub import hf_hub_download

    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    try:
        fp = Path(hf_hub_download(repo_id, "opening_book.json", local_dir=str(run)))
    except Exception:
        return None
    target = run / "opening_book.json"
    if fp != target:
        shutil.move(str(fp), str(target))
    return str(target)


def pull_base_model(repo_id: str, run_dir: str) -> str | None:
    """Download the served base net (pretrain/model.pt, legacy root model.pt)."""
    from huggingface_hub import hf_hub_download

    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    for repo_path in (f"{PRETRAIN_DIR}/model.pt", "model.pt"):
        try:
            fp = Path(hf_hub_download(repo_id, repo_path, local_dir=str(run)))
        except Exception:
            continue
        target = run / "model.pt"
        if fp != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(fp), str(target))
        return str(target)
    return None


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
                     private: bool = False, with_handler: bool = True,
                     dest: str | None = None) -> str:
    """Upload a checkpoint (+ card + Inference Endpoint handler) to the HF Hub.

    Requires ``huggingface_hub`` and a logged-in token (``huggingface-cli login``
    or ``HF_TOKEN``). ``dest`` sets the path in the repo (e.g. ``pretrain/model.pt``);
    defaults to the file's name at the root. When ``with_handler`` is set, also
    uploads ``handler.py`` and ``requirements.txt`` from ``deploy/hf_endpoint/``
    so the repo can be served as a custom Inference Endpoint. Returns the repo URL.
    """
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, exist_ok=True, private=private)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=dest or Path(model_path).name,
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
