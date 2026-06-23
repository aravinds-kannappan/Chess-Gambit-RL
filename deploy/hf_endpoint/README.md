# Hugging Face Inference Endpoint

Files to place in the Hugging Face **model repo** (alongside the uploaded
`model.pt`):

- `handler.py` - custom `EndpointHandler` that loads `ChessNet` and serves moves.
- `requirements.txt` - installs torch, python-chess, and this package from GitHub.

`sgambit export --hf <repo_id>` uploads all of these automatically.

## Request / response

```http
POST https://<endpoint>.endpoints.huggingface.cloud
Authorization: Bearer hf_xxx
Content-Type: application/json

{ "inputs": { "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" } }
```

```json
{
  "best_move": "e2e4",
  "wdl": { "loss": 0.04, "draw": 0.00, "win": 0.96 },
  "value": 0.97,
  "rating": 2577.3,
  "policy_entropy_bits": 1.09,
  "top_moves": [{ "uci": "e2e4", "prob": 0.76 }]
}
```

This is exactly the shape `web/app/lib/hf.ts` parses, so the web app works once
`HF_ENDPOINT_URL` (and `HF_API_TOKEN` if private) are set.
