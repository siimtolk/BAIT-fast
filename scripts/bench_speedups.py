#!/usr/bin/env python3
"""Benchmark + equivalence check for the opt-in BAIT speedups (KV cache, candidate batching, fast warm-up).

Runs BAIT.scan_init_token (warm-up + full inversion, the per-batch unit of the scan) on the first N batches of the
real data pipeline, once per configuration, on identical inputs, and reports wall-clock plus whether the per-batch
(q_score, inverted target) matches the original code path. The judge is not called.

  python scripts/bench_speedups.py --model Qwen/Qwen3-0.6B --data-dir DIR --batches 3 [--device cuda:0]
"""
import argparse, os, sys, time, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "unused")
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config.arguments import BAITArguments, DataArguments
from src.data.dataset import build_data_module

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--dataset", default="alpaca")
ap.add_argument("--batches", type=int, default=3)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--dtype", default="float16")
ap.add_argument("--cand-batch", type=int, default=8)
ap.add_argument("--out", default="")
a = ap.parse_args()
logger.remove()

# Imported after the env is set; the patched module reads BAIT_* variables at call time.
from src.core.detector import BAIT

tok = AutoTokenizer.from_pretrained(a.model, padding_side="left", truncation_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# --device auto: shard the model over all visible GPUs (e.g. 8B fp16 on 2 x 16 GB); inputs/timing use cuda:0.
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=("balanced" if a.device == "auto" else {"": a.device})).eval()
if a.device == "auto":
    a.device = "cuda:0"
bargs = BAITArguments()
dargs = DataArguments(dataset=a.dataset, data_dir=a.data_dir)
bargs.batch_size, bargs.prompt_size = dargs.batch_size, dargs.prompt_size  # as BAITWrapper does
if bargs.warmup_batch_size > dargs.prompt_size:
    bargs.warmup_batch_size = dargs.prompt_size
_, loader = build_data_module(dargs, tok, logger)
batches = []
for i, b in enumerate(loader):
    if i >= a.batches:
        break
    batches.append(b)
det = BAIT(model, tok, loader, bargs, logger, torch.device(a.device))

CONFIGS = [
    ("original", {"BAIT_FAST_WARMUP": "0", "BAIT_CAND_BATCH": "0"}, True),
    ("kv-cache (default)", {"BAIT_FAST_WARMUP": "0", "BAIT_CAND_BATCH": "0"}, False),
    ("+ fast warm-up", {"BAIT_FAST_WARMUP": "1", "BAIT_CAND_BATCH": "0"}, False),
    (f"+ candidate batching K={a.cand_batch}", {"BAIT_FAST_WARMUP": "1", "BAIT_CAND_BATCH": str(a.cand_batch)}, False),
]
import types
results = []
for name, env, disable_cache in CONFIGS:
    os.environ.update(env)
    # "original" = the unpatched per-step re-prefill: drop any cache handed to generate().
    gen = BAIT._BAIT__generate
    if disable_cache:
        def nocache(self, ids, mask, max_new_tokens=1, past_key_values=None, _g=gen):
            return _g(self, ids, mask, max_new_tokens, None)[0], None
        BAIT._BAIT__generate = nocache
    outs, times = [], []
    with torch.no_grad():
        for b in batches:
            torch.cuda.synchronize(a.device)
            t = time.time()
            q, tgt = det.scan_init_token(b["input_ids"], b["attention_mask"], b["index_map"])
            torch.cuda.synchronize(a.device)
            times.append(time.time() - t)
            outs.append((float(q), tgt))
    BAIT._BAIT__generate = gen
    results.append((name, sum(times), times, outs, torch.cuda.max_memory_allocated(a.device) / 2**30))
    torch.cuda.reset_peak_memory_stats(a.device)

base = results[0]
print(f"model={a.model} device={a.device} dtype={a.dtype} batches={a.batches}")
print(f"{'config':32} {'sec/batch':>10} {'speedup':>8} {'peak GiB':>9}  matches original (q within 1e-2, same target)")
rows = []
for name, tot, times, outs, peak in results:
    ok = all(abs(o[0] - b[0]) < 1e-2 and o[1] == b[1] for o, b in zip(outs, base[3]))
    print(f"{name:32} {tot/len(times):10.2f} {base[1]/tot:7.2f}x {peak:9.2f}  {ok}")
    rows.append({"config": name, "sec_per_batch": tot/len(times), "speedup": base[1]/tot, "peak_gib": peak,
                 "matches_original": ok, "q": [o[0] for o in outs], "targets": [o[1] for o in outs]})
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
