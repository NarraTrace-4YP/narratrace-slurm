"""Checks a job's GPUs can talk to each other before a model is split across them.

jobs/llm_serve.sbatch runs this when SLURM gives it more than one GPU, under a time limit,
before loading the model:

    python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=<GPUs> check_gpus.py

One process per GPU sets up NCCL (the GPU-to-GPU library vLLM uses) and Gloo (the CPU-side one
vLLM also uses), adds up one number per GPU, and waits at a barrier on each. If NCCL can't set
up, this hangs or fails in seconds, not after the several minutes a model takes to load
(Blythe, 25 Nov 2026: test job 1389794 passed this in about 20 s on sea003).
"""

from __future__ import annotations

import os
import time
from datetime import timedelta


def main() -> None:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    started = time.monotonic()
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=90))
    try:
        number = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
        dist.all_reduce(number, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        expected = world * (world + 1) / 2
        if number.item() != expected:
            raise SystemExit(f"rank {rank}: NCCL all-reduce gave {number.item()}, not {expected}")
        dist.barrier(device_ids=[local_rank])
        gloo = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
        dist.barrier(group=gloo)
        dist.destroy_process_group(gloo)
        print(
            f"rank {rank}: {torch.cuda.get_device_name(local_rank)}, NCCL and Gloo work"
            f" ({time.monotonic() - started:.1f} s)",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
