"""Benchmark blowtorch LIF vs snnTorch and Norse - sequence and per-step modes.

Run from the repo root:

    uv run --group bench python benchmarks/bench_all_vs.py [--steps 1000]

Rows:
    blowtorch LIF   sequence (forward_sequence) and per-step (forward),
                    each in hidden/explicit x eager/compiled.
                    Compiled sequences go through fast_sequence_()
                    (compile-the-scan, mode="default").
    snntorch Leaky  canonical per-step record-and-stack loop (no sequence
                    module).
    norse LIF       real sequence module; LIFCell for per-step.

Results are printed to the console and exported to a CSV:

    benchmarks/results/bench_s<steps>_b<batch>_f<features>_<timestamp>.csv
    columns: name, variant, compiled, ms_per_step, steps_per_sec

Compilation is lazy, so it happens during warmup (the first call); only
steady-state run time is reported.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch

import blowtorch as bt
from blowtorch.snn import LIF as BtLIF

BETA = 0.9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=10000, help="timesteps per run (default 1000)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--features", type=int, default=1024, help="feature dim; compile pays off at >=512")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="CSV output directory (default: benchmarks/results)",
    )
    return p.parse_args()



# Timing



def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device: torch.device, warmup: int, reps: int) -> tuple[float, float]:
    """Return (best run time in seconds, peak GPU MiB during the timed reps)."""
    for _ in range(warmup):
        fn()
        _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _sync(device)

    best = float("inf")
    for _ in range(reps):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, time.perf_counter() - start)

    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    return best, peak



# Blowtorch LIF



def bt_sequence(x: torch.Tensor, device: torch.device, hidden: bool, compiled: bool):
    m = BtLIF(init_hidden=hidden, validate=False).to(device)
    if compiled:
        m.fast_sequence_()
    state = None if hidden else m.initial_state_for_sequence(x)

    def run():
        with torch.no_grad():
            return m.forward_sequence(x, state)

    return run


def bt_step(x: torch.Tensor, device: torch.device, hidden: bool, compiled: bool):
    m = BtLIF(init_hidden=hidden, validate=False).to(device)
    steps, batch, features = x.shape

    if compiled:
        if hidden:
            # Hidden buffers must exist before the compiled graph traces them.
            m.allocate_like(x[0])
        cell = torch.compile(m, mode="default")
    else:
        cell = m

    if hidden:
        def run():
            with torch.no_grad():
                for t in range(steps):
                    cell(x[t])
    else:
        state = m.zero_state((batch, features), device=device)

        def run():
            with torch.no_grad():
                s = state
                for t in range(steps):
                    s = cell(x[t], *s)[1:]

    return run



# snnTorch / Norse



def snn_run(x: torch.Tensor, device: torch.device, compiled: bool):
    import snntorch as snn

    layer = snn.Leaky(beta=BETA).to(device)
    cell = torch.compile(layer, mode="default") if compiled else layer
    steps, batch, features = x.shape

    def run():
        with torch.no_grad():
            mem = torch.zeros(batch, features, device=device)
            outs = []
            for t in range(steps):
                spk, mem = cell(x[t], mem)
                outs.append(spk)
            return torch.stack(outs)

    return run


def norse_sequence(x: torch.Tensor, device: torch.device, compiled: bool):
    import norse.torch as nt

    seq = nt.LIF(p=_norse_params()).to(device)
    if compiled:
        seq = torch.compile(seq, mode="default")

    def run():
        with torch.no_grad():
            spikes, _ = seq(x)
            return spikes

    return run


def norse_step(x: torch.Tensor, device: torch.device, compiled: bool):
    import norse.torch as nt

    cell = nt.LIFCell(p=_norse_params()).to(device)
    if compiled:
        cell = torch.compile(cell, mode="default")
    steps, _, _ = x.shape

    def run():
        with torch.no_grad():
            s = None
            for t in range(steps):
                spk, s = cell(x[t], s)
            return spk

    return run


def _norse_params():
    import norse.torch as nt

    # Euler mapping to blowtorch LIF(beta=0.9): membrane decays 0.9/step,
    # input current fully consumed each step (single compartment).
    return nt.LIFParameters(
        tau_syn_inv=torch.as_tensor(1000.0),
        tau_mem_inv=torch.as_tensor(100.0),
    )



# Measuring and report



def measure(make, device: torch.device, warmup: int, reps: int):
    """Run one row; return (best, peak) or None if the framework is missing/broken."""
    torch._dynamo.reset()  # give each compiled variant a clean compile cache
    try:
        return timeit(make(), device, warmup, reps)
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 - surface failures loudly
        print(f"      ERROR: {exc}", flush=True)
        return None


def report(label: str, best: float, peak: float, steps: int, base: float | None = None) -> None:
    line = f"{label:<40} {best * 1e3:>10.3f} ms  {steps / best:>11,.0f} steps/s  {peak:>7.1f} MiB"
    if base is not None:
        line += f"  {best / base:>6.2f}x"
    print(line, flush=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    steps, batch, features = args.steps, args.batch, args.features
    x = torch.randn(steps, batch, features, device=device) * 0.1

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"bench_s{steps}_b{batch}_f{features}_{stamp}.csv"

    print(f"device={device}  batch={batch}  features={features}  steps={steps}")
    print(f"exporting to {path}")
    print("(compile happens lazily during warmup; only steady-state run time is reported)")

    blowtorch_rows = [
        ("blowtorch LIF", "seq hidden", False, lambda: bt_sequence(x, device, True, False)),
        ("blowtorch LIF", "seq hidden", True, lambda: bt_sequence(x, device, True, True)),
        ("blowtorch LIF", "seq explicit", False, lambda: bt_sequence(x, device, False, False)),
        ("blowtorch LIF", "seq explicit", True, lambda: bt_sequence(x, device, False, True)),
        ("blowtorch LIF", "step hidden", False, lambda: bt_step(x, device, True, False)),
        ("blowtorch LIF", "step hidden", True, lambda: bt_step(x, device, True, True)),
        ("blowtorch LIF", "step explicit", False, lambda: bt_step(x, device, False, False)),
        ("blowtorch LIF", "step explicit", True, lambda: bt_step(x, device, False, True)),
    ]

    other_rows = [
        ("snntorch", "seq", False, lambda: snn_run(x, device, False)),
        ("snntorch", "seq", True, lambda: snn_run(x, device, True)),
        ("norse", "seq", False, lambda: norse_sequence(x, device, False)),
        ("norse", "seq", True, lambda: norse_sequence(x, device, True)),
        ("norse", "step", False, lambda: norse_step(x, device, False)),
        ("norse", "step", True, lambda: norse_step(x, device, True)),
    ]

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "variant", "compiled", "ms_per_step", "steps_per_sec"])

        base = None
        print("\n[ blowtorch LIF ]")
        for name, variant, compiled, make in blowtorch_rows:
            label = f"{name} {variant}" + (" compile" if compiled else " eager")
            res = measure(make, device, args.warmup, args.reps)
            if res is None:
                print(f"{label:<40} SKIPPED or ERROR", flush=True)
                continue
            best, peak = res
            if base is None:
                base = best
            report(label, best, peak, steps)
            writer.writerow(
                [name, variant, str(compiled), f"{best * 1000 / steps:.6f}", f"{steps / best:.0f}"]
            )

        print("\n[ snnTorch / Norse ]")
        for name, variant, compiled, make in other_rows:
            label = f"{name} {variant}" + (" compile" if compiled else " eager")
            res = measure(make, device, args.warmup, args.reps)
            if res is None:
                print(f"{label:<40} SKIPPED or ERROR", flush=True)
                continue
            best, peak = res
            report(label, best, peak, steps, base)
            writer.writerow(
                [name, variant, str(compiled), f"{best * 1000 / steps:.6f}", f"{steps / best:.0f}"]
            )


if __name__ == "__main__":
    main()