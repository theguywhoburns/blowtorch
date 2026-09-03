"""Benchmark a multilayer LIF network: crematorium Sequential vs snnTorch/Norse.

The network is deliberately LIF-heavy and Linear-light (fewer affine
transforms, more neuron layers), because that is where a fused scan wins:

    Linear(F, F) -> LIF x HIDDEN -> Linear(F, OUT) -> LIF

Run from the repo root:

    uv run --group bench python benchmarks/bench_multilayer_sequential_vs_all.py
        [--steps 1000] [--batch 32] [--features 512] [--hidden 4] [--out 10]

Rows:
    crematorium Sequential  the whole network as one scan (forward_sequence),
                          eager and compiled (fast_sequence_); plus the
                          per-step loop as a reference.
    snntorch             nn.Sequential of Linear/Leaky, canonical per-step
                         record-and-stack loop (no sequence module).
    norse                SequentialState of Linear/LIF, native whole-sequence
                         call.

Results are printed and exported to a CSV:

    benchmarks/results/bench_multi_s<steps>_b<batch>_f<features>_h<hidden>_<ts>.csv
    columns: name, variant, compiled, ms_per_step, steps_per_sec

Compilation is lazy (happens during warmup); only steady-state time is
reported.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from crematorium.nn import Sequential
from crematorium.snn import LIF as PkLIF

BETA = 0.9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=1000, help="timesteps per run")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--features", type=int, default=512, help="hidden width F")
    p.add_argument("--hidden", type=int, default=4, help="LIF layers between the two Linears")
    p.add_argument("--out", type=int, default=10, help="output features")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="CSV output directory (default: benchmarks/results)",
    )
    return p.parse_args()


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


# crematorium Sequential


def pk_seq(x: torch.Tensor, device: torch.device, hidden: int, out: int, compiled: bool):
    layers = [nn.Linear(x.shape[2], x.shape[2])] + [PkLIF(validate=False)] * hidden
    layers += [nn.Linear(x.shape[2], out), PkLIF(validate=False)]
    m = Sequential(*layers).to(device)
    if compiled:
        m.fast_sequence_()
    state = m.initial_state_for_sequence(x)

    def run():
        with torch.no_grad():
            return m.forward_sequence(x, state)

    return run


def pk_step(x: torch.Tensor, device: torch.device, hidden: int, out: int):
    layers = [nn.Linear(x.shape[2], x.shape[2])] + [PkLIF(validate=False)] * hidden
    layers += [nn.Linear(x.shape[2], out), PkLIF(validate=False)]
    m = Sequential(*layers).to(device)
    steps, batch, features = x.shape
    state = m.initial_state((batch, features), device=device)

    def run():
        with torch.no_grad():
            s = state
            for t in range(steps):
                s = m.step(x[t], s)[1]

    return run


# snnTorch


def snn_run(x: torch.Tensor, device: torch.device, hidden: int, out: int, compiled: bool):
    import snntorch as snn

    features = x.shape[2]
    steps, batch = x.shape[0], x.shape[1]

    lin_in = nn.Linear(features, features).to(device)
    lin_out = nn.Linear(features, out).to(device)
    hiddens = [snn.Leaky(beta=BETA).to(device) for _ in range(hidden)]
    readout = snn.Leaky(beta=BETA).to(device)

    def step(x_t, h_mems, r_mem):
        cur = lin_in(x_t)
        next_mems = []
        for layer, mem in zip(hiddens, h_mems, strict=True):
            spk, mem = layer(cur, mem)
            next_mems.append(mem)
            cur = spk
        cur = lin_out(cur)
        spk, r_mem = readout(cur, r_mem)
        return spk, tuple(next_mems), r_mem

    cell = torch.compile(step, mode="default") if compiled else step

    def run():
        with torch.no_grad():
            h_mems = tuple(torch.zeros(batch, features, device=device) for _ in hiddens)
            r_mem = torch.zeros(batch, out, device=device)
            spks = []
            for t in range(steps):
                spk, h_mems, r_mem = cell(x[t], h_mems, r_mem)
                spks.append(spk)
            return torch.stack(spks)

    return run


# Norse


def norse_run(x: torch.Tensor, device: torch.device, hidden: int, out: int, compiled: bool):
    import norse.torch as nt

    def p():
        return nt.LIFParameters(
            tau_syn_inv=torch.as_tensor(1000.0),
            tau_mem_inv=torch.as_tensor(100.0),
        )

    seq = nt.SequentialState(
        nn.Linear(x.shape[2], x.shape[2]),
        *[nt.LIF(p=p()) for _ in range(hidden)],
        nn.Linear(x.shape[2], out),
        nt.LIF(p=p()),
    ).to(device)
    if compiled:
        seq = torch.compile(seq, mode="default")

    def run():
        with torch.no_grad():
            return seq(x)

    return run


def measure(make, device: torch.device, warmup: int, reps: int):
    """Run one row; return (best, peak) or None if the framework is missing/broken."""
    torch._dynamo.reset()  # give each compiled variant a clean compile cache
    try:
        return timeit(make(), device, warmup, reps)
    except ImportError:
        return None
    except Exception as exc:
        print(f"      ERROR: {exc}", flush=True)
        return None


def report(label: str, best: float, peak: float, steps: int, base: float | None = None) -> None:
    line = f"{label:<46} {best * 1e3:>10.3f} ms  {steps / best:>11,.0f} steps/s  {peak:>7.1f} MiB"
    if base is not None:
        line += f"  {best / base:>6.2f}x"
    print(line, flush=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    steps, batch, features, hidden, out = args.steps, args.batch, args.features, args.hidden, args.out
    x = torch.randn(steps, batch, features, device=device) * 0.1

    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"bench_multi_s{steps}_b{batch}_f{features}_h{hidden}_{stamp}.csv"

    print(f"device={device}  batch={batch}  features={features}  hidden={hidden}  out={out}  steps={steps}")
    print(f"network: Linear({features},{features}) -> LIF x{hidden} -> Linear({features},{out}) -> LIF")
    print(f"exporting to {path}")
    print("(compile happens lazily during warmup; only steady-state run time is reported)")

    crematorium_rows = [
        ("crematorium Sequential", "seq", False, lambda: pk_seq(x, device, hidden, out, False)),
        ("crematorium Sequential", "seq", True, lambda: pk_seq(x, device, hidden, out, True)),
        ("crematorium Sequential", "step", False, lambda: pk_step(x, device, hidden, out)),
    ]

    other_rows = [
        ("snntorch", "step loop", False, lambda: snn_run(x, device, hidden, out, False)),
        ("snntorch", "step loop", True, lambda: snn_run(x, device, hidden, out, True)),
        ("norse", "seq", False, lambda: norse_run(x, device, hidden, out, False)),
        ("norse", "seq", True, lambda: norse_run(x, device, hidden, out, True)),
    ]

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "variant", "compiled", "ms_per_step", "steps_per_sec"])

        base = None
        print("\n[ crematorium Sequential ]")
        for name, variant, compiled, make in crematorium_rows:
            label = f"{name} {variant}" + (" compile" if compiled else " eager")
            res = measure(make, device, args.warmup, args.reps)
            if res is None:
                print(f"{label:<46} SKIPPED or ERROR", flush=True)
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
                print(f"{label:<46} SKIPPED or ERROR", flush=True)
                continue
            best, peak = res
            report(label, best, peak, steps, base)
            writer.writerow(
                [name, variant, str(compiled), f"{best * 1000 / steps:.6f}", f"{steps / best:.0f}"]
            )


if __name__ == "__main__":
    main()