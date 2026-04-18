"""
Proper speedup measurement: FD solver vs surrogate inference.
Averages over 20 cases with std deviation, for both legacy and LXCat modes.
"""
import os, sys, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
os.environ['MPLBACKEND'] = 'Agg'

from solver import TELSolver

OUT = os.path.join(REPO, 'results', 'speedup_measurement')
os.makedirs(OUT, exist_ok=True)

R_PROC, Z_TOP = 0.105, 0.234

# Load surrogate_v4 for inference timing
V4_DIR = os.path.join(REPO, 'results', 'surrogate_v4')
LXCAT_DIR = os.path.join(REPO, 'results', 'surrogate_lxcat_v3')

# 20 diverse test conditions
CONDITIONS = [
    (200, 3, 0.0), (300, 5, 0.0), (500, 7, 0.0), (700, 10, 0.0), (1000, 15, 0.0),
    (200, 10, 0.1), (500, 10, 0.25), (700, 10, 0.5), (1000, 5, 0.1), (1200, 20, 0.5),
    (300, 3, 0.05), (400, 8, 0.15), (600, 12, 0.3), (800, 15, 0.4), (900, 18, 0.45),
    (1100, 7, 0.2), (350, 20, 0.0), (750, 6, 0.35), (550, 14, 0.1), (1050, 9, 0.25),
]


class SurrogateNet(torch.nn.Module):
    def __init__(self, n_out=2, nh=128, nl=4, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(5, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        layers = [torch.nn.Linear(ni, nh), torch.nn.GELU(), torch.nn.Dropout(drop)]
        for _ in range(nl - 1):
            layers += [torch.nn.Linear(nh, nh), torch.nn.GELU(), torch.nn.Dropout(drop)]
        self.bb = torch.nn.Sequential(*layers)
        self.proj = torch.nn.Linear(ni, nh)
        self.head = torch.nn.Linear(nh, n_out)

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))


def time_solver(mode, n_cases=20):
    times = []
    for P, p, Ar in CONDITIONS[:n_cases]:
        t0 = time.time()
        solver = TELSolver(Nr=30, Nz=50, P_rf=P, p_mTorr=p,
                           frac_Ar=Ar, rate_mode=mode)
        solver.solve(n_iter=80, w=0.12, verbose=False)
        times.append(time.time() - t0)
    return times


def time_surrogate(model_dir, n_cases=20, n_repeats=50):
    """Time surrogate inference including data prep."""
    model = SurrogateNet()
    ckpt = os.path.join(model_dir, 'model_0.pt')
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval()

    # Build a representative input batch (642 active cells per case)
    r = np.linspace(0.001, R_PROC, 30)
    z = np.linspace(0.001, Z_TOP, 50)

    times = []
    for P, p, Ar in CONDITIONS[:n_cases]:
        # Repeat many times for stable timing (inference is sub-ms)
        t0 = time.time()
        for _ in range(n_repeats):
            # Include data prep in timing
            X = []
            for ri in r:
                for zi in z:
                    X.append([ri / R_PROC, zi / Z_TOP, P / 1200, p / 20, Ar])
            X_t = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                _ = model(X_t)
        dt = (time.time() - t0) / n_repeats
        times.append(dt)
    return times


def main():
    print("=== Speedup Measurement ===\n", flush=True)

    # Legacy solver
    print("Timing legacy solver (20 cases)...", flush=True)
    leg_times = time_solver('legacy')
    print(f"  Legacy: {np.mean(leg_times):.3f} +/- {np.std(leg_times):.3f} s\n", flush=True)

    # LXCat solver
    print("Timing LXCat solver (20 cases)...", flush=True)
    lx_times = time_solver('lxcat')
    print(f"  LXCat: {np.mean(lx_times):.3f} +/- {np.std(lx_times):.3f} s\n", flush=True)

    # Surrogate v4
    print("Timing surrogate_v4 inference (20 cases x 50 repeats)...", flush=True)
    v4_times = time_surrogate(V4_DIR)
    print(f"  Surrogate v4: {np.mean(v4_times)*1000:.2f} +/- {np.std(v4_times)*1000:.2f} ms\n", flush=True)

    # Surrogate lxcat_v3
    print("Timing surrogate_lxcat_v3 inference (20 cases x 50 repeats)...", flush=True)
    lx_surr_times = time_surrogate(LXCAT_DIR)
    print(f"  Surrogate lxcat_v3: {np.mean(lx_surr_times)*1000:.2f} +/- {np.std(lx_surr_times)*1000:.2f} ms\n", flush=True)

    results = {
        'n_cases': 20,
        'n_inference_repeats': 50,
        'legacy_solver': {
            'mean_s': float(np.mean(leg_times)),
            'std_s': float(np.std(leg_times)),
            'min_s': float(np.min(leg_times)),
            'max_s': float(np.max(leg_times)),
        },
        'lxcat_solver': {
            'mean_s': float(np.mean(lx_times)),
            'std_s': float(np.std(lx_times)),
            'min_s': float(np.min(lx_times)),
            'max_s': float(np.max(lx_times)),
        },
        'surrogate_v4': {
            'mean_ms': float(np.mean(v4_times) * 1000),
            'std_ms': float(np.std(v4_times) * 1000),
        },
        'surrogate_lxcat_v3': {
            'mean_ms': float(np.mean(lx_surr_times) * 1000),
            'std_ms': float(np.std(lx_surr_times) * 1000),
        },
        'speedups': {
            'legacy_vs_v4': float(np.mean(leg_times) / np.mean(v4_times)),
            'lxcat_vs_lxcat_surr': float(np.mean(lx_times) / np.mean(lx_surr_times)),
            'lxcat_vs_v4_surr': float(np.mean(lx_times) / np.mean(v4_times)),
        },
    }

    with open(os.path.join(OUT, 'speedup.json'), 'w') as f:
        json.dump(results, f, indent=2)

    md = f"""# Speedup Measurement

## Solver timing (20 diverse conditions, 30x50 mesh)

| Mode | Mean | Std | Min | Max |
|---|---|---|---|---|
| Legacy FD | {results['legacy_solver']['mean_s']:.3f}s | {results['legacy_solver']['std_s']:.3f}s | {results['legacy_solver']['min_s']:.3f}s | {results['legacy_solver']['max_s']:.3f}s |
| LXCat FD | {results['lxcat_solver']['mean_s']:.3f}s | {results['lxcat_solver']['std_s']:.3f}s | {results['lxcat_solver']['min_s']:.3f}s | {results['lxcat_solver']['max_s']:.3f}s |

## Surrogate inference (20 cases, 50 repeats each, CPU)

| Model | Mean | Std |
|---|---|---|
| surrogate_v4 | {results['surrogate_v4']['mean_ms']:.2f}ms | {results['surrogate_v4']['std_ms']:.2f}ms |
| surrogate_lxcat_v3 | {results['surrogate_lxcat_v3']['mean_ms']:.2f}ms | {results['surrogate_lxcat_v3']['std_ms']:.2f}ms |

## Speedups

| Comparison | Speedup |
|---|---|
| Legacy solver vs surrogate_v4 | {results['speedups']['legacy_vs_v4']:.0f}x |
| LXCat solver vs LXCat surrogate | {results['speedups']['lxcat_vs_lxcat_surr']:.0f}x |
| LXCat solver vs surrogate_v4 | {results['speedups']['lxcat_vs_v4_surr']:.0f}x |
"""

    with open(os.path.join(OUT, 'speedup.md'), 'w') as f:
        f.write(md)

    print(f"\n{'='*60}")
    print(f"  Legacy speedup:  {results['speedups']['legacy_vs_v4']:.0f}x")
    print(f"  LXCat speedup:   {results['speedups']['lxcat_vs_lxcat_surr']:.0f}x")
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
