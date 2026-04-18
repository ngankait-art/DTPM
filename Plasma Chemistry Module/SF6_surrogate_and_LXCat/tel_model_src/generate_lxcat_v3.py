"""
Generate pinn_dataset_lxcat_v3: all 221 conditions from v4, solved with rate_mode='lxcat'.

This is the first full-coverage LXCat dataset enabling a fair comparison
against surrogate_v4 (which was trained on 221 legacy-rate cases).
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

from solver import TELSolver

DS_V4 = os.path.join(REPO, 'results', 'pinn_dataset_v4')
DS_OUT = os.path.join(REPO, 'results', 'pinn_dataset_lxcat_v3')


def main():
    os.makedirs(DS_OUT, exist_ok=True)

    with open(os.path.join(DS_V4, 'metadata.json')) as f:
        v4_meta = json.load(f)

    n_total = len(v4_meta)
    print(f"Generating LXCat v3 dataset: {n_total} cases", flush=True)
    print(f"Output: {DS_OUT}", flush=True)

    # Check which cases already exist (resume support)
    existing = set()
    for entry in v4_meta:
        fpath = os.path.join(DS_OUT, entry['file'])
        if os.path.exists(fpath):
            existing.add(entry['idx'])
    if existing:
        print(f"  Resuming: {len(existing)}/{n_total} already done", flush=True)

    meta_out = []
    t0 = time.time()
    n_failed = 0

    for entry in v4_meta:
        idx = entry['idx']
        fname = entry['file']
        P_rf = entry['P_rf']
        p_mTorr = entry['p_mTorr']
        frac_Ar = entry['frac_Ar']

        outpath = os.path.join(DS_OUT, fname)

        if idx in existing:
            # Load existing to get metadata
            d = np.load(outpath)
            meta_entry = {
                'idx': idx, 'file': fname,
                'P_rf': P_rf, 'p_mTorr': p_mTorr, 'frac_Ar': frac_Ar,
                'ne_avg': float(d['ne_avg']),
                'F_drop_pct': float(d['F_drop_pct']),
            }
            meta_out.append(meta_entry)
            continue

        tc = time.time()
        try:
            solver = TELSolver(
                Nr=30, Nz=50,
                P_rf=P_rf, p_mTorr=p_mTorr, frac_Ar=frac_Ar,
                rate_mode='lxcat',
            )
            result = solver.solve(n_iter=80, w=0.12, verbose=False)

            m = solver.mesh
            np.savez_compressed(outpath,
                nF=result['nF'], nSF6=result['nSF6'],
                Te=result['Te'], ne=result['ne'],
                rc=m.rc, zc=m.zc,
                inside=solver.inside, bc_type=solver.bc_type,
                ne_avg=result['ne_avg'],
                F_drop_pct=result['F_drop_pct'],
                P_rf=P_rf, p_mTorr=p_mTorr, frac_Ar=frac_Ar,
            )

            dt = time.time() - tc
            meta_entry = {
                'idx': idx, 'file': fname,
                'P_rf': P_rf, 'p_mTorr': p_mTorr, 'frac_Ar': frac_Ar,
                'ne_avg': float(result['ne_avg']),
                'F_drop_pct': float(result['F_drop_pct']),
                'time_s': dt,
            }
            meta_out.append(meta_entry)

            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - t0
                rate = (idx + 1 - len(existing)) / max(elapsed, 1)
                remaining = (n_total - idx - 1) / max(rate, 0.01)
                print(f"  [{idx+1:3d}/{n_total}] P={P_rf:4d}W p={p_mTorr:2d}mT Ar={frac_Ar:.2f} "
                      f"| ne={result['ne_avg']:.2e} drop={result['F_drop_pct']:.0f}% "
                      f"| {dt:.1f}s | ETA {remaining:.0f}s", flush=True)

        except Exception as exc:
            n_failed += 1
            print(f"  [{idx+1:3d}/{n_total}] FAILED: {exc}", flush=True)
            meta_entry = {
                'idx': idx, 'file': fname,
                'P_rf': P_rf, 'p_mTorr': p_mTorr, 'frac_Ar': frac_Ar,
                'error': str(exc),
            }
            meta_out.append(meta_entry)

    # Write metadata
    with open(os.path.join(DS_OUT, 'metadata.json'), 'w') as f:
        json.dump(meta_out, f, indent=2)

    # Provenance
    provenance = {
        'rate_source': 'lxcat',
        'rate_mode': 'lxcat',
        'lxcat_files': ['data/lxcat/SF6_Biagi_full.txt'],
        'solver': 'TELSolver(rate_mode="lxcat")',
        'mesh': '30x50',
        'n_cases': n_total,
        'n_failed': n_failed,
        'total_time_s': time.time() - t0,
        'note': 'True LXCat-backed solver dataset. Ionization and attachment rates from Biagi cross sections; dissociation/neutral channels on legacy fallback.',
    }
    with open(os.path.join(DS_OUT, 'provenance.json'), 'w') as f:
        json.dump(provenance, f, indent=2)

    total_t = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  DONE: {n_total - n_failed}/{n_total} cases in {total_t:.0f}s", flush=True)
    print(f"  Failed: {n_failed}", flush=True)
    print(f"  Output: {DS_OUT}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
