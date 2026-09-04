"""Sequential, resumable run queue for the full GLips retraining sweep; skips any run whose final_model.pth already exists.

    python run_all.py            # run/continue the whole queue
    python run_all.py --force    # ignore final_model.pth, re-run everything
    python run_all.py --only train_15,train_15_transfer   # substring filter

Per-run stdout/stderr go to logs/<script>.log; a one-line status per run is
appended to logs/run_all.log.
"""
import os
import sys
import time
import argparse
import subprocess

LIP = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(LIP, 'logs')

# (script, save_dir) in execution order. save_dir/final_model.pth => already done.
QUEUE = [
    ('Transformer_based/train.py',              'Transformer_based/checkpoints_500'),
    ('mstcn_baseline/train.py',                 'mstcn_baseline/checkpoints_500'),
    ('Transformer_based/train_15.py',           'Transformer_based/checkpoints_15'),
    ('mstcn_baseline/train_15.py',              'mstcn_baseline/checkpoints_15'),
    ('Transformer_based/train_15_ameer.py',     'Transformer_based/checkpoints_15_ameer'),
    ('mstcn_baseline/train_15_ameer.py',        'mstcn_baseline/checkpoints_15_ameer'),
    ('Transformer_based/train_15_transfer.py',  'Transformer_based/checkpoints_15_transfer'),
    ('mstcn_baseline/train_15_transfer.py',     'mstcn_baseline/checkpoints_15_transfer'),
    ('Transformer_based/train_15_meanpool.py',  'Transformer_based/checkpoints_15_meanpool'),
    ('Transformer_based/train_15_nostem.py',    'Transformer_based/checkpoints_15_nostem'),
    ('Transformer_based/train_15_noroi.py',     'Transformer_based/checkpoints_15_noroi'),
]


def _log_status(msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, 'run_all.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='re-run even if final_model.pth exists')
    ap.add_argument('--only', default='', help='comma-separated substrings; run only matching scripts')
    args = ap.parse_args()
    only = [s for s in args.only.split(',') if s.strip()]

    os.makedirs(LOG_DIR, exist_ok=True)
    _log_status(f"=== run_all start ({len(QUEUE)} jobs) ===")
    for script, save_dir in QUEUE:
        if only and not any(s in script for s in only):
            continue
        done_marker = os.path.join(LIP, save_dir, 'final_model.pth')
        if os.path.exists(done_marker) and not args.force:
            _log_status(f"SKIP (done): {script}")
            continue

        log_path = os.path.join(LOG_DIR, os.path.basename(script).replace('.py', '') + '.log')
        _log_status(f"START: {script}  (log -> {os.path.relpath(log_path, LIP)})")
        t0 = time.time()
        with open(log_path, 'a', encoding='utf-8', buffering=1) as lf:
            lf.write(f"\n===== run {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            proc = subprocess.run([sys.executable, script], cwd=LIP,
                                  stdout=lf, stderr=subprocess.STDOUT,
                                  env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        dt = (time.time() - t0) / 60.0
        if proc.returncode == 0:
            _log_status(f"DONE ({dt:.1f} min): {script}")
        else:
            _log_status(f"FAIL (rc={proc.returncode}, {dt:.1f} min): {script} — see log; continuing")
    _log_status("=== run_all finished ===")


if __name__ == '__main__':
    main()
