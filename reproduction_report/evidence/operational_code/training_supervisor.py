"""Supervise one full author-default training run independently of SSH clients."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'training_runs' / 'from_scratch_20261003'
STATE_OUTSIDE = RUN.with_suffix('.supervisor.json')
LOG = RUN.with_suffix('.train.log')
SUPERVISOR_LOG = RUN.with_suffix('.supervisor.log')
PYTHON = '/root/autodl-tmp/envs/dtiuie-legacy/bin/python'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_state(state):
    state['updated_at_utc'] = utc()
    paths = [STATE_OUTSIDE]
    if (RUN / 'run_metadata.json').exists():
        paths.append(RUN / 'supervisor_state.json')
    for path in paths:
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(path)


def finalize(state, environment):
    state.update(status='finalizing', completed_at_utc=utc(), warnings=[])
    write_state(state)
    with SUPERVISOR_LOG.open('ab', buffering=0) as export_log:
        result = subprocess.run([PYTHON, '-u', '-B', 'cloud_reproduction/finalize_training.py'],
                                cwd=ROOT, env=environment, stdout=export_log,
                                stderr=subprocess.STDOUT)
    if result.returncode == 0:
        archive = Path('/root/autodl-tmp/DTIUIE-training-results-20261003.tar.gz')
        digest = archive.with_name(archive.name + '.sha256').read_text().split()[0]
        state.update(status='complete', result_archive=str(archive), result_archive_sha256=digest)
    else:
        state.update(status='export_failed', export_exit_code=result.returncode)
    write_state(state)
    return result.returncode


def main():
    os.chdir(ROOT)
    environment = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1')
    state = {'status': 'launching', 'supervisor_pid': os.getpid(), 'restarts': 0,
             'started_at_utc': utc(), 'run_dir': str(RUN), 'training_log': str(LOG),
             'target_epochs': 100, 'batch_size': 4, 'algorithm': 'official code defaults'}
    resume = None
    if (RUN / 'run_metadata.json').exists():
        if (RUN / 'completed.json').exists():
            completion = read_json(RUN / 'completed.json')
            if completion.get('epoch') != 100:
                raise RuntimeError('Unexpected completion epoch')
            return finalize(state, environment)
        resume = RUN / 'latest_full.pth'
        if not resume.exists():
            raise RuntimeError('Existing run has no complete checkpoint; refusing to start over')
    while True:
        command = [PYTHON, '-u', '-B', 'cloud_reproduction/underwater_train_monitored.py',
                   '--database', 'TIUIED', '--model_E', 'model_dtiuie', '--model_S', 'vgg16',
                   '--batch_size', '4', '--epochs', '100', '--crop_size', '256',
                   '--warmup_epochs', '10', '--run_dir', str(RUN), '--cudnn_benchmark']
        if resume:
            command += ['--resume_full', str(resume)]
        with LOG.open('ab', buffering=0) as log:
            log.write(('\nSUPERVISOR_START ' + utc() + '\n').encode())
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', child_pid=process.pid, command=command,
                         resume_full=str(resume) if resume else None)
            write_state(state)
            while process.poll() is None:
                heartbeat = read_json(RUN / 'heartbeat.json')
                state['last_observed_heartbeat'] = heartbeat
                warnings = []
                timestamp = heartbeat.get('updated_at_utc')
                if timestamp:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()
                    state['heartbeat_age_seconds'] = age
                    if age > 900:
                        warnings.append('No heartbeat update for more than 15 minutes')
                free = shutil.disk_usage(ROOT).free
                state['disk_free_bytes'] = free
                if free < 2 * 1024**3:
                    warnings.append('Less than 2 GiB disk space remains')
                state['warnings'] = warnings
                state['last_checkpoint_epoch'] = heartbeat.get('last_saved_epoch')
                write_state(state)
                time.sleep(60)
            exit_code = process.returncode
        state['child_exit_code'] = exit_code
        completed = read_json(RUN / 'completed.json')
        if exit_code == 0 and completed.get('epoch') == 100:
            return finalize(state, environment)
        failure = read_json(RUN / 'failed.json')
        state['failure'] = failure
        checkpoint = RUN / 'latest_full.pth'
        # Recover external termination only. Numerical or code errors need investigation.
        if exit_code < 0 and checkpoint.exists() and state['restarts'] < 2:
            state['restarts'] += 1
            state['status'] = 'recovering'
            write_state(state)
            resume = checkpoint
            time.sleep(30)
            continue
        state.update(status='failed', warnings=['Training stopped; inspect failed.json and training log'])
        write_state(state)
        return exit_code or 1


if __name__ == '__main__':
    raise SystemExit(main())
