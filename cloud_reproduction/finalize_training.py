"""Verify epoch 100, export trained-model test images, and archive the run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid
from datetime import datetime, timezone
from PIL import Image

import torch
from training_observer import json_value

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'training_runs' / 'from_scratch_20261003'
ARCHIVE = Path('/root/autodl-tmp/DTIUIE-training-results-20261003.tar.gz')


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def checkpoint_path(value):
    path = (ROOT / value).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError('Missing or out-of-project checkpoint')
    return path


def verified_export(output, sha, expected_names):
    try:
        manifest = json.loads((output / 'run_manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['completed_count'] == 110
        assert manifest['guidance'] == 'model_SA' and manifest['checkpoint_sha256'] == sha
        assert {p.name for p in (output / 'jpeg').glob('*.jpg')} == expected_names
        assert {p.name for p in (output / 'png').glob('*.png')} == {Path(n).stem + '.png' for n in expected_names}
        for folder in ('jpeg', 'png'):
            for path in (output / folder).iterdir():
                with Image.open(path) as image:
                    image.load()
                    assert image.mode == 'RGB' and image.size == (256, 256)
        return True
    except (OSError, ValueError, KeyError, AssertionError):
        return False


def main():
    completion = json.loads((RUN / 'completed.json').read_text())
    metadata = json.loads((RUN / 'run_metadata.json').read_text())
    assert completion['epoch'] == 100 and metadata['completed_epoch'] == 100
    settings = metadata['args']
    checkpoints = {
        'final': checkpoint_path(settings['final_model_file']),
        'best_ssim': checkpoint_path(settings['best_model_file'] + '_ssim.pth'),
        'best_miou': checkpoint_path(settings['best_model_file'] + '_miou.pth'),
    }
    expected_names = {p.name for p in (ROOT / 'dataset/test/images').glob('*.jpg')}
    assert len(expected_names) == 110
    summary = {
        'scope': 'Author-default 100-epoch training; internal validation differs from paper downstream retraining tables',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_epochs': 100, 'batch_size': 4, 'pretrained_dtiuie_loaded': False,
        'vgg_initialization': 'Official ImageNet VGG16-BN weights',
        'training_metadata': metadata, 'exports': {},
    }
    for label, path in checkpoints.items():
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
        epoch = int(checkpoint['epoch'])
        for key in ('model_E', 'model_S', 'model_SA'):
            assert key in checkpoint and all(torch.isfinite(value).all().item()
                for value in checkpoint[key].values() if value.is_floating_point())
        if label == 'final':
            assert epoch == 100
        del checkpoint
        sha = digest(path)
        output = ROOT / 'outputs' / ('trained_' + label + '_model_SA_20261003')
        if not verified_export(output, sha, expected_names):
            if output.exists():
                output.rename(output.with_name(output.name + '.previous-' + uuid.uuid4().hex[:12]))
            subprocess.run([
                sys.executable, '-u', '-B', str(ROOT / 'cloud_reproduction/infer_trained_checkpoint.py'),
                '--guidance', 'model_SA', '--device', 'cuda:0', '--checkpoint', str(path),
                '--trusted-checkpoint-sha256', sha, '--output-dir', str(output),
            ], cwd=ROOT, check=True)
        assert verified_export(output, sha, expected_names)
        summary['exports'][label] = {'checkpoint': str(path.relative_to(ROOT)), 'sha256': sha,
                                    'epoch': epoch, 'output': str(output.relative_to(ROOT)),
                                    'guidance': 'model_SA', 'completed_count': 110}
    full = torch.load(str(RUN / 'latest_full.pth'), map_location='cpu', weights_only=False)
    assert full['epoch'] == 100 and full['global_step'] == 38100
    summary['best_internal_validation'] = full['best_criterions']
    del full
    (RUN / 'training_result_summary.json').write_text(json.dumps(json_value(summary), indent=2))
    paths = [RUN, RUN.with_suffix('.train.log'), RUN.with_suffix('.supervisor.log'),
             Path(settings['log_format_str']), *checkpoints.values()]
    paths = [path if path.is_absolute() else ROOT / path for path in paths]
    paths.extend(ROOT / details['output'] for details in summary['exports'].values())
    paths.extend(ROOT / 'cloud_reproduction' / name for name in (
        'underwater_train_monitored.py', 'training_observer.py', 'build_monitored_train.py',
        'PATCH_NOTES.md', 'training_supervisor.py', 'finalize_training.py',
        'infer_trained_checkpoint.py', 'data_audit.json'))
    all_files = set()
    for path in paths:
        if path.is_dir():
            all_files.update(item for item in path.rglob('*') if item.is_file())
        elif path.is_file():
            all_files.add(path)
    partial = ARCHIVE.with_name(ARCHIVE.name + '.part')
    with tarfile.open(partial, 'w:gz', compresslevel=3) as bundle:
        for path in sorted(all_files):
            bundle.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    partial.replace(ARCHIVE)
    sha = digest(ARCHIVE)
    ARCHIVE.with_name(ARCHIVE.name + '.sha256').write_text(sha + '  ' + ARCHIVE.name + '\n')
    print(json.dumps({'archive': str(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
                      'sha256': sha, 'training_epochs': 100, 'exported_images': 330}, indent=2), flush=True)


if __name__ == '__main__':
    main()
