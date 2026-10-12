"""Short isolated author-training checks; the formal run is started separately."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import underwater_train as author


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-loss', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA must be available'
    torch.backends.cudnn.benchmark = True
    original_dataset = author.Paired_Set

    class SmallSet(original_dataset):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            count = 8 if self.status == 'train' else 4
            for attribute in ('images_paths', 'images_gt_paths', 'images_mask_paths'):
                setattr(self, attribute, getattr(self, attribute)[:count])

    author.Paired_Set = SmallSet
    sys.argv = [sys.argv[0], '--database', 'TIUIED', '--batch_size', '4', '--epochs', '1']
    settings = author.get_args()
    settings.warmup_epochs = 0 if args.feature_loss else 10
    settings.save_best_checkpoint = False
    settings.save_final_checkpoint = False
    settings.save_iter_checkpoint = False
    settings.log_dir = 'training_runs/preflight_' + ('tdp' if args.feature_loss else 'warmup')
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    author.train_model(settings)
    torch.cuda.synchronize()
    result = {
        'scope': 'Isolated short diagnostic; not part of the formal 100-epoch run',
        'feature_loss_enabled': args.feature_loss,
        'batch_size': 4, 'training_batches': 2, 'validation_images': 4,
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'elapsed_seconds': time.monotonic() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'success': True,
    }
    output = Path('training_runs') / ('preflight_tdp.json' if args.feature_loss else 'preflight_warmup.json')
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
