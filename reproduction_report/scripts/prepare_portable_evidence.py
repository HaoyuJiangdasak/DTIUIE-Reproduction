"""Copy hash-verified small source evidence and selected real image samples."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
EXTRACTED=PROJECT/'cloud_training_results/results-20261003-ebcddb62d624'
EVIDENCE=ROOT/'evidence'


def main():
    provenance=json.loads((ROOT/'data/provenance.json').read_text(encoding='utf-8'))
    manifest=[]
    for item in provenance['read_only_source_sha256']:
        source=Path(item['path'])
        if source.stat().st_size>50*1024*1024:
            raise ValueError('Unexpected large source file; weights should remain in original results')
        if any(part in {'credentials','.ssh'} for part in source.parts):
            raise ValueError('Credentials are not report evidence')
        parent='tensorboard' if source.name.startswith('events.out.tfevents') else ('monitor' if source.name.startswith('history-') else 'run')
        target=EVIDENCE/parent/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            actual=hashlib.sha256(target.read_bytes()).hexdigest()
            if actual!=item['sha256']:raise ValueError('Evidence filename collision')
        else:
            shutil.copyfile(source,target)
            actual=hashlib.sha256(target.read_bytes()).hexdigest()
            assert actual==item['sha256']
        manifest.append({'original_path':str(source),'portable_path':target.relative_to(ROOT).as_posix(),
                         'sha256':actual,'size_bytes':source.stat().st_size})
    names=['d_r_122_.jpg','d_r_129_.jpg','d_r_144_.jpg','d_r_166_.jpg']
    for label,folder in [('raw',PROJECT/'dataset/test/images'),('reference',PROJECT/'dataset/test/reference'),
                        ('best_ssim',EXTRACTED/'outputs/trained_best_ssim_model_SA_20261003/png'),
                        ('final',EXTRACTED/'outputs/trained_final_model_SA_20261003/png')]:
        target=EVIDENCE/'samples'/label;target.mkdir(parents=True,exist_ok=True)
        for name in names:
            filename=name if label in {'raw','reference'} else Path(name).stem+'.png'
            shutil.copyfile(folder/filename,target/filename)
    code=EVIDENCE/'official_code';code.mkdir(parents=True,exist_ok=True)
    for name in ['underwater_train.py','uie_eval.py','seg_eval.py','dataloader_seg.py','loss.py','environment.yaml','README.md']:
        shutil.copyfile(PROJECT/name,code/name)
    for folder in ['models','utils']:
        for source in (PROJECT/folder).rglob('*.py'):
            if '__pycache__' in source.parts:continue
            target=code/source.relative_to(PROJECT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    (EVIDENCE/'source_map.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Portable source files verified:',len(manifest))


if __name__=='__main__':
    main()
