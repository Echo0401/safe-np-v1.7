"""Print a recovered WESAD CNN/Prototype reference command, without running training."""
from pathlib import Path
import argparse,json,sys,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
def command(seed,out):
 c=json.loads((ROOT/f'configs/frozen/wesad_reference_seed{seed}.json').read_text(encoding='utf-8'))
 mapping={'data-npz':'data_npz','manifest':'manifest_csv','input-mode':'input_mode','embedding-dim':'embedding_dim','repeats':'repeats','seed':'random_seed','calibration-fraction':'calibration_fraction','gap-rr':'gap_rr','prototype-metric':'prototype_metric','prototype-temperature':'prototype_temperature','batch-size':'batch_size','max-epochs':'max_epochs','patience':'patience','min-epochs':'min_epochs','learning-rate':'learning_rate','weight-decay':'weight_decay','dropout':'dropout','n-val-subjects':'n_val_subjects','num-workers':'num_workers','minimum-retrain-epochs':'minimum_retrain_epochs','device':'device_used'}
 argv=[sys.executable,'-B',str(ROOT/'src/reference/cnn_prototype_loso_v2.py'),'--output-dir',str(out),'--k-values',','.join(map(str,c['k_values']))]
 for flag,key in mapping.items():argv+=['--'+flag,str(c[key])]
 for flag,key in [('no-balance-nonstress','balance_nonstress_states'),('allow-single-class-test','require_both_test_classes'),('non-deterministic','deterministic')]:
  if not c[key]:argv+=['--'+flag]
 return argv,c
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[2026,2036,2046,2056,2066],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();argv,c=command(a.seed,a.out);print(json.dumps({'argv':argv,'execution_requested':a.execute},indent=2))
 if a.execute:
  if a.out.exists():raise SystemExit('Choose a fresh output directory.')
  for key in ['data_npz','manifest_csv']:
   path=ROOT/c[key]
   if not path.is_file():raise SystemExit('Missing restricted input: '+c[key])
   expected=next(x['sha256'] for x in json.loads((ROOT/'provenance/restricted_input_hashes.json').read_text(encoding='utf-8')) if x['logical_path']==c[key])
   if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise SystemExit('Restricted input hash differs from the frozen reference: '+c[key])
  subprocess.run(argv,cwd=ROOT,check=True)
