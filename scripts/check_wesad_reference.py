"""Recompute the five-run reference mean from bundled group summaries."""
from pathlib import Path
import csv,json,statistics
ROOT=Path(__file__).resolve().parents[1]
def main():
 selected=[]
 for seed in [2026,2036,2046,2056,2066]:
  with (ROOT/f'results/reference/wesad_seed{seed}_group_summary.csv').open(encoding='utf-8-sig',newline='') as f:
   rows=[r for r in csv.DictReader(f) if r['method']=='Global FUSION Encoder' and int(r['k_shot_per_class'])==0]
  assert len(rows)==1 and int(rows[0]['n_subjects'])==15;selected.append(rows[0])
 values={m:statistics.mean(float(r[m+'_mean']) for r in selected) for m in ['balanced_accuracy','macro_f1','auroc']}
 assert [round(values[m],3) for m in ['balanced_accuracy','macro_f1','auroc']]==[0.834,0.809,0.918]
 print(json.dumps({'status':'passed','runs':5,'participants_per_run':15,'means':values,'scope':'saved group means, not new training'}))
if __name__=='__main__':main()
