"""Validate packaged aggregate evidence, without individual-level data or ML dependencies."""
from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parents[1]
def main():
 with (ROOT/'results/reference/ablation_query_key_audit.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 assert {r['ablation_id'] for r in rows}=={'A'+str(i) for i in range(7)}
 for r in rows:
  assert int(r['n_rows'])==2565 and int(r['n_seeds'])==5 and int(r['n_participants'])==13
  assert int(r['n_duplicate_keys'])==0 and r['contexts']=='2;4;6' and r['exact_match_to_A6']=='True'
 assert len({r['key_sha256'] for r in rows})==1
 manifest=json.loads((ROOT/'results/reference/ablation_analysis_manifest.json').read_text(encoding='utf-8'))
 assert manifest['expected_formal_design']['queries_per_seed_context']==171
 assert manifest['bootstrap_replicates']==10000
 print(json.dumps({'status':'passed','variants':7,'records_per_variant':2565,'contexts':[2,4,6],'scope':'saved aggregate design evidence only'}))
if __name__=='__main__':main()
