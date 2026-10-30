"""Recompute participant-level statistics from authorized saved predictions, without training."""
from pathlib import Path
import argparse,importlib.util,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['relax','polar'],required=True);p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.exists():raise SystemExit('Choose a new output directory; existing results will not be overwritten.')
 for name in ['matched_query_results.csv','np_safety_audit.csv','fold_training_summary.csv']:
  if not (a.source/name).is_file():raise SystemExit('Missing restricted file: '+name)
 spec=importlib.util.spec_from_file_location('participant_statistics',ROOT/'src/analysis/participant_statistics.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 m.SOURCE=a.source;m.OUT=a.out;m.OUT.mkdir(parents=True)
 if a.dataset=='relax':
  m.EXPECTED_PARTICIPANTS=[13,16,18,22,25,28,38,43,45,46,48,49,56]
  m.ANALYSIS_STATUS='posthoc_exploratory_on_viewed_relax_cohort';m.RUN_LABEL='RELAX v1.7 safe fine-grid';m.OUTPUT_PREFIX='relax_v17'
  m.CONFIRMATORY_WARNING='The fine gamma grid and minimum-NLL rule were introduced after viewing earlier target results; this is post-hoc/exploratory.'
 m.main()
if __name__=='__main__':main()
