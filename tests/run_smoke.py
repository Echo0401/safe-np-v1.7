"""Data-free contract checks of the actual archived code; no optimization/training."""
from pathlib import Path
import importlib.util,sys,unittest,json,subprocess,ast
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 for p in ROOT.rglob('*.py'):
  if '__pycache__' not in p.parts:compile(p.read_text(encoding='utf-8-sig'),p.name,'exec')
 gamma=load('safe_gamma_tests','tests/test_safe_gamma.py')
 gamma.test_gamma_zero_is_exact_global();gamma.test_small_improvement_falls_back();gamma.test_sufficient_improvement_activates()
 ab=load('ablation_contract_tests','tests/test_ablation_contracts.py');ab.run_checks()
 times=load('time_contract_tests','tests/test_label_availability.py')
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(times))
 if not result.wasSuccessful():raise SystemExit(1)
 runner=load('experiment_runner','scripts/run_experiment.py')
 # Parse every generated command using each actual archived CLI, without main().
 for profile in ['relax_a6','polar_a6','relax_a1','relax_a2','relax_a3']:
  argv,config=runner.command(profile,Path('runs')/('dry_'+profile))
  script=Path(argv[2]);m=load('parse_'+profile,str(script.relative_to(ROOT)))
  saved=sys.argv
  try:sys.argv=[str(script)]+argv[3:];args=m.parse_args()
  finally:sys.argv=saved
  assert args.context_per_class=='1,2,3' and args.seeds=='2026,2027,2028,2029,2030'
  assert args.encoder_max_epochs==30 and args.np_max_epochs==40
  assert args.encoder_patience==8 and args.np_patience==10
  assert args.target_subjects==','.join(map(str,config['target_subjects']))
 print(json.dumps({'status':'passed','safe_gamma_tests':3,'time_contract_tests':result.testsRun,'ablation_structural_suite':1,'frozen_cli_profiles':5,'training_steps':0}))
if __name__=='__main__':main()
