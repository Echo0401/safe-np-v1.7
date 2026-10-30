"""Check each recovered WESAD argv against the actual archived parser; no main loop."""
from pathlib import Path
import importlib.util,sys,argparse,json
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
runner=load('wesad_runner','scripts/run_wesad_reference.py');module=load('wesad_reference','src/reference/cnn_prototype_loso_v2.py')
for seed in [2026,2036,2046,2056,2066]:
 argv,cfg=runner.command(seed,Path('runs')/f'wesad_cli_check_{seed}')
 saved=sys.argv
 try:sys.argv=[argv[2]]+argv[3:];args=module.build_parser().parse_args()
 finally:sys.argv=saved
 assert args.seed==seed and tuple(args.k_values)==(1,3,5) and args.max_epochs==80
 assert not args.no_balance_nonstress and not args.allow_single_class_test and not args.non_deterministic
print(json.dumps({'status':'passed','wesad_cli_profiles':5,'training_steps':0}))
