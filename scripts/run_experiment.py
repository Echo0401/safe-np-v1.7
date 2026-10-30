"""Print an explicit frozen experiment command; training needs --execute."""
from pathlib import Path
import argparse,json,subprocess,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]

def command(profile,output):
    config=json.loads((ROOT/'configs/frozen'/f'{profile}_run_config.json').read_text(encoding='utf-8'))
    enc=config['encoder'];np=config['neural_process'];cal=np['adaptation_strength_calibration'];ft=config['fine_tuning']
    script='src/frozen/v1_7_safe_source_calibrated.py' if profile.endswith('a6') else 'src/ablations/controlled_ablation_v17.py'
    values={
        'data-npz':config['data_npz'],'manifest':config['manifest'],'output-dir':str(output),
        'context-per-class':','.join(map(str,config['context_per_class'])),
        'target-subjects':','.join(map(str,config['target_subjects'])),'seeds':','.join(map(str,config['seeds'])),
        'device':config['device'],'inner-val-fraction':enc['inner_val_fraction'],'batch-size':64,
        'encoder-lr':enc['lr'],'encoder-weight-decay':enc['weight_decay'],'encoder-max-epochs':enc['max_epochs'],'encoder-patience':enc['patience'],
        'np-lr':np['lr'],'np-weight-decay':np['weight_decay'],'np-max-epochs':np['max_epochs'],'np-patience':np['patience'],
        'episodes-per-epoch':np['episodes_per_epoch'],'np-val-episodes':64,'meta-query-per-class':np['meta_query_per_class'],
        'np-latent-dim':np['latent_dim'],'np-hidden-dim':np['hidden_dim'],'np-attention-dim':np['attention_dim'],
        'np-context-aux-weight':np['context_aux_weight'],'np-posterior-weight':np['posterior_predictive_weight'],
        'np-alpha-max':np['alpha_max'],'np-residual-max':np['residual_max_logodds'],
        'np-gate-init-fraction':np['gate_init_fraction'],'np-gate-reg-weight':np['gate_reg_weight'],'np-residual-reg-weight':np['residual_reg_weight'],
        'np-contrast-scale':np['contrast_scale'],'np-residual-init-fraction':np['residual_init_fraction'],'np-beta-kl':np['beta_kl'],
        'mc-samples-eval':np['mc_samples_eval'],'adaptation-gamma-candidates':','.join(map(str,cal['candidates'])),
        'adaptation-calibration-primary':cal['primary'],'adaptation-min-nll-improvement':cal.get('minimum_balanced_nll_improvement_vs_gamma0',0.001),
        'prototype-temperature':config['prototype_temperature'],'finetune-lr':ft['lr'],'finetune-steps':ft['steps'],'finetune-anchor-lambda':ft['anchor_lambda'],
        'max-target-subjects':0,
    }
    argv=[sys.executable,'-B',str(ROOT/script)]
    if not profile.endswith('a6'):argv += ['--ablation-id',profile[-2:].upper()]
    for key,value in values.items():argv += ['--'+key,str(value)]
    argv += ['--drop-overlap' if config['drop_overlap'] else '--no-drop-overlap','--save-checkpoints']
    return argv,config

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=['relax_a6','polar_a6','relax_a1','relax_a2','relax_a3'],required=True)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--execute',action='store_true');args=p.parse_args()
    argv,config=command(args.profile,args.out)
    # JSON is an unambiguous argv representation on Windows and POSIX.
    print(json.dumps({'argv':argv,'execution_requested':bool(args.execute)},indent=2))
    if args.execute:
        if args.out.resolve().exists():raise SystemExit('Output already exists; choose a new directory to preserve prior runs.')
        for key in ['data_npz','manifest']:
            path=ROOT/config[key]
            if not path.is_file():raise SystemExit('Missing restricted input: '+config[key])
            expected=next(x['sha256'] for x in json.loads((ROOT/'provenance/restricted_input_hashes.json').read_text(encoding='utf-8')) if x['logical_path']==config[key])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise SystemExit('Restricted input hash differs from the frozen profile: '+config[key])
        subprocess.run(argv,cwd=ROOT,check=True)
if __name__=='__main__':main()
