"""Static publication scan of tracked candidate files; excludes local restricted inputs/runs."""
from pathlib import Path
import re,csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.idea','.venv','__pycache__','runs','restricted','private_review_do_not_publish'}
BANNED={'.npz','.npy','.pt','.pth','.ckpt','.parquet','.pkl','.pickle','.zip','.7z','.docx','.pdf','.xlsx','.sqlite','.jsonl'}
PATTERNS={
 'absolute_windows_path':re.compile(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]'),
 'home_directory_path':re.compile(r'/(?:Users|home)/[^/\s]+/'),
 'email_address':re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
 'credential_token':re.compile(r'\b(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{24,})\b'),
 'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}
def scan():
 findings=[];count=0;total=0;largest=0
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or any(part in SKIP for part in p.relative_to(ROOT).parts):continue
  count+=1;size=p.stat().st_size;total+=size;largest=max(largest,size);rel=p.relative_to(ROOT).as_posix()
  if size>50*1024*1024:findings.append({'file':rel,'kind':'large_file_over_50MiB'})
  if p.suffix.lower() in BANNED:findings.append({'file':rel,'kind':'excluded_data_or_private_artifact_type'})
  try:text=p.read_text(encoding='utf-8-sig')
  except UnicodeDecodeError:findings.append({'file':rel,'kind':'unreviewed_binary'});continue
  # This scanner's literal patterns are configuration, not leaked source records.
  if rel!='scripts/scan_release.py':
   for kind,pat in PATTERNS.items():
    for m in pat.finditer(text):findings.append({'file':rel,'line':text[:m.start()].count('\n')+1,'kind':kind})
  if p.suffix.lower()=='.csv':
   header=next(csv.reader(text.splitlines()),[])
   risky={'query_anchor_utc','anchor_utc','ifb_submission_utc','p_stress','target_subject','participant_name','phone','email'} & set(header)
   if risky:findings.append({'file':rel,'kind':'individual_level_csv_columns','columns':sorted(risky)})
 return {'status':'passed' if not findings else 'review_required','files':count,'bytes':total,'largest_file_bytes':largest,'findings':findings,'scope':'static content checks; not proof of ownership or anonymity'}
if __name__=='__main__':
 result=scan();print(json.dumps(result,indent=2));sys.exit(0 if result['status']=='passed' else 1)
