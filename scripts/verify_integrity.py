"""Verify the exact reviewed candidate file set and SHA-256 content hashes."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
MANIFEST='provenance/FILE_SHA256.json'
SKIP={'.git','.idea','.venv','venv','__pycache__','runs','restricted'}
def main():
 expected=json.loads((ROOT/MANIFEST).read_text(encoding='utf-8'));findings=[]
 actual={p.relative_to(ROOT).as_posix():p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts) and p.relative_to(ROOT).as_posix()!=MANIFEST}
 for rel,info in expected.items():
  if rel not in actual:findings.append({'file':rel,'status':'missing'});continue
  if hashlib.sha256(actual[rel].read_bytes()).hexdigest()!=info['sha256']:findings.append({'file':rel,'status':'changed'})
 for rel in sorted(set(actual)-set(expected)):findings.append({'file':rel,'status':'unexpected'})
 print(json.dumps({'status':'passed' if not findings else 'failed','verified_files':len(expected),'findings':findings},indent=2))
 return 1 if findings else 0
if __name__=='__main__':raise SystemExit(main())
