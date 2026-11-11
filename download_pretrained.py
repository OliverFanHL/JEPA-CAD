"""Download the pinned pretrained comparison artifacts and verify SHA-256 hashes."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parent

def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir',type=Path,default=ROOT,
                        help='JEPA-CAD checkout directory; defaults to this script directory')
    parser.add_argument('--verify-only',action='store_true',help='Check existing downloads without network access')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'REMOTE_PRETRAINED_ARTIFACTS.json').read_text())
    destination=args.local_dir.expanduser().resolve()
    if not args.verify_only:
        try:from huggingface_hub import hf_hub_download
        except ImportError:raise SystemExit('Install the download client: python -m pip install --upgrade huggingface_hub')
    for record in manifest['files']:
        path=destination/record['path']
        if not args.verify_only:
            print('Downloading '+record['path'],flush=True)
            hf_hub_download(repo_id=manifest['repo_id'],repo_type=manifest['repo_type'],
                filename=record['path'],revision=manifest['revision'],local_dir=destination)
        if not path.is_file() or path.stat().st_size!=record['bytes']:
            raise SystemExit('Missing file or wrong size: '+str(path))
        print('Verifying '+record['path'],flush=True)
        if file_sha256(path)!=record['sha256']:
            raise SystemExit('SHA-256 mismatch: '+str(path))
        print('Verified: '+str(path),flush=True)
    print('Both pretrained artifacts are ready. No preprocessing or training was run.',flush=True)

if __name__=='__main__':main()
