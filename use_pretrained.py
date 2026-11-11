"""Check or extract the existing pretrained comparison arm; never train/preprocess."""
from pathlib import Path
import argparse
import json
import sys
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT/'histcad_large_joint_jepa_v8'
sys.path.insert(0,str(PROJECT/'src'))
from histcad_large_jepa.data import PairedGraphDataset, collate_paired
from histcad_large_jepa.model import WideJointJEPA
from histcad_large_jepa.train import RunLogger, extract_representations

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract',action='store_true',help='Run frozen encoder inference on copied graphs; no training or preprocessing')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('--batch-size must be positive')
    cache_path=PROJECT/'data/processed/full_paired_graphs_v7.pt'
    checkpoint_path=PROJECT/'outputs/full_dataset_4096/best.pt'
    for path in [cache_path,checkpoint_path]:
        if not path.is_file():raise FileNotFoundError(f'Existing artifact required; preprocessing/training is disabled: {path}')
    cache=torch.load(cache_path,map_location='cpu',weights_only=False,mmap=True)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
    if cache.get('format_version')!=7 or checkpoint.get('format_version')!=7:
        raise ValueError('Both artifacts must use format version 7')
    if tuple(cache['step_entity_types'])!=tuple(checkpoint['step_entity_types']):
        raise ValueError('STEP entity vocabulary differs between cache and checkpoint')
    model_config=checkpoint['model_config']
    if model_config['embedding_dim']!=4096:raise ValueError('Expected the 4096-D pretrained arm')
    if len(checkpoint['dimension_target_mean'])!=len(cache['dimension_target_names']):
        raise ValueError('Dimension target schema differs')
    report={'arm':'jepa_joint','mode':'existing_pretrained_frozen',
        'cache_path':str(cache_path),'checkpoint_path':str(checkpoint_path),
        'checkpoint_epoch':checkpoint['epoch'],'cache_parts':len(cache['uids']),
        'split_sizes':cache['split_sizes'],'embedding_dim':model_config['embedding_dim'],
        'preprocessing_run':False,'training_run':False,'optimizer_created':False}
    print(json.dumps(report,indent=2),flush=True)
    if not args.extract:return
    output=checkpoint_path.parent/'representations.pt'
    if output.exists():raise FileExistsError(f'Preserving existing representations: {output}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
    model=WideJointJEPA(len(cache['step_entity_types']),model_config,len(cache['dimension_target_names'])).to(device)
    model.load_state_dict(checkpoint['model']);model.eval();model.requires_grad_(False)
    datasets={split:PairedGraphDataset(cache,split,checkpoint['dimension_target_mean'],checkpoint['dimension_target_std'])
              for split in ['train','val','test']}
    loaders={split:DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0,
                             collate_fn=collate_paired) for split,dataset in datasets.items()}
    extract_representations(model,loaders,cache,device,checkpoint['dimension_target_mean'],
        checkpoint['dimension_target_std'],output,RunLogger(output.parent/'pretrained_inference.log',append=True),
        'float16',checkpoint_path,checkpoint['epoch'],model_config)
    print(json.dumps({'arm':'jepa_joint','representations':str(output),
                      'preprocessing_run':False,'training_run':False}),flush=True)

if __name__=='__main__':main()
