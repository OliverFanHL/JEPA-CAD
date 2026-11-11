"""Reproduce the released full-cohort TMCAD classifier without source CAD."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
TASK = ROOT / 'histcad_large_joint_jepa_v8/downstream_classifier_tmcad_final'
sys.path.insert(0, str(TASK))
from model import ResidualAdapter

def main():
    torch.set_num_threads(1)
    checkpoint=torch.load(TASK/'outputs/default/physical_residual/best.pt',
                          map_location='cpu',weights_only=False)
    model=ResidualAdapter(len(checkpoint['classes']),checkpoint['settings'])
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    assert sum(p.numel() for p in model.parameters())==33418
    data=np.load(TASK/'data/processed/test_frozen_inputs.npz')
    with torch.inference_mode():
        logits,_=model(*[torch.from_numpy(data[k]) for k in ['baseline_logits','geometry','mesh']])
    final=logits.argmax(1).numpy(); baseline=data['baseline_logits'].argmax(1)
    with (TASK/'outputs/default/test_predictions.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len(final)==1404
    classes=checkpoint['classes']
    for i,row in enumerate(rows):
        assert classes[int(data['label'][i])]==row['true_class']
        assert classes[int(baseline[i])]==row['baseline_prediction']
        assert classes[int(final[i])]==row['final_prediction']
    truth=data['label'];corrected=int(((baseline!=truth)&(final==truth)).sum())
    worsened=int(((baseline==truth)&(final!=truth)).sum())
    assert (corrected,worsened)==(55,7)
    print(json.dumps({'status':'passed','reproduced_test_predictions':len(rows),
        'checkpoint_epoch':checkpoint['epoch'],'residual_parameters':33418,
        'baseline_accuracy':float((baseline==truth).mean()),'final_accuracy':float((final==truth).mean()),
        'corrected':corrected,'worsened':worsened},indent=2))

if __name__=='__main__':main()
