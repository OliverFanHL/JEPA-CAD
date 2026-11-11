"""Read-only verification of source/results, final matrix, and classifier metrics."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent
V8=ROOT/'histcad_large_joint_jepa_v8'
BENCH=V8/'benchmarking/outputs'

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def equal(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10), (a,b)
def follow(value,keys):
    for key in keys.split('/'):value=value[key]
    return value

def main():
    manifest=load(ROOT/'FILE_MANIFEST.json')
    assert not manifest['source_dataset_included']
    for item in manifest['files']:
        p=ROOT/item['path']
        assert p.stat().st_size==item['bytes'],item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
        assert p.suffix.lower() not in {'.stp','.step','.stl','.obj','.ply','.iges','.igs'}
        assert 'jev_tmcad' not in item['path']
    inventory=load(BENCH/'inventory.json')
    assert inventory['ready']==inventory['active_required']==12
    matrix=rows(ROOT/'FINAL_EXPERIMENT_MATRIX.csv')
    assert len(matrix)==38 and len({(r['stage'],r['representation']) for r in matrix})==38
    checks={
      'frozen_representation':{'dimensions':'dimensions','effective_rank':'latent/effective_rank_entropy',
        'knn_test_macro_f1':'knn/test/macro_f1','linear_test_macro_f1':'linear_probe/test/macro_f1',
        'adapter_test_macro_f1':'adapter_probe/test/macro_f1'},
      'tmcad_classification':{'frozen_probe_test_macro_f1':'frozen_linear_probe/test/macro_f1',
        'adapter_test_macro_f1':'controlled_adapter/test/macro_f1',
        'adapter_validation_macro_f1':'controlled_adapter/validation/macro_f1',
        'adapter_trainable_parameters':'controlled_adapter/trainable_parameters'},
      'retrieval_adaptation':{'direct_dimension_mae_at_1':'direct/dimension_retrieval/mae_at_1',
        'adapted_dimension_mae_at_1':'adapted/dimension_retrieval/mae_at_1',
        'direct_dimension_recall_at_10':'direct/dimension_neighbor_retrieval/recall_at_10',
        'adapted_dimension_recall_at_10':'adapted/dimension_neighbor_retrieval/recall_at_10',
        'adapter_parameters':'training/trainable_parameters'},
      'semantic_retrieval':{'direct_semantic_hit_at_1':'direct/function_at_1_eligible',
        'adapted_semantic_hit_at_1':'semantic_adapter/function_at_1_eligible',
        'direct_dimension_recall_at_10':'direct/dimension_recall_at_10',
        'adapted_dimension_recall_at_10':'semantic_adapter/dimension_recall_at_10',
        'adapter_parameters':'training/trainable_parameters'}}
    checked=0
    for stage,mapping in checks.items():
        data=load(BENCH/stage/'results.json')['representations']; summary=rows(BENCH/stage/'summary.csv')
        assert {r['representation'] for r in summary}==set(data)
        assert {r['representation'] for r in matrix if r['stage']==stage}==set(data)
        for row in summary:
            for col,keys in mapping.items():equal(row[col],follow(data[row['representation']],keys));checked+=1
    efficiency=load(BENCH/'efficiency/results.json')['methods']
    for row in rows(BENCH/'efficiency/summary.csv'):
        saved=efficiency[row['method']]
        for k in ['training_seconds','epochs','parameters','embedding_seconds_per_sample']:
            if row[k] and saved[k] is not None:equal(row[k],saved[k]);checked+=1
    task=V8/'downstream_classifier_tmcad_final/outputs/default'
    data=rows(task/'test_predictions.csv');assert len(data)==len({r['uid'] for r in data})==1404
    evaluation=load(task/'evaluation.json')['test']
    classes=sorted({r['true_class'] for r in data})
    for column,key in [('baseline_prediction','frozen_baseline'),('final_prediction','physical_residual')]:
        counts=collections.Counter()
        for row in data:
            y,pred=row['true_class'],row[column];counts['true',y]+=1;counts['pred',pred]+=1
            if y==pred:counts['tp',y]+=1
        equal(sum(counts['tp',c] for c in classes)/len(data),evaluation[key]['accuracy'])
        equal(sum(2*counts['tp',c]/(counts['true',c]+counts['pred',c]) for c in classes)/len(classes),evaluation[key]['macro_f1'])
        equal(sum(counts['tp',c]/counts['true',c] for c in classes)/len(classes),evaluation[key]['balanced_accuracy'])
    corrected=sum(r['baseline_prediction']!=r['true_class'] and r['final_prediction']==r['true_class'] for r in data)
    worsened=sum(r['baseline_prediction']==r['true_class'] and r['final_prediction']!=r['true_class'] for r in data)
    assert (corrected,worsened)==(55,7)
    figures=[r for r in manifest['files'] if r['category']=='manuscript_figure'];assert len(figures)==39
    print(json.dumps({'status':'passed','hashed_files':len(manifest['files']),
      'active_representations':12,'matrix_entries':len(matrix),'summary_metric_cells_verified':checked,
      'full_cohort_test_predictions':len(data),'corrected':corrected,'worsened':worsened,
      'manuscript_figures':len(figures),'source_datasets_included':False},indent=2))

if __name__=='__main__':main()
