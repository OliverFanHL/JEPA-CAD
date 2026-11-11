# Source datasets

Download datasets from their original sources. No source dataset is redistributed
in this repository.

| Dataset | Source | Used for |
| --- | --- | --- |
| HistCAD | [Authors' dataset repository](https://huggingface.co/datasets/DongXintong/HistCAD), [paper](https://arxiv.org/abs/2602.19171) | Paired parametric-history/STEP pretraining; dimensional, history, and functional retrieval |
| TMCAD | [Authors' dataset documentation](https://github.com/Qiang-Zou/BRT/blob/main/DATASET.md) | STEP-only classification transfer |

Use the academic HistCAD collection compatible with the saved audit, not the
subsequently expanded corpus without checking identities. The local full study
retained 162,131 valid pairs (145,915 training / 8,130 validation / 8,086 test).
The BRepNet-shared benchmark retained 162,064 pairs. TMCAD's prepared split has
9,312 parts (6,522 / 1,386 / 1,404); shared external-method coverage retains
7,252 (5,085 / 1,077 / 1,090).

The checked-in split manifests, coverage exclusions, source paths, and hashes
support alignment; they do not contain source CAD geometries. Historical absolute
paths in those records identify original files and must be remapped to your local
download when regenerating inputs. Upstream dataset versions can change; match
sample identities and coverage before comparing metrics.
