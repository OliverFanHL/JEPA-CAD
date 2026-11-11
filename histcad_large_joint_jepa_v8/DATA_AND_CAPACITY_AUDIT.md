# Full HistCAD input and capacity audit

## What v6 actually encoded

The v6 `1280-D` parametric and `1024-D` STEP representations were not raw vectors padded to fixed
lengths. Graphs were concatenated during batching, so no node-sequence padding was present. The branch
vectors were learned attention summaries:

- parametric: four learned 256-D summaries plus one 256-D global/history token;
- STEP: three learned 256-D summaries plus one 256-D global/type token.

All branch coordinates were learned. The sparse area was the type-specific 24-field parametric numeric
schema: a node used 9.27% of its numeric fields on average, and the accompanying mask explicitly told the
network which fields were present. That is schema sparsity, not unused padding in the 1280-D branch.

An exact full-corpus coordinate audit of the trained v6 representations found zero constant-zero columns
and zero columns with standard deviation below `1e-3` in the 1280-D parametric, 1024-D STEP, or 1024-D
joint matrices. Thus the old branch widths were active learned coordinates, although activity alone does
not prove that every coordinate carried independent information.

## Measurements on all 162,131 retained samples

| Quantity | min | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Parametric nodes | 3 | 14 | 16 | 32 | 58 | 89 | 306 | 5,531 |
| Parametric edges | 9 | 88 | 108 | 218 | 407 | 631 | 2,064 | 32,901 |
| Feature operations | 1 | 1 | 1 | 2 | 3 | 3 | 6 | 100 |
| Sketch primitives | 1 | 4 | 5 | 10 | 20 | 32 | 119 | 2,348 |
| Constraints | 0 | 7 | 10 | 20 | 34 | 52 | 155 | 2,351 |
| STEP entities before selection | 36 | 189 | 234 | 399 | 718 | 1,146 | 4,207 | 77,957 |
| B-Rep candidate instances | 3 | 85 | 107 | 195 | 373 | 599 | 2,216 | 42,693 |

The cached v6 STEP input had a median of 155 nodes: typically 46 untruncated entity-type nodes, 107
selected B-Rep instance nodes, and one global node. Its 12 STEP numeric fields were nonzero 74.91% of
the time. Every graph retained the complete entity-type vocabulary present in the file and all type-level
reference counts, while instance-level nodes were capped.

The old 128-instance cap truncated 71,625 parts (44.18%). The new default cap of 256 truncates 27,970
parts (17.25%) and fully retains 82.75%. A cap of 512 would reduce truncation to 6.20%, but substantially
increases cache size, graph-message traffic, and full-data training cost. The cap remains configurable.

## Why the new dimensions are justified

Adding more pooling slots would be poorly matched to the median 16-node parametric graph and could create
redundant summaries. Instead, the design keeps four semantically defined parametric summaries and widens
the relational node state from 256 to 512. This gives each operation, primitive, constraint, and relation
more learned channels while preserving the heavy tail of large histories without truncation. The result is
`5 x 512 = 2560-D`, with every coordinate learned.

STEP graphs are much larger and 44.18% previously reached the instance cap. The new version both raises the
cap to 256 and widens STEP relational states to 512. Three justified summaries—entity types, B-Rep instances,
and connectivity—plus the global token produce `4 x 512 = 2048-D`. A fifth STEP token was not added because
the current parsed schema does not expose a fifth clean, non-overlapping role.

The normalized branch concatenation is therefore 4608-D of learned source features. The joint network maps
`4608 -> 4096 -> 4096`; it does not create the final size by zero-padding, coordinate repetition, or a large
expansion from a small source vector.

## Remaining limitation

The 256-instance default is a deliberate compute/coverage tradeoff, not a claim that larger STEP files have
only 256 useful entities. Typed aggregate nodes and type-reference connectivity still cover the complete STEP
file, but instance-reference detail beyond the selected 256 nodes is summarized rather than retained
individually. Increase `preprocessing.max_step_instances` if hardware permits.
