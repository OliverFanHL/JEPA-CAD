# Manual review of weak-class validation failures

These are **validation** examples selected from the eight lowest-recall leaf
classes, not test examples used for model design. The
[`validation_failure_sheet.png`](outputs/attention_regularized_v2/validation_failure_sheet.png)
contains OpenCASCADE renders of the corresponding STEP solids. All eight STEP
imports and renders succeeded. Observations below distinguish an obviously
wrong rule from a label that is textually correct but not uniquely inferable
from form alone.

| UID | Annotation label → prediction | Review |
|---|---|---|
| `0054/00544517` | cylinder → tube | The annotation says *hollow cylinder* with inner and outer diameter. The rendered part looks like a tube; both words describe its shape. The label is not false, but the leaf distinction is not reliable from CAD geometry. |
| `0013/00135003` | beam → bar | The long rectangular solid is plausibly either an elongated bar or a structural beam. Functional use would be needed to decide. |
| `0054/00547121` | sleeve → bushing | The stepped hollow sleeve has the same visible form as many bushings; the rendered bore and flange do not establish intended use. |
| `0062/00620491` | plug → disc | A short round plug renders as a solid disc-like cylinder; its *plug* use is not visible in the standalone CAD. |
| `0100/01005661` | profile → shim | The description is *small semicircular extrusion (0.6 mm thick)*. `profile` is a generic fallback for *extrusion* and is not an independently verified functional class; the model's thin-plate/shim interpretation is plausible. |
| `0057/00572534` | bushing → plug | A flanged cylindrical bushing is visually similar to a plug from this view. The generated label may be correct, but the view alone cannot verify its intended function. |
| `0057/00578065` | shim → plate | The rendered part is a very thin rectangular plate. *Shim* denotes an application/thickness role; the prediction describes its visible form. |
| `0004/00046033` | shaft → flange | The description explicitly says *shaft with a larger flanged head*. The model over-emphasizes the prominent flange. This is a genuine whole-part classification mistake, though the predicted feature exists. |

The v1 rule error was checked separately on training/validation text: *square
block extrusion* and *dual pins extrusion* had been labelled `profile` solely
because *extrusion* occurred last. The refined v2 rule now chooses the more
specific explicit noun. It does not pretend that cylinder/tube, shim/plate,
or plug/disc have a universally recoverable boundary. Such cases are retained
and reported rather than silently hidden in an `other` class.
