"""Human-readable width comparison from evaluated predictions and real STEP files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from data import TMCADSource, load_config, save_json
from step_render import render_step


OUTCOMES = ("512_corrected", "512_worsened", "both_wrong", "both_correct")
OUTCOME_COLORS = {
    "512_corrected": "#157347", "512_worsened": "#b42318",
    "both_wrong": "#a15c00", "both_correct": "#315477",
}


def _font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _metric_chart(evaluation: dict, classes: list[str], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    selected = int(evaluation["selected_width_by_validation"])
    width_scores = {
        width: evaluation["test"][str(width)]["with_physical_residual"]
        for width in (256, 512)
    }
    positions = np.arange(len(classes))
    figure, (overview, recall_axis) = plt.subplots(
        2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1, 2]},
    )
    try:
        names = ("accuracy", "balanced_accuracy", "macro_f1")
        for width, shift, color in ((256, -0.18, "#3565a2"),
                                    (512, 0.18, "#dd8124")):
            values = [width_scores[width][name] for name in names]
            overview.bar(np.arange(len(names)) + shift, values, width=0.36,
                         color=color, label=f"{width}-D supervised hidden")
            recalls = [width_scores[width]["per_class"][name]["recall"]
                       for name in classes]
            recall_axis.bar(positions + shift, recalls, width=0.36, color=color)
        overview.set_xticks(np.arange(len(names)),
                            ["Accuracy", "Balanced accuracy", "Macro-F1"])
        recall_axis.set_xticks(positions, classes, rotation=35, ha="right")
        overview.set_ylim(0, 1.04)
        recall_axis.set_ylim(0, 1.04)
        overview.set_ylabel("Held-out test score")
        recall_axis.set_ylabel("Held-out test recall")
        overview.legend(loc="lower right")
        overview.grid(axis="y", alpha=0.2)
        recall_axis.grid(axis="y", alpha=0.2)
        figure.suptitle(
            f"256-D vs 512-D hidden width | validation selected {selected}-D",
            fontsize=15,
        )
        figure.text(0.5, 0.01,
                    "Same frozen JEPA inputs and physical-residual design; "
                    "test values are descriptive, not used for width selection.",
                    ha="center", fontsize=10)
        figure.tight_layout(rect=(0, 0.035, 1, 0.95))
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, facecolor="white")
    finally:
        plt.close(figure)


def _read_predictions(source: TMCADSource, path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Run width evaluation before visualization: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indices = source.indices["test"]
    if len(rows) != len(indices):
        raise ValueError("Prediction CSV does not match the held-out test split")
    for row, index in zip(rows, indices, strict=True):
        uid = source.metadata["uids"][int(index)]
        expected_path = source.sample_path(int(index))
        if (row["uid"] != uid
                or Path(row["step_path"]).resolve() != expected_path
                or row["true_class"] != source.classes[int(source.labels[index])]
                or row["outcome"] not in OUTCOMES):
            raise ValueError(f"Qualitative CSV row differs from frozen TMCAD sample {uid}")
    return rows


def _select(rows: list[dict], classes: list[str], settings: dict) -> list[dict]:
    per_group = int(settings["cases_per_class_outcome"])
    if not 1 <= per_group <= 5:
        raise ValueError("cases_per_class_outcome must be between 1 and 5")
    selected = []
    for class_name in classes:
        for outcome in OUTCOMES:
            matches = [row for row in rows if row["true_class"] == class_name
                       and row["outcome"] == outcome]
            # Predeclared outcome/class strata, deterministic ordering, no
            # hand-picked models or test-driven architecture choice.
            matches.sort(key=lambda row: (
                -max(float(row["confidence_256"]),
                     float(row["confidence_512"])), row["uid"],
            ))
            selected.extend(matches[:per_group])
    return selected


def _case_card(row: dict, views: list[str], destination: Path) -> None:
    size = 300
    card = Image.new("RGB", (2 * size + 330, size + 72), "white")
    draw = ImageDraw.Draw(card)
    color = OUTCOME_COLORS[row["outcome"]]
    draw.rectangle((0, 0, card.width - 1, card.height - 1),
                   outline=color, width=3)
    draw.rectangle((0, 0, card.width, 38), fill=color)
    draw.text((12, 9), f"{row['true_class']} | {row['outcome']}",
              fill="white", font=_font(17, True))
    for view_index, path in enumerate(views):
        with Image.open(path) as source:
            picture = source.convert("RGB")
            picture.thumbnail((size - 12, size - 12), Image.Resampling.LANCZOS)
            x = view_index * size + (size - picture.width) // 2
            y = 42 + (size - picture.height) // 2
            card.paste(picture, (x, y))
    x = 2 * size + 12
    draw.text((x, 53), f"Ground truth: {row['true_class']}",
              fill="#172b4d", font=_font(17, True))
    for width, top in ((256, 104), (512, 164)):
        prediction = row[f"prediction_{width}"]
        is_correct = prediction == row["true_class"]
        draw.text((x, top), f"{width}-D: {prediction}",
                  fill="#157347" if is_correct else "#b42318",
                  font=_font(19, True))
        draw.text((x, top + 25),
                  f"confidence: {float(row[f'confidence_{width}']):.3f}",
                  fill="#526176", font=_font(15))
    uid = row["uid"]
    if len(uid) > 38:
        uid = uid[:35] + "..."
    draw.text((x, 250), f"ID: {uid}", fill="#526176", font=_font(13))
    draw.text((x, 278), "Actual STEP, opposite views",
              fill="#526176", font=_font(13))
    destination.parent.mkdir(parents=True, exist_ok=True)
    card.save(destination)


def _class_sheet(class_name: str, cases: list[dict], destination: Path) -> None:
    if not cases:
        return
    width, title_height, gap = 930, 55, 10
    card_height = 372
    sheet = Image.new("RGB", (width, title_height + len(cases) * (card_height + gap)),
                      "#edf2f6")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 15), f"{class_name}: 256-D versus 512-D predictions",
              fill="#172b4d", font=_font(23, True))
    for index, case in enumerate(cases):
        with Image.open(case["card_path"]) as card:
            sheet.paste(card.convert("RGB"), (0, title_height + index * (card_height + gap)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def visualize(config: dict, settings: dict) -> None:
    output = settings["output"]
    evaluation_path = output / "evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Run width evaluation first: {evaluation_path}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    source = TMCADSource(config)
    if evaluation.get("provenance") != source.provenance:
        raise ValueError("Saved evaluation does not match the current TMCAD inputs")
    rows = _read_predictions(source, output / "test_predictions_256_vs_512.csv")
    visual_root = output / "visualization"
    _metric_chart(evaluation, source.classes, visual_root / "metric_comparison.png")
    class_summary = []
    for class_name in source.classes:
        own = [row for row in rows if row["true_class"] == class_name]
        first = evaluation["test"]["256"]["with_physical_residual"]["per_class"][class_name]
        second = evaluation["test"]["512"]["with_physical_residual"]["per_class"][class_name]
        class_summary.append({
            "class": class_name, "test_n": len(own),
            "recall_256": first["recall"], "recall_512": second["recall"],
            "recall_change_pp": 100 * (second["recall"] - first["recall"]),
            **{outcome: sum(row["outcome"] == outcome for row in own)
               for outcome in OUTCOMES},
        })
    visual_root.mkdir(parents=True, exist_ok=True)
    with (visual_root / "class_change_summary.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_summary[0]))
        writer.writeheader()
        writer.writerows(class_summary)
    selected = _select(rows, source.classes, settings["qualitative"])
    if not selected:
        raise ValueError("No qualitative TMCAD test cases are available")
    try:
        import OCP  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "STEP rendering needs OCP. Install cadquery-ocp in the active environment, "
            "then rerun python run.py --stage visualize. The metric chart was saved."
        ) from error
    render_settings = settings["qualitative"]
    image_size = int(render_settings["image_size"])
    triangle_limit = int(render_settings["max_render_triangles"])
    edge_limit = int(render_settings["max_render_edges"])
    if (not 200 <= image_size <= 700 or not 1000 <= triangle_limit <= 100000
            or not 0 <= edge_limit <= 2000):
        raise ValueError("Invalid qualitative render limits")
    audit_cases = []
    for position, row in enumerate(selected, 1):
        path = Path(row["step_path"])
        stem = hashlib.sha256(row["uid"].encode("utf-8")).hexdigest()[:16]
        target = visual_root / "parts" / stem
        try:
            details = render_step(path, target, image_size, triangle_limit,
                                  edge_limit)
            card_path = visual_root / "cards" / f"{stem}.png"
            _case_card(row, details["views"], card_path)
            details["card_path"] = str(card_path)
        except Exception as error:
            details = {"status": "invalid", "step_path": str(path),
                       "error": f"{type(error).__name__}: {error}"}
        audit_cases.append({**row, **details})
        print(f"STEP visualization {position}/{len(selected)} "
              f"{row['true_class']} {row['outcome']} {details['status']}", flush=True)
    valid = [case for case in audit_cases if case["status"] == "valid"]
    sheets = []
    for class_name in source.classes:
        own = [case for case in valid if case["true_class"] == class_name]
        if own:
            destination = visual_root / f"class_{class_name}.png"
            _class_sheet(class_name, own, destination)
            sheets.append(str(destination))
    fieldnames = list(selected[0]) + [
        "status", "card_path", "views", "triangles_in_step_tessellation",
        "triangles_rendered", "topological_edges", "edges_rendered", "error",
    ]
    with (visual_root / "rendered_cases.csv").open("w", encoding="utf-8",
                                                        newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit_cases)
    audit = {
        "source": "Held-out test predictions; actual imported TMCAD STEP files",
        "selection": "At most one highest-confidence sample per class and outcome by default; "
                     "all four outcomes are attempted, including worsened cases.",
        "requested": len(selected), "valid": len(valid),
        "invalid": len(audit_cases) - len(valid),
        "classes_with_sheets": [Path(path).stem for path in sheets],
        "metric_chart": str(visual_root / "metric_comparison.png"),
        "cases": audit_cases,
    }
    save_json(visual_root / "render_audit.json", audit)
    links = "\n".join(
        f'<li><a href="{html.escape(Path(path).name)}">'
        f'{html.escape(Path(path).stem.removeprefix("class_"))}</a></li>'
        for path in sheets
    )
    summary_table = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['class'])}</td>"
        f"<td>{item['test_n']}</td>"
        f"<td>{item['recall_256']:.3f}</td>"
        f"<td>{item['recall_512']:.3f}</td>"
        f"<td>{item['recall_change_pp']:+.1f}</td>"
        f"<td>{item['512_corrected']}</td>"
        f"<td>{item['512_worsened']}</td>"
        "</tr>"
        for item in class_summary
    )
    (visual_root / "index.html").write_text(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        "<title>TMCAD width comparison</title>"
        "<style>body{font:16px system-ui;max-width:1050px;margin:32px auto;"
        "color:#172b4d}img{max-width:100%}li{margin:6px 0}"
        "table{border-collapse:collapse}th,td{padding:7px 10px;"
        "border-bottom:1px solid #d4dce5;text-align:right}"
        "td:first-child,th:first-child{text-align:left}</style>"
        "<h1>256-D versus 512-D downstream hidden width</h1>"
        f"<p>Validation-selected width: "
        f"{int(evaluation['selected_width_by_validation'])}-D. "
        "The two width arms use the same frozen JEPA features and physical "
        "residual. Test visual cases illustrate errors; they were not used "
        "to select the width.</p>"
        '<img src="metric_comparison.png" alt="Class-wise width comparison">'
        "<h2>Which classes changed?</h2>"
        "<p>Recall change is in percentage points. Corrected means 256-D was "
        "wrong and 512-D was right; worsened means the reverse.</p>"
        "<table><tr><th>Class</th><th>Test parts</th><th>256 recall</th>"
        "<th>512 recall</th><th>Change (pp)</th><th>Corrected</th>"
        f"<th>Worsened</th></tr>{summary_table}</table>"
        "<h2>Actual STEP test parts, shown from two sides</h2>"
        f"<p>Valid renders: {len(valid)}/{len(selected)}. "
        "See render_audit.json for missing or invalid STEP images.</p>"
        f"<ul>{links}</ul></html>", encoding="utf-8",
    )
    print(f"Visual comparison: {visual_root / 'index.html'}; "
          f"valid STEP renders={len(valid)}/{len(selected)}", flush=True)
    if not valid:
        raise RuntimeError(f"No STEP cases rendered; inspect {visual_root / 'render_audit.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the finalized TMCAD 256-D versus 512-D width comparison",
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "width_ablation" / "config.yaml",
    )
    args = parser.parse_args()
    path = args.config.resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_config((path.parent / settings["base_config"]).resolve())
    settings["output"] = (path.parent / settings["output"]).resolve()
    visualize(config, settings)


if __name__ == "__main__":
    main()
