from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.histcad_large_jepa.rendering import render_step  # noqa: E402, I001


OUTPUT = ROOT / "outputs"
RENDER_DIR = OUTPUT / "rendered_parts"
SHEET_DIR = OUTPUT / "retrieval_sheets"


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def canonical_label(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value.lower())
    value = re.sub(r"(?<![a-z0-9])(?:with|and|for|of|the|a|an)(?![a-z0-9])", " ", value)
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", value))))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def short(value: str, length: int = 42) -> str:
    return value if len(value) <= length else value[: length - 1] + "…"


def nearest(values: torch.Tensor, count: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    values = F.normalize(values.float(), dim=-1)
    scores, indices = [], []
    for start in range(0, len(values), 512):
        stop = min(start + 512, len(values))
        similarity = values[start:stop] @ values.T
        similarity[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
        chunk_scores, chunk_indices = similarity.topk(count, dim=1)
        scores.append(chunk_scores)
        indices.append(chunk_indices)
    return torch.cat(scores), torch.cat(indices)


def render_cached(item: dict[str, Any], audit: dict[str, Any]) -> Path:
    uid = item["uid"]
    path = RENDER_DIR / f"{uid.replace('/', '__')}.png"
    if uid not in audit:
        try:
            audit[uid] = render_step(Path(item["step_path"]), path)
        except Exception as error:  # Preserve exact per-file failures in the audit.
            audit[uid] = {"status": "invalid", "error": f"{type(error).__name__}: {error}"}
    if audit[uid]["status"] != "valid":
        raise ValueError(audit[uid]["error"])
    with Image.open(path) as image:
        image.verify()
    return path


def make_sheet(
    case: int,
    query: int,
    results: dict[str, tuple[list[int], list[float]]],
    metadata: list[dict[str, Any]],
    structural: list[dict[str, Any]],
    targets: torch.Tensor,
    renders: dict[str, Path],
) -> Path:
    cell_w, cell_h, left, header = 310, 278, 175, 90
    canvas = Image.new("RGB", (left + 4 * cell_w, header + 2 * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, head_font, body_font, detail_font = (
        font(22, True),
        font(17, True),
        font(14),
        font(12),
    )
    query_item = metadata[query]
    draw.text((18, 14), f"Case {case}: {query_item['uid']}", fill="#10233f", font=title_font)
    draw.text(
        (18, 46),
        short(query_item.get("functional_type", "Unlabeled"), 105),
        fill="#334155",
        font=body_font,
    )
    for column, label in enumerate(("Query", "Nearest 1", "Nearest 2", "Nearest 3")):
        draw.text((left + column * cell_w + 10, header - 27), label, fill="#10233f", font=head_font)
    for row, (name, (neighbors, scores)) in enumerate(results.items()):
        y0 = header + row * cell_h
        draw.rectangle((0, y0, canvas.width, y0 + cell_h), outline="#d7deea", width=1)
        draw.text((14, y0 + 18), name, fill="#10233f", font=head_font)
        indices, similarities = [query, *neighbors], [1.0, *scores]
        for column, (index, similarity) in enumerate(zip(indices, similarities, strict=True)):
            x0 = left + column * cell_w
            item = metadata[index]
            with Image.open(renders[item["uid"]]) as source:
                image = source.convert("RGB")
                image.thumbnail((cell_w - 24, 185), Image.Resampling.LANCZOS)
                canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 5))
            if column == 0:
                border, first, second = "#2563eb", "reference", ""
            else:
                function_match = canonical_label(
                    item.get("functional_type", "")
                ) == canonical_label(query_item.get("functional_type", ""))
                history_match = (
                    structural[index]["operation_signature"]
                    == structural[query]["operation_signature"]
                )
                count_match = structural[index]["num_features"] == structural[query]["num_features"]
                distance = float((targets[index] - targets[query]).abs().mean())
                border = "#16a34a" if function_match else "#dc6b19"
                first = f"sim {similarity:.3f}  |  dimensional distance {distance:.2f}"
                second = (
                    f"feature count {'yes' if count_match else 'no'}  |  "
                    f"history {'yes' if history_match else 'no'}"
                )
            draw.rectangle(
                (x0 + 5, y0 + 4, x0 + cell_w - 5, y0 + cell_h - 5), outline=border, width=3
            )
            draw.text((x0 + 12, y0 + 195), item["uid"], fill="#172033", font=head_font)
            draw.text(
                (x0 + 12, y0 + 215),
                short(item.get("functional_type", "Unlabeled")),
                fill="#475569",
                font=body_font,
            )
            draw.text((x0 + 12, y0 + 233), first, fill="#334155", font=detail_font)
            if second:
                draw.text((x0 + 12, y0 + 247), second, fill="#334155", font=detail_font)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    path = SHEET_DIR / f"case_{case:02d}.png"
    canvas.save(path, optimize=True)
    return path


def main() -> None:
    global OUTPUT, RENDER_DIR, SHEET_DIR
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    OUTPUT = resolve(config["paths"]["output_dir"])
    RENDER_DIR = OUTPUT / "rendered_parts"
    SHEET_DIR = OUTPUT / "retrieval_sheets"
    source = torch.load(
        resolve(config["paths"]["representations"]),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    adapted = torch.load(
        OUTPUT / "adapted_representations.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    test_indices = [
        index for index, item in enumerate(source["metadata"]) if item["split"] == "test"
    ]
    metadata = [source["metadata"][index] for index in test_indices]
    structural = [source["structural_metadata"][index] for index in test_indices]
    targets = (
        source["dimension_targets"][test_indices] - source["dimension_target_mean"]
    ) / source["dimension_target_std"]
    representations = {
        "Direct v7 joint": adapted["embeddings"]["direct_joint"][test_indices],
        "Adapted v7 joint": adapted["embeddings"]["adapted_joint"][test_indices],
    }
    retrieval = {name: nearest(values) for name, values in representations.items()}
    direct_top = retrieval["Direct v7 joint"][1][:, 0]
    adapted_top = retrieval["Adapted v7 joint"][1][:, 0]
    direct_distance = (targets[direct_top] - targets).abs().mean(1)
    adapted_distance = (targets[adapted_top] - targets).abs().mean(1)
    direct_feature = torch.tensor(
        [structural[int(index)]["num_features"] for index in direct_top]
    ) == torch.tensor([item["num_features"] for item in structural])
    adapted_feature = torch.tensor(
        [structural[int(index)]["num_features"] for index in adapted_top]
    ) == torch.tensor([item["num_features"] for item in structural])
    direct_history = [
        structural[int(direct_top[index])]["operation_signature"]
        == structural[index]["operation_signature"]
        for index in range(len(metadata))
    ]
    adapted_history = [
        structural[int(adapted_top[index])]["operation_signature"]
        == structural[index]["operation_signature"]
        for index in range(len(metadata))
    ]
    priority = []
    for index in range(len(metadata)):
        gain = float(direct_distance[index] - adapted_distance[index])
        gain += 0.4 * float(adapted_feature[index] and not direct_feature[index])
        gain += 0.4 * float(adapted_history[index] and not direct_history[index])
        priority.append((gain, index))
    priority.sort(reverse=True)

    audit: dict[str, Any] = {}
    cases, manifest, used_labels = [], [], set()
    for _, query in priority:
        label = canonical_label(metadata[query].get("functional_type", ""))
        if not label or label in used_labels:
            continue
        rows, required = {}, {query}
        for name, (scores, indices) in retrieval.items():
            neighbors = [int(value) for value in indices[query, :3]]
            rows[name] = (neighbors, [float(value) for value in scores[query, :3]])
            required.update(neighbors)
        try:
            renders = {
                metadata[index]["uid"]: render_cached(metadata[index], audit) for index in required
            }
        except ValueError:
            continue
        case = len(cases) + 1
        sheet = make_sheet(case, query, rows, metadata, structural, targets, renders)
        cases.append(sheet)
        used_labels.add(label)
        for name, (neighbors, scores) in rows.items():
            for rank, (neighbor, score) in enumerate(zip(neighbors, scores, strict=True), start=1):
                manifest.append(
                    {
                        "case": case,
                        "representation": name,
                        "query_uid": metadata[query]["uid"],
                        "rank": rank,
                        "neighbor_uid": metadata[neighbor]["uid"],
                        "cosine_similarity": score,
                        "dimension_mae": float((targets[query] - targets[neighbor]).abs().mean()),
                        "feature_count_match": structural[query]["num_features"]
                        == structural[neighbor]["num_features"],
                        "history_match": structural[query]["operation_signature"]
                        == structural[neighbor]["operation_signature"],
                        "functional_match": label
                        == canonical_label(metadata[neighbor].get("functional_type", "")),
                    }
                )
        if len(cases) == 6:
            break
    audit_payload = {
        "renderer": "OpenCASCADE STEP import + B-Rep triangulation + topological edges",
        "cases": len(cases),
        "unique_step_files_attempted": len(audit),
        "valid": sum(item["status"] == "valid" for item in audit.values()),
        "invalid": sum(item["status"] != "valid" for item in audit.values()),
        "files": audit,
    }
    (OUTPUT / "render_audit.json").write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    if len(cases) != 6:
        failures = [
            f"{uid}: {item.get('error', 'unknown rendering error')}"
            for uid, item in audit.items()
            if item["status"] != "valid"
        ]
        raise RuntimeError(
            f"Expected 6 valid cases, rendered {len(cases)}. "
            f"First failures: {failures[:3]}. Full details: {OUTPUT / 'render_audit.json'}"
        )
    with (OUTPUT / "qualitative_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    lines = [
        "# Direct versus adapted v7 retrieval",
        "",
        "Blue is the query. Green indicates a canonical functional-label match; orange indicates a "
        "mismatch. Functional labels were not used to train the adapter. Each result also reports "
        "standardized dimensional distance, feature-count agreement, and exact history agreement.",
        "",
    ]
    for case, sheet in enumerate(cases, start=1):
        lines.extend([f"## Case {case}", "", f"![Case {case}]({sheet.name})", ""])
    (SHEET_DIR / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "cases": len(cases),
                "valid_step_renders": audit_payload["valid"],
                "invalid_step_renders": audit_payload["invalid"],
            }
        )
    )


if __name__ == "__main__":
    main()
