"""STEP-rendered semantic wins and trade-offs, selected after fixed evaluation."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from evaluate import canonical
from PIL import Image, ImageDraw, ImageFont

from data import ROOT, config, payload, resolve, split_indices

sys.path.insert(0, str(ROOT.parent / "src"))
from histcad_large_jepa.rendering import render_step  # noqa: E402, I001


def font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def short(value: str, maximum: int) -> str:
    return value if len(value) <= maximum else value[: maximum - 1] + "…"


def choices(top: dict, labels: list[str], targets: torch.Tensor) -> list[tuple[str, int]]:
    original = top["direct_jepa"]["indices"][:, 0].tolist()
    old = top["existing_geometry_adapter"]["indices"][:, 0].tolist()
    semantic = top["semantic_only"]["indices"][:, 0].tolist()
    fused = top["existing_geometry_adapter_plus_semantic"]["indices"][:, 0].tolist()
    counts = Counter(labels)
    pools = {"semantic_win": [], "balanced_win": [], "geometry_tradeoff": [], "random_eligible": []}
    for index, label in enumerate(labels):
        if not label or counts[label] < 2:
            continue
        pools["random_eligible"].append(index)
        old_distance = float((targets[index] - targets[old[index]]).abs().mean())
        semantic_distance = float((targets[index] - targets[semantic[index]]).abs().mean())
        fused_distance = float((targets[index] - targets[fused[index]]).abs().mean())
        if (
            labels[semantic[index]] == label
            and labels[original[index]] != label
            and labels[old[index]] != label
        ):
            pools["semantic_win"].append(index)
        if (
            labels[fused[index]] == label
            and labels[old[index]] != label
            and fused_distance <= old_distance + 0.1
        ):
            pools["balanced_win"].append(index)
        if (
            labels[semantic[index]] == label
            and semantic_distance > old_distance + 0.2
            and old_distance <= 0.2
        ):
            pools["geometry_tradeoff"].append(index)
    rng = random.Random(431)
    selected = []
    used = set()
    for category in pools:
        candidates = [item for item in pools[category] if item not in used]
        rng.shuffle(candidates)
        for candidate in candidates[:20]:
            selected.append((category, candidate))
            used.add(candidate)
    return selected


def main() -> None:
    settings = config()
    output_dir = resolve(settings["paths"]["output_dir"])
    source = payload(settings)
    indices = split_indices(source)["test"]
    metadata = [source["metadata"][index] for index in indices.tolist()]
    structural = [source["structural_metadata"][index] for index in indices.tolist()]
    labels = [canonical(item.get("functional_type", "")) for item in metadata]
    targets = (source["dimension_targets"][indices] - source["dimension_target_mean"]) / source[
        "dimension_target_std"
    ]
    saved = torch.load(output_dir / "test_neighbors.pt", map_location="cpu", weights_only=False)
    top = saved["neighbors"]
    categories = choices(top, labels, targets)
    render_dir = output_dir / "rendered_parts"
    sheet_dir = output_dir / "retrieval_sheets"
    render_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    audit = {}
    manifest = []
    representations = (
        ("Frozen JEPA", "direct_jepa"),
        ("Old geometry adapter", "existing_geometry_adapter"),
        ("Semantic-only adapter", "semantic_only"),
        ("Old + semantic", "existing_geometry_adapter_plus_semantic"),
    )
    successful = 0
    rendered_categories = set()
    for category, query in categories:
        if category in rendered_categories:
            continue
        needed = {query}
        for _, name in representations:
            needed.update(top[name]["indices"][query, :3].tolist())
        images = {}
        for index in needed:
            item = metadata[index]
            uid = item["uid"]
            path = render_dir / f"{uid.replace('/', '__')}.png"
            if not path.is_file():
                try:
                    audit[uid] = render_step(Path(item["step_path"]), path)
                except Exception as error:
                    audit[uid] = {"status": "invalid", "error": f"{type(error).__name__}: {error}"}
            else:
                audit[uid] = {"status": "valid", "cached": True}
            if audit[uid]["status"] != "valid":
                break
            with Image.open(path) as image:
                image.verify()
            images[index] = path
        if len(images) != len(needed):
            continue
        successful += 1
        rendered_categories.add(category)
        cell_w, cell_h, left, header = 310, 272, 175, 90
        canvas = Image.new(
            "RGB", (left + 4 * cell_w, header + len(representations) * cell_h), "white"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (15, 12), f"{category}: {metadata[query]['uid']}", fill="#10233f", font=font(21, True)
        )
        draw.text(
            (15, 44),
            short(metadata[query].get("functional_type", ""), 110),
            fill="#334155",
            font=font(14),
        )
        for column, title in enumerate(("Query", "Nearest 1", "Nearest 2", "Nearest 3")):
            draw.text((left + column * cell_w + 10, 64), title, fill="#10233f", font=font(16, True))
        for row, (display, name) in enumerate(representations):
            y0 = header + row * cell_h
            draw.text((10, y0 + 14), display, fill="#10233f", font=font(15, True))
            neighbors = top[name]["indices"][query, :3].tolist()
            scores = top[name]["scores"][query, :3].tolist()
            for column, index in enumerate([query, *neighbors]):
                x0 = left + column * cell_w
                with Image.open(images[index]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((cell_w - 20, 182), Image.Resampling.LANCZOS)
                    canvas.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 5))
                function_match = column > 0 and labels[query] == labels[index]
                border = "#2563eb" if column == 0 else "#16a34a" if function_match else "#dc6b19"
                draw.rectangle(
                    (x0 + 5, y0 + 4, x0 + cell_w - 5, y0 + cell_h - 4), outline=border, width=3
                )
                draw.text(
                    (x0 + 12, y0 + 190), metadata[index]["uid"], fill="#172033", font=font(15, True)
                )
                draw.text(
                    (x0 + 12, y0 + 210),
                    short(metadata[index].get("functional_type", ""), 42),
                    fill="#475569",
                    font=font(13),
                )
                if column > 0:
                    distance = float((targets[query] - targets[index]).abs().mean())
                    history = (
                        structural[query]["operation_signature"]
                        == structural[index]["operation_signature"]
                    )
                    count = structural[query]["num_features"] == structural[index]["num_features"]
                    draw.text(
                        (x0 + 12, y0 + 230),
                        f"cos {scores[column - 1]:.3f} | dim {distance:.2f}",
                        fill="#334155",
                        font=font(12),
                    )
                    draw.text(
                        (x0 + 12, y0 + 246),
                        f"history {'yes' if history else 'no'} | count {'yes' if count else 'no'}",
                        fill="#334155",
                        font=font(12),
                    )
                    manifest.append(
                        {
                            "case": successful,
                            "category": category,
                            "representation": name,
                            "query_uid": metadata[query]["uid"],
                            "rank": column,
                            "neighbor_uid": metadata[index]["uid"],
                            "functional_match": function_match,
                            "dimension_mae": distance,
                            "history_match": history,
                            "feature_count_match": count,
                        }
                    )
        canvas.save(sheet_dir / f"case_{successful:02d}_{category}.png", optimize=True)
        if len(rendered_categories) == 4:
            break
    if manifest:
        with (output_dir / "qualitative_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    (output_dir / "render_audit.json").write_text(
        json.dumps(
            {
                "selected_categories": 4,
                "rendered_categories": sorted(rendered_categories),
                "rendered": successful,
                "files": audit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"selected_categories": 4, "rendered": successful}), flush=True)


if __name__ == "__main__":
    main()
