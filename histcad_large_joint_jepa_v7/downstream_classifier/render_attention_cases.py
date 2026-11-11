"""Render audited STEP geometry for selected held-out classification cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from histcad_large_jepa.rendering import _edges, _shape_bounds, _step_shape, _triangles  # noqa: E402

OUTPUT = ROOT / "outputs" / "attention_role_tokens_32ep"


def _render_step_robust(source: Path, destination: Path, image_size: int = 260) -> dict:
    """Same OCC geometry as v7, with each variable-length edge plotted separately."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    shape = _step_shape(source)
    bounds = _shape_bounds(shape)
    lower, upper = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    diagonal = float(np.linalg.norm(upper - lower))
    if diagonal <= 1e-9:
        raise ValueError("STEP shape has degenerate bounds")
    triangles = _triangles(shape, diagonal)
    edges = _edges(shape, diagonal)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    shading = np.clip(0.52 + 0.42 * np.abs(normals @ np.array([0.35, -0.45, 0.82])), 0, 1)
    colors = np.clip(np.array([0.23, 0.55, 0.91])[None, :] * shading[:, None] + 0.12, 0, 1)
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    try:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_proj_type("ortho")
        ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors="none"))
        for edge in edges:
            # Matplotlib cannot autoscale one ragged collection containing
            # curves with different vertex counts. Individual lines are valid.
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="#18324d", linewidth=0.45)
        center = (lower + upper) / 2
        radius = max(upper - lower) / 2 * 1.08
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1), zoom=1.35)
        ax.view_init(elev=24, azim=-48)
        ax.set_axis_off()
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=100, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    finally:
        plt.close(fig)
    return {"status": "valid", "step_path": str(source), "render_path": str(destination),
            "triangles": int(len(triangles)), "topological_edges_rendered": len(edges),
            "bounds": list(bounds)}


def _font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _short(text: str, limit: int = 49) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output = parser.parse_args().output.resolve()
    rows = list(csv.DictReader((output / "qualitative_cases.csv").open(encoding="utf-8", newline="")))
    render_dir = output / "rendered_cases"
    render_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, dict] = {}
    selected: dict[str, list[tuple[dict, Path]]] = {
        "correct": [], "within_parent_error": [], "cross_parent_error": []
    }
    for row in rows:
        bucket = row["case_type"]
        if len(selected[bucket]) >= 4:
            continue
        uid = row["uid"]
        source = Path(row["step_path"])
        destination = render_dir / f"{uid.replace('/', '__')}.png"
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            details = _render_step_robust(source, destination, image_size=260)
            with Image.open(destination) as sample:
                sample.verify()
            audit[uid] = details
            selected[bucket].append((row, destination))
        except Exception as error:
            audit[uid] = {"status": "invalid", "step_path": str(source),
                          "error": f"{type(error).__name__}: {error}"}
    total = sum(map(len, selected.values()))
    if total == 0:
        raise RuntimeError("No selected STEP files could be rendered; inspect render_audit.json")
    cell_w, cell_h = 325, 365
    canvas = Image.new("RGB", (cell_w * 3, cell_h * 4 + 50), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = _font(20, True), _font(15, True), _font(13)
    draw.text((12, 9), "Frozen v7 attention classifier: held-out STEP examples", fill="#18263d", font=title_font)
    categories = list(selected)
    colors = {"correct": "#17834a", "within_parent_error": "#d28b16", "cross_parent_error": "#c54336"}
    for column, category in enumerate(categories):
        for row_number, (item, image_path) in enumerate(selected[category]):
            x, y = column * cell_w, 50 + row_number * cell_h
            draw.rectangle((x + 3, y + 3, x + cell_w - 3, y + cell_h - 3), outline=colors[category], width=3)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((cell_w - 14, 245), Image.Resampling.LANCZOS)
                canvas.paste(image, (x + (cell_w - image.width) // 2, y + 5))
            draw.text((x + 12, y + 253), f"{category.replace('_', ' ')} | {item['uid']}", fill=colors[category], font=body_font)
            draw.text((x + 12, y + 277), _short(item["functional_description"]), fill="#253449", font=small_font)
            draw.text((x + 12, y + 301), f"true: {item['true_form']}   predicted: {item['predicted_form']}", fill="#253449", font=small_font)
            draw.text((x + 12, y + 325), f"confidence: {float(item['prediction_confidence']):.3f}", fill="#526273", font=small_font)
    canvas.save(output / "qualitative_sheet.png")
    (output / "render_audit.json").write_text(json.dumps({
        "renderer": "OpenCASCADE STEP import, triangulated faces and B-Rep edges",
        "selected_cases": {bucket: len(items) for bucket, items in selected.items()},
        "attempted": len(audit),
        "valid": sum(item["status"] == "valid" for item in audit.values()),
        "invalid": sum(item["status"] != "valid" for item in audit.values()),
        "files": audit,
    }, indent=2), encoding="utf-8")
    print(f"Rendered {total} valid STEP parts into {output / 'qualitative_sheet.png'}")


if __name__ == "__main__":
    main()
