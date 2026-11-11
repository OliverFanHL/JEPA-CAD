"""Optional OpenCASCADE renders of test cases and their frozen-feature neighbors."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from common import ROOT, load_config, save_json, v7_modules
from PIL import Image, ImageDraw, ImageFont


def _render_step(source: Path, destination: Path, image_size: int, config: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v7_modules(config)
    from src.histcad_large_jepa.rendering import _edges, _shape_bounds, _step_shape, _triangles

    shape = _step_shape(source)
    bounds = _shape_bounds(shape)
    lower, upper = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    diagonal = float(np.linalg.norm(upper - lower))
    if diagonal <= 1e-9:
        raise ValueError("Degenerate STEP shape bounds")
    triangles = _triangles(shape, diagonal)
    edges = _edges(shape, diagonal)
    if len(triangles) == 0:
        raise ValueError("OpenCASCADE imported the shape but produced no triangles")
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(min=1e-12)
    shading = np.clip(0.52 + 0.42 * np.abs(normal @ np.array([0.35, -0.45, 0.82])), 0, 1)
    colors = np.clip(np.array([0.23, 0.55, 0.91])[None, :] * shading[:, None] + 0.12, 0, 1)
    figure = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    try:
        axis = figure.add_subplot(111, projection="3d")
        axis.set_proj_type("ortho")
        axis.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors="none"))
        # Plot edges individually; OCC returns differently sampled curves.
        for edge in edges:
            axis.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="#18324d", linewidth=0.45)
        center = (lower + upper) / 2
        radius = max(upper - lower) / 2 * 1.08
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1), zoom=1.35)
        axis.view_init(elev=24, azim=-48)
        axis.set_axis_off()
        figure.patch.set_facecolor("white")
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=100, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    finally:
        plt.close(figure)
    with Image.open(destination) as check:
        check.verify()
    return {"status": "valid", "triangles": int(len(triangles)),
            "topological_edges": len(edges), "bounds": list(bounds)}


def _font(size: int, bold: bool = False):
    candidate = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(candidate), size) if candidate.is_file() else ImageFont.load_default()


def render_model(config: dict, model_name: str) -> dict:
    try:
        import OCP  # noqa: F401
    except ImportError as error:
        raise ImportError("OCP/cadquery-ocp is required to render actual STEP solids") from error
    model_dir = config["paths"]["output_dir"] / model_name
    cases_file = model_dir / "qualitative_cases.csv"
    if not cases_file.is_file():
        raise FileNotFoundError(f"Evaluate {model_name} before rendering: {cases_file}")
    cases = list(csv.DictReader(cases_file.open(encoding="utf-8", newline="")))
    render_dir = model_dir / "rendered_parts"
    cache: dict[str, tuple[Path | None, dict]] = {}

    def render_one(path: Path) -> tuple[Path | None, dict]:
        key = str(path.resolve())
        if key not in cache:
            destination = render_dir / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".png")
            try:
                if not path.is_file():
                    raise FileNotFoundError(path)
                details = _render_step(path, destination, int(config["qualitative"]["image_size"]), config)
                cache[key] = (destination, details)
            except Exception as error:
                cache[key] = (None, {"status": "invalid", "error": f"{type(error).__name__}: {error}"})
        return cache[key]

    by_class: dict[str, list[dict]] = {}
    for item in cases:
        by_class.setdefault(item["true_class"], []).append(item)
    sheets = []
    for class_name, rows in by_class.items():
        rendered = []
        for row in rows:
            query, _ = render_one(Path(row["step_path"]))
            neighbor, _ = render_one(Path(row["nearest_train_step_path"]))
            if query is not None and neighbor is not None:
                rendered.append((row, query, neighbor))
        if not rendered:
            continue
        cell_w, cell_h = 310, 310
        sheet = Image.new("RGB", (2 * cell_w, 60 + len(rendered) * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 12), f"{model_name}: {class_name} held-out examples", fill="#192b42", font=_font(19, True))
        for row_number, (item, query, neighbor) in enumerate(rendered):
            top = 60 + row_number * cell_h
            for column, (path, title) in enumerate((
                (query, f"Test: {item['true_class']} -> {item['predicted_class']}"),
                (neighbor, f"Nearest train: {item['nearest_train_class']}"),
            )):
                x = column * cell_w
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((cell_w - 16, 245), Image.Resampling.LANCZOS)
                    sheet.paste(image, (x + (cell_w - image.width) // 2, top + 2))
                draw.text((x + 10, top + 250), title, fill="#24364e", font=_font(14, True))
                subtitle = (f"{item['case_type']} confidence={float(item['confidence']):.3f}"
                            if column == 0 else f"cosine={float(item['nearest_train_cosine']):.3f}")
                draw.text((x + 10, top + 272), subtitle, fill="#54647a", font=_font(12))
        output = model_dir / f"qualitative_{class_name}.png"
        sheet.save(output)
        sheets.append(str(output))
    audit = {"model": model_name, "requested_cases": len(cases), "rendered_class_sheets": sheets,
             "valid_parts": sum(details["status"] == "valid" for _, details in cache.values()),
             "invalid_parts": sum(details["status"] != "valid" for _, details in cache.values()),
             "files": {path: details for path, (_, details) in cache.items()},
             "warning": "Only successfully imported and triangulated STEP models appear in sheets; failures are listed here."}
    save_json(model_dir / "render_audit.json", audit)
    print(f"{model_name}: {len(sheets)} class sheets, {audit['valid_parts']} valid renders, "
          f"{audit['invalid_parts']} failures", flush=True)
    if not sheets:
        raise RuntimeError(f"No valid qualitative STEP renders; see {model_dir / 'render_audit.json'}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--model", choices=("linear", "attention"), default="attention")
    args = parser.parse_args()
    render_model(load_config(args.config), args.model)


if __name__ == "__main__":
    main()
