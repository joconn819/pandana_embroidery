"""Command-line photo → SVG → embroidery workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app import (image_to_svg, kmeans_quantize, remove_background_with_model,
                 simplify_color_regions, trim_transparent_margins)
from stitch_generator import (rasterize_region_svg, render_stitch_preview,
                              svg_to_stitch_file)


def convert_photo(source_path: Path, output_dir: Path, *, colors: int = 6,
                  size_mm: float = 90.0, max_stitch_mm: float = 4.0,
                  model: str = "u2netp", formats: tuple[str, ...] = ("DST", "PES"),
                  process_pixels: int = 240) -> dict:
    if not 1 <= colors <= 15:
        raise ValueError("Photo designs require between 1 and 15 colors.")
    if not 10 <= size_mm <= 300 or not 0.4 <= max_stitch_mm <= 4.0 or process_pixels < 64:
        raise ValueError("Use a size of 10–300 mm, stitch length 0.4–4 mm, and ≥ 64 pixels.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
    image.thumbnail((process_pixels, process_pixels), Image.Resampling.LANCZOS)
    if image.getchannel("A").getextrema()[0] < 255:
        foreground = trim_transparent_margins(image)
    else:
        foreground = trim_transparent_margins(remove_background_with_model(image, model))
    quantized = simplify_color_regions(kmeans_quantize(foreground, colors))
    stem = source_path.stem
    foreground.save(output_dir / f"{stem}-cutout.png")
    quantized.save(output_dir / f"{stem}-quantized.png")
    svg_path = output_dir / f"{stem}.svg"
    image_to_svg(quantized, svg_path, max_dimension=size_mm,
                 max_stitch_mm=max_stitch_mm, direction_image=foreground)
    svg_image, svg_angles = rasterize_region_svg(svg_path, with_angles=True)
    svg_image.save(output_dir / f"{stem}-svg.png")

    original = np.asarray(quantized)
    vector = np.asarray(svg_image)
    visible_a = original[:, :, 3] > 0
    visible_b = vector[:, :, 3] > 0
    silhouette_iou = float(np.logical_and(visible_a, visible_b).sum() /
                           np.logical_or(visible_a, visible_b).sum())
    exact_pixels = float(np.all(original == vector, axis=2).mean())
    if silhouette_iou < 0.90 or exact_pixels < 0.85:
        raise RuntimeError("SVG lost too much of the quantized dog image.")

    designs = {}
    thread_colors = []
    for format_name in formats:
        destination = output_dir / f"{stem}.{format_name.lower()}"
        metrics = svg_to_stitch_file(svg_path, destination, format_name)
        if not thread_colors:
            thread_colors = ["#{:02X}{:02X}{:02X}".format(*color)
                             for color in metrics["palette"]]
        designs[format_name] = {key: value for key, value in metrics.items()
                                if key != "palette"}
        render_stitch_preview(destination, format_name, metrics["palette"]).save(
            output_dir / f"{stem}-{format_name.lower()}-preview.png")
    report = {
        "source": str(source_path),
        "svg": svg_path.name,
        "requested_colors": colors,
        "actual_colors": len({tuple(rgb) for rgb in original[visible_a, :3]}),
        "thread_colors": thread_colors,
        "svg_silhouette_iou": round(silhouette_iou, 4),
        "svg_exact_pixel_fraction": round(exact_pixels, 4),
        "stitch_direction_angles_deg": {
            str(int(angle)): int(count) for angle, count in zip(
                *np.unique(svg_angles[visible_b], return_counts=True))
        },
        "designs": designs,
    }
    (output_dir / f"{stem}-metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / f"{stem}-thread-colors.txt").write_text(
        "Thread sequence (approximate RGB references)\n" +
        "\n".join(f"{index}. {color}" for index, color in enumerate(thread_colors, 1)) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs="+", type=Path, help="JPEG or PNG dog photos")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--colors", type=int, default=6)
    parser.add_argument("--size-mm", type=float, default=90.0)
    parser.add_argument("--max-stitch-mm", type=float, default=4.0)
    parser.add_argument("--model", default="u2netp",
                        help="rembg model, such as bria-rmbg or u2netp")
    parser.add_argument("--format", action="append", choices=("DST", "PES"),
                        help="Repeat for both formats (default: both)")
    parser.add_argument("--process-pixels", type=int, default=240)
    args = parser.parse_args()
    for photo in args.photos:
        destination = args.output_dir / photo.stem
        result = convert_photo(
            photo, destination, colors=args.colors, size_mm=args.size_mm,
            max_stitch_mm=args.max_stitch_mm, model=args.model,
            formats=tuple(args.format or ("DST", "PES")),
            process_pixels=args.process_pixels,
        )
        print(f"{photo.name}: {result['actual_colors']} colors; "
              f"SVG silhouette IoU {result['svg_silhouette_iou']:.3f}; "
              f"output {destination}")


if __name__ == "__main__":
    main()
