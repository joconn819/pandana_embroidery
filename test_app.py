"""Checks that the actual SVG and machine-file pipeline preserves its design."""

import tempfile
import unittest
from pathlib import Path

import pyembroidery
from PIL import Image, ImageDraw

from app import (image_to_svg, kmeans_quantize, remove_background,
                 simplify_color_regions, trim_transparent_margins)
from stitch_generator import (rasterize_region_svg, render_stitch_preview,
                              svg_to_stitch_file)


class ConversionTests(unittest.TestCase):
    def test_background_and_palette_keep_subject_alpha(self):
        image = Image.new("RGB", (20, 20), "white")
        ImageDraw.Draw(image).rectangle((4, 4, 15, 15), fill="brown")
        cutout = trim_transparent_margins(remove_background(image))
        self.assertEqual(cutout.size, (12, 12))
        quantized = simplify_color_regions(kmeans_quantize(cutout, 2))
        self.assertEqual(quantized.getchannel("A").getextrema(), (255, 255))
        self.assertLessEqual(len(set(quantized.getdata())), 2)

    def test_svg_geometry_and_machine_files(self):
        image = Image.new("RGBA", (80, 60), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 5, 74, 54), fill=(180, 90, 45, 255))
        draw.rectangle((27, 20, 52, 39), fill=(0, 0, 0, 0))
        draw.ellipse((32, 24, 47, 35), fill=(40, 30, 25, 255))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            svg = folder / "dog.svg"
            image_to_svg(image, svg, max_dimension=40)
            raster = rasterize_region_svg(svg)
            self.assertEqual(raster.size, image.size)
            self.assertEqual(raster.getpixel((39, 21))[3], 0)
            self.assertEqual(raster.getpixel((39, 30))[:3], (40, 30, 25))

            previews = []
            for format_name in ("DST", "PES"):
                machine_file = folder / f"dog.{format_name.lower()}"
                metrics = svg_to_stitch_file(svg, machine_file, format_name)
                self.assertGreater(metrics["stitches"], 100)
                self.assertLessEqual(metrics["max_stitch_mm"], 4.1)
                self.assertLessEqual(metrics["max_jump_mm"], 12.1)
                self.assertGreaterEqual(metrics["stitch_coverage"], 0.9)
                self.assertEqual(metrics["color_changes"], 1)
                if format_name == "PES":
                    actual = [thread.color & 0xFFFFFF for thread in
                              pyembroidery.read_pes(str(machine_file)).threadlist]
                    expected = [(red << 16) | (green << 8) | blue
                                for red, green, blue in metrics["palette"]]
                    self.assertEqual(actual, expected)
                previews.append(render_stitch_preview(
                    machine_file, format_name, metrics["palette"]))
            self.assertEqual(previews[0].tobytes(), previews[1].tobytes())


if __name__ == "__main__":
    unittest.main()
