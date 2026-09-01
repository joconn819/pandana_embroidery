import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app import (detect_stitch_directions, image_to_directional_dst,
                 image_to_svg, kmeans_quantize, remove_background,
                 svg_to_dst, trim_transparent_margins)


class ConversionTests(unittest.TestCase):
    def test_background_removal_keeps_foreground(self):
        image = Image.new("RGB", (5, 5), "white")
        for x in range(5):
            image.putpixel((x, 0), (230, 230, 230))
        image.putpixel((2, 2), (255, 0, 0))
        result = remove_background(image)
        self.assertEqual(result.getpixel((0, 0))[3], 0)
        self.assertEqual(result.getpixel((2, 2)), (255, 0, 0, 255))

    def test_transparent_margins_are_trimmed(self):
        image = Image.new("RGBA", (5, 5), (0, 0, 0, 0))
        image.putpixel((2, 3), (255, 0, 0, 255))
        self.assertEqual(trim_transparent_margins(image).size, (1, 1))

    def test_quantization_uses_requested_palette(self):
        image = Image.new("RGB", (4, 1), "red")
        image.putpixel((1, 0), (0, 255, 0))
        image.putpixel((2, 0), (0, 0, 255))
        result = kmeans_quantize(image, 2)
        self.assertLessEqual(len(set(result.getdata())), 2)
        self.assertEqual(result.mode, "RGBA")

    def test_stitch_direction_field_matches_image_size(self):
        directions = detect_stitch_directions(Image.new("RGB", (4, 3), "red"))
        self.assertEqual((len(directions[0]), len(directions)), (4, 3))

    def test_directional_dst_contains_stitches(self):
        image = Image.new("RGBA", (20, 10), (255, 0, 0, 255))
        directions = detect_stitch_directions(image)
        with tempfile.TemporaryDirectory() as directory:
            dst = Path(directory) / "directional.dst"
            image_to_directional_dst(image, directions, dst)
            self.assertGreater(dst.stat().st_size, 512)
            self.assertEqual(dst.read_bytes()[:512].count(b"LA:"), 1)
            jef = Path(directory) / "directional.jef"
            image_to_directional_dst(image, directions, jef, format_name="JEF")
            self.assertGreater(jef.stat().st_size, 100)

    def test_svg_can_be_converted_to_dst(self):
        with tempfile.TemporaryDirectory() as directory:
            svg = Path(directory) / "image.svg"
            dst = Path(directory) / "image.dst"
            image_to_svg(Image.new("RGB", (3, 2), "red"), svg)
            self.assertIn("<path", svg.read_text())
            svg_to_dst(svg, dst)
            self.assertGreater(dst.stat().st_size, 512)
            self.assertEqual(dst.read_bytes()[:2], b"LA")


if __name__ == "__main__":
    unittest.main()
