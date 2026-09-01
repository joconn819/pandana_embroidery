"""Tkinter photo quantizer and embroidery converter.

The conversion pipeline is:
PNG/JPEG -> background removal -> k-means colour quantization -> SVG -> DST.
"""

from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
import math
from pathlib import Path
from typing import Any

from PIL import Image

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]
MAX_PROCESS_DIMENSION = 240
MIN_PROCESS_DIMENSION = 180


def _distance_sq(left: RGB, right: RGB) -> int:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def remove_background(image: Image.Image, tolerance: int = 55) -> Image.Image:
    """Make border-connected pixels matching corner colours transparent."""
    source = image.convert("RGBA")
    width, height = source.size
    pixels = list(source.getdata())
    references = [pixels[0][:3], pixels[width - 1][:3],
                  pixels[(height - 1) * width][:3],
                  pixels[-1][:3]]
    threshold = tolerance * tolerance
    transparent = bytearray(width * height)
    pending = [(x, 0) for x in range(width)] + [(x, height - 1) for x in range(width)]
    pending += [(0, y) for y in range(height)] + [(width - 1, y) for y in range(height)]
    while pending:
        x, y = pending.pop()
        index = y * width + x
        if transparent[index]:
            continue
        pixel = pixels[index]
        if min(_distance_sq(pixel[:3], reference) for reference in references) > threshold:
            continue
        transparent[index] = 1
        pending.extend((neighbor_x, neighbor_y) for neighbor_x, neighbor_y in
                       ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                       if 0 <= neighbor_x < width and 0 <= neighbor_y < height)
    output = source.copy()
    output.putdata([
        (pixel[0], pixel[1], pixel[2], 0 if transparent[index] else pixel[3])
        for index, pixel in enumerate(pixels)
    ])
    return output


def remove_background_with_model(image: Image.Image) -> Image.Image:
    """Remove the subject with rembg's local segmentation model."""
    try:
        from rembg import remove
    except ImportError as error:
        raise RuntimeError(
            "The rembg package is required for model-based background removal. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from error
    result = remove(image.convert("RGBA"))
    if not isinstance(result, Image.Image):
        raise RuntimeError("The background-removal model returned an invalid image.")
    return result.convert("RGBA")


def trim_transparent_margins(image: Image.Image) -> Image.Image:
    """Crop transparent padding so exported designs have no blank margins."""
    source = image.convert("RGBA")
    bounds = source.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("Background removal produced no visible subject.")
    return source.crop(bounds)


def kmeans_quantize(image: Image.Image, k: int, iterations: int = 12) -> Image.Image:
    """Return an RGBA image using at most *k* deterministic colour clusters."""
    if k < 1:
        raise ValueError("k must be at least 1")
    source = image.convert("RGBA")
    pixels = list(source.getdata())
    visible = [pixel[:3] for pixel in pixels if pixel[3] > 0]
    if not visible:
        raise ValueError("the image has no visible pixels")

    # Sampling keeps a large photograph responsive while retaining its palette.
    sample_step = max(1, len(visible) // 10000)
    sample = visible[::sample_step]
    unique = list(dict.fromkeys(sample))
    if len(unique) <= k:
        palette = unique
    else:
        palette = [unique[(index * len(unique)) // k] for index in range(k)]
        for _ in range(iterations):
            buckets: list[list[RGB]] = [[] for _ in palette]
            for pixel in sample:
                nearest = min(range(len(palette)),
                              key=lambda index: _distance_sq(pixel, palette[index]))
                buckets[nearest].append(pixel)
            updated = [
                tuple(round(sum(pixel[channel] for pixel in bucket) / len(bucket))
                      for channel in range(3))
                if bucket else palette[index]
                for index, bucket in enumerate(buckets)
            ]
            if updated == palette:
                break
            palette = updated

    output = Image.new("RGBA", source.size)
    output.putdata([
        (*min(palette, key=lambda colour: _distance_sq(pixel[:3], colour)), pixel[3])
        if pixel[3] > 0 else (0, 0, 0, 0)
        for pixel in pixels
    ])
    return output


def detect_stitch_directions(image: Image.Image) -> list[list[float]]:
    """Estimate local texture tangent angles in radians for future fill stitches."""
    try:
        import numpy as np
        from skimage.filters import sobel_h, sobel_v
    except ImportError as error:
        raise RuntimeError(
            "scikit-image and numpy are required for stitch direction detection."
        ) from error
    source = np.asarray(image.convert("RGB"), dtype=float)
    luminance = (0.299 * source[:, :, 0] + 0.587 * source[:, :, 1] +
                 0.114 * source[:, :, 2])
    horizontal = sobel_h(luminance)
    vertical = sobel_v(luminance)
    # Gradient direction is perpendicular to the dominant texture tangent.
    angles = np.arctan2(vertical, horizontal) + (np.pi / 2)
    return angles.tolist()


def image_to_directional_dst(image: Image.Image, directions: list[list[float]],
                             path: str | os.PathLike[str],
                             spacing: int = 1, segment_length: int = 10,
                             format_name: str = "DST") -> None:
    """Write fill stitches aligned to the local texture tangent field."""
    source = image.convert("RGBA")
    width, height = source.size
    points: list[tuple[float, float, int]] = []
    position = (0, 0)
    colours = sorted({pixel[:3] for pixel in source.getdata() if pixel[3] > 0})
    for colour_index, colour in enumerate(colours):
        if colour_index:
            points.append((position[0], position[1], 5))
        for y in range(0, height, max(1, spacing)):
            x = 0
            while x < width:
                while (x < width and
                       (source.getpixel((x, y))[3] == 0 or
                        source.getpixel((x, y))[:3] != colour)):
                    x += 1
                start = x
                while (x < width and source.getpixel((x, y))[3] > 0 and
                       source.getpixel((x, y))[:3] == colour):
                    x += 1
                if start == x:
                    continue
                for segment_start in range(start, x, max(2, segment_length)):
                    segment_end = min(x, segment_start + max(2, segment_length))
                    angles = directions[y][segment_start:segment_end]
                    if not angles:
                        continue
                    sine = sum(math.sin(angle) for angle in angles)
                    cosine = sum(math.cos(angle) for angle in angles)
                    angle = math.atan2(sine, cosine)
                    half_length = (segment_end - segment_start) / 2
                    center_x = (segment_start + segment_end) / 2
                    center_y = y
                    first = (round(center_x - math.cos(angle) * half_length),
                             round(center_y - math.sin(angle) * half_length))
                    second = (round(center_x + math.cos(angle) * half_length),
                              round(center_y + math.sin(angle) * half_length))
                    for target in (first, second):
                        target = (max(0, min(width - 1, target[0])),
                                  max(0, min(height - 1, target[1])))
                        command = 1 if target == first and position != (0, 0) else 0
                        points.append((target[0], target[1], command))
                        position = target
    _write_embroidery_points(points, path, format_name, colours)


def _write_embroidery_points(points: list[tuple[float, float, int]],
                             path: str | os.PathLike[str],
                             format_name: str = "JEF",
                             thread_colors: list[RGB] | None = None) -> None:
    try:
        import pyembroidery
    except ImportError as error:
        raise RuntimeError(
            "pyembroidery is required for valid DST output. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from error
    pattern = pyembroidery.EmbPattern()
    for colour in thread_colors or []:
        thread = pyembroidery.EmbThread()
        thread.color = (colour[0] << 16) | (colour[1] << 8) | colour[2]
        pattern.add_thread(thread)
    for x, y, command in points:
        pattern.add_stitch_absolute(command, x, y)
    pattern.add_stitch_absolute(pyembroidery.END, 0, 0)
    writers = {"JEF": pyembroidery.write_jef, "DST": pyembroidery.write_dst}
    try:
        writers[format_name](pattern, str(path))
    except KeyError as error:
        raise ValueError(f"Unsupported embroidery format: {format_name}") from error


def _write_dst_points(points: list[tuple[float, float, int]],
                      path: str | os.PathLike[str]) -> None:
    _write_embroidery_points(points, path, "DST")


def _hex(colour: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)


def image_to_svg(image: Image.Image, path: str | os.PathLike[str],
                 max_dimension: int = 120) -> None:
    """Write connected same-colour regions as SVG contour paths."""
    try:
        import numpy as np
        from skimage.measure import find_contours
    except ImportError as error:
        raise RuntimeError(
            "scikit-image is required for smooth SVG contours. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from error
    source = image.convert("RGBA")
    width, height = source.size
    scale = max_dimension / max(width, height)
    svg_width = max(1, round(width * scale))
    svg_height = max(1, round(height * scale))
    root = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": f"{svg_width}mm", "height": f"{svg_height}mm",
        "viewBox": f"0 0 {width} {height}",
    })
    groups: dict[RGB, ET.Element] = {}
    pixels = list(source.getdata())
    colours = {pixel[:3] for pixel in pixels if pixel[3] > 0}
    for colour in colours:
        mask = [[pixels[y * width + x][:3] == colour and
                 pixels[y * width + x][3] > 0 for x in range(width)]
                for y in range(height)]
        group = groups.setdefault(colour, ET.SubElement(root, "g", {
            "id": f"color-{len(groups) + 1}", "fill": _hex(colour)
        }))
        padded_mask = np.pad(np.asarray(mask, dtype=float), 1, constant_values=0)
        for contour in find_contours(padded_mask, 0.5):
            if len(contour) < 3:
                continue
            points = [(float(point[1] - 1), float(point[0] - 1)) for point in contour]
            path_data = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points) + " Z"
            ET.SubElement(group, "path", {"d": path_data})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _dst_record(dx: int, dy: int, *, jump: bool = False,
                trim: bool = False, end: bool = False) -> bytes:
    """Encode one Tajima DST stitch record."""
    dx = max(-121, min(121, int(round(dx))))
    dy = max(-121, min(121, int(round(dy))))
    record = [0, 0, 0]
    # DST uses 7-bit signed movement split across three bytes.
    for value, first, second, third in ((dx, 0x01, 0x02, 0x04), (dy, 0x10, 0x20, 0x40)):
        if value < 0:
            value = -value
            record[2] |= 0x80 if first == 0x01 else 0x08
        if value & 1:
            record[0] |= first
        if value & 2:
            record[0] |= second
        if value & 4:
            record[0] |= third
        if value & 8:
            record[1] |= first
        if value & 16:
            record[1] |= second
        if value & 32:
            record[1] |= third
        if value & 64:
            record[2] |= first
        if value & 128:
            record[2] |= second
    record[2] |= 0x83
    if jump:
        record[2] |= 0x10
    if trim:
        record[2] |= 0x20
    if end:
        return b"\x00\x00\xf3"
    return bytes(record)


def svg_to_dst(svg_path: str | os.PathLike[str],
               dst_path: str | os.PathLike[str]) -> None:
    """Convert SVG contour paths into a valid DST running-stitch file."""
    root = ET.parse(svg_path).getroot()
    points: list[tuple[float, float, int]] = []
    position = (0, 0)
    groups = list(root)
    for group_index, group in enumerate(groups):
        if group_index:
            points.append((position[0], position[1], 1))
        for element in list(group):
            values = [float(value) for value in
                      element.attrib["d"].replace("M", "").replace("L", "")
                      .replace("Z", "").replace(",", " ").split()]
            contour_points = list(zip(values[::2], values[1::2]))
            for target in contour_points:
                target_position = (round(target[0]), round(target[1]))
                points.append((target_position[0], target_position[1], 0))
                position = target_position
    _write_dst_points(points, dst_path)


class ConverterApp:
    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self._tk = tk
        self._ttk = ttk
        self.root.title("Pandana Embroidery Converter")
        self.root.minsize(700, 500)
        self.image: Image.Image | None = None
        self.quantized: Image.Image | None = None
        self.foreground: Image.Image | None = None
        self.stitch_directions: list[list[float]] | None = None
        self.input_path: Path | None = None
        self.svg_path: Path | None = None
        self.dst_path: Path | None = None
        self.format_value = tk.StringVar(value="JEF")
        self.aspect_ratio = 1.0
        self._syncing_dimensions = False
        self.preview = self._ttk.Label(root, text="Choose a PNG or JPEG photo", anchor="center")
        self.preview.pack(fill="both", expand=True, padx=12, pady=12)
        controls = self._ttk.Frame(root)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        self._ttk.Button(controls, text="Choose photo…", command=self.choose_photo).pack(side="left")
        self._ttk.Label(controls, text="Colours (k):").pack(side="left", padx=(16, 4))
        self.k_value = tk.IntVar(value=6)
        self._ttk.Spinbox(controls, from_=1, to=32, width=4, textvariable=self.k_value).pack(side="left")
        self._ttk.Label(controls, text="Width (mm):").pack(side="left", padx=(12, 4))
        self.width_value = tk.DoubleVar(value=100)
        self._ttk.Entry(controls, textvariable=self.width_value, width=7).pack(side="left")
        self._ttk.Label(controls, text="Height (mm):").pack(side="left", padx=(8, 4))
        self.height_value = tk.DoubleVar(value=100)
        self._ttk.Entry(controls, textvariable=self.height_value, width=7).pack(side="left")
        self._ttk.Label(controls, text="Max stitch length (mm):").pack(side="left", padx=(8, 4))
        self.stitch_value = tk.DoubleVar(value=2.5)
        self._ttk.Entry(controls, textvariable=self.stitch_value, width=6).pack(side="left")
        self._ttk.Label(controls, text="Format:").pack(side="left", padx=(8, 4))
        self._ttk.Combobox(
            controls, textvariable=self.format_value, values=("JEF", "DST"),
            state="readonly", width=5).pack(side="left")
        self.width_value.trace_add("write", lambda *_: self._sync_dimension("width"))
        self.height_value.trace_add("write", lambda *_: self._sync_dimension("height"))
        self.go_button = self._ttk.Button(controls, text="Go", command=self.convert)
        self.go_button.pack(side="left", padx=(16, 0))
        self.action_frame = self._ttk.Frame(root)
        self.workflow_button = self._ttk.Button(
            self.action_frame, text="See workflow", command=self.show_workflow)
        self.workflow_button.pack(side="left")
        self.save_button = self._ttk.Button(
            self.action_frame, text="Save", command=self.save_embroidery)
        self.save_button.pack(side="left", padx=(8, 0))
        self.action_frame.pack_forget()
        self.status = self._ttk.Label(root, text="Ready", anchor="w")
        self.status.pack(fill="x", padx=12, pady=(0, 8))

    def choose_photo(self) -> None:
        from tkinter import filedialog, messagebox

        selected = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("PNG", "*.png"),
                       ("JPEG", "*.jpg *.jpeg")])
        if not selected:
            return
        try:
            self.image = Image.open(selected).convert("RGBA")
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot open image", str(error))
            return
        self.input_path = Path(selected)
        self.aspect_ratio = self.image.width / self.image.height
        self.width_value.set(100)
        self.height_value.set(round(100 / self.aspect_ratio, 1))
        self.action_frame.pack_forget()
        self._show(self.image)
        self.status.config(text=f"Loaded {self.input_path.name} ({self.image.width}×{self.image.height})")

    def _sync_dimension(self, changed: str) -> None:
        if self._syncing_dimensions or self.aspect_ratio <= 0:
            return
        try:
            self._syncing_dimensions = True
            if changed == "width":
                self.height_value.set(round(self.width_value.get() / self.aspect_ratio, 1))
            else:
                self.width_value.set(round(self.height_value.get() * self.aspect_ratio, 1))
        except (self._tk.TclError, ValueError):
            pass
        finally:
            self._syncing_dimensions = False

    def _show(self, image: Image.Image) -> None:
        from PIL import ImageTk

        displayed = self._preview_image(image, (900, 550))
        self.photo = ImageTk.PhotoImage(displayed)
        self.preview.configure(image=self.photo, text="")

    @staticmethod
    def _preview_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        displayed = image.convert("RGBA")
        alpha_bounds = displayed.getchannel("A").getbbox()
        if alpha_bounds and alpha_bounds != (0, 0, displayed.width, displayed.height):
            displayed = displayed.crop(alpha_bounds)
        displayed.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", size, (255, 255, 255, 255))
        offset = ((size[0] - displayed.width) // 2,
                  (size[1] - displayed.height) // 2)
        canvas.alpha_composite(displayed, offset)
        return canvas

    def show_workflow(self) -> None:
        from PIL import ImageTk

        if self.image is None or self.foreground is None or self.quantized is None:
            return
        workflow = self._tk.Toplevel(self.root)
        workflow.title("Conversion workflow")
        workflow.geometry("1100x360")
        row = self._ttk.Frame(workflow)
        row.pack(fill="both", expand=True, padx=12, pady=12)
        images = [self.image, self.foreground, self.quantized, self.quantized]
        labels = ["Original photo", "Background removed", "K-means colours", "SVG result"]
        workflow_items = []
        for image, label in zip(images, labels):
            frame = self._ttk.Frame(row)
            frame.pack(side="left", fill="both", expand=True, padx=4)
            image_label = self._ttk.Label(frame)
            image_label.pack(fill="both", expand=True)
            self._ttk.Label(frame, text=label).pack(pady=(6, 0))
            workflow_items.append((image, image_label))

        def resize_workflow(_event: Any = None) -> None:
            panel_width = max(80, (workflow.winfo_width() - 48) // 4)
            panel_height = max(80, workflow.winfo_height() - 80)
            workflow.photos = [
                ImageTk.PhotoImage(self._preview_image(image, (panel_width, panel_height)))
                for image, _label in workflow_items
            ]
            for photo, (_image, image_label) in zip(workflow.photos, workflow_items):
                image_label.configure(image=photo)

        workflow.bind("<Configure>", resize_workflow)
        workflow.after_idle(resize_workflow)

    def save_embroidery(self) -> None:
        from tkinter import filedialog, messagebox

        if self.dst_path is None:
            return
        extension = self.dst_path.suffix
        destination = filedialog.asksaveasfilename(
            defaultextension=extension,
            initialfile=self.dst_path.name,
            filetypes=[(f"Embroidery {extension[1:].upper()}", f"*{extension}")])
        if not destination:
            return
        try:
            shutil.copyfile(self.dst_path, destination)
        except OSError as error:
            messagebox.showerror("Save failed", str(error))
            return
        self.status.config(text=f"Saved embroidery file: {Path(destination).name}")

    def convert(self) -> None:
        from tkinter import messagebox

        if self.image is None or self.input_path is None:
            messagebox.showinfo("Choose a photo", "Select a PNG or JPEG photo first.")
            return
        try:
            k = self.k_value.get()
            if not 1 <= k <= 32:
                raise ValueError("Colours must be between 1 and 32.")
            final_width = self.width_value.get()
            final_height = self.height_value.get()
            average_stitch = self.stitch_value.get()
            if final_width <= 0 or final_height <= 0 or average_stitch <= 0:
                raise ValueError("Final dimensions and average stitch size must be positive.")
            self.status.config(text="Removing background with segmentation model…")
            self.root.update_idletasks()
            working_image = self.image.copy()
            target_width = max(1, round(final_width / average_stitch))
            target_height = max(1, round(final_height / average_stitch))
            minimum_scale = max(MIN_PROCESS_DIMENSION / target_width,
                                MIN_PROCESS_DIMENSION / target_height, 1)
            maximum_scale = min(MAX_PROCESS_DIMENSION / target_width,
                                MAX_PROCESS_DIMENSION / target_height)
            scale = min(max(minimum_scale, 1), maximum_scale)
            working_image = working_image.resize(
                (max(16, round(target_width * scale)),
                 max(16, round(target_height * scale))), Image.Resampling.LANCZOS)
            foreground = trim_transparent_margins(
                remove_background_with_model(working_image))
            self.status.config(text="Quantizing image…")
            self.root.update_idletasks()
            self.quantized = kmeans_quantize(foreground, k)
            self.stitch_directions = detect_stitch_directions(self.quantized)
            svg_path = self.input_path.with_name(self.input_path.stem + "_kmeans.svg")
            format_name = self.format_value.get()
            extension = format_name.lower()
            dst_path = self.input_path.with_name(
                self.input_path.stem + f"_embroidery.{extension}")
            self.foreground = foreground
            self.svg_path = svg_path
            self.dst_path = dst_path
            image_to_svg(self.quantized, svg_path, max_dimension=final_width)
            image_to_directional_dst(
                self.quantized, self.stitch_directions, dst_path,
                format_name=format_name)
            self._show(self.quantized)
            self.action_frame.pack(fill="x", padx=12, pady=(0, 8))
            self.status.config(text=f"Saved {svg_path.name} and {dst_path.name}")
            messagebox.showinfo("Conversion complete",
                                f"SVG and DST files saved beside {self.input_path.name}.")
        except (OSError, ValueError, ET.ParseError) as error:
            messagebox.showerror("Conversion failed", str(error))


def main() -> None:
    import tkinter as tk

    root = tk.Tk()
    ConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
