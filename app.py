"""Tkinter photo quantizer and embroidery converter.

The conversion pipeline is:
PNG/JPEG -> background removal -> k-means colour quantization -> SVG -> DST/PES.
"""

from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from functools import lru_cache
from queue import Empty, Queue
from threading import Thread
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from stitch_generator import (FILL_MAX_STITCH_MM, FILL_SPACING_MM,
                              render_stitch_preview, svg_to_stitch_file)

RGB = tuple[int, int, int]
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


@lru_cache(maxsize=2)
def _background_session(model_name: str) -> Any:
    from rembg import new_session

    return new_session(model_name)


def remove_background_with_model(image: Image.Image,
                                 model_name: str = "u2netp") -> Image.Image:
    """Remove the subject with a cached local rembg model session."""
    try:
        from rembg import remove
    except ImportError as error:
        raise RuntimeError(
            "The rembg package is required for model-based background removal. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from error
    result = remove(image.convert("RGBA"), session=_background_session(model_name))
    if not isinstance(result, Image.Image):
        raise RuntimeError("The background-removal model returned an invalid image.")
    return result.convert("RGBA")


def trim_transparent_margins(image: Image.Image) -> Image.Image:
    """Crop transparent padding so exported designs have no blank margins."""
    source = image.convert("RGBA")
    bounds = source.getchannel("A").point(lambda value: 255 if value >= 128 else 0).getbbox()
    if bounds is None:
        raise ValueError("Background removal produced no visible subject.")
    return source.crop(bounds)


def kmeans_quantize(image: Image.Image, k: int, iterations: int = 12) -> Image.Image:
    """Return an RGBA image using at most *k* deterministic colour clusters."""
    if k < 1:
        raise ValueError("k must be at least 1")
    source = image.convert("RGBA")
    pixels = list(source.getdata())
    visible = [pixel[:3] for pixel in pixels if pixel[3] >= 128]
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
        (*min(palette, key=lambda colour: _distance_sq(pixel[:3], colour)), 255)
        if pixel[3] >= 128 else (0, 0, 0, 0)
        for pixel in pixels
    ])
    return output


def simplify_color_regions(image: Image.Image, minimum_pixels: int = 8) -> Image.Image:
    """Smooth palette labels and absorb tiny islands without changing the silhouette."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from skimage.measure import label
    from PIL import ImageFilter

    source = np.asarray(image.convert("RGBA"))
    visible = source[:, :, 3] >= 128
    if not visible.any():
        raise ValueError("The image has no visible subject to simplify.")
    colors = sorted({tuple(map(int, rgb)) for rgb in source[visible, :3]})
    palette = {color: index for index, color in enumerate(colors)}
    labels = np.full(visible.shape, len(colors), dtype=np.uint8)
    for color, index in palette.items():
        labels[visible & np.all(source[:, :, :3] == color, axis=2)] = index

    # Filter palette indices, never separate RGB and alpha channels. Restore the
    # source silhouette so a transparent neighbor cannot erase the dog.
    filtered = np.asarray(Image.fromarray(labels).filter(ImageFilter.ModeFilter(3)))
    labels = np.where(visible & (filtered < len(colors)), filtered, labels)
    removable = np.zeros(visible.shape, dtype=bool)
    for index in range(len(colors)):
        components = label(visible & (labels == index), connectivity=2)
        counts = np.bincount(components.ravel())
        if len(counts) > 1:
            removable |= (components > 0) & (counts[components] < minimum_pixels)
    stable = visible & ~removable
    if stable.any() and removable.any():
        nearest = distance_transform_edt(~stable, return_distances=False,
                                         return_indices=True)
        labels[removable] = labels[tuple(axis[removable] for axis in nearest)]

    result = np.zeros_like(source)
    for color, index in palette.items():
        result[(labels == index) & visible, :3] = color
    result[visible, 3] = 255
    return Image.fromarray(result, "RGBA")


def _hex(colour: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)


def texture_direction_map(image: Image.Image, tile_pixels: int = 24):
    """Estimate local line orientation from the photo before color reduction.

    A structure tensor measures the direction across image detail; the stitch
    direction is perpendicular to that gradient. Ambiguous tiles keep the
    conventional vertical fill instead of following noise or cutout edges.
    """
    import numpy as np
    from scipy.ndimage import binary_erosion, gaussian_filter, sobel

    rgba = np.asarray(image.convert("RGBA"))
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    visible = rgba[:, :, 3] >= 128
    interior = binary_erosion(visible, iterations=3, border_value=0)
    smooth = gaussian_filter(gray, 1.0)
    dx = sobel(smooth, axis=1)
    dy = sobel(smooth, axis=0)
    directions = np.full(visible.shape, 90, dtype=np.uint8)
    bins = np.arange(0, 180, 30)
    for top in range(0, gray.shape[0], tile_pixels):
        for left in range(0, gray.shape[1], tile_pixels):
            area = np.s_[top:top + tile_pixels, left:left + tile_pixels]
            valid = interior[area]
            if valid.sum() < max(24, valid.size // 5):
                continue
            gx, gy = dx[area][valid], dy[area][valid]
            xx, xy, yy = float(np.dot(gx, gx)), float(np.dot(gx, gy)), float(np.dot(gy, gy))
            energy = xx + yy
            coherence = np.hypot(xx - yy, 2 * xy) / max(energy, 1e-9)
            if energy / valid.sum() < 0.004 or coherence < 0.28:
                continue
            tangent = (0.5 * np.degrees(np.arctan2(2 * xy, xx - yy)) + 90) % 180
            separation = np.abs((bins - tangent + 90) % 180 - 90)
            directions[area] = bins[np.argmin(separation)]
    return directions


def image_to_svg(image: Image.Image, path: str | os.PathLike[str],
                 max_dimension: int = 120,
                 max_stitch_mm: float = FILL_MAX_STITCH_MM,
                 row_spacing_mm: float = FILL_SPACING_MM,
                 direction_image: Image.Image | None = None) -> None:
    """Write connected same-colour regions as SVG contour paths."""
    try:
        import numpy as np
        from skimage.measure import approximate_polygon, find_contours, label
        from scipy.ndimage import find_objects
    except ImportError as error:
        raise RuntimeError(
            "scikit-image is required for smooth SVG contours. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from error
    source = image.convert("RGBA")
    width, height = source.size
    direction_map = (texture_direction_map(direction_image)
                     if direction_image is not None else None)
    if direction_map is not None and direction_map.shape != (height, width):
        raise ValueError("The direction photo must match the quantized image dimensions.")
    scale = max_dimension / max(width, height)
    svg_width = width * scale
    svg_height = height * scale
    inkscape_namespace = "http://www.inkscape.org/namespaces/inkscape"
    sodipodi_namespace = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
    svg_namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("inkscape", inkscape_namespace)
    ET.register_namespace("sodipodi", sodipodi_namespace)
    ET.register_namespace("svg", svg_namespace)
    root = ET.Element("svg", {
        "xmlns": svg_namespace,
        "xmlns:inkscape": inkscape_namespace,
        "xmlns:sodipodi": sodipodi_namespace,
        "xmlns:svg": svg_namespace,
        "width": f"{svg_width:.3f}mm", "height": f"{svg_height:.3f}mm",
        "viewBox": f"0 0 {width} {height}",
        "version": "1.1",
        "id": "svg1",
        "sodipodi:docname": Path(path).name,
        "inkscape:version": "1.4",
        "data-inkstitch-version": "3.3.0",
    })
    inkstitch_namespace = "http://inkstitch.org/namespace"
    ET.register_namespace("inkstitch", inkstitch_namespace)
    groups: dict[RGB, ET.Element] = {}
    pixels = list(source.getdata())
    colours = sorted({pixel[:3] for pixel in pixels if pixel[3] >= 128})
    for colour in colours:
        mask = [[pixels[y * width + x][:3] == colour and
                 pixels[y * width + x][3] >= 128 for x in range(width)]
                for y in range(height)]
        mask_array = np.asarray(mask, dtype=bool)
        group_attributes = {"id": f"color-{len(groups) + 1}", "fill": _hex(colour)}
        group = ET.SubElement(root, "g", group_attributes)
        groups[colour] = group
        angles = sorted(set(map(int, direction_map[mask_array]))) if direction_map is not None else [90]
        for angle in angles:
            angle_mask = mask_array & (direction_map == angle) if direction_map is not None else mask_array
            components = label(angle_mask, connectivity=2)
            for component_id, bounds in enumerate(find_objects(components), start=1):
                if bounds is None:
                    continue
                y_slice, x_slice = bounds
                component = components[y_slice, x_slice] == component_id
                if component.sum() < 8 and width * height > 100:
                    continue
                padded_mask = np.pad(component.astype(float), 1, constant_values=0)
                path_data_parts = []
                for contour in find_contours(padded_mask, 0.5):
                    if len(contour) < 3:
                        continue
                    contour = approximate_polygon(contour, tolerance=0.5)
                    points = [(float(point[1] - 1 + x_slice.start),
                               float(point[0] - 1 + y_slice.start)) for point in contour]
                    area = abs(sum(
                        points[index][0] * points[(index + 1) % len(points)][1] -
                        points[(index + 1) % len(points)][0] * points[index][1]
                        for index in range(len(points))
                    )) / 2
                    if len(points) < 3 or area < 1:
                        continue
                    path_data_parts.append(
                        "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points) + " Z")
                if not path_data_parts:
                    continue
                path_attributes = {
                    "d": " ".join(path_data_parts),
                    "fill-rule": "evenodd",
                    f"{{{inkstitch_namespace}}}auto_fill": "true",
                    f"{{{inkstitch_namespace}}}angle": str(angle),
                    f"{{{inkstitch_namespace}}}row_spacing_mm": str(row_spacing_mm),
                    f"{{{inkstitch_namespace}}}max_stitch_length_mm": str(max_stitch_mm),
                    f"{{{inkstitch_namespace}}}expand_mm": "0.2",
                    f"{{{inkstitch_namespace}}}fill_underlay": "true",
                    f"{{{inkstitch_namespace}}}fill_underlay_angle": str((angle + 90) % 180),
                    f"{{{inkstitch_namespace}}}fill_underlay_row_spacing_mm": "0.8",
                    f"{{{inkstitch_namespace}}}fill_underlay_max_stitch_length_mm": str(max_stitch_mm),
                    f"{{{inkstitch_namespace}}}ties": "true",
                }
                ET.SubElement(group, "path", path_attributes)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def svg_to_dst(svg_path: str | os.PathLike[str],
               dst_path: str | os.PathLike[str]) -> None:
    """Convert the SVG's filled color regions into a checked DST file."""
    svg_to_stitch_file(svg_path, dst_path, "DST")


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
        self.stitch_preview: Image.Image | None = None
        self.stitch_metrics: dict[str, float | int | list[tuple[int, int, int]]] | None = None
        self.input_path: Path | None = None
        self.svg_path: Path | None = None
        self.dst_path: Path | None = None
        self.format_value = tk.StringVar(value="DST")
        self.model_value = tk.StringVar(value="u2netp")
        self.aspect_ratio = 1.0
        self._syncing_dimensions = False
        self._conversion_queue: Queue = Queue()
        self._converting = False
        self.preview = self._ttk.Label(root, text="Choose a PNG or JPEG photo", anchor="center")
        self.preview.pack(fill="both", expand=True, padx=12, pady=12)
        controls = self._ttk.Frame(root)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        self._ttk.Button(controls, text="Choose photo…", command=self.choose_photo).pack(side="left")
        self._ttk.Label(controls, text="Background model:").pack(side="left", padx=(8, 4))
        self._ttk.Combobox(
            controls, textvariable=self.model_value, values=("bria-rmbg", "u2netp"),
            state="readonly", width=10).pack(side="left")
        self._ttk.Label(controls, text="Colours (k):").pack(side="left", padx=(16, 4))
        self.k_value = tk.IntVar(value=6)
        self._ttk.Spinbox(controls, from_=1, to=12, width=4, textvariable=self.k_value).pack(side="left")
        sizing = self._ttk.Frame(root)
        sizing.pack(fill="x", padx=12, pady=(0, 12))
        self._ttk.Label(sizing, text="Width (mm):").pack(side="left", padx=(0, 4))
        self.width_value = tk.DoubleVar(value=100)
        self._ttk.Entry(sizing, textvariable=self.width_value, width=7).pack(side="left")
        self._ttk.Label(sizing, text="Height (mm):").pack(side="left", padx=(8, 4))
        self.height_value = tk.DoubleVar(value=100)
        self._ttk.Entry(sizing, textvariable=self.height_value, width=7).pack(side="left")
        self._ttk.Label(sizing, text="Max stitch length (mm):").pack(side="left", padx=(8, 4))
        self.stitch_value = tk.DoubleVar(value=FILL_MAX_STITCH_MM)
        self._ttk.Entry(sizing, textvariable=self.stitch_value, width=6).pack(side="left")
        self._ttk.Label(sizing, text="Format:").pack(side="left", padx=(8, 4))
        self._ttk.Combobox(
            sizing, textvariable=self.format_value, values=("DST", "PES", "JEF"),
            state="readonly", width=5).pack(side="left")
        self.width_value.trace_add("write", lambda *_: self._sync_dimension("width"))
        self.height_value.trace_add("write", lambda *_: self._sync_dimension("height"))
        self.go_button = self._ttk.Button(sizing, text="Go", command=self.convert)
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

        if self._converting:
            return
        selected = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("PNG", "*.png"),
                       ("JPEG", "*.jpg *.jpeg")])
        if not selected:
            return
        try:
            with Image.open(selected) as opened:
                self.image = ImageOps.exif_transpose(opened).convert("RGBA")
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot open image", str(error))
            return
        self.input_path = Path(selected)
        self.aspect_ratio = self.image.width / self.image.height
        self.foreground = None
        self.quantized = None
        self.stitch_preview = None
        self.stitch_metrics = None
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

        if (self.image is None or self.foreground is None or self.quantized is None
                or self.stitch_preview is None):
            return
        workflow = self._tk.Toplevel(self.root)
        workflow.title("Conversion workflow")
        workflow.geometry("1100x360")
        row = self._ttk.Frame(workflow)
        row.pack(fill="both", expand=True, padx=12, pady=12)
        images = [self.image, self.foreground, self.quantized, self.stitch_preview]
        labels = ["Original photo", "Background removed", "K-means colours", "Stitch preview"]
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

        if self._converting:
            return
        if self.image is None or self.input_path is None:
            messagebox.showinfo("Choose a photo", "Select a PNG or JPEG photo first.")
            return
        try:
            k = self.k_value.get()
            if not 1 <= k <= 15:
                raise ValueError("Photo designs require between 1 and 15 colours.")
            final_width = self.width_value.get()
            final_height = self.height_value.get()
            max_stitch = self.stitch_value.get()
            if final_width <= 0 or final_height <= 0 or not 0.4 <= max_stitch <= 4.0:
                raise ValueError("Use positive dimensions and max stitch length 0.4–4 mm.")
            options = (self.image.copy(), self.input_path, k, final_width,
                       final_height, max_stitch, self.format_value.get(),
                       self.model_value.get())
        except (ValueError, self._tk.TclError) as error:
            messagebox.showerror("Conversion failed", str(error))
            return
        self._converting = True
        self.go_button.configure(state="disabled")
        self.status.config(text="Removing background…")
        Thread(target=self._convert_worker, args=options, daemon=True).start()
        self.root.after(100, self._poll_conversion)

    def _convert_worker(self, image: Image.Image, input_path: Path, k: int,
                        final_width: float, final_height: float,
                        max_stitch: float, format_name: str, model_name: str) -> None:
        try:
            target_width = max(1, round(final_width / max_stitch))
            target_height = max(1, round(final_height / max_stitch))
            minimum_scale = max(MIN_PROCESS_DIMENSION / target_width,
                                MIN_PROCESS_DIMENSION / target_height, 1)
            maximum_scale = min(MAX_PROCESS_DIMENSION / target_width,
                                MAX_PROCESS_DIMENSION / target_height)
            scale = min(max(minimum_scale, 1), maximum_scale)
            working_image = image.resize(
                (max(16, round(target_width * scale)),
                 max(16, round(target_height * scale))), Image.Resampling.LANCZOS)
            if working_image.getchannel("A").getextrema()[0] < 255:
                foreground = trim_transparent_margins(working_image)
            else:
                foreground = trim_transparent_margins(
                    remove_background_with_model(working_image, model_name))
            self._conversion_queue.put(("status", "Quantizing and tracing the dog…"))
            quantized = simplify_color_regions(kmeans_quantize(foreground, k))
            svg_path = input_path.with_name(input_path.stem + "_kmeans.svg")
            extension = format_name.lower()
            dst_path = input_path.with_name(input_path.stem + f"_embroidery.{extension}")
            design_max_dimension = max(final_width, final_height)
            image_to_svg(quantized, svg_path, max_dimension=design_max_dimension,
                         max_stitch_mm=max_stitch, direction_image=foreground)
            self._conversion_queue.put(("status", "Digitizing and checking stitches…"))
            metrics = svg_to_stitch_file(svg_path, dst_path, format_name=format_name)
            chart_path = input_path.with_name(input_path.stem + "_thread_colors.txt")
            chart_path.write_text(
                "Thread sequence (approximate RGB references)\n" +
                "\n".join(
                    f"{index}. #{red:02X}{green:02X}{blue:02X}"
                    for index, (red, green, blue) in enumerate(metrics["palette"], 1)
                ) + "\n"
            )
            stitch_preview = render_stitch_preview(
                dst_path, format_name, metrics["palette"])
            preview_path = input_path.with_name(input_path.stem + "_stitch_preview.png")
            stitch_preview.save(preview_path)
            self._conversion_queue.put(("done", (foreground, quantized, svg_path,
                                                dst_path, stitch_preview, metrics)))
        except Exception as error:
            self._conversion_queue.put(("error", error))

    def _poll_conversion(self) -> None:
        from tkinter import messagebox

        try:
            while True:
                kind, payload = self._conversion_queue.get_nowait()
                if kind == "status":
                    self.status.config(text=payload)
                    continue
                self._converting = False
                self.go_button.configure(state="normal")
                if kind == "error":
                    messagebox.showerror("Conversion failed", str(payload))
                    return
                (self.foreground, self.quantized, self.svg_path, self.dst_path,
                 self.stitch_preview, self.stitch_metrics) = payload
                self._show(self.stitch_preview)
                self.action_frame.pack(fill="x", padx=12, pady=(0, 8))
                self.status.config(
                    text=(f"Saved {self.svg_path.name} and {self.dst_path.name} · "
                          f"{self.stitch_metrics['stitches']:,} stitches · "
                          f"{self.stitch_metrics['width_mm']:.1f} × "
                          f"{self.stitch_metrics['height_mm']:.1f} mm"))
                messagebox.showinfo("Conversion complete",
                                    "SVG, embroidery, and stitch preview saved beside the photo.")
                return
        except Empty:
            self.root.after(100, self._poll_conversion)


def main() -> None:
    import tkinter as tk

    root = tk.Tk()
    ConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
