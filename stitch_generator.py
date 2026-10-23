"""Create a checked, multi-colour fill stitch plan from a quantized photo."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


FILL_SPACING_MM = 0.40
UNDERLAY_SPACING_MM = 0.80
FILL_MAX_STITCH_MM = 4.0
MIN_STITCH_MM = 0.4
MAX_JUMP_MM = 12.0
MIN_COMPONENT_PIXELS = 16
MAX_UNTRIMMED_TRAVEL_MM = 10.0


def rasterize_region_svg(svg_path: str | Path) -> Image.Image:
    """Rasterize the M/L/Z color regions emitted by ``image_to_svg``.

    This deliberately supports the converter's SVG subset. Rejecting another
    SVG path type is safer than silently producing a different stitch design.
    """
    import numpy as np

    root = ET.parse(svg_path).getroot()
    view_box = root.attrib.get("viewBox", "").split()
    if len(view_box) != 4 or float(view_box[0]) != 0 or float(view_box[1]) != 0:
        raise ValueError("Expected a converter SVG with a zero-origin viewBox.")
    width, height = round(float(view_box[2])), round(float(view_box[3]))
    if width < 1 or height < 1 or width * height > 4_000_000:
        raise ValueError("SVG viewBox has invalid or excessive dimensions.")
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    coordinate = re.compile(rf"({number}),({number})")
    svg_namespace = "{http://www.w3.org/2000/svg}"
    for group in root.findall(f"{svg_namespace}g"):
        fill = group.attrib.get("fill", "")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", fill):
            raise ValueError("SVG color group has no supported fill color.")
        rgb = tuple(int(fill[index:index + 2], 16) for index in (1, 3, 5))
        for path in group.findall(f"{svg_namespace}path"):
            if path.attrib.get("fill-rule") != "evenodd":
                raise ValueError("Expected evenodd path geometry in SVG.")
            path_data = path.attrib.get("d", "")
            polygons = re.findall(r"M\s*([^M]*?)\s*Z", path_data)
            if not polygons or len(polygons) != path_data.count("M"):
                raise ValueError("SVG contains an unsupported or empty path.")
            parity = np.zeros((height, width), dtype=bool)
            for polygon in polygons:
                points = [(float(x), float(y)) for x, y in coordinate.findall(polygon)]
                if len(points) < 3:
                    continue
                layer = Image.new("1", (width, height))
                ImageDraw.Draw(layer).polygon(points, fill=1)
                parity ^= np.asarray(layer, dtype=bool)
            pixels[parity, :3] = rgb
            pixels[parity, 3] = 255
    if not pixels[:, :, 3].any():
        raise ValueError("The SVG has no fill regions to stitch.")
    return Image.fromarray(pixels, "RGBA")


def svg_to_stitch_file(svg_path: str | Path, output_path: str | Path,
                       format_name: str = "DST", max_dimension_mm: float | None = None,
                       max_stitch_mm: float | None = None,
                       row_spacing_mm: float | None = None) -> dict:
    """Generate stitches from SVG geometry and its physical stitch parameters."""
    root = ET.parse(svg_path).getroot()
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    if not paths:
        raise ValueError("The SVG contains no stitchable paths.")
    stitch_ns = "{http://inkstitch.org/namespace}"
    if max_dimension_mm is None:
        width = float(root.attrib["width"].removesuffix("mm"))
        height = float(root.attrib["height"].removesuffix("mm"))
        max_dimension_mm = max(width, height)
    if max_stitch_mm is None:
        max_stitch_mm = float(paths[0].attrib[stitch_ns + "max_stitch_length_mm"])
    if row_spacing_mm is None:
        row_spacing_mm = float(paths[0].attrib[stitch_ns + "row_spacing_mm"])
    for path in paths[1:]:
        if (float(path.attrib[stitch_ns + "max_stitch_length_mm"]) != max_stitch_mm
                or float(path.attrib[stitch_ns + "row_spacing_mm"]) != row_spacing_mm):
            raise ValueError("SVG paths have inconsistent stitch parameters.")
    return create_stitch_file(
        rasterize_region_svg(svg_path), output_path, format_name,
        max_dimension_mm, max_stitch_mm, row_spacing_mm,
    )


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive intervals of true pixels in a one-dimensional mask."""
    import numpy as np

    positions = np.flatnonzero(values)
    if positions.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(positions) > 1) + 1
    return [(int(part[0]), int(part[-1])) for part in np.split(positions, cuts)]


def _color_key(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _normalize_pes_jump_flags(path: str | Path) -> None:
    """Keep PEC jumps as jumps; trim once before a travel longer than 10 mm.

    pyembroidery 1.5.x writes every post-start JUMP as a PEC trim-jump and
    ignores separate TRIM commands. The stitch plan already marks long travel
    with TRIM, so preserve that policy in the written PES/PEC command stream.
    """
    data = bytearray(Path(path).read_bytes())
    if not (data.startswith(b"#PES0001") or data.startswith(b"#PES0060")):
        raise RuntimeError("Unsupported PES header; cannot audit travel flags.")
    pec_offset = int.from_bytes(data[8:12], "little")
    block = pec_offset + 512
    if data[block + 5:block + 8] != b"\x31\xff\xf0":
        raise RuntimeError("Unsupported PEC stitch block in PES output.")
    position = block + 16
    pending: list[tuple[int, int, float]] = []
    has_sewn = False

    def finish_travel() -> None:
        nonlocal pending
        if not pending:
            return
        total = sum(item[2] for item in pending)
        for x_offset, y_offset, _ in pending:
            data[x_offset] = (data[x_offset] & ~0x30) | 0x10
            data[y_offset] = (data[y_offset] & ~0x30) | 0x10
        if has_sewn and total > MAX_UNTRIMMED_TRAVEL_MM * 10:
            x_offset, y_offset, _ = pending[0]
            data[x_offset] = (data[x_offset] & ~0x30) | 0x20
            data[y_offset] = (data[y_offset] & ~0x30) | 0x20
        pending = []

    def read_value(offset: int) -> tuple[int, int, int | None]:
        first = data[offset]
        if first & 0x80:
            value = ((first << 8) | data[offset + 1]) & 0xFFF
            if value & 0x800:
                value -= 0x1000
            return value, offset + 2, offset
        value = first & 0x7F
        if value & 0x40:
            value -= 0x80
        return value, offset + 1, None

    while position < len(data):
        if data[position] == 0xFF:
            finish_travel()
            break
        if data[position:position + 2] == b"\xfe\xb0":
            finish_travel()
            has_sewn = False
            position += 3
            continue
        dx, following, x_offset = read_value(position)
        dy, following, y_offset = read_value(following)
        flag = ((data[x_offset] if x_offset is not None else 0) |
                (data[y_offset] if y_offset is not None else 0)) & 0x30
        if flag:
            if x_offset is None or y_offset is None:
                raise RuntimeError("PEC travel command has inconsistent coordinates.")
            pending.append((x_offset, y_offset, math.hypot(dx, dy)))
        else:
            finish_travel()
            has_sewn = True
        position = following
    else:
        raise RuntimeError("PEC stitch block has no end command.")
    Path(path).write_bytes(data)


def _move(pattern: Any, command: int, point: tuple[float, float]) -> None:
    import pyembroidery

    pattern.add_stitch_absolute(command, point[0], point[1])
    if command in (pyembroidery.TRIM, pyembroidery.COLOR_CHANGE):
        pattern._needs_tie = True
    elif command == pyembroidery.STITCH:
        pattern._needs_tie = False


def _lock_before_trim(pattern: Any, current: tuple[float, float]) -> None:
    """Tie a short backtrack into the last sewn line before cutting thread."""
    import pyembroidery

    if not pattern.stitches or int(pattern.stitches[-1][2]) != pyembroidery.STITCH:
        return
    for x, y, command in reversed(pattern.stitches[:-1]):
        if int(command) != pyembroidery.STITCH:
            break
        distance = math.hypot(current[0] - x, current[1] - y)
        if distance >= MIN_STITCH_MM * 10:
            fraction = min(6.0 / distance, 1.0)
            anchor = (current[0] + (x - current[0]) * fraction,
                      current[1] + (y - current[1]) * fraction)
            for point in (anchor, current, anchor, current):
                _move(pattern, pyembroidery.STITCH, point)
            return


def _jump_to(pattern: Any, start: tuple[float, float] | None,
             target: tuple[float, float]) -> None:
    """Split travel moves below the DST per-record movement limit."""
    import pyembroidery

    origin = start or (0.0, 0.0)
    distance = math.hypot(target[0] - origin[0], target[1] - origin[1])
    steps = max(1, math.ceil(distance / 115.0))
    for step in range(1, steps + 1):
        fraction = step / steps
        point = (origin[0] + (target[0] - origin[0]) * fraction,
                 origin[1] + (target[1] - origin[1]) * fraction)
        _move(pattern, pyembroidery.JUMP, point)


def _fill_pass(pattern: Any, component: np.ndarray, *, horizontal: bool,
               step_px: int, units_per_px: float, max_stitch_mm: float,
               current: tuple[float, float] | None,
               offset_x: int = 0, offset_y: int = 0) -> tuple[float, float] | None:
    """Sew a serpentine raster fill; short row transitions stay under the fill."""
    import pyembroidery
    import numpy as np

    ys, xs = np.nonzero(component)
    if not len(xs):
        return current
    low, high = (int(ys.min()), int(ys.max())) if horizontal else (int(xs.min()), int(xs.max()))
    scan_positions = list(range(low, high + 1, max(1, step_px)))
    reverse_scan = False
    max_step = max_stitch_mm * 10.0

    def inside_connector(start: tuple[float, float], end: tuple[float, float]) -> bool:
        """A short travel stitch is safe only when it stays inside this color."""
        sx, sy = start[0] / units_per_px - offset_x, start[1] / units_per_px - offset_y
        ex, ey = end[0] / units_per_px - offset_x, end[1] / units_per_px - offset_y
        samples = max(2, math.ceil(math.hypot(ex - sx, ey - sy) * 2))
        for index in range(samples + 1):
            fraction = index / samples
            x = round(sx + (ex - sx) * fraction)
            y = round(sy + (ey - sy) * fraction)
            if (not 0 <= x < component.shape[1] or not 0 <= y < component.shape[0]
                    or not component[y, x]):
                return False
        return True

    def inside_route(start: tuple[float, float], end: tuple[float, float]
                     ) -> list[tuple[float, float]] | None:
        """Route short travel under the same fill instead of trimming thread."""
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        if distance < MIN_STITCH_MM * 10:
            return []
        if distance <= max_step and inside_connector(start, end):
            return [end]
        if distance > 80:  # Long hidden travel would distort the fill.
            return None
        origin = (round(start[0] / units_per_px - offset_x),
                  round(start[1] / units_per_px - offset_y))
        target = (round(end[0] / units_per_px - offset_x),
                  round(end[1] / units_per_px - offset_y))
        for x, y in (origin, target):
            if (not 0 <= x < component.shape[1] or not 0 <= y < component.shape[0]
                    or not component[y, x]):
                return None
        frontier = [(0.0, 0.0, origin)]
        cost = {origin: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1))
        while frontier and len(cost) < 3000:
            _, route_length, point = heappop(frontier)
            if route_length != cost[point]:
                continue
            if point == target:
                break
            for dx, dy in neighbors:
                next_point = (point[0] + dx, point[1] + dy)
                x, y = next_point
                if (not 0 <= x < component.shape[1] or not 0 <= y < component.shape[0]
                        or not component[y, x]):
                    continue
                new_cost = route_length + math.hypot(dx, dy)
                if new_cost > 30 or new_cost >= cost.get(next_point, float("inf")):
                    continue
                cost[next_point] = new_cost
                parent[next_point] = point
                estimate = math.hypot(target[0] - x, target[1] - y)
                heappush(frontier, (new_cost + estimate, new_cost, next_point))
        if target not in cost:
            return None
        route = [target]
        while route[-1] != origin:
            route.append(parent[route[-1]])
        route.reverse()
        positions = [((x + offset_x) * units_per_px,
                      (y + offset_y) * units_per_px) for x, y in route]
        positions[0], positions[-1] = start, end
        stitches = []
        index = 0
        while index < len(positions) - 1:
            last = index + 1
            for candidate in range(index + 2, len(positions)):
                if (math.hypot(positions[candidate][0] - positions[index][0],
                               positions[candidate][1] - positions[index][1]) > max_step
                        or not inside_connector(positions[index], positions[candidate])):
                    break
                last = candidate
            stitches.append(positions[last])
            index = last
        return stitches

    for scan in scan_positions:
        line = component[scan, :] if horizontal else component[:, scan]
        intervals = _runs(line)
        if reverse_scan:
            intervals.reverse()
        for start, end in intervals:
            if horizontal:
                a = (start + offset_x, scan + offset_y)
                b = (end + offset_x, scan + offset_y)
            else:
                a = (scan + offset_x, start + offset_y)
                b = (scan + offset_x, end + offset_y)
            if reverse_scan:
                a, b = b, a
            first = (a[0] * units_per_px, a[1] * units_per_px)
            last = (b[0] * units_per_px, b[1] * units_per_px)
            distance = math.hypot(last[0] - first[0], last[1] - first[1])
            if distance < MIN_STITCH_MM * 10:
                continue
            if current is None:
                _jump_to(pattern, None, first)
            else:
                travel = math.hypot(first[0] - current[0], first[1] - current[1])
                route = inside_route(current, first)
                if route is not None:
                    for point in route:
                        _move(pattern, pyembroidery.STITCH, point)
                else:
                    if travel > MAX_UNTRIMMED_TRAVEL_MM * 10:
                        _lock_before_trim(pattern, current)
                        _move(pattern, pyembroidery.TRIM, current)
                    _jump_to(pattern, current, first)

            if getattr(pattern, "_needs_tie", True):
                fraction = min(6.0 / distance, 1.0)
                anchor = (first[0] + (last[0] - first[0]) * fraction,
                          first[1] + (last[1] - first[1]) * fraction)
                for point in (anchor, first, anchor, first):
                    _move(pattern, pyembroidery.STITCH, point)
            segments = max(1, math.ceil(distance / max_step))
            for segment in range(1, segments + 1):
                fraction = segment / segments
                point = (first[0] + (last[0] - first[0]) * fraction,
                         first[1] + (last[1] - first[1]) * fraction)
                _move(pattern, pyembroidery.STITCH, point)
            current = last
        reverse_scan = not reverse_scan
    return current


def create_stitch_file(image: Image.Image, output_path: str | Path,
                       format_name: str = "DST", max_dimension_mm: float = 90.0,
                       max_stitch_mm: float = FILL_MAX_STITCH_MM,
                       row_spacing_mm: float = FILL_SPACING_MM) -> dict[str, float | int | list[tuple[int, int, int]]]:
    """Convert color regions into satin-free tatami fills and verify the written file.

    The image should already be background-removed and k-means quantized. Each
    raster pixel becomes at most one color region; transparent pixels stay empty.
    """
    try:
        import pyembroidery
        import numpy as np
        from skimage.measure import label
    except ImportError as error:
        raise RuntimeError("Install project requirements to create embroidery files.") from error

    if max_dimension_mm <= 0 or not MIN_STITCH_MM <= max_stitch_mm <= FILL_MAX_STITCH_MM:
        raise ValueError("Use a positive size and maximum stitch from 0.4 to 4 mm.")
    if not 0.3 <= row_spacing_mm <= 0.5:
        raise ValueError("Fill row spacing must be between 0.3 and 0.5 mm.")
    writers = {"DST": pyembroidery.write_dst, "PES": pyembroidery.write_pes,
               "JEF": pyembroidery.write_jef}
    readers = {"DST": pyembroidery.read_dst, "PES": pyembroidery.read_pes,
               "JEF": pyembroidery.read_jef}
    format_name = format_name.upper()
    if format_name not in writers:
        raise ValueError(f"Unsupported embroidery format: {format_name}")

    rgba = image.convert("RGBA")
    # Semi-transparent segmentation halos cannot be represented by thread.
    alpha = rgba.getchannel("A").point(lambda value: 255 if value >= 128 else 0)
    rgba.putalpha(alpha)
    bounds = rgba.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("No foreground pixels remain after removing the background.")
    rgba = rgba.crop(bounds)
    width, height = rgba.size
    target_long_edge = max(2, round(max_dimension_mm / row_spacing_mm))
    scale = target_long_edge / max(width, height)
    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgba = rgba.resize(target_size, Image.Resampling.NEAREST)
    pixels = np.asarray(rgba)
    height, width = pixels.shape[:2]
    units_per_px = max_dimension_mm * 10.0 / max(width, height)

    visible = pixels[:, :, 3] > 0
    candidates = [tuple(map(int, color)) for color, count in Counter(
        map(tuple, pixels[visible, :3].tolist())
    ).most_common() if count >= MIN_COMPONENT_PIXELS]
    if not candidates:
        raise ValueError("No stitchable color regions remain after cleanup.")

    from scipy.ndimage import find_objects

    color_layers = []
    for color in candidates:
        color_mask = visible & np.all(pixels[:, :, :3] == color, axis=2)
        components = label(color_mask, connectivity=2)
        component_ids, counts = np.unique(components[components > 0], return_counts=True)
        slices = find_objects(components)
        kept = [(int(component_id), slices[int(component_id) - 1])
                for component_id, count in zip(component_ids, counts)
                if count >= MIN_COMPONENT_PIXELS]
        if kept:
            color_layers.append((color, components, kept))
    colors = [layer[0] for layer in color_layers]
    if not colors:
        raise ValueError("No foreground region is large enough to embroider.")

    pattern = pyembroidery.EmbPattern()
    for color in colors:
        thread = pyembroidery.EmbThread()
        thread.color = (color[0] << 16) | (color[1] << 8) | color[2]
        thread.description = _color_key(color)
        pattern.add_thread(thread)

    current: tuple[float, float] | None = None
    for color_index, (color, components, kept) in enumerate(color_layers):
        if color_index:
            if current is not None:
                _lock_before_trim(pattern, current)
                _move(pattern, pyembroidery.TRIM, current)
                _move(pattern, pyembroidery.COLOR_CHANGE, current)
        pending = list(kept)
        while pending:
            if current is None:
                anchor = (0.0, 0.0)
            else:
                anchor = (current[0] / units_per_px, current[1] / units_per_px)
            component_id, bounds = min(
                pending,
                key=lambda item: math.hypot(
                    (item[1][1].start + item[1][1].stop - 1) / 2 - anchor[0],
                    (item[1][0].start + item[1][0].stop - 1) / 2 - anchor[1],
                ),
            )
            pending.remove((component_id, bounds))
            y_slice, x_slice = bounds
            component = components[y_slice, x_slice] == component_id
            # Light cross underlay first, then denser top fill at right angles.
            current = _fill_pass(
                pattern, component, horizontal=True,
                step_px=max(1, round(UNDERLAY_SPACING_MM / (max_dimension_mm / max(width, height)))),
                units_per_px=units_per_px, max_stitch_mm=max_stitch_mm, current=current,
                offset_x=x_slice.start, offset_y=y_slice.start,
            )
            current = _fill_pass(
                pattern, component, horizontal=False, step_px=1,
                units_per_px=units_per_px, max_stitch_mm=max_stitch_mm, current=current,
                offset_x=x_slice.start, offset_y=y_slice.start,
            )

    if not colors or current is None:
        raise ValueError("No stitches were generated.")
    _lock_before_trim(pattern, current)
    _move(pattern, pyembroidery.TRIM, current)
    _move(pattern, pyembroidery.END, current)
    if format_name == "PES":
        pyembroidery.write_pes(pattern, str(output_path), settings={"version": 6})
    else:
        writers[format_name](pattern, str(output_path))
    if format_name == "PES":
        _normalize_pes_jump_flags(output_path)

    parsed = readers[format_name](str(output_path))
    if format_name == "PES":
        actual_colors = [thread.color & 0xFFFFFF for thread in parsed.threadlist]
        expected_colors = [(red << 16) | (green << 8) | blue
                           for red, green, blue in colors]
        if actual_colors[:len(colors)] != expected_colors:
            Path(output_path).unlink(missing_ok=True)
            raise RuntimeError("PES file did not preserve the thread color order.")
    previous = (0.0, 0.0)
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    stitches = jumps = trims = color_changes = 0
    longest_stitch = longest_jump = 0.0
    shortest_stitch = float("inf")
    zero_commands = 0
    for x, y, command in parsed.stitches:
        command = int(command)
        point = (float(x), float(y))
        if command in (pyembroidery.STITCH, pyembroidery.JUMP, pyembroidery.TRIM):
            distance = math.hypot(point[0] - previous[0], point[1] - previous[1]) / 10.0
            if command == pyembroidery.STITCH:
                stitches += 1
                longest_stitch = max(longest_stitch, distance)
                if distance == 0 and format_name in ("PES", "JEF"):
                    # These formats insert a zero-move command after travel.
                    zero_commands += 1
                else:
                    shortest_stitch = min(shortest_stitch, distance)
            elif command == pyembroidery.JUMP:
                jumps += 1
                longest_jump = max(longest_jump, distance)
            else:
                trims += 1
            previous = point
            min_x, max_x = min(min_x, point[0]), max(max_x, point[0])
            min_y, max_y = min(min_y, point[1]), max(max_y, point[1])
        elif command == pyembroidery.COLOR_CHANGE:
            color_changes += 1

    bounds_mm = ((max_x - min_x) / 10.0, (max_y - min_y) / 10.0)
    if stitches == 0:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError("The embroidery file contains no needle-down stitches.")
    if longest_stitch > max_stitch_mm + 0.11:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"Longest stitch is {longest_stitch:.2f} mm, above {max_stitch_mm:.2f} mm.")
    if shortest_stitch < MIN_STITCH_MM - 0.05:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"Shortest stitch is {shortest_stitch:.2f} mm, below {MIN_STITCH_MM:.2f} mm.")
    if longest_jump > MAX_JUMP_MM + 0.11:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"Longest jump is {longest_jump:.2f} mm, above {MAX_JUMP_MM:.2f} mm.")
    if max(bounds_mm) > max_dimension_mm + 1.0:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"Design bounds {bounds_mm[0]:.1f} x {bounds_mm[1]:.1f} mm exceed the limit.")

    # Check that the needle-down plan actually reaches the vector fill areas.
    # One raster pixel represents roughly one top-fill row of thread.
    from scipy.ndimage import binary_dilation

    covered_image = Image.new("1", (width, height))
    covered_draw = ImageDraw.Draw(covered_image)
    previous_stitch = None
    for x, y, command in parsed.stitches:
        if int(command) == pyembroidery.STITCH:
            point = (round(x / units_per_px), round(y / units_per_px))
            if previous_stitch is not None:
                covered_draw.line((*previous_stitch, *point), fill=1)
            previous_stitch = point
        else:
            previous_stitch = None
    nearby_thread = binary_dilation(np.asarray(covered_image, dtype=bool), iterations=1)
    stitch_coverage = float((nearby_thread & visible).sum() / visible.sum())
    if stitch_coverage < 0.90:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"Only {stitch_coverage:.1%} of the SVG fill has nearby stitches.")
    return {
        "stitches": stitches,
        "jumps": jumps,
        "trims": trims,
        "color_changes": color_changes,
        "width_mm": bounds_mm[0],
        "height_mm": bounds_mm[1],
        "max_stitch_mm": longest_stitch,
        "min_stitch_mm": shortest_stitch,
        "zero_move_commands": zero_commands,
        "max_jump_mm": longest_jump,
        "row_spacing_mm": row_spacing_mm,
        "stitch_coverage": stitch_coverage,
        "palette": colors,
    }


def render_stitch_preview(path: str | Path, format_name: str,
                          palette: list[tuple[int, int, int]],
                          pixels_per_mm: float = 5.0) -> Image.Image:
    """Render only needle-down DST/PES/JEF movements as a thread preview."""
    import pyembroidery

    readers = {"DST": pyembroidery.read_dst, "PES": pyembroidery.read_pes,
               "JEF": pyembroidery.read_jef}
    pattern = readers[format_name.upper()](str(path))
    movements = [(float(x), float(y), int(command)) for x, y, command in pattern.stitches
                 if int(command) in (pyembroidery.STITCH, pyembroidery.JUMP,
                                     pyembroidery.TRIM, pyembroidery.COLOR_CHANGE)]
    if not movements:
        raise ValueError("Cannot preview an empty embroidery file.")
    xs = [movement[0] for movement in movements]
    ys = [movement[1] for movement in movements]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    scale = pixels_per_mm / 10.0
    padding = round(pixels_per_mm * 4)
    size = (max(80, round((max_x - min_x) * scale + padding * 2)),
            max(80, round((max_y - min_y) * scale + padding * 2)))
    result = Image.new("RGB", size, (237, 230, 211))
    draw = ImageDraw.Draw(result)
    current = None
    color_index = 0
    thread = palette[0] if palette else (40, 40, 40)
    line_width = max(1, round(pixels_per_mm * 0.45))
    for x, y, command in movements:
        point = (padding + round((x - min_x) * scale),
                 padding + round((y - min_y) * scale))
        if command == pyembroidery.COLOR_CHANGE:
            color_index = min(color_index + 1, len(palette) - 1) if palette else 0
            if palette:
                thread = palette[color_index]
            current = None
            continue
        if command == pyembroidery.STITCH and current is not None:
            draw.line((current, point), fill=thread, width=line_width)
        current = point if command == pyembroidery.STITCH else None
    return result
