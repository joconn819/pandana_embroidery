# Pandana Embroidery Converter

Convert a dog photo into a background-free, six-color SVG and then into DST or
PES fill stitches. The included Tkinter app is for interactive use; `convert.py`
offers the same pipeline from a terminal. The SVG is the geometry used to
generate the machine files.

## Install and run

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python convert.py dog.jpg --output-dir output
# or launch the desktop app:
python app.py
```

Windows users can build a local executable with `build_windows.bat`. Build on
Windows; PyInstaller does not reliably cross-compile a Windows executable from
Linux.

`rembg` performs background removal locally. The default `u2netp` model is
about 4.6 MB and ran much faster than the larger model on the three dog photos.
`--model bria-rmbg` can give cleaner fur edges, but downloads about 1 GB and
can take tens of seconds on CPU. The [BRIA RMBG-2.0 weights](https://huggingface.co/briaai/RMBG-2.0)
are licensed for noncommercial use unless you have a separate agreement; see
the model's license before selecting it for a product. Model sessions are reused while
the program runs. The desktop app offers the same model choice and runs
conversion in a worker thread so the window remains responsive. A PNG that
already has transparency skips model inference.

## Output

For `dog.jpg`, the command writes into `output/dog/`:

| File | Meaning |
| --- | --- |
| `dog-cutout.png` | Subject with transparent background |
| `dog-quantized.png` | Cleaned K-means color regions |
| `dog.svg` | Filled vector regions with physical stitch parameters |
| `dog-svg.png` | SVG geometry rendered for inspection |
| `dog.dst`, `dog.pes` | Machine embroidery files |
| `dog-dst-preview.png`, `dog-pes-preview.png` | Preview rendered from the written stitch files |
| `dog-thread-colors.txt` | Thread order for DST machines, which do not store RGB colors |
| `dog-metrics.json` | Measured dimensions, travel, stitch count, and SVG fidelity |

Use `--colors 8`, `--size-mm 110`, or `--max-stitch-mm 3.5` to change the
design. Each color is sewn as one thread block. Six colors and a 90 mm longest
dimension are the photo preset. The desktop app writes its outputs beside the
selected image; **See workflow** displays the original, cutout, color regions,
and actual stitch preview.

## Stitch requirements and checks

The current fill profile uses 0.4 mm spacing between top rows, a perpendicular
underlay at 0.8 mm spacing, stitches from 0.4 to 4 mm, jumps no longer than
12 mm, and lock stitches at thread starts and trims. Travel inside a color
region is routed beneath its fill; longer moves use jumps and trims. PES version
6 preserves the RGB thread colors; DST users should follow the separate thread
chart. Dimensions,
maximum stitch and jump length, color changes, and counts are checked after
reading the written DST/PES file back. At least 90% of vector fill pixels must
have a needle-down path within one fill-row pixel. PES jump flags are adjusted after
`pyembroidery` export so short jumps do not all become trims.

Top stitches follow a local texture angle estimated from the background-removed
photo before K-means removes its fine detail. A structure tensor measures the
dominant line orientation in 24-pixel areas, rounded to 30-degree increments.
The angle is saved on each SVG fill path and used by the DST/PES generator;
underlay runs across that angle. Areas with weak or conflicting texture use a
90-degree fill. The metrics file lists the number of SVG pixels at each angle.
This estimates visible line orientation, not the biological root-to-tip
direction of fur. Very smooth, blurred, or tiny image areas cannot provide a
reliable direction; inspect the preview and adjust the SVG angles manually if
a specific coat flow matters.

The converter checks that the traced SVG retains at least 90% of the dog's
silhouette by intersection-over-union and that at least 85% of image pixels
match their quantized colors exactly. These checks catch major tracing loss;
they do not prove that a photo will be recognizable. Inspect the stitch preview
before sewing. Physical coverage, thread color, fabric stretch, hoop size, and
stabilizer also matter, so make a sample stitch-out before using a design on a
finished item.

## GitHub Pages website

`web/` is a static upload site. It runs the small U2NetP ONNX model in the
browser, then uses Pyodide to run these same Python color, SVG, and stitch
modules on the transparent image. Uploaded photos are processed on the
visitor's device. A first visit downloads the 4.6 MB model, the WebAssembly
inference runtime, Pyodide, and Python packages, so it takes longer than later
visits. The generated DST and PES files are checked in Python and offered as
downloads alongside an SVG and stitch preview.

```bash
cd web
npm ci
npm run dev
npm run build
```

The production static files are in `web/dist/`. The build fetches the U2NetP
model with a pinned SHA-256 and copies the Python modules into the site. The
GitHub Pages Actions workflow builds and publishes this directory after it is
merged to `main` and Pages is configured to use GitHub Actions. The website is
not currently hosted by this repository branch.

Browser components: [U-2-Net model and Apache 2.0 license](https://github.com/xuebinqin/U-2-Net),
[rembg model conversion](https://github.com/danielgatis/rembg),
[ONNX Runtime Web](https://github.com/microsoft/onnxruntime),
[Pyodide](https://github.com/pyodide/pyodide), and
[pyembroidery](https://github.com/EmbroidePy/pyembroidery).
