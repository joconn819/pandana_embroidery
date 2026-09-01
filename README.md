# Pandana Embroidery Converter

A desktop Tkinter application that removes a photo background with the local
`rembg` segmentation model, converts a PNG or JPEG photograph into a
colour-reduced SVG, and creates a Tajima DST embroidery input file.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

To install Inkscape and Ink/Stitch, run `./install_inkstitch.sh` and enter
your administrator password in the terminal when prompted. Credentials should
never be placed in the script or shared with the application.

## Build a Windows executable

Build the `.exe` on Windows (PyInstaller does not reliably cross-compile a
Windows executable from Linux):

```bat
build_windows.bat
```

The script creates `dist\PandanaEmbroidery.exe`. Copy that executable to the
Windows computer and run it normally. The first run downloads the local
`rembg` model weights (approximately 1 GB), so the computer needs internet
access once; later runs use the cached model. Windows Defender may briefly
scan a newly built unsigned executable.

Choose a photo, enter the desired number of colours (`k`), and select **Go**.
Set the final width and height in millimetres; editing either value keeps the
photo’s aspect ratio. Set **Max stitch length (mm)** to the target maximum stitch length: smaller
values create a finer, more detailed working image, while larger values reduce
detail and stitch count. Transparent margins are trimmed after segmentation,
so the exported design is cropped to the visible subject.
The selected JEF/DST file contains one thread colour for each k-means colour,
with colour-change commands between colour sections.
After conversion, **See workflow** opens a side-by-side view of the original
photo, background-removed image, k-means result, and SVG result. Select
**Save** to choose where to write the generated embroidery file. JEF is the
default format; DST is also available from the Format selector.
Intermediate SVG and embroidery files are also written next to the source:

The conversion first runs `rembg` (U²-Net) locally. Its transparent mask is
excluded from both the SVG and embroidery stitches, so backgrounds such as the
garden in the puppy sample are not converted into thread. The first run may
download the model weights; subsequent runs use the local cache.

- `<name>_kmeans.svg` contains one colour group per k-means cluster.
- `<name>_embroidery.dst` contains running stitches suitable for embroidery
  software that accepts Tajima DST files.

The conversion is deterministic. Images are resized to a maximum dimension of
240 pixels before vectorization. This conservative cap avoids excessive memory
use from the segmentation model and contour arrays; use a smaller input photo
or increase Max stitch length (mm) if the machine is memory-constrained. Each
quantized colour is converted into
connected contour paths rather than horizontal pixel strips, eliminating the
scanline appearance while keeping the SVG and embroidery geometry connected.
More colours (`k`) preserve more detail; fewer colours produce a cleaner, more
graphic embroidery design.

The code computes a local stitch-direction field from image gradients and uses
it to orient short fill-stitch segments in the generated DST file. The
directional fill is clipped to visible foreground scanlines; the SVG remains
the colour-region preview, while Ink/Stitch is recommended for inspecting
stitch angle, order, and density.