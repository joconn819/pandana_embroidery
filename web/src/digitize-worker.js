const pyodideBase = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'
let pythonPromise

async function python(assetBase) {
  if (!pythonPromise) {
    pythonPromise = (async () => {
      self.postMessage({ type: 'status', message: 'Loading the embroidery engine…' })
      const { loadPyodide } = await import(`${pyodideBase}pyodide.mjs`)
      const runtime = await loadPyodide({ indexURL: pyodideBase })
      await runtime.loadPackage(['numpy', 'scipy', 'Pillow', 'scikit-image', 'micropip'])
      const micropip = runtime.pyimport('micropip')
      await micropip.install('pyembroidery==1.5.1')
      runtime.FS.mkdirTree('/workspace')
      for (const name of ['app.py', 'stitch_generator.py', 'convert.py']) {
        const response = await fetch(`${assetBase}python/${name}`)
        if (!response.ok) throw new Error(`Could not load ${name}`)
        runtime.FS.writeFile(`/workspace/${name}`, await response.text())
      }
      runtime.runPython("import sys; sys.path.insert(0, '/workspace')")
      return runtime
    })()
  }
  return pythonPromise
}

self.onmessage = async event => {
  const { image, assetBase, colors, sizeMm } = event.data
  try {
    const runtime = await python(assetBase)
    self.postMessage({ type: 'status', message: 'Tracing colors and planning stitches…' })
    runtime.FS.writeFile('/workspace/input.png', new Uint8Array(image))
    runtime.FS.mkdirTree('/workspace/output')
    runtime.globals.set('settings_json', JSON.stringify({ colors, size_mm: sizeMm }))
    runtime.runPython(`
import json
from pathlib import Path
from convert import convert_photo
settings = json.loads(settings_json)
convert_photo(Path('/workspace/input.png'), Path('/workspace/output'),
              colors=settings['colors'], size_mm=settings['size_mm'],
              formats=('DST', 'PES'), process_pixels=512)
`)
    const names = runtime.FS.readdir('/workspace/output').filter(name => !name.startsWith('.'))
    const files = names.map(name => ({
      name,
      bytes: runtime.FS.readFile(`/workspace/output/${name}`).slice().buffer,
    }))
    const report = JSON.parse(new TextDecoder().decode(
      runtime.FS.readFile('/workspace/output/input-metrics.json')))
    self.postMessage({ type: 'done', files, report }, files.map(file => file.bytes))
  } catch (error) {
    pythonPromise = null
    self.postMessage({ type: 'error', message: error?.message || String(error) })
  }
}
