import './style.css'
import { removeDogBackground } from './background.js'

const form = document.querySelector('#convert-form')
const input = document.querySelector('#photo-input')
const dropZone = document.querySelector('#drop-zone')
const fileLabel = document.querySelector('#file-label')
const colorsInput = document.querySelector('#colors')
const sizeInput = document.querySelector('#size')
const button = document.querySelector('#convert-button')
const status = document.querySelector('#status')
const resultPanel = document.querySelector('#results')
const downloadPanel = document.querySelector('#downloads')
const metrics = document.querySelector('#metrics')
const palette = document.querySelector('#palette')
const previews = {
  original: document.querySelector('#original-preview'),
  cutout: document.querySelector('#cutout-preview'),
  stitch: document.querySelector('#stitch-preview'),
}
const assetBase = new URL(import.meta.env.BASE_URL, document.baseURI).href
let currentFile
let worker
let objectUrls = []
let busy = false

function updateStatus(message, isError = false) {
  status.textContent = message
  status.classList.toggle('error', isError)
}

function objectUrl(blob) {
  const url = URL.createObjectURL(blob)
  objectUrls.push(url)
  return url
}

function showPreview(image, blob) {
  image.src = objectUrl(blob)
  image.hidden = false
  image.nextElementSibling.hidden = true
}

function selectFile(file) {
  if (!file || busy) return
  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    updateStatus('Choose a JPG or PNG photo.', true)
    return
  }
  if (file.size > 20 * 1024 * 1024) {
    updateStatus('The photo must be under 20 MB.', true)
    return
  }
  for (const url of objectUrls) URL.revokeObjectURL(url)
  objectUrls = []
  for (const image of [previews.cutout, previews.stitch]) {
    image.hidden = true
    image.nextElementSibling.hidden = false
  }
  currentFile = file
  fileLabel.textContent = file.name
  showPreview(previews.original, file)
  resultPanel.hidden = true
  updateStatus('Ready to create your design.')
}

input.addEventListener('change', () => selectFile(input.files?.[0]))
for (const name of ['dragenter', 'dragover']) {
  dropZone.addEventListener(name, event => {
    event.preventDefault()
    dropZone.classList.add('dragging')
  })
}
for (const name of ['dragleave', 'drop']) {
  dropZone.addEventListener(name, event => {
    event.preventDefault()
    dropZone.classList.remove('dragging')
  })
}
dropZone.addEventListener('drop', event => selectFile(event.dataTransfer?.files?.[0]))

function digitize(image, colors, sizeMm) {
  if (!worker) worker = new Worker(new URL('./digitize-worker.js', import.meta.url), { type: 'module' })
  return new Promise((resolve, reject) => {
    worker.onmessage = event => {
      if (event.data.type === 'status') updateStatus(event.data.message)
      if (event.data.type === 'done') resolve(event.data)
      if (event.data.type === 'error') reject(new Error(event.data.message))
    }
    worker.onerror = event => {
      worker.terminate()
      worker = null
      reject(new Error(event.message || 'Embroidery engine failed.'))
    }
    worker.postMessage({ image, assetBase, colors, sizeMm }, [image])
  })
}

function showResults(result, filename) {
  const stem = filename.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]+/g, '-') || 'dog'
  const files = Object.fromEntries(result.files.map(item => [item.name, item.bytes]))
  const preview = new Blob([files['input-dst-preview.png']], { type: 'image/png' })
  showPreview(previews.stitch, preview)
  const dst = result.report.designs.DST
  metrics.textContent = `${dst.stitches.toLocaleString()} stitches · ${dst.width_mm.toFixed(1)} × ${dst.height_mm.toFixed(1)} mm · ${result.report.actual_colors} colors`
  palette.replaceChildren()
  for (const [index, color] of result.report.thread_colors.entries()) {
    const chip = document.createElement('span')
    chip.className = 'thread-chip'
    chip.title = `Thread ${index + 1}: ${color}`
    const swatch = document.createElement('i')
    swatch.style.backgroundColor = color
    chip.append(swatch, document.createTextNode(`${index + 1}. ${color}`))
    palette.append(chip)
  }
  downloadPanel.replaceChildren()
  for (const [label, name, suffix, type] of [
    ['Download DST', 'input.dst', 'dst', 'application/octet-stream'],
    ['Download PES', 'input.pes', 'pes', 'application/octet-stream'],
    ['Download SVG', 'input.svg', 'svg', 'image/svg+xml'],
    ['Thread chart', 'input-thread-colors.txt', 'txt', 'text/plain'],
    ['Download preview', 'input-dst-preview.png', 'png', 'image/png'],
  ]) {
    if (!files[name]) continue
    const link = document.createElement('a')
    link.className = 'download-button'
    link.textContent = label
    link.href = objectUrl(new Blob([files[name]], { type }))
    link.download = `${stem}-embroidery.${suffix}`
    downloadPanel.append(link)
  }
  resultPanel.hidden = false
}

form.addEventListener('submit', async event => {
  event.preventDefault()
  if (!currentFile) {
    updateStatus('Choose a dog photo first.', true)
    return
  }
  const colors = colorsInput.valueAsNumber
  const sizeMm = sizeInput.valueAsNumber
  if (!Number.isInteger(colors) || colors < 1 || colors > 15 ||
      !Number.isFinite(sizeMm) || sizeMm < 10 || sizeMm > 300) {
    updateStatus('Enter 1–15 thread colors and a longest side from 10 to 300 mm.', true)
    return
  }
  busy = true
  button.disabled = true
  resultPanel.hidden = true
  try {
    updateStatus('Preparing the photo…')
    const cutout = await removeDogBackground(currentFile, updateStatus)
    showPreview(previews.cutout, cutout)
    const result = await digitize(
      await cutout.arrayBuffer(),
      colors,
      sizeMm,
    )
    showResults(result, currentFile.name)
    updateStatus('Your embroidery files are ready.')
  } catch (error) {
    updateStatus(error.message || 'Conversion failed.', true)
  } finally {
    busy = false
    button.disabled = false
  }
})
