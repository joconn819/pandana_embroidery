import * as ort from 'onnxruntime-web/wasm'

const INPUT_SIZE = 320
// Keep enough source detail for narrow features such as ears, tails, and legs.
// The segmentation model still runs at 320px, but we resample its mask back to
// this larger source before vectorization and stitch planning.
const PROCESS_SIZE = 512
const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]
let modelSession

function canvas(width, height) {
  const result = document.createElement('canvas')
  result.width = width
  result.height = height
  return result
}

async function toBlob(element) {
  return new Promise((resolve, reject) => {
    element.toBlob(blob => blob ? resolve(blob) : reject(new Error('Could not encode PNG')), 'image/png')
  })
}

async function session() {
  if (!modelSession) {
    ort.env.wasm.numThreads = 1
    modelSession = ort.InferenceSession.create(`${import.meta.env.BASE_URL}model/u2netp.onnx`, {
      executionProviders: ['wasm'],
    })
  }
  return modelSession
}

export async function removeDogBackground(file, status = () => {}) {
  const bitmap = await createImageBitmap(file)
  const scale = Math.min(1, PROCESS_SIZE / Math.max(bitmap.width, bitmap.height))
  const width = Math.max(1, Math.round(bitmap.width * scale))
  const height = Math.max(1, Math.round(bitmap.height * scale))
  const source = canvas(width, height)
  const context = source.getContext('2d', { willReadFrequently: true })
  context.imageSmoothingQuality = 'high'
  context.drawImage(bitmap, 0, 0, width, height)
  bitmap.close()
  const sourcePixels = context.getImageData(0, 0, width, height)

  // A prepared transparent PNG can enter the same Python digitizer directly.
  let hasTransparency = false
  for (let index = 3; index < sourcePixels.data.length; index += 4) {
    if (sourcePixels.data[index] < 255) { hasTransparency = true; break }
  }
  if (!hasTransparency) {
    status('Loading the local background model…')
    const model = await session()
    const resized = canvas(INPUT_SIZE, INPUT_SIZE)
    const resizedContext = resized.getContext('2d', { willReadFrequently: true })
    resizedContext.imageSmoothingQuality = 'high'
    resizedContext.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE)
    const pixels = resizedContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data
    let maxValue = 1
    for (let index = 0; index < pixels.length; index += 4) {
      maxValue = Math.max(maxValue, pixels[index], pixels[index + 1], pixels[index + 2])
    }
    const input = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE)
    const plane = INPUT_SIZE * INPUT_SIZE
    for (let index = 0; index < plane; index++) {
      for (let channel = 0; channel < 3; channel++) {
        input[channel * plane + index] =
          (pixels[index * 4 + channel] / maxValue - MEAN[channel]) / STD[channel]
      }
    }
    status('Removing the background on this device…')
    const output = await model.run({
      [model.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    })
    const scores = output[model.outputNames[0]].data
    let minimum = Infinity
    let maximum = -Infinity
    for (const value of scores) {
      minimum = Math.min(minimum, value)
      maximum = Math.max(maximum, value)
    }
    if (!(maximum > minimum)) throw new Error('The background model returned an empty mask.')
    const mask = canvas(INPUT_SIZE, INPUT_SIZE)
    const maskContext = mask.getContext('2d')
    const maskPixels = maskContext.createImageData(INPUT_SIZE, INPUT_SIZE)
    for (let index = 0; index < scores.length; index++) {
      const value = Math.round(255 * (scores[index] - minimum) / (maximum - minimum))
      maskPixels.data[index * 4] = value
      maskPixels.data[index * 4 + 1] = value
      maskPixels.data[index * 4 + 2] = value
      maskPixels.data[index * 4 + 3] = 255
    }
    maskContext.putImageData(maskPixels, 0, 0)
    const resampled = canvas(width, height)
    const resampledContext = resampled.getContext('2d', { willReadFrequently: true })
    resampledContext.imageSmoothingQuality = 'high'
    resampledContext.drawImage(mask, 0, 0, width, height)
    const alpha = resampledContext.getImageData(0, 0, width, height).data
    for (let index = 0; index < width * height; index++) {
      sourcePixels.data[index * 4 + 3] = alpha[index * 4]
    }
    context.putImageData(sourcePixels, 0, 0)
  }
  return toBlob(source)
}
