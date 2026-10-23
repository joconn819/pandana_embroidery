import { createHash } from 'node:crypto'
import { createWriteStream } from 'node:fs'
import { copyFile, mkdir, readFile, rm, stat } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join, resolve } from 'node:path'
import { pipeline } from 'node:stream/promises'

const here = dirname(fileURLToPath(import.meta.url))
const web = resolve(here, '..')
const repo = resolve(web, '..')
const modelDirectory = join(web, 'public', 'model')
const modelPath = join(modelDirectory, 'u2netp.onnx')
const expectedSha256 = '309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8'
const modelUrl = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx'

for (const name of ['app.py', 'stitch_generator.py', 'convert.py']) {
  const destination = join(web, 'public', 'python', name)
  await mkdir(dirname(destination), { recursive: true })
  await copyFile(join(repo, name), destination)
}

await rm(join(web, 'public', 'ort'), { recursive: true, force: true })
await rm(join(web, 'public', 'pyodide'), { recursive: true, force: true })

async function hashFile(path) {
  const bytes = await readFile(path)
  return createHash('sha256').update(bytes).digest('hex')
}

async function ensureFile(path, url, expectedHash) {
  await mkdir(dirname(path), { recursive: true })
  let valid = false
  try {
    await stat(path)
    valid = await hashFile(path) === expectedHash
  } catch { /* Download below. */ }
  if (!valid) {
    const response = await fetch(url)
    if (!response.ok || !response.body) {
      throw new Error(`Could not download ${url}: HTTP ${response.status}`)
    }
    await pipeline(response.body, createWriteStream(path))
    if (await hashFile(path) !== expectedHash) {
      throw new Error(`Downloaded ${url} failed SHA-256 verification`)
    }
  }
}

await ensureFile(modelPath, modelUrl, expectedSha256)
