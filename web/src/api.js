/**
 * Thin wrapper over the FastAPI JSON API.
 *
 * Every field name below is taken from app/main.py and app/db.py rather than
 * guessed, so the shapes here track the backend exactly. In dev the Vite proxy
 * forwards /api to uvicorn; in the built bundle the app is served from the same
 * origin, so a relative base works in both cases.
 */

const BASE = '/api'

class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function qs(params) {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params || {})) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

async function request(path, options = {}) {
  const response = await fetch(BASE + path, options)
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body && body.detail) detail = body.detail
    } catch {
      /* a non JSON error body is not worth a second failure */
    }
    throw new ApiError(detail, response.status)
  }
  return response.json()
}

export const api = {
  health: () => request('/health'),
  stats: () => request('/stats'),
  documents: () => request('/documents'),
  documentStatus: (docId) => request(`/documents/${docId}/status`),
  facts: (params) => request(`/facts${qs(params)}`),
  fact: (factId) => request(`/facts/${factId}`),
  links: (params) => request(`/links${qs(params)}`),
  link: (linkId) => request(`/links/${linkId}`),
  quarantine: (params) => request(`/quarantine${qs(params)}`),
  registry: () => request('/registry'),

  upload(file, collection, onProgress) {
    // XHR rather than fetch, because upload progress events are the point.
    return new Promise((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      form.append('collection', collection || '')

      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/documents`)
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(event.loaded / event.total)
        }
      })
      xhr.addEventListener('load', () => {
        let body = null
        try {
          body = JSON.parse(xhr.responseText)
        } catch {
          /* handled below */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body)
        else reject(new ApiError((body && body.detail) || `upload failed (${xhr.status})`, xhr.status))
      })
      xhr.addEventListener('error', () => reject(new ApiError('network error during upload', 0)))
      xhr.send(form)
    })
  },
}

export const RELATIONS = [
  'corroborates',
  'contradicts',
  'reconciled_by_period',
  'reconciled_by_scope',
  'reconciled_by_unit',
  'reconciled_by_vintage',
  'refines',
  'unrelated',
]

export const PIPELINE_STAGES = [
  'parse',
  'chunk',
  'docmeta',
  'extract',
  'entities',
  'embed',
  'link',
  'adjudicate',
]

/** Glyphs stand in for the colour this monochrome palette cannot spend. */
export const RELATION_GLYPH = {
  corroborates: '=',
  contradicts: '≠',
  refines: '⊂',
  reconciled_by_period: '◴',
  reconciled_by_scope: '◫',
  reconciled_by_unit: '×',
  reconciled_by_vintage: '↻',
  unrelated: '·',
}

export const RELATION_LABEL = {
  corroborates: 'corroborates',
  contradicts: 'contradicts',
  refines: 'refines',
  reconciled_by_period: 'period',
  reconciled_by_scope: 'scope',
  reconciled_by_unit: 'unit',
  reconciled_by_vintage: 'vintage',
  unrelated: 'unrelated',
}

/** The relations that mean "this looked like a conflict but is not one". */
export const RECONCILED = RELATIONS.filter((r) => r.startsWith('reconciled_by_'))

export { ApiError }
