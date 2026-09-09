import { useCallback, useMemo, useRef, useState } from 'react'
import { api, PIPELINE_STAGES } from '../api'
import { useApi, usePolling } from '../hooks'
import {
  Empty,
  ErrorState,
  Icon,
  Panel,
  Reveal,
  SectionHead,
  Skeleton,
  Stat,
  compact,
  shortDate,
  useToast,
} from '../components/ui'

/** The stage line under a document that is mid-ingest. */
function StageLine({ current }) {
  const activeIndex = PIPELINE_STAGES.indexOf(current)
  return (
    <div className="stage-line">
      {PIPELINE_STAGES.map((stage, index) => {
        const state =
          activeIndex < 0 ? 'idle' : index < activeIndex ? 'done' : index === activeIndex ? 'active' : 'idle'
        return (
          <span key={stage} className="stage-dot" data-state={state}>
            <i />
            {stage}
          </span>
        )
      })}
    </div>
  )
}

function Dropzone({ collections, onUploaded }) {
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [collection, setCollection] = useState('')
  const inputRef = useRef(null)
  const toast = useToast()

  const send = useCallback(
    async (files) => {
      const pdfs = Array.from(files || []).filter((file) => file.name.toLowerCase().endsWith('.pdf'))
      if (!pdfs.length) {
        toast('Only PDF files are accepted', 'error')
        return
      }
      setBusy(true)
      try {
        for (const file of pdfs) {
          const result = await api.upload(file, collection)
          toast(
            result.already_ingested
              ? `${file.name} was already ingested — re-running costs no model calls`
              : `${file.name} queued for ingest`,
          )
        }
        onUploaded()
      } catch (error) {
        toast(error.message || 'Upload failed', 'error')
      } finally {
        setBusy(false)
      }
    },
    [collection, onUploaded, toast],
  )

  return (
    <div>
      <div
        className="dropzone"
        data-drag={dragging}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            inputRef.current?.click()
          }
        }}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          send(event.dataTransfer.files)
        }}
      >
        <div className="dropzone-icon">
          <Icon.upload size={17} />
        </div>
        <div style={{ fontSize: 13.5, marginBottom: 4 }}>
          {dragging ? 'Release to ingest' : 'Drop a PDF here, or click to choose'}
        </div>
        <div className="faint" style={{ fontSize: 12 }}>
          Parsed, chunked, extracted and linked in the background. Re-uploading the same bytes rewrites the
          same rows and makes zero model calls.
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(event) => {
            send(event.target.files)
            event.target.value = ''
          }}
        />
      </div>

      <div className="controls" style={{ marginTop: 12, marginBottom: 0 }}>
        <input
          className="input"
          list="collection-options"
          placeholder="Collection (defaults to uploads)"
          value={collection}
          onChange={(event) => setCollection(event.target.value)}
          style={{ width: 260 }}
        />
        <datalist id="collection-options">
          {collections.map((name) => (
            <option key={name} value={name} />
          ))}
        </datalist>
        {busy ? (
          <div style={{ flex: 1, minWidth: 140 }}>
            <div className="progress" />
          </div>
        ) : null}
      </div>
    </div>
  )
}

export default function Documents({ onNavigate }) {
  const { data, error, loading, refresh } = useApi(() => api.documents(), [])
  const documents = data?.documents || []
  const collections = data?.collections || []

  // Poll only while something is actually moving, so an idle page is silent.
  const busy = documents.some((doc) => ['queued', 'parsing', 'ingesting'].includes(doc.status) || doc.progress_stage)
  usePolling(refresh, 1800, busy)

  const grouped = useMemo(() => {
    const map = new Map()
    for (const doc of documents) {
      if (!map.has(doc.collection)) map.set(doc.collection, [])
      map.get(doc.collection).push(doc)
    }
    return Array.from(map.entries())
  }, [documents])

  const totals = useMemo(
    () =>
      documents.reduce(
        (acc, doc) => ({
          pages: acc.pages + (doc.n_pages || 0),
          chunks: acc.chunks + (doc.n_chunks || 0),
          facts: acc.facts + (doc.n_facts || 0),
          verified: acc.verified + (doc.n_verified_facts || 0),
        }),
        { pages: 0, chunks: 0, facts: 0, verified: 0 },
      ),
    [documents],
  )

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Documents</h1>
        <p className="page-sub">
          Every PDF the system has read, with the page and chunk counts it produced and how many of its
          extracted facts survived the evidence gate.
        </p>
      </div>

      <Dropzone collections={collections} onUploaded={refresh} />

      {documents.length > 0 ? (
        <div className="stat-grid" style={{ marginTop: 26 }}>
          <Stat index={0} label="Documents" value={documents.length} />
          <Stat index={1} label="Pages" value={totals.pages} />
          <Stat index={2} label="Chunks" value={totals.chunks} />
          <Stat index={3} label="Facts" value={totals.facts} />
          <Stat
            index={4}
            label="Verified"
            value={totals.verified}
            foot={totals.facts ? `${((totals.verified / totals.facts) * 100).toFixed(1)}% passed the gate` : null}
          />
        </div>
      ) : null}

      <div className="section">
        <SectionHead title="Corpus" note={collections.length ? `${collections.length} collections` : null} />

        {loading && !data ? (
          <Panel>
            <Skeleton rows={6} />
          </Panel>
        ) : error ? (
          <Panel>
            <ErrorState error={error} onRetry={refresh} />
          </Panel>
        ) : documents.length === 0 ? (
          <Panel>
            <Empty
              title="No documents ingested yet"
              note="Drop a PDF above, or run python scripts/ingest_all.py to load the starter collections."
            />
          </Panel>
        ) : (
          <div className="row-gap">
            {grouped.map(([name, docs], groupIndex) => (
              <Reveal key={name} index={groupIndex}>
                <Panel>
                  <div className="link-head">
                    <Icon.layers size={15} />
                    <span className="link-title">{name}</span>
                    <span className="faint" style={{ fontSize: 12 }}>
                      {docs.length} {docs.length === 1 ? 'document' : 'documents'}
                    </span>
                    <div className="right">
                      <button className="btn ghost sm" onClick={() => onNavigate('facts', { collection: name })}>
                        View facts
                      </button>
                    </div>
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Document</th>
                          <th>Type</th>
                          <th>Publisher</th>
                          <th className="nowrap">As of</th>
                          <th style={{ textAlign: 'right' }}>Pages</th>
                          <th style={{ textAlign: 'right' }}>Chunks</th>
                          <th style={{ textAlign: 'right' }}>Facts</th>
                          <th style={{ textAlign: 'right' }}>Verified</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {docs.map((doc, index) => (
                          <tr key={doc.doc_id} className="reveal" style={{ '--i': index }}>
                            <td style={{ maxWidth: 330 }}>
                              <div style={{ fontWeight: 500 }}>{doc.title || doc.filename}</div>
                              {doc.title ? <div className="faint trunc" style={{ fontSize: 11.5 }}>{doc.filename}</div> : null}
                              {doc.progress_stage ? (
                                <div style={{ marginTop: 7 }}>
                                  <StageLine current={doc.progress_stage} />
                                </div>
                              ) : null}
                            </td>
                            <td className="muted">{doc.doc_type || '—'}</td>
                            <td className="muted trunc">{doc.publisher || '—'}</td>
                            <td className="muted num nowrap">{shortDate(doc.as_of_date)}</td>
                            <td className="num" style={{ textAlign: 'right' }}>{doc.n_pages ?? 0}</td>
                            <td className="num muted" style={{ textAlign: 'right' }}>{doc.n_chunks ?? 0}</td>
                            <td className="num" style={{ textAlign: 'right' }}>{compact(doc.n_facts)}</td>
                            <td className="num" style={{ textAlign: 'right' }}>
                              {compact(doc.n_verified_facts)}
                              {doc.n_facts ? (
                                <span className="faint" style={{ fontSize: 11 }}>
                                  {' '}
                                  {Math.round((doc.n_verified_facts / doc.n_facts) * 100)}%
                                </span>
                              ) : null}
                            </td>
                            <td>
                              <span className="tag">{doc.progress_stage || doc.status || 'unknown'}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Panel>
              </Reveal>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
