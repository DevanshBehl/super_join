import { useEffect, useState } from 'react'
import { api } from '../api'
import { useApi, useDebounced } from '../hooks'
import {
  Chip,
  CopyButton,
  Empty,
  ErrorState,
  Icon,
  Pager,
  Panel,
  SectionHead,
  Skeleton,
  formatNumber,
  normalizedValue,
  pageRef,
  periodLabel,
} from '../components/ui'

/**
 * The expanded row. This is the part that matters: a fact is only worth
 * anything if the reader can see the sentence it came from, so the quote gets
 * the most space and everything else is arranged around it.
 */
function FactDetail({ fact }) {
  return (
    <div className="detail-body">
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.35fr) minmax(0, 1fr)', gap: 24 }}>
        <div>
          <div className="eyebrow" style={{ marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
            Verbatim evidence
            <CopyButton text={fact.quote || ''} title="Copy quote" />
          </div>
          <blockquote className="quote" style={{ margin: 0 }}>
            {fact.quote || 'No quote recorded.'}
          </blockquote>
          <div className="inline wrap" style={{ marginTop: 10, fontSize: 11.5 }}>
            <span className="tag">{pageRef(fact)}</span>
            <span className="tag">
              offsets {fact.evidence_char_start ?? '?'}–{fact.evidence_char_end ?? '?'}
            </span>
            {fact.match_method ? <span className="tag">{fact.match_method}</span> : null}
            {fact.match_score !== null && fact.match_score !== undefined ? (
              <span className="tag">score {Number(fact.match_score).toFixed(1)}</span>
            ) : null}
            <span className="tag">{fact.verified ? 'verified' : 'quarantined'}</span>
          </div>
        </div>

        <div>
          <div className="eyebrow" style={{ marginBottom: 8 }}>Normalized record</div>
          <dl className="kv">
            <dt>subject</dt>
            <dd>
              <span className="strong">{fact.subject_raw || '—'}</span>
              {fact.subject_canonical && fact.subject_canonical !== fact.subject_raw ? (
                <div className="faint mono" style={{ fontSize: 11 }}>{fact.subject_canonical}</div>
              ) : null}
            </dd>

            <dt>predicate</dt>
            <dd className="mono">{fact.predicate_canonical || fact.predicate || '—'}</dd>

            <dt>raw value</dt>
            <dd className="mono strong">{fact.value_raw || '—'}</dd>

            <dt>normalized</dt>
            <dd className="mono strong">
              {normalizedValue(fact)}
              {fact.magnitude_label ? <span className="faint"> (magnitude {fact.magnitude_label})</span> : null}
            </dd>

            <dt>type</dt>
            <dd>
              {fact.value_type || '—'}
              {fact.unit_dimension ? <span className="faint"> · {fact.unit_dimension}</span> : null}
              {fact.currency ? <span className="faint"> · {fact.currency}</span> : null}
            </dd>

            <dt>period</dt>
            <dd>
              {periodLabel(fact)}
              {fact.period_basis ? <span className="faint"> · basis {fact.period_basis}</span> : null}
            </dd>

            <dt>scope</dt>
            <dd>
              {fact.scope_tags?.length
                ? fact.scope_tags.map((tag) => (
                    <span key={tag} className="tag">
                      {tag}
                    </span>
                  ))
                : '—'}
            </dd>

            <dt>source</dt>
            <dd>
              {fact.document_title || fact.filename}
              <div className="faint" style={{ fontSize: 11 }}>{fact.collection}</div>
            </dd>

            <dt>confidence</dt>
            <dd className="mono">{fact.confidence !== null && fact.confidence !== undefined ? Number(fact.confidence).toFixed(2) : '—'}</dd>

            <dt>fact id</dt>
            <dd className="mono faint" style={{ fontSize: 11, wordBreak: 'break-all' }}>
              {fact.fact_id}
            </dd>
          </dl>

          {fact.qualifiers && Object.keys(fact.qualifiers).length ? (
            <div style={{ marginTop: 14 }}>
              <div className="eyebrow" style={{ marginBottom: 7 }}>Qualifiers</div>
              <dl className="kv">
                {Object.entries(fact.qualifiers).map(([key, value]) => (
                  <div key={key} style={{ display: 'contents' }}>
                    <dt className="mono">{key}</dt>
                    <dd>{String(value)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

export default function Facts({ initial = {}, collections }) {
  const [query, setQuery] = useState('')
  const [collection, setCollection] = useState(initial.collection || '')
  const [predicate, setPredicate] = useState('')
  const [period, setPeriod] = useState('')
  const [includeUnverified, setIncludeUnverified] = useState(false)
  const [page, setPage] = useState(1)
  const [open, setOpen] = useState(null)

  const debouncedQuery = useDebounced(query)
  const debouncedPeriod = useDebounced(period)

  // Any filter change invalidates the current page number.
  useEffect(() => {
    setPage(1)
  }, [debouncedQuery, collection, predicate, debouncedPeriod, includeUnverified])

  const { data, error, loading, refresh } = useApi(
    () =>
      api.facts({
        q: debouncedQuery,
        collection,
        predicate,
        period: debouncedPeriod,
        include_unverified: includeUnverified,
        page,
        page_size: 50,
      }),
    [debouncedQuery, collection, predicate, debouncedPeriod, includeUnverified, page],
  )

  const registry = useApi(() => api.registry(), [])
  const predicates = registry.data?.predicates || []

  const facts = data?.facts || []
  const total = data?.total || 0

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Facts</h1>
        <p className="page-sub">
          Every assertion the system kept. Expand a row to see the exact sentence it came from, the page it sits
          on, and the character offsets that locate it in the document buffer.
        </p>
      </div>

      <div className="controls">
        <div className="field">
          <span className="icon">
            <Icon.search size={14} />
          </span>
          <input
            className="input with-icon search"
            placeholder="Search subject, predicate, value or quote"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>

        <select className="select" value={collection} onChange={(event) => setCollection(event.target.value)}>
          <option value="">All collections</option>
          {(collections || []).map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>

        <select className="select" value={predicate} onChange={(event) => setPredicate(event.target.value)}>
          <option value="">All predicates</option>
          {predicates.map((entry) => (
            <option key={entry.predicate_canonical} value={entry.predicate_canonical}>
              {entry.predicate_canonical} ({entry.n_facts})
            </option>
          ))}
        </select>

        <input
          className="input"
          placeholder="Period contains…"
          value={period}
          onChange={(event) => setPeriod(event.target.value)}
          style={{ width: 160 }}
        />

        <Chip on={includeUnverified} onChange={setIncludeUnverified}>
          Include quarantined
        </Chip>

        <button className="btn ghost sm" onClick={refresh} title="Refresh" style={{ marginLeft: 'auto' }}>
          <Icon.refresh /> Refresh
        </button>
      </div>

      <Panel>
        <SectionHead
          title="Fact table"
          note={loading ? 'loading…' : `${total.toLocaleString()} matching`}
        />
        {loading && !data ? (
          <Skeleton rows={9} />
        ) : error ? (
          <ErrorState error={error} onRetry={refresh} />
        ) : facts.length === 0 ? (
          <Empty
            title="No facts match these filters"
            note={
              total === 0 && !debouncedQuery && !collection
                ? 'Nothing has been extracted yet. Ingest a document with a working GEMINI_API_KEY, or clear the filters.'
                : 'Try widening the search, clearing the predicate, or including quarantined facts.'
            }
          />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th style={{ width: 30 }} />
                  <th>Subject</th>
                  <th>Predicate</th>
                  <th style={{ textAlign: 'right' }}>Value</th>
                  <th>Period</th>
                  <th>Scope</th>
                  <th>Source</th>
                  <th className="nowrap">Page</th>
                </tr>
              </thead>
              <tbody>
                {facts.map((fact, index) => {
                  const isOpen = open === fact.fact_id
                  return (
                    <>
                      <tr
                        key={fact.fact_id}
                        className={`row-clickable reveal ${isOpen ? 'expanded' : ''}`}
                        style={{ '--i': index }}
                        onClick={() => setOpen(isOpen ? null : fact.fact_id)}
                      >
                        <td>
                          <span className="caret" data-open={isOpen}>
                            <Icon.chevron size={13} />
                          </span>
                        </td>
                        <td style={{ maxWidth: 260 }}>
                          <span className="trunc" title={fact.subject_raw}>
                            {fact.subject_raw || '—'}
                          </span>
                        </td>
                        <td className="mono muted" style={{ maxWidth: 220 }}>
                          <span className="trunc" title={fact.predicate_canonical}>
                            {fact.predicate_canonical || fact.predicate}
                          </span>
                        </td>
                        <td className="num nowrap" style={{ textAlign: 'right' }}>
                          {fact.value_num !== null && fact.value_num !== undefined
                            ? formatNumber(fact.value_num)
                            : fact.value_text || fact.value_raw || '—'}
                          {fact.unit_canonical ? <span className="faint"> {fact.unit_canonical}</span> : null}
                        </td>
                        <td className="muted nowrap">{periodLabel(fact)}</td>
                        <td>
                          {fact.scope_tags?.length ? (
                            <span className="tag">{fact.scope_tags[0]}</span>
                          ) : (
                            <span className="faint">—</span>
                          )}
                          {fact.scope_tags?.length > 1 ? (
                            <span className="faint" style={{ fontSize: 11 }}> +{fact.scope_tags.length - 1}</span>
                          ) : null}
                        </td>
                        <td className="muted" style={{ maxWidth: 190 }}>
                          <span className="trunc" title={fact.document_title || fact.filename}>
                            {fact.document_title || fact.filename}
                          </span>
                        </td>
                        <td className="num muted nowrap">{fact.page_label || fact.page_index}</td>
                      </tr>
                      <tr key={`${fact.fact_id}-detail`}>
                        <td colSpan={8} className="detail-cell">
                          <div className="expand" data-open={isOpen}>
                            <div className="expand-inner">{isOpen ? <FactDetail fact={fact} /> : null}</div>
                          </div>
                        </td>
                      </tr>
                    </>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        <div style={{ padding: '0 16px 16px' }}>
          <Pager page={page} pageSize={data?.page_size || 50} total={total} onPage={setPage} />
        </div>
      </Panel>
    </div>
  )
}
