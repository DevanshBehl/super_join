import { useEffect, useState } from 'react'
import { api, RECONCILED, RELATIONS } from '../api'
import { useApi } from '../hooks'
import {
  CopyButton,
  Empty,
  ErrorState,
  Icon,
  Pager,
  Panel,
  RelationBadge,
  Segmented,
  Skeleton,
  formatNumber,
  normalizedValue,
  pageRef,
  periodLabel,
  titleCase,
} from '../components/ui'

/** One side of the face-off. Both sides render identically so the eye can diff them. */
function Side({ fact, label }) {
  if (!fact) {
    return (
      <div className="side">
        <div className="side-label">{label}</div>
        <div className="faint">Fact not available.</div>
      </div>
    )
  }
  return (
    <div className="side">
      <div className="side-label">
        {label}
        <span className="faint" style={{ textTransform: 'none', letterSpacing: 0 }}>
          {fact.collection}
        </span>
      </div>
      <div className="side-value">{normalizedValue(fact)}</div>
      <div className="side-subject">{fact.subject_raw}</div>
      <div className="side-meta">
        <span className="mono">{fact.predicate_canonical}</span>
        {' · '}
        {periodLabel(fact)}
        {fact.scope_tags?.length ? ` · ${fact.scope_tags.join(', ')}` : ''}
      </div>

      <blockquote className="quote" style={{ margin: 0 }}>
        {fact.quote || 'No quote recorded.'}
      </blockquote>

      <div className="inline wrap" style={{ marginTop: 9, fontSize: 11 }}>
        <span className="tag">{fact.filename}</span>
        <span className="tag">{pageRef(fact)}</span>
        <span className="tag">raw: {fact.value_raw}</span>
        <CopyButton text={fact.quote || ''} title="Copy quote" />
      </div>
    </div>
  )
}

/**
 * The ordered rule chain. Each check is one step, and the step that actually
 * produced the verdict is marked, because "which check decided this" is the
 * question a reader is really asking.
 */
function RuleTrace({ trace, decidedBy }) {
  if (!trace?.length) {
    return (
      <div className="faint" style={{ fontSize: 12 }}>
        No rule trace recorded{decidedBy === 'llm' ? ' — this pair went straight to the adjudicator.' : '.'}
      </div>
    )
  }
  return (
    <div className="trace">
      {trace.map((step, index) => {
        const decisive = index === trace.length - 1
        return (
          <div
            key={`${step.name}-${index}`}
            className={`trace-step ${decisive ? 'decisive' : ''}`}
            style={{ animationDelay: `${index * 55}ms` }}
          >
            <div className="trace-name">
              {step.name || `step ${index + 1}`}
              {step.outcome ? <span className="trace-outcome">→ {String(step.outcome)}</span> : null}
              {decisive ? <span className="decider">decisive</span> : null}
            </div>
            {step.detail ? <div className="trace-detail">{String(step.detail)}</div> : null}
            {step.inputs && Object.keys(step.inputs).length ? (
              <div className="trace-inputs">
                {Object.entries(step.inputs)
                  .map(([key, value]) => `${key}=${typeof value === 'object' ? JSON.stringify(value) : value}`)
                  .join('  ')}
              </div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}

function LinkCard({ link, index }) {
  const [open, setOpen] = useState(index < 2)
  const a = link.fact_a_view
  const b = link.fact_b_view

  return (
    <Panel className="link-card reveal" style={{ '--i': index }}>
      <div className="link-head">
        <RelationBadge relation={link.relation} />
        <span className="link-title">
          {a?.subject_raw || 'Fact A'}
          <span className="faint" style={{ fontWeight: 400 }}> · </span>
          <span className="mono" style={{ fontWeight: 400, fontSize: 12 }}>{a?.predicate_canonical || ''}</span>
        </span>
        <div className="right">
          {link.relative_delta !== null && link.relative_delta !== undefined ? (
            <span className="faint num" style={{ fontSize: 11.5 }}>
              Δ {(Number(link.relative_delta) * 100).toFixed(2)}%
            </span>
          ) : null}
          {link.numeric_delta !== null && link.numeric_delta !== undefined ? (
            <span className="faint num" style={{ fontSize: 11.5 }}>
              abs {formatNumber(link.numeric_delta)}
            </span>
          ) : null}
          <span className="decider">{link.decided_by === 'llm' ? 'adjudicated' : link.decided_by}</span>
          <span className="faint num" style={{ fontSize: 11.5 }}>
            {Number(link.confidence ?? 0).toFixed(2)}
          </span>
          <button className="btn ghost sm" onClick={() => setOpen(!open)} aria-expanded={open}>
            <span className="caret" data-open={open}>
              <Icon.chevron size={12} />
            </span>
            {open ? 'Hide' : 'Show'}
          </button>
        </div>
      </div>

      <div className="expand" data-open={open}>
        <div className="expand-inner">
          <div>
            <div className="face-off">
              <Side fact={a} label="Fact A" />
              <div className="divider">
                <span className="vs">vs</span>
              </div>
              <Side fact={b} label="Fact B" />
            </div>

            <div style={{ borderTop: '1px solid var(--line)', padding: 16 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: 24 }}>
                <div>
                  <div className="eyebrow" style={{ marginBottom: 11 }}>
                    Rule trace · {link.rule_trace?.length || 0} checks
                  </div>
                  {open ? <RuleTrace trace={link.rule_trace} decidedBy={link.decided_by} /> : null}
                </div>
                <div>
                  <div className="eyebrow" style={{ marginBottom: 11 }}>
                    Verdict
                  </div>
                  <p className="explanation">{link.explanation || 'No explanation recorded.'}</p>
                  <div className="inline wrap" style={{ marginTop: 12, fontSize: 11 }}>
                    <span className="tag">{titleCase(link.relation)}</span>
                    <span className="tag">
                      {link.decided_by === 'rule' ? 'zero tokens' : 'model adjudicated'}
                    </span>
                    <CopyButton text={link.link_id} title="Copy link id" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Panel>
  )
}

const VIEWS = [
  { value: 'interesting', label: 'Needs attention' },
  { value: 'contradicts', label: 'Contradictions' },
  { value: 'reconciled', label: 'Reconciled' },
  { value: 'all', label: 'All' },
]

export default function Conflicts({ collections }) {
  const [view, setView] = useState('interesting')
  const [relation, setRelation] = useState('')
  const [collection, setCollection] = useState('')
  const [decidedBy, setDecidedBy] = useState('')
  const [page, setPage] = useState(1)

  useEffect(() => {
    setPage(1)
  }, [view, relation, collection, decidedBy])

  // "Needs attention" and "Reconciled" are client-side groupings over the
  // relation vocabulary; the API filters one relation at a time, so those two
  // views fetch unfiltered and narrow here.
  const effectiveRelation = view === 'contradicts' ? 'contradicts' : relation

  const { data, error, loading, refresh } = useApi(
    () =>
      api.links({
        relation: effectiveRelation,
        collection,
        decided_by: decidedBy,
        page,
        page_size: 25,
      }),
    [effectiveRelation, collection, decidedBy, page],
  )

  const all = data?.links || []
  const links = all.filter((link) => {
    if (view === 'interesting') return link.relation === 'contradicts' || link.relation === 'refines'
    if (view === 'reconciled') return RECONCILED.includes(link.relation)
    return true
  })

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Conflict inbox</h1>
        <p className="page-sub">
          Pairs of facts the funnel brought together, each with both quotes side by side and the ordered chain of
          checks that produced the verdict. A reconciled pair is one that looked like a disagreement until a rule
          named the period, scope, unit or vintage difference behind it.
        </p>
      </div>

      <div className="controls">
        <Segmented value={view} options={VIEWS} onChange={setView} />

        <select
          className="select"
          value={relation}
          onChange={(event) => setRelation(event.target.value)}
          disabled={view === 'contradicts'}
        >
          <option value="">Any relation</option>
          {RELATIONS.map((name) => (
            <option key={name} value={name}>
              {titleCase(name)}
            </option>
          ))}
        </select>

        <select className="select" value={collection} onChange={(event) => setCollection(event.target.value)}>
          <option value="">All collections</option>
          {(collections || []).map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>

        <select className="select" value={decidedBy} onChange={(event) => setDecidedBy(event.target.value)}>
          <option value="">Rule or model</option>
          <option value="rule">Decided by rule</option>
          <option value="llm">Adjudicated by model</option>
        </select>

        <button className="btn ghost sm" onClick={refresh} style={{ marginLeft: 'auto' }}>
          <Icon.refresh /> Refresh
        </button>
      </div>

      {loading && !data ? (
        <Panel>
          <Skeleton rows={8} height={20} />
        </Panel>
      ) : error ? (
        <Panel>
          <ErrorState error={error} onRetry={refresh} />
        </Panel>
      ) : links.length === 0 ? (
        <Panel>
          <Empty
            icon="scales"
            title={all.length ? 'Nothing in this view' : 'No links yet'}
            note={
              all.length
                ? 'This page of results held no links of that kind. Try "All", or move to the next page.'
                : 'Links appear once two documents in the same collection produce facts about the same thing. Ingest at least two documents to see them.'
            }
          />
        </Panel>
      ) : (
        <div className="row-gap">
          {links.map((link, index) => (
            <LinkCard key={link.link_id} link={link} index={index} />
          ))}
        </div>
      )}

      <Pager page={page} pageSize={data?.page_size || 25} total={data?.total || 0} onPage={setPage} />
    </div>
  )
}
