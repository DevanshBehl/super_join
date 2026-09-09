import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { useApi } from '../hooks'
import {
  Empty,
  ErrorState,
  Icon,
  Pager,
  Panel,
  RelationBadge,
  SectionHead,
  Skeleton,
  Stat,
  compact,
  formatNumber,
  pageRef,
  titleCase,
} from '../components/ui'

/** Bars grow from zero on mount, so the funnel's shape is read as a motion. */
function useGrowOnMount(delay = 60) {
  const [grown, setGrown] = useState(false)
  useEffect(() => {
    const timer = setTimeout(() => setGrown(true), delay)
    return () => clearTimeout(timer)
  }, [delay])
  return grown
}

function Funnel({ funnel }) {
  const grown = useGrowOnMount(120)
  const rows = [
    ['Blocked pairs', funnel.blocked_pairs, 'survived the SQL blocking join'],
    ['Retrieved', funnel.candidate_pairs, 'above the cosine threshold'],
    ['Settled by rules', funnel.rule_decided, 'zero tokens'],
    ['Sent to model', funnel.residue_for_llm, 'the residue only'],
  ]
  const max = Math.max(...rows.map(([, value]) => Number(value) || 0), 1)

  return (
    <div className="funnel">
      {rows.map(([label, value, note], index) => (
        <div className="funnel-row" key={label} title={note}>
          <span className="funnel-label">{label}</span>
          <div className="funnel-track">
            <div
              className="funnel-bar"
              style={{
                width: grown ? `${Math.max(1.5, ((Number(value) || 0) / max) * 100)}%` : '0%',
                transitionDelay: `${index * 90}ms`,
              }}
            />
          </div>
          <span className="funnel-value">{(Number(value) || 0).toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

function Breakdown({ entries, renderLabel, format = (v) => v.toLocaleString() }) {
  const grown = useGrowOnMount(160)
  const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1)
  if (!entries.length) {
    return <div className="faint" style={{ fontSize: 12 }}>Nothing recorded yet.</div>
  }
  return (
    <div className="bars">
      {entries.map(([key, value], index) => (
        <div className="bar-row" key={key}>
          <div>
            <div className="bar-top">{renderLabel ? renderLabel(key) : <span style={{ fontSize: 12.5 }}>{titleCase(key)}</span>}</div>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: grown ? `${((Number(value) || 0) / max) * 100}%` : '0%',
                  transitionDelay: `${index * 60}ms`,
                }}
              />
            </div>
          </div>
          <span className="num" style={{ textAlign: 'right', fontSize: 12.5 }}>{format(Number(value) || 0)}</span>
        </div>
      ))}
    </div>
  )
}

function Quarantine() {
  const [page, setPage] = useState(1)
  const { data, error, loading } = useApi(() => api.quarantine({ page, page_size: 15 }), [page])
  const facts = data?.facts || []

  if (loading && !data) return <Skeleton rows={5} />
  if (error) return <ErrorState error={error} />
  if (!facts.length) {
    return (
      <Empty
        title="Quarantine is empty"
        note="Every extracted fact located its quote in the source text. Facts whose quote cannot be found are kept here rather than deleted, so the cost of the evidence gate stays visible."
      />
    )
  }

  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Subject</th>
              <th>Predicate</th>
              <th>Claimed value</th>
              <th>Unfindable quote</th>
              <th className="nowrap">Best match</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {facts.map((fact, index) => (
              <tr key={fact.fact_id} className="reveal" style={{ '--i': index }}>
                <td style={{ maxWidth: 180 }}><span className="trunc">{fact.subject_raw || '—'}</span></td>
                <td className="mono muted" style={{ maxWidth: 170 }}><span className="trunc">{fact.predicate_canonical}</span></td>
                <td className="num nowrap">{fact.value_raw || '—'}</td>
                <td style={{ maxWidth: 380 }}>
                  <span className="mono faint trunc" style={{ fontSize: 11.5, maxWidth: 380 }} title={fact.quote}>
                    {fact.quote || '—'}
                  </span>
                </td>
                <td className="num muted nowrap">
                  {fact.match_score !== null && fact.match_score !== undefined
                    ? Number(fact.match_score).toFixed(1)
                    : '—'}
                  {fact.match_method ? <span className="faint"> {fact.match_method}</span> : null}
                </td>
                <td className="muted" style={{ maxWidth: 170 }}>
                  <span className="trunc">{fact.filename}</span>
                  <div className="faint" style={{ fontSize: 11 }}>{pageRef(fact)}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ padding: '0 16px 16px' }}>
        <Pager page={page} pageSize={data?.page_size || 15} total={data?.total || 0} onPage={setPage} />
      </div>
    </>
  )
}

function Registry() {
  const { data, error, loading } = useApi(() => api.registry(), [])
  const predicates = data?.predicates || []

  if (loading && !data) return <Skeleton rows={5} />
  if (error) return <ErrorState error={error} />
  if (!predicates.length) {
    return <Empty title="The registry is empty" note="Predicates are coined as facts are extracted, and the most frequent are fed back into later extraction prompts so the vocabulary converges." />
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Predicate</th>
            <th style={{ textAlign: 'right' }}>Facts</th>
            <th>Expects</th>
            <th>Dimension</th>
            <th>First seen as</th>
          </tr>
        </thead>
        <tbody>
          {predicates.slice(0, 40).map((entry, index) => (
            <tr key={entry.predicate_canonical} className="reveal" style={{ '--i': index }}>
              <td className="mono">{entry.predicate_canonical}</td>
              <td className="num" style={{ textAlign: 'right' }}>{entry.n_facts}</td>
              <td className="muted">{entry.expected_value_type || '—'}</td>
              <td className="muted">{entry.expected_unit_dimension || '—'}</td>
              <td className="faint" style={{ maxWidth: 400 }}>
                <span className="trunc" style={{ maxWidth: 400 }} title={entry.description}>
                  {entry.description || '—'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Stats() {
  const { data, error, loading, refresh } = useApi(() => api.stats(), [])
  const stats = data || {}

  if (loading && !data) {
    return (
      <Panel>
        <Skeleton rows={10} />
      </Panel>
    )
  }
  if (error) {
    return (
      <Panel>
        <ErrorState error={error} onRetry={refresh} />
      </Panel>
    )
  }

  const funnel = stats.funnel || {}
  const relations = Object.entries(stats.links_by_relation || {})
  const deciders = Object.entries(stats.links_by_decider || {})
  const stages = Object.entries(stats.seconds_by_stage || {})
  const totalSeconds = stages.reduce((sum, [, value]) => sum + Number(value || 0), 0)
  const ruleShare = stats.n_links ? ((stats.links_by_decider?.rule || 0) / stats.n_links) * 100 : 0

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Stats</h1>
        <p className="page-sub">
          The funnel counts, the verification rate, and the quarantine — every number read from the tables rather
          than written by hand.
        </p>
      </div>

      <div className="stat-grid">
        <Stat index={0} label="Documents" value={stats.n_documents} foot={`${compact(stats.n_pages)} pages`} />
        <Stat index={1} label="Chunks" value={stats.n_chunks} />
        <Stat index={2} label="Facts" value={stats.n_facts} />
        <Stat
          index={3}
          label="Verified"
          value={(stats.verification_rate || 0) * 100}
          unit="%"
          decimals={1}
          foot={`${compact(stats.n_verified_facts)} of ${compact(stats.n_facts)}`}
        />
        <Stat index={4} label="Quarantined" value={stats.n_quarantined} foot="quote not locatable" />
        <Stat index={5} label="Entities" value={stats.n_entities} />
        <Stat index={6} label="Predicates" value={stats.n_predicates} />
        <Stat index={7} label="Links" value={stats.n_links} foot={`${ruleShare.toFixed(0)}% settled by rule`} />
      </div>

      <div className="section">
        <SectionHead title="Candidate funnel" note="each stage narrows the set; only the last costs tokens" />
        <Panel pad>
          <Funnel funnel={funnel} />
          <div className="banner" style={{ marginTop: 18 }}>
            <Icon.layers size={15} />
            <span>
              Naive all-pairs comparison is quadratic and almost entirely wasted. Blocking runs inside the SQL
              join, so similarity is only ever computed for pairs that already share a collection, an entity or
              predicate, and a compatible unit dimension.{' '}
              <b>
                {(funnel.rule_decided || 0).toLocaleString()} pairs were settled without a single model call
              </b>
              {funnel.residue_for_llm ? `, and ${Number(funnel.residue_for_llm).toLocaleString()} reached the adjudicator.` : '.'}
            </span>
          </div>
        </Panel>
      </div>

      <div className="section">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 18 }}>
          <div>
            <SectionHead title="Links by relation" />
            <Panel pad>
              <Breakdown entries={relations} renderLabel={(key) => <RelationBadge relation={key} />} />
            </Panel>
          </div>
          <div>
            <SectionHead title="Time by stage" note={`${totalSeconds.toFixed(1)}s total`} />
            <Panel pad>
              <Breakdown
                entries={stages}
                format={(value) => `${value.toFixed(1)}s`}
                renderLabel={(key) => <span className="mono" style={{ fontSize: 12.5 }}>{key}</span>}
              />
            </Panel>
          </div>
        </div>
      </div>

      <div className="section">
        <SectionHead title="Model usage" />
        <div className="stat-grid">
          <Stat index={0} label="Model calls" value={stats.n_llm_calls} />
          <Stat index={1} label="Cache hits" value={stats.n_cache_hits} foot="re-ingest costs nothing" />
          <Stat index={2} label="Tokens in" value={stats.tokens_in} />
          <Stat index={3} label="Tokens out" value={stats.tokens_out} />
          <Stat
            index={4}
            label="Rule decided"
            value={deciders.find(([key]) => key === 'rule')?.[1] || 0}
            foot="zero tokens"
          />
          <Stat index={5} label="Adjudicated" value={deciders.find(([key]) => key === 'llm')?.[1] || 0} />
        </div>
      </div>

      <div className="section">
        <SectionHead title="Predicate registry" note="the open part of the schema, converging" />
        <Panel>
          <Registry />
        </Panel>
      </div>

      <div className="section">
        <SectionHead
          title="Quarantine"
          note={stats.n_quarantined ? `${stats.n_quarantined.toLocaleString()} facts held back` : 'empty'}
        />
        <Panel>
          <Quarantine />
        </Panel>
      </div>
    </div>
  )
}
