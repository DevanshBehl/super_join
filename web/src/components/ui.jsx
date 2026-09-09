import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { useCopy, useCountUp } from '../hooks'
import { RELATION_GLYPH, RELATION_LABEL } from '../api'

/* ==========================================================================
   Icons. Hand-rolled at 1.5px stroke so they sit at the same optical weight
   as the type, and inherit currentColor so the ramp applies to them too.
   ========================================================================== */

const Svg = ({ children, size = 16, ...rest }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...rest}
  >
    {children}
  </svg>
)

export const Icon = {
  search: (p) => (
    <Svg {...p}>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </Svg>
  ),
  chevron: (p) => (
    <Svg {...p}>
      <path d="m9 6 6 6-6 6" />
    </Svg>
  ),
  upload: (p) => (
    <Svg {...p}>
      <path d="M12 16V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2" />
    </Svg>
  ),
  doc: (p) => (
    <Svg {...p}>
      <path d="M14 3H7a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V7z" />
      <path d="M14 3v4h4" />
    </Svg>
  ),
  clip: (p) => (
    <Svg {...p} size={13}>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15V5a1 1 0 0 1 1-1h9" />
    </Svg>
  ),
  tick: (p) => (
    <Svg {...p} size={13}>
      <path d="m5 12.5 4.5 4.5L19 7" />
    </Svg>
  ),
  tickSmall: (p) => (
    <Svg {...p} size={9} strokeWidth="2.8" stroke="#000">
      <path d="m5 12.5 4.5 4.5L19 7" />
    </Svg>
  ),
  scales: (p) => (
    <Svg {...p}>
      <path d="M12 4v16" />
      <path d="M6 8h12" />
      <path d="m6 8-3 6h6z" />
      <path d="m18 8-3 6h6z" />
    </Svg>
  ),
  layers: (p) => (
    <Svg {...p}>
      <path d="m12 3 9 5-9 5-9-5z" />
      <path d="m3 13 9 5 9-5" />
    </Svg>
  ),
  alert: (p) => (
    <Svg {...p}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7.5v5" />
      <path d="M12 16.2h.01" />
    </Svg>
  ),
  refresh: (p) => (
    <Svg {...p} size={14}>
      <path d="M20 11A8 8 0 0 0 6.3 6.3L4 8.5" />
      <path d="M4 4v4.5h4.5" />
      <path d="M4 13a8 8 0 0 0 13.7 4.7L20 15.5" />
      <path d="M20 20v-4.5h-4.5" />
    </Svg>
  ),
}

/* ==========================================================================
   Primitives
   ========================================================================== */

export function Reveal({ index = 0, as: Tag = 'div', className = '', children, ...rest }) {
  return (
    <Tag className={`reveal ${className}`} style={{ '--i': index }} {...rest}>
      {children}
    </Tag>
  )
}

export function Panel({ className = '', pad = false, children, ...rest }) {
  return (
    <div className={`panel ${pad ? 'panel-pad' : ''} ${className}`} {...rest}>
      {children}
    </div>
  )
}

export function SectionHead({ title, note, children }) {
  return (
    <div className="section-head">
      <span className="section-title">{title}</span>
      {children}
      {note ? <span className="section-note">{note}</span> : null}
    </div>
  )
}

export function Stat({ label, value, unit, foot, index = 0, decimals = 0 }) {
  const animated = useCountUp(Number(value) || 0)
  const shown = decimals
    ? animated.toFixed(decimals)
    : Math.round(animated).toLocaleString('en-US')
  return (
    <Reveal index={index} className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {shown}
        {unit ? <span className="unit">{unit}</span> : null}
      </div>
      {foot ? <div className="stat-foot">{foot}</div> : null}
    </Reveal>
  )
}

export function RelationBadge({ relation }) {
  const key = relation || 'unrelated'
  return (
    <span className={`badge rel-${key}`} title={key}>
      <span className="glyph">{RELATION_GLYPH[key] || '·'}</span>
      {RELATION_LABEL[key] || key}
    </span>
  )
}

export function CopyButton({ text, title = 'Copy' }) {
  const [copied, copy] = useCopy()
  return (
    <button
      type="button"
      className="copy"
      data-copied={copied}
      title={copied ? 'Copied' : title}
      aria-label={copied ? 'Copied' : title}
      onClick={(event) => {
        event.stopPropagation()
        copy(text)
      }}
    >
      <span className="clip">
        <Icon.clip />
      </span>
      <span className="tick">
        <Icon.tick />
      </span>
    </button>
  )
}

export function Chip({ on, onChange, children }) {
  return (
    <button type="button" className="chip" data-on={!!on} onClick={() => onChange(!on)} aria-pressed={!!on}>
      <span className="box">
        <Icon.tickSmall />
      </span>
      {children}
    </button>
  )
}

export function Segmented({ value, options, onChange }) {
  const [ref, setRef] = useState(null)
  const [thumb, setThumb] = useState({ width: 0, transform: 'translateX(0)', opacity: 0 })

  useEffect(() => {
    if (!ref) return
    const measure = () => {
      const active = ref.querySelector('[aria-pressed="true"]')
      if (!active) return setThumb((t) => ({ ...t, opacity: 0 }))
      setThumb({
        width: active.offsetWidth,
        transform: `translateX(${active.offsetLeft - 3}px)`,
        opacity: 1,
      })
    }
    measure()
    const raf = requestAnimationFrame(measure)
    return () => cancelAnimationFrame(raf)
  }, [ref, value, options])

  return (
    <div className="segmented" ref={setRef}>
      <span className="thumb" style={thumb} />
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

export function Skeleton({ rows = 6, height = 15 }) {
  return (
    <div className="row-gap" style={{ padding: 16 }}>
      {Array.from({ length: rows }).map((_, index) => (
        <div
          key={index}
          className="skeleton"
          style={{
            height,
            width: `${100 - (index % 4) * 11}%`,
            animationDelay: `${index * 70}ms`,
          }}
        />
      ))}
    </div>
  )
}

export function Empty({ icon = 'doc', title, note, action }) {
  const Glyph = Icon[icon] || Icon.doc
  return (
    <div className="empty">
      <div className="empty-icon">
        <Glyph size={19} />
      </div>
      <div className="empty-title">{title}</div>
      {note ? <div className="empty-note">{note}</div> : null}
      {action ? <div style={{ marginTop: 18 }}>{action}</div> : null}
    </div>
  )
}

export function ErrorState({ error, onRetry }) {
  return (
    <Empty
      icon="alert"
      title="Could not reach the API"
      note={`${error?.message || 'Unknown error'}. Check that uvicorn is running on port 8000.`}
      action={
        onRetry ? (
          <button className="btn ghost sm" onClick={onRetry}>
            <Icon.refresh /> Retry
          </button>
        ) : null
      }
    />
  )
}

export function Pager({ page, pageSize, total, onPage }) {
  const pages = Math.max(1, Math.ceil((total || 0) / (pageSize || 1)))
  if (total === 0) return null
  const from = (page - 1) * pageSize + 1
  const to = Math.min(page * pageSize, total)
  return (
    <div className="pager">
      <span className="count">
        <span className="num">{from.toLocaleString()}</span>–<span className="num">{to.toLocaleString()}</span> of{' '}
        <span className="num">{total.toLocaleString()}</span>
      </span>
      <button className="btn ghost sm" disabled={page <= 1} onClick={() => onPage(page - 1)}>
        Previous
      </button>
      <span className="faint" style={{ fontSize: 12 }}>
        {page} / {pages}
      </span>
      <button className="btn ghost sm" disabled={page >= pages} onClick={() => onPage(page + 1)}>
        Next
      </button>
    </div>
  )
}

/* ==========================================================================
   Toasts
   ========================================================================== */

const ToastContext = createContext(() => {})
export const useToast = () => useContext(ToastContext)

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const push = useCallback((message, tone = 'info') => {
    const id = Math.random().toString(36).slice(2)
    setToasts((current) => [...current, { id, message, tone }])
    setTimeout(() => {
      setToasts((current) => current.map((t) => (t.id === id ? { ...t, leaving: true } : t)))
      setTimeout(() => setToasts((current) => current.filter((t) => t.id !== id)), 240)
    }, 4200)
  }, [])

  const value = useMemo(() => push, [push])

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toast-wrap">
        {toasts.map((toast) => (
          <div key={toast.id} className="toast" data-leaving={!!toast.leaving}>
            {toast.tone === 'error' ? <Icon.alert size={15} /> : <Icon.tick size={15} />}
            <span>{toast.message}</span>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

/* ==========================================================================
   Formatting helpers, shared by every page
   ========================================================================== */

export function formatNumber(value, options = {}) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  const n = Number(value)
  const abs = Math.abs(n)
  // Six significant figures matches what demo_cases.py prints, so the web view
  // and the CLI view of the same fact read identically.
  if (abs !== 0 && (abs >= 1e9 || abs < 1e-4)) return n.toExponential(3)
  return n.toLocaleString('en-US', { maximumFractionDigits: 6, ...options })
}

export function compact(value) {
  const n = Number(value) || 0
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}

export function normalizedValue(fact) {
  if (!fact) return '—'
  if (fact.value_num !== null && fact.value_num !== undefined) {
    return `${formatNumber(fact.value_num)}${fact.unit_canonical ? ` ${fact.unit_canonical}` : ''}`
  }
  return fact.value_text || fact.value_raw || '—'
}

export function periodLabel(fact) {
  if (!fact) return '—'
  if (fact.period_label_raw) return fact.period_label_raw
  if (fact.period_start || fact.period_end) {
    return `${fact.period_start || '?'} → ${fact.period_end || '?'}`
  }
  return 'unstated'
}

export function pageRef(fact) {
  if (!fact) return '—'
  const index = fact.page_index
  if (index === null || index === undefined) return '—'
  return fact.page_label ? `p. ${fact.page_label} (index ${index})` : `index ${index}`
}

export function shortDate(value) {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return String(value).slice(0, 10)
  return parsed.toISOString().slice(0, 10)
}

export function titleCase(text) {
  return String(text || '').replace(/_/g, ' ')
}
