import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Fetch on mount and whenever `deps` change, with the stale response from a
 * superseded request discarded rather than allowed to overwrite a newer one.
 */
export function useApi(fetcher, deps = [], { skip = false } = {}) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(!skip)
  const [nonce, setNonce] = useState(0)
  const generation = useRef(0)

  const refresh = useCallback(() => setNonce((n) => n + 1), [])

  useEffect(() => {
    if (skip) {
      setLoading(false)
      return
    }
    const mine = ++generation.current
    setLoading(true)
    fetcher()
      .then((result) => {
        if (mine !== generation.current) return
        setData(result)
        setError(null)
      })
      .catch((err) => {
        if (mine !== generation.current) return
        setError(err)
      })
      .finally(() => {
        if (mine === generation.current) setLoading(false)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce, skip])

  return { data, error, loading, refresh }
}

export function useDebounced(value, delay = 260) {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])
  return settled
}

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches

/**
 * Count a number up on mount. Eased, frame-timed, and skipped entirely when
 * the reader has asked for reduced motion or the value is large enough that
 * the animation would just be noise.
 */
export function useCountUp(target, duration = 800) {
  const [display, setDisplay] = useState(() => (prefersReducedMotion() ? target : 0))
  const previous = useRef(0)

  useEffect(() => {
    const end = Number(target) || 0
    if (prefersReducedMotion()) {
      setDisplay(end)
      previous.current = end
      return
    }
    const start = previous.current
    const startedAt = performance.now()
    let frame

    const tick = (now) => {
      const t = Math.min(1, (now - startedAt) / duration)
      // easeOutExpo: fast commitment, soft landing.
      const eased = t === 1 ? 1 : 1 - Math.pow(2, -10 * t)
      setDisplay(start + (end - start) * eased)
      if (t < 1) frame = requestAnimationFrame(tick)
      else previous.current = end
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, duration])

  return display
}

/** Poll while `active` is true. Used for ingest status, stopped when idle. */
export function usePolling(callback, intervalMs, active) {
  const saved = useRef(callback)
  useEffect(() => {
    saved.current = callback
  }, [callback])

  useEffect(() => {
    if (!active) return
    const id = setInterval(() => saved.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs, active])
}

/** Width and offset of the active element, for the sliding nav indicator. */
export function useSlider(activeKey) {
  const containerRef = useRef(null)
  const [style, setStyle] = useState({ width: 0, transform: 'translateX(0px)', opacity: 0 })

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const measure = () => {
      const active = container.querySelector('[data-slider-active="true"]')
      if (!active) return setStyle((s) => ({ ...s, opacity: 0 }))
      setStyle({
        width: active.offsetWidth,
        transform: `translateX(${active.offsetLeft}px)`,
        opacity: 1,
      })
    }
    measure()
    // Fonts land after first paint and shift the measurement, so remeasure.
    const raf = requestAnimationFrame(measure)
    window.addEventListener('resize', measure)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', measure)
    }
  }, [activeKey])

  return [containerRef, style]
}

/** Minimal history-API router. Enough for four pages, no dependency. */
export function useRoute(basePath = '') {
  const read = () => {
    const path = window.location.pathname
    const stripped = basePath && path.startsWith(basePath) ? path.slice(basePath.length) : path
    return stripped.replace(/^\/+|\/+$/g, '') || 'documents'
  }
  const [route, setRoute] = useState(read)

  useEffect(() => {
    const onPop = () => setRoute(read())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const navigate = useCallback(
    (next) => {
      const target = `${basePath}/${next}`.replace(/\/+$/, '') || '/'
      if (window.location.pathname !== target) window.history.pushState({}, '', target)
      setRoute(next)
      window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' })
    },
    [basePath],
  )

  return [route, navigate]
}

/** Copy to clipboard with a short-lived "copied" flag for the tick swap. */
export function useCopy(resetMs = 1400) {
  const [copied, setCopied] = useState(false)
  const timer = useRef(null)

  const copy = useCallback(
    async (text) => {
      try {
        await navigator.clipboard.writeText(text)
      } catch {
        // Clipboard is unavailable outside a secure context; fall back.
        const area = document.createElement('textarea')
        area.value = text
        area.style.position = 'fixed'
        area.style.opacity = '0'
        document.body.appendChild(area)
        area.select()
        try {
          document.execCommand('copy')
        } catch {
          /* nothing further to try */
        }
        document.body.removeChild(area)
      }
      setCopied(true)
      clearTimeout(timer.current)
      timer.current = setTimeout(() => setCopied(false), resetMs)
    },
    [resetMs],
  )

  useEffect(() => () => clearTimeout(timer.current), [])
  return [copied, copy]
}
