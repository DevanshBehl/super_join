import { useEffect, useState } from 'react'
import { api } from './api'
import { useApi, useRoute, useSlider } from './hooks'
import { Icon, ToastProvider } from './components/ui'
import Documents from './pages/Documents'
import Facts from './pages/Facts'
import Conflicts from './pages/Conflicts'
import Stats from './pages/Stats'

// Served by FastAPI under /app, so the router strips that prefix. Vite's dev
// server serves at the root, where the prefix is simply absent.
const BASE_PATH = window.location.pathname.startsWith('/app') ? '/app' : ''

const ROUTES = [
  { key: 'documents', label: 'Documents' },
  { key: 'facts', label: 'Facts' },
  { key: 'conflicts', label: 'Conflict inbox' },
  { key: 'stats', label: 'Stats' },
]

function Topbar({ route, navigate, health }) {
  const [navRef, underline] = useSlider(route)

  return (
    <header className="topbar">
      <button className="brand" onClick={() => navigate('documents')}>
        <span className="mark">
          <i />
          <i />
          <i />
        </span>
        Fact Knowledge Layer
      </button>

      <nav className="nav" ref={navRef}>
        {ROUTES.map((item) => (
          <button
            key={item.key}
            className="nav-item"
            data-slider-active={route === item.key}
            aria-current={route === item.key ? 'page' : undefined}
            onClick={() => navigate(item.key)}
          >
            {item.label}
          </button>
        ))}
        <span className="nav-underline" style={underline} />
      </nav>

      <div className="topbar-right">
        <span className="pill" title={health ? `Database: ${health.db_path}` : 'Checking the API'}>
          <span className={`dot ${health?.llm_available ? 'live' : ''}`} />
          {health ? (health.llm_available ? 'Model layer live' : 'Deterministic only') : 'Connecting'}
        </span>
        <span className="byline">Devansh Behl</span>
      </div>
    </header>
  )
}

export default function App() {
  const [route, navigate] = useRoute(BASE_PATH)
  const [factsFilter, setFactsFilter] = useState({})

  const health = useApi(() => api.health(), [])
  const documents = useApi(() => api.documents(), [])
  const collections = documents.data?.collections || []

  // Cross-page navigation that carries a filter, e.g. "view facts" on a
  // collection card.
  const go = (next, filter) => {
    if (filter) setFactsFilter(filter)
    navigate(next)
  }

  useEffect(() => {
    document.title =
      route === 'documents'
        ? 'Documents · Fact Knowledge Layer'
        : `${ROUTES.find((r) => r.key === route)?.label || 'Fact'} · Fact Knowledge Layer`
  }, [route])

  return (
    <ToastProvider>
      <div className="shell">
        <Topbar route={route} navigate={navigate} health={health.data} />
        <main className="main" key={route}>
          {route === 'facts' ? (
            <Facts initial={factsFilter} collections={collections} />
          ) : route === 'conflicts' ? (
            <Conflicts collections={collections} />
          ) : route === 'stats' ? (
            <Stats />
          ) : (
            <Documents onNavigate={go} />
          )}
        </main>
      </div>
    </ToastProvider>
  )
}
