import { useRoute } from '../app/AppContext'
import { Route } from '../types'
import logo from '../assets/logo.png'

const NAV: { route: Route; label: string; icon: string }[] = [
  { route: 'patients', label: '患者管理', icon: '👤' },
  { route: 'assessment', label: '康复评估', icon: '✚' },
  { route: 'records', label: '评估记录总览', icon: '🗂' },
  { route: 'system', label: '系统管理', icon: '⚙' },
]

export default function Sidebar() {
  const { route, navigate } = useRoute()

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <img className="sidebar-org-logo" src={logo} alt="珠海复旦创新研究院" />
        <span>珠海复旦创新研究院康复评估系统</span>
      </div>
      <nav className="sidebar-nav">
        {NAV.map((item) => (
          <button
            key={item.route}
            className={`sidebar-item ${route === item.route ? 'active' : ''}`}
            onClick={() => navigate(item.route)}
          >
            <span className="sidebar-icon" aria-hidden="true">
              {item.icon}
            </span>
            {item.label}
          </button>
        ))}
      </nav>
    </aside>
  )
}
