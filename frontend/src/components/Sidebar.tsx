import { useRoute } from '../app/AppContext'
import { Route } from '../types'
import logo from '../assets/logo.png'

const NAV: { route: Route; label: string; icon: string }[] = [
  { route: 'patients', label: '患者管理', icon: '👤' },
  { route: 'assessment', label: 'FMA 分数预测', icon: '✚' },
  { route: 'records', label: 'FMA 分数预测记录总览', icon: '🗂' },
  { route: 'system', label: '系统管理', icon: '⚙' },
]

export default function Sidebar() {
  const { route, navigate } = useRoute()

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <img className="sidebar-org-logo" src={logo} alt="珠海复旦创新研究院" />
        <span>基于脑电肌电和运动信号的手部康复FMA评分预测软件 V1.0</span>
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
