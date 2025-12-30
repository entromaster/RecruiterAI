import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
    LayoutDashboard,
    Briefcase,
    Users,
    Settings,
    Sparkles,
    ChevronRight
} from 'lucide-react'
import './Sidebar.css'

const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/jobs', icon: Briefcase, label: 'Jobs' },
    { path: '/candidates', icon: Users, label: 'Candidates' },
    { path: '/settings', icon: Settings, label: 'Settings' },
]

export default function Sidebar() {
    const location = useLocation()

    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <motion.div
                    className="logo"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.3 }}
                >
                    <div className="logo-icon">
                        <Sparkles size={24} />
                    </div>
                    <div className="logo-text">
                        <span className="logo-title">RecruiterAI</span>
                        <span className="logo-subtitle">Smart Hiring</span>
                    </div>
                </motion.div>
            </div>

            <nav className="sidebar-nav">
                {navItems.map((item, index) => {
                    const isActive = location.pathname === item.path ||
                        (item.path !== '/' && location.pathname.startsWith(item.path))

                    return (
                        <motion.div
                            key={item.path}
                            initial={{ x: -20, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <NavLink
                                to={item.path}
                                className={`nav-item ${isActive ? 'active' : ''}`}
                            >
                                <item.icon size={20} />
                                <span>{item.label}</span>
                                {isActive && (
                                    <motion.div
                                        className="nav-indicator"
                                        layoutId="activeIndicator"
                                        initial={false}
                                        transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                    />
                                )}
                                <ChevronRight size={16} className="nav-arrow" />
                            </NavLink>
                        </motion.div>
                    )
                })}
            </nav>

            <div className="sidebar-footer">
                <div className="sidebar-stats">
                    <div className="stat">
                        <span className="stat-value">12</span>
                        <span className="stat-label">Jobs</span>
                    </div>
                    <div className="stat">
                        <span className="stat-value">48</span>
                        <span className="stat-label">Candidates</span>
                    </div>
                </div>
            </div>
        </aside>
    )
}
