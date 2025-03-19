import React from "react";
import { NavLink } from "react-router-dom";
import { NavItem } from "../types/types";

interface SidebarProps {
  navItems: NavItem[];
  isOpen: boolean;
  toggleSidebar: () => void; // Add toggle function prop
}

const Sidebar: React.FC<SidebarProps> = ({
  navItems,
  isOpen,
  toggleSidebar,
}) => {
  return (
    <div
      className={`fixed h-full bg-green-500 shadow-lg ${
        isOpen ? "w-64" : "w-20"
      } transition-all duration-300 z-10`}
    >
      {/* Header with toggle button instead of logo */}
      <div className="flex items-center justify-between h-16 border-b border-green-400 px-4">
        {isOpen ? (
          <h1 className="text-white text-xl font-bold">FarmTech</h1>
        ) : (
          <span className="text-white text-xl font-bold mx-auto">FT</span>
        )}

        {isOpen && (
          <button
            onClick={toggleSidebar}
            className="text-white hover:bg-green-800 bg-green-600 p-2 rounded-md"
            title="Collapse Sidebar"
          >
            <i className="fa fa-chevron-left"></i>
          </button>
        )}
      </div>

      <nav className="p-4">
        {/* If sidebar is collapsed, show expand button at top */}
        {!isOpen && (
          <div className="mb-6 flex justify-center">
            <button
              onClick={toggleSidebar}
              className="text-white hover:bg-green-700 bg-green-600 p-2 rounded-full w-10 h-10 flex items-center justify-center"
              title="Expand Sidebar"
            >
              <i className="fa fa-chevron-right"></i>
            </button>
          </div>
        )}

        <ul>
          {navItems.map((item) => (
            <li key={item.id} className="mb-4">
              <NavLink
                to={`/${item.id}`}
                className={({ isActive }) =>
                  `flex items-center ${
                    isOpen ? "px-4" : "justify-center"
                  } py-2 rounded-md transition-all ${
                    isActive
                      ? "bg-white text-green-500"
                      : "text-white hover:bg-green-600"
                  }`
                }
              >
                {/* Always show icon */}
                <i className={`fa fa-${item.icon} text-xl`}></i>

                {/* Only show text when sidebar is open */}
                {isOpen && <span className="ml-3">{item.title}</span>}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;
