import React from "react";

const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-md py-3 px-4">
      <div className="flex justify-between items-center">
        {/* Logo/Title */}
        <div className="text-2xl text-green-600 flex items-center font-mine">
          <span className="mr-2">🌱</span>
          <h1 className="font-small">Farm</h1>
          <h1 className="font-bold">Tech</h1>
        </div>

        {/* Search Bar */}
        <div className="flex-1 max-w-md mx-6">
          <div className="flex items-center bg-gray-100 rounded-full px-4 py-2 hover:bg-gray-200 transition-colors">
            <i className="fa fa-search text-gray-500 mr-2"></i>
            <input
              type="text"
              placeholder="Search crops, guides, or tools..."
              className="bg-transparent border-none outline-none w-full text-sm text-gray-700"
            />
          </div>
        </div>

        {/* User Area */}
        <div className="flex items-center">
          {/* Notification */}
          <button className="relative mr-3 w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-100">
            <i className="fa fa-bell text-gray-600"></i>
            <span className="absolute top-0 right-0 w-3 h-3 bg-red-500 rounded-full text-xs">
              3
            </span>
          </button>

          {/* User */}
          <div className="flex items-center cursor-pointer group relative">
            <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center text-white">
              <span>U</span>
            </div>
            <div className="hidden md:block ml-2">
              <div className="text-sm font-medium">User</div>
              <div className="text-xs text-gray-500">Farmer</div>
            </div>

            {/* Dropdown */}
            <div className="absolute top-full right-0 mt-1 w-40 bg-white rounded shadow-lg invisible group-hover:visible z-50">
              <ul className="py-1">
                <li>
                  <a
                    href="/profile"
                    className="block px-3 py-1 text-sm hover:bg-gray-100"
                  >
                    Profile
                  </a>
                </li>
                <li>
                  <a
                    href="/settings"
                    className="block px-3 py-1 text-sm hover:bg-gray-100"
                  >
                    Settings
                  </a>
                </li>
                <li>
                  <a
                    href="/logout"
                    className="block px-3 py-1 text-sm text-red-500 hover:bg-gray-100"
                  >
                    Logout
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
