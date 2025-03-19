import { useState } from "react";
import Logbtn from "./Logbtn";

const ProfilePage = () => {
  const [isLoggedOut, setIsLoggedOut] = useState(false);
  const [user] = useState({
    name: "Srishanth",
    email: "Srishanth@example.com",
    role: "Farm AI Manager",
    joined: "March 2025",
    profileImage: "/sri.jpeg", // Placeholder image
  });

  const handleLogout = () => {
    console.log("Logging out...");
    // Implement actual logout logic here
    setIsLoggedOut(true);
  };

  if (isLoggedOut) {
    return <Logbtn />; // Replace content with Logbtn component
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-green-50 p-4">
      <div className="w-full max-w-md bg-white rounded-lg shadow-lg p-8">
        <div className="flex flex-col items-center">
          {/* Profile Photo */}
          <div className="relative mb-6">
            <div className="w-32 h-32 rounded-full overflow-hidden border-4 border-green-100">
              <img
                src={user.profileImage}
                alt="Profile"
                className="w-full object-center"
              />
            </div>
            <button
              className="absolute bottom-0 right-0 bg-green-600 text-white p-2 rounded-full hover:bg-green-700"
              title="Change profile photo"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                ></path>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
                ></path>
              </svg>
            </button>
          </div>

          {/* User Information */}
          <h1 className="text-2xl font-bold text-gray-800 mb-1">{user.name}</h1>
          <p className="text-gray-600 mb-1">{user.email}</p>
          <div className="inline-block px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm mb-6">
            {user.role}
          </div>

          <div className="w-full border-t border-gray-200 my-4"></div>

          {/* Profile Stats */}
          <div className="grid grid-cols-2 gap-4 w-full mb-6">
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <p className="text-sm text-gray-600">Member Since</p>
              <p className="font-semibold">{user.joined}</p>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <p className="text-sm text-gray-600">Last Login</p>
              <p className="font-semibold">Today</p>
            </div>
          </div>

          {/* Logout Button */}
          <button
            onClick={handleLogout}
            className="w-full py-2 px-4 bg-red-500 hover:bg-red-600 text-white rounded-md font-medium transition duration-200 flex items-center justify-center"
          >
            <svg
              className="w-5 h-5 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
              ></path>
            </svg>
            Logout
          </button>
        </div>
      </div>

      <div className="mt-8 text-center text-sm text-gray-500">
        <p>
          &copy; {new Date().getFullYear()} AgriTech Platform. All rights
          reserved.
        </p>
      </div>
    </div>
  );
};

export default ProfilePage;
