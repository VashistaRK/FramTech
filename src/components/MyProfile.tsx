import React from "react";

const MyProfile: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold mb-6 border-b pb-4">My Profile</h2>

      <div className="flex flex-col md:flex-row">
        <div className="md:w-1/3 flex flex-col items-center mb-6 md:mb-0">
          <div className="w-32 h-32 rounded-full bg-green-100 flex items-center justify-center text-green-600 text-5xl mb-4">
            <i className="fas fa-user"></i>
          </div>
          <button className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 transition-colors">
            Change Photo
          </button>
        </div>

        <div className="md:w-2/3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-700 text-sm font-medium mb-2">
                First Name
              </label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                defaultValue="John"
              />
            </div>

            <div>
              <label className="block text-gray-700 text-sm font-medium mb-2">
                Last Name
              </label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                defaultValue="Farmer"
              />
            </div>

            <div>
              <label className="block text-gray-700 text-sm font-medium mb-2">
                Email
              </label>
              <input
                type="email"
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                defaultValue="john.farmer@example.com"
              />
            </div>

            <div>
              <label className="block text-gray-700 text-sm font-medium mb-2">
                Phone Number
              </label>
              <input
                type="tel"
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                defaultValue="(123) 456-7890"
              />
            </div>

            <div>
              <label className="block text-gray-700 text-sm font-medium mb-2">
                Farm Name
              </label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                defaultValue="Green Valley Farm"
              />
            </div>

            <div>
              <label className="block text-gray-700 text-sm font-medium mb-2">
                Farm Size (acres)
              </label>
              <input
                type="number"
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                defaultValue="120"
              />
            </div>
          </div>

          <div className="mt-6">
            <label className="block text-gray-700 text-sm font-medium mb-2">
              Address
            </label>
            <textarea
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
              rows={3}
              defaultValue="123 Farm Road, Agricultural County, AC 12345"
            ></textarea>
          </div>

          <div className="mt-6 flex justify-end">
            <button className="bg-gray-300 text-gray-700 py-2 px-4 rounded hover:bg-gray-400 transition-colors mr-2">
              Cancel
            </button>
            <button className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 transition-colors">
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MyProfile;
