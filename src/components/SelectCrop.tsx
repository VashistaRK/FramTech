import React, { useState } from "react";

interface CropOption {
  id: string;
  name: string;
  season: string;
  growthPeriod: string;
  waterRequirement: string;
}

const SelectCrop: React.FC = () => {
  const [selectedCrop, setSelectedCrop] = useState<string>("");

  const cropOptions: CropOption[] = [
    {
      id: "wheat",
      name: "Wheat",
      season: "Winter",
      growthPeriod: "110-130 days",
      waterRequirement: "Medium",
    },
    {
      id: "rice",
      name: "Rice",
      season: "Rainy",
      growthPeriod: "90-150 days",
      waterRequirement: "High",
    },
    {
      id: "corn",
      name: "Corn",
      season: "Spring/Summer",
      growthPeriod: "60-100 days",
      waterRequirement: "Medium-High",
    },
    {
      id: "potato",
      name: "Potato",
      season: "Spring",
      growthPeriod: "70-120 days",
      waterRequirement: "Medium",
    },
    {
      id: "tomato",
      name: "Tomato",
      season: "Spring/Summer",
      growthPeriod: "60-80 days",
      waterRequirement: "Medium",
    },
    {
      id: "soybean",
      name: "Soybean",
      season: "Spring/Summer",
      growthPeriod: "80-120 days",
      waterRequirement: "Medium",
    },
  ];

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold mb-6 border-b pb-4">
        Select Your Crop
      </h2>

      <div className="mb-6">
        <label className="block text-gray-700 text-sm font-medium mb-2">
          Field Name
        </label>
        <input
          type="text"
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
          placeholder="E.g., North Field, South Field"
        />
      </div>

      <div className="mb-6">
        <label className="block text-gray-700 text-sm font-medium mb-2">
          Field Size (acres)
        </label>
        <input
          type="number"
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
          placeholder="Enter field size"
        />
      </div>

      <div className="mb-6">
        <label className="block text-gray-700 text-sm font-medium mb-2">
          Select Crop
        </label>
        <select
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
          value={selectedCrop}
          onChange={(e) => setSelectedCrop(e.target.value)}
        >
          <option value="">-- Select a crop --</option>
          {cropOptions.map((crop) => (
            <option key={crop.id} value={crop.id}>
              {crop.name}
            </option>
          ))}
        </select>
      </div>

      {selectedCrop && (
        <div className="mb-6 bg-green-50 p-4 rounded-md">
          <h3 className="text-lg font-medium text-green-700 mb-2">
            Crop Information
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-gray-500">Growing Season</div>
              <div className="font-medium">
                {cropOptions.find((c) => c.id === selectedCrop)?.season}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Growth Period</div>
              <div className="font-medium">
                {cropOptions.find((c) => c.id === selectedCrop)?.growthPeriod}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Water Requirement</div>
              <div className="font-medium">
                {
                  cropOptions.find((c) => c.id === selectedCrop)
                    ?.waterRequirement
                }
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="mb-6">
        <label className="block text-gray-700 text-sm font-medium mb-2">
          Planting Date
        </label>
        <input
          type="date"
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
        />
      </div>

      <div className="mb-6">
        <label className="block text-gray-700 text-sm font-medium mb-2">
          Notes
        </label>
        <textarea
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
          rows={3}
          placeholder="Additional information about this crop planting"
        ></textarea>
      </div>

      <div className="flex justify-end">
        <button className="bg-gray-300 text-gray-700 py-2 px-4 rounded hover:bg-gray-400 transition-colors mr-2">
          Cancel
        </button>
        <button className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 transition-colors">
          Add Crop
        </button>
      </div>
    </div>
  );
};

export default SelectCrop;
