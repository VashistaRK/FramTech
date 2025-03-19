import React from "react";

interface CropData {
  id: string;
  name: string;
  field: string;
  plantingDate: string;
  status: string;
  progress: number;
}

const CropTracker: React.FC = () => {
  const crops: CropData[] = [
    {
      id: "1",
      name: "Wheat",
      field: "North Field",
      plantingDate: "2024-10-15",
      status: "Growing",
      progress: 40,
    },
    {
      id: "2",
      name: "Corn",
      field: "East Field",
      plantingDate: "2024-09-01",
      status: "Harvesting",
      progress: 90,
    },
    {
      id: "3",
      name: "Soybean",
      field: "West Field",
      plantingDate: "2024-08-10",
      status: "Mature",
      progress: 65,
    },
    {
      id: "4",
      name: "Rice",
      field: "South Field",
      plantingDate: "2024-09-20",
      status: "Growing",
      progress: 30,
    },
  ];

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold mb-6 border-b pb-4">
        Crop Tracker
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <div className="text-green-600 mb-1 text-sm">Total Crops</div>
          <div className="text-2xl font-bold">{crops.length}</div>
        </div>

        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <div className="text-blue-600 mb-1 text-sm">Growing</div>
          <div className="text-2xl font-bold">
            {crops.filter((c) => c.status === "Growing").length}
          </div>
        </div>

        <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
          <div className="text-yellow-600 mb-1 text-sm">Mature</div>
          <div className="text-2xl font-bold">
            {crops.filter((c) => c.status === "Mature").length}
          </div>
        </div>

        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
          <div className="text-purple-600 mb-1 text-sm">Harvesting</div>
          <div className="text-2xl font-bold">
            {crops.filter((c) => c.status === "Harvesting").length}
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full bg-white">
          <thead>
            <tr className="bg-gray-50 border-b">
              <th className="py-3 px-4 text-left text-sm font-medium text-gray-600">
                Crop
              </th>
              <th className="py-3 px-4 text-left text-sm font-medium text-gray-600">
                Field
              </th>
              <th className="py-3 px-4 text-left text-sm font-medium text-gray-600">
                Planting Date
              </th>
              <th className="py-3 px-4 text-left text-sm font-medium text-gray-600">
                Status
              </th>
              <th className="py-3 px-4 text-left text-sm font-medium text-gray-600">
                Progress
              </th>
              <th className="py-3 px-4 text-left text-sm font-medium text-gray-600">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {crops.map((crop) => (
              <tr key={crop.id} className="hover:bg-gray-50">
                <td className="py-3 px-4 text-sm">{crop.name}</td>
                <td className="py-3 px-4 text-sm">{crop.field}</td>
                <td className="py-3 px-4 text-sm">{crop.plantingDate}</td>
                <td className="py-3 px-4 text-sm">
                  <span
                    className={`
                    inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                    ${
                      crop.status === "Growing"
                        ? "bg-green-100 text-green-800"
                        : ""
                    }
                    ${
                      crop.status === "Mature"
                        ? "bg-yellow-100 text-yellow-800"
                        : ""
                    }
                    ${
                      crop.status === "Harvesting"
                        ? "bg-purple-100 text-purple-800"
                        : ""
                    }
                  `}
                  >
                    {crop.status}
                  </span>
                </td>
                <td className="py-3 px-4 text-sm">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                      className={`h-2.5 rounded-full ${
                        crop.status === "Growing"
                          ? "bg-green-600"
                          : crop.status === "Mature"
                          ? "bg-yellow-500"
                          : "bg-purple-600"
                      }`}
                      style={{ width: `${crop.progress}%` }}
                    ></div>
                  </div>
                  <div className="text-xs text-right mt-1">
                    {crop.progress}%
                  </div>
                </td>
                <td className="py-3 px-4 text-sm">
                  <button className="text-blue-600 hover:text-blue-800 mr-3">
                    <i className="fas fa-edit"></i>
                  </button>
                  <button className="text-red-600 hover:text-red-800">
                    <i className="fas fa-trash-alt"></i>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-6 flex justify-end">
        <button className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 transition-colors flex items-center">
          <i className="fas fa-plus mr-2"></i> Add New Crop
        </button>
      </div>
    </div>
  );
};

export default CropTracker;
