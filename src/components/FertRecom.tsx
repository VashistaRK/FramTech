import React, { useState } from "react";

const FertilizerRecommendation: React.FC = () => {
  const [formData, setFormData] = useState({
    temperature: "",
    humidity: "",
    moisture: "",
    soil_type: "Sandy",
    crop_type: "Maize",
    nitrogen: "",
    potassium: "",
    phosphorous: "",
  });

  const [recommendation, setRecommendation] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const soilClasses = ["Sandy", "Loamy", "Black", "Red", "Clayey"];
  const cropClasses = [
    "Maize",
    "Sugarcane",
    "Cotton",
    "Tobacco",
    "Paddy",
    "Barley",
    "Wheat",
    "Millets",
    "Oil seeds",
    "Pulses",
    "Ground Nuts",
  ];

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setRecommendation(null);
    setError(null);

    try {
      const response = await fetch("http://localhost:8080/api/fertilizerRec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error(`Error: ${response.status}`);
      const result = await response.text();
      setRecommendation(result);
    } catch (err) {
      setError(
        `Failed to fetch recommendation: ${
          err instanceof Error ? err.message : "Unknown error"
        }`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-5 min-h-screen flex flex-col">
      <header className="text-center mb-10 pb-5 border-b border-gray-200">
        <h1 className="text-3xl font-bold text-green-700 mb-3">
          Fertilizer Recommendation Tool
        </h1>
        <p className="text-gray-600 text-lg">
          Enter soil conditions to get the best fertilizer recommendation.
        </p>
      </header>

      <main className="flex-1">
        <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-5 mb-8">
          {Object.keys(formData).map((key) =>
            key === "soil_type" || key === "crop_type" ? (
              <div key={key} className="flex flex-col">
                <label className="font-medium text-gray-700 capitalize">
                  {key.replace("_", " ")}
                </label>
                <select
                  name={key}
                  value={formData[key as keyof typeof formData]}
                  onChange={handleChange}
                  className="border p-2 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
                >
                  {(key === "soil_type" ? soilClasses : cropClasses).map(
                    (option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    )
                  )}
                </select>
              </div>
            ) : (
              <div key={key} className="flex flex-col">
                <label className="font-medium text-gray-700 capitalize">
                  {key.replace("_", " ")}
                </label>
                <input
                  type="number"
                  name={key}
                  value={formData[key as keyof typeof formData]}
                  onChange={handleChange}
                  className="border p-2 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
                />
              </div>
            )
          )}

          <button
            type="submit"
            className="col-span-2 bg-green-700 text-white py-3 rounded hover:bg-green-800 transition-colors"
          >
            {loading ? "Processing..." : "Get Recommendation"}
          </button>
        </form>

        {loading && (
          <p className="text-center text-gray-700">
            Fetching recommendation...
          </p>
        )}
        {recommendation && (
          <div className="mt-5 p-5 bg-green-50 border border-green-200 rounded-lg text-center">
            <h2 className="text-xl font-bold text-green-700 mb-3">
              Recommended Fertilizer
            </h2>
            <p className="text-lg text-gray-800 font-medium">
              {recommendation}
            </p>
          </div>
        )}
        {error && (
          <div className="mt-5 p-5 bg-red-50 border border-red-200 rounded-lg text-center">
            <p className="text-red-700 font-medium">{error}</p>
          </div>
        )}
      </main>

      <footer className="mt-10 pt-4 text-center text-gray-500 text-sm border-t border-gray-200">
        <p>© 2025 Fertilizer Recommendation Tool</p>
      </footer>
    </div>
  );
};

export default FertilizerRecommendation;
