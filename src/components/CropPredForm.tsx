import { useState } from "react";

interface FormData {
  N: number;
  P: number;
  K: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
  soilType: string;
}

interface PredictionResult {
  crop: string;
  probability: number;
}

const soilMapping = {
  loamy: 0,
  clayey: 1,
  sandy: 2,
  black: 3,
  red: 4,
};

const CropPredictionForm: React.FC = () => {
  const [formData, setFormData] = useState<FormData>({
    N: 0,
    P: 0,
    K: 0,
    temperature: 0,
    humidity: 0,
    ph: 0,
    rainfall: 0,
    soilType: "loamy",
  });

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: name === "soilType" ? value : parseFloat(value),
    }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8080/cropRec", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }

      const data = await response.json();
      setResult(data);
      console.log(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An unknown error occurred"
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl p-6">
      <h2 className="text-2xl font-bold text-green-700 mb-6">
        Crop Prediction Tool
      </h2>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.keys(formData).map(
            (key) =>
              key !== "soilType" && (
                <div key={key}>
                  <label
                    htmlFor={key}
                    className="block text-sm font-medium text-gray-700"
                  >
                    {key.charAt(0).toUpperCase() + key.slice(1)}
                  </label>
                  <input
                    type="number"
                    id={key}
                    name={key}
                    value={
                      formData[key as keyof FormData] === 0
                        ? ""
                        : formData[key as keyof FormData]
                    }
                    onChange={handleInputChange}
                    placeholder="Enter your Data Here..."
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-green-500 focus:border-green-500"
                    required
                  />
                </div>
              )
          )}
          <div>
            <label
              htmlFor="soilType"
              className="block text-sm font-medium text-gray-700"
            >
              Soil Type
            </label>
            <select
              id="soilType"
              name="soilType"
              value={formData.soilType}
              onChange={handleInputChange}
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-green-500 focus:border-green-500"
            >
              {Object.keys(soilMapping).map((soil) => (
                <option key={soil} value={soil}>
                  {soil.charAt(0).toUpperCase() + soil.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex items-center justify-center pt-4">
          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
          >
            {isLoading ? "Predicting..." : "Predict Suitable Crop"}
          </button>
        </div>
      </form>

      {result && (
        <div className="mt-6 p-4 bg-green-50 rounded-lg border border-green-200">
          <h3 className="font-semibold text-green-800 mb-2">
            Prediction Result
          </h3>
          <div className="text-gray-700">
            {result.map((crop, index) => (
              <li key={index} className="flex justify-between">
                <span>{crop.crop}</span>
                <span className="font-semibold">
                  {(crop.probability * 100).toFixed(2)}%
                </span>
              </li>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-red-50 rounded-lg border border-red-200">
          <p className="text-red-700">{error}</p>
        </div>
      )}
    </div>
  );
};

export default CropPredictionForm;
