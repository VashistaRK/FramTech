import { useState } from "react";

interface FormData {
  n: number;
  p: number;
  k: number;
  ph: number;
  ec: number;
  s: number;
  cu: number;
  fe: number;
  mn: number;
  zn: number;
  b: number;
}

interface PredictionResult {
  crop: string;
  probability: number;
}

const CropReqNut: React.FC = () => {
  const [formData, setFormData] = useState<FormData>({
    n: 0,
    p: 0,
    k: 0,
    ph: 0,
    ec: 0,
    s: 0,
    cu: 0,
    fe: 0,
    mn: 0,
    zn: 0,
    b: 0,
  });

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: parseFloat(value),
    }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        "http://localhost:8080/api/cropRecNutrients",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(formData),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An unknown error occurred"
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto bg-green-50 border border-green-200 rounded-xl shadow-lg p-8">
      <h2 className="text-3xl font-bold text-green-700 text-center mb-6">
        Crop Prediction Tool
      </h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.keys(formData).map((key) => (
            <div key={key}>
              <label
                htmlFor={key}
                className="block text-sm font-medium text-gray-700 capitalize"
              >
                {key}
              </label>
              <input
                type="number"
                id={key}
                name={key}
                value={formData[key as keyof FormData] || ""}
                onChange={handleInputChange}
                placeholder="Enter value"
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:ring-green-500 focus:border-green-500"
                required
              />
            </div>
          ))}
        </div>

        <div className="flex items-center justify-center">
          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-md focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
          >
            {isLoading ? "Predicting..." : "Predict Suitable Crop"}
          </button>
        </div>
      </form>

      {result && (
        <div className="mt-6 p-4 bg-green-100 rounded-lg border border-green-300">
          <h3 className="font-semibold text-green-800 mb-2">
            Prediction Result
          </h3>
          <ul className="text-gray-700">
            {result.map((crop, index) => (
              <li key={index} className="flex justify-between py-1">
                <span>{crop.crop}</span>
                <span className="font-semibold">
                  {(crop.probability * 100).toFixed(2)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-red-100 rounded-lg border border-red-300">
          <p className="text-red-700">{error}</p>
        </div>
      )}
    </div>
  );
};

export default CropReqNut;
