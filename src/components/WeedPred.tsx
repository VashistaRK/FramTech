import React, { useState, ChangeEvent, FormEvent } from "react";

const WeedPre: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{
    predicted_class: string;
    confidence: number;
  } | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please select an image file");
      return;
    }

    setLoading(true);
    setPrediction(null);
    setError(null);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("http://localhost:8080/api/weedPred", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(
        `Failed to get prediction: ${
          err instanceof Error ? err.message : "Unknown error"
        }`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-5 h-screen flex flex-col">
      <header className="text-center mb-10 pb-5 border-b border-gray-200">
        <h1 className="text-3xl font-bold text-green-700 mb-3">
          Weed Prediction Tool
        </h1>
        <p className="text-gray-600 text-lg">
          Upload a plant image to identify if it's a weed
        </p>
      </header>

      <main className="flex-1">
        <form
          onSubmit={handleSubmit}
          className="flex flex-col items-center gap-5 mb-8"
        >
          <div className="w-full text-center">
            <label
              htmlFor="file-upload"
              className="inline-block px-5 py-3 bg-green-700 text-white rounded cursor-pointer font-medium transition-colors hover:bg-green-800"
            >
              {selectedFile ? "Change Image" : "Select Image"}
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </div>

          {previewUrl && (
            <div className="w-full max-w-md flex flex-col items-center gap-2">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full h-auto max-h-64 object-contain rounded-lg border border-gray-300"
              />
              <p className="text-sm text-gray-600">{selectedFile?.name}</p>
            </div>
          )}

          <button
            type="submit"
            className={`px-8 py-3 rounded text-white font-medium transition-colors ${
              !selectedFile || loading
                ? "bg-gray-500 cursor-not-allowed"
                : "bg-green-700 hover:bg-green-800"
            }`}
            disabled={!selectedFile || loading}
          >
            {loading ? "Analyzing..." : "Predict Weed"}
          </button>
        </form>

        {loading && (
          <div className="flex flex-col items-center gap-4 my-5">
            <div className="w-10 h-10 border-4 border-gray-200 border-l-green-700 rounded-full animate-spin"></div>
            <p className="text-gray-700">Analyzing image...</p>
          </div>
        )}

        {prediction && (
          <div className="mt-5 p-5 bg-green-50 border border-green-200 rounded-lg text-center">
            <h2 className="text-xl font-bold text-green-700 mb-3">
              Prediction Result
            </h2>
            <p className="text-lg text-gray-800 font-medium">
              Weed Type: {prediction.predicted_class}
            </p>
            <p className="text-lg text-gray-700">
              Confidence: {(prediction.confidence * 100).toFixed(2)}%
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
        <p>© 2025 Weed Prediction Tool</p>
      </footer>
    </div>
  );
};

export default WeedPre;
