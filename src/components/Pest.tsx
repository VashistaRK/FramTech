import React, { useState, ChangeEvent, FormEvent } from "react";

const PestRecommendation: React.FC = () => {
  const [pest, setPest] = useState<string>("");
  const [recommendation, setRecommendation] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    setPest(event.target.value);
    setRecommendation(null);
    setError(null);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!pest) {
      setError("Please enter a pest name");
      return;
    }

    setLoading(true);
    setRecommendation(null);
    setError(null);

    try {
      const response = await fetch("http://localhost:8080/api/PestRec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pest }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const result = await response.text();
      setRecommendation(result);
    } catch (err) {
      setError(
        `Failed to get recommendation: ${
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
        <h1 className="text-3xl font-bold text-blue-700 mb-3">
          Pest Recommendation Tool
        </h1>
        <p className="text-gray-600 text-lg">
          Enter a pest name to get recommended pesticides
        </p>
      </header>

      <main className="flex-1">
        <form
          onSubmit={handleSubmit}
          className="flex flex-col items-center gap-5 mb-8"
        >
          <input
            type="text"
            value={pest}
            onChange={handleInputChange}
            placeholder="Enter pest name"
            className="px-4 py-2 border border-gray-300 rounded-lg w-full max-w-md"
          />

          <button
            type="submit"
            className={`px-8 py-3 rounded text-white font-medium transition-colors ${
              !pest || loading
                ? "bg-gray-500 cursor-not-allowed"
                : "bg-blue-700 hover:bg-blue-800"
            }`}
            disabled={!pest || loading}
          >
            {loading ? "Fetching..." : "Get Recommendation"}
          </button>
        </form>

        {loading && (
          <p className="text-center text-gray-700">
            Fetching recommendations...
          </p>
        )}

        {recommendation && (
          <div className="mt-5 p-5 bg-blue-50 border border-blue-200 rounded-lg text-center">
            <h2 className="text-xl font-bold text-blue-700 mb-3">
              Recommended Pesticides
            </h2>
            <p className="text-lg text-gray-800 font-medium">
              {recommendation}
            </p>
          </div>
        )}

        {error && <p className="mt-5 text-red-700 text-center">{error}</p>}
      </main>

      <footer className="mt-10 pt-4 text-center text-gray-500 text-sm border-t border-gray-200">
        <p>© 2025 Pest Recommendation Tool</p>
      </footer>
    </div>
  );
};

export default PestRecommendation;
