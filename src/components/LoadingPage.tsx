import React, { useState, useEffect, useRef } from "react";
/*eslint-disable*/

const LoadingPage: React.FC = () => {
  const [recommendation, setRecommendation] = useState<string>("");
  const intervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const recommendations = [
    "Rotate your crops regularly to prevent soil-borne diseases.",
    "Consider using companion planting to naturally repel pests.",
    "Morning is the best time to water your plants to prevent fungal growth.",
    "Test your soil pH before planting to ensure optimal growing conditions.",
    "Add organic matter to your soil annually to improve its structure and fertility.",
    "Mulch around plants to conserve moisture and suppress weeds.",
    "Use drip irrigation to save water and reduce leaf wetness.",
    "Keep track of planting dates to plan for succession planting.",
    "Save seeds from your best-performing plants for next season.",
    "Monitor for pests regularly - early detection makes control easier.",
    "Apply fertilizers based on soil test results, not guesswork.",
    "Use floating row covers to protect seedlings from insects.",
    "Prune damaged or diseased plant parts promptly.",
    "Consider raised beds for better drainage and soil warming.",
    "Start a compost pile to recycle garden and kitchen waste.",
    "Plant cover crops in the off-season to improve soil health.",
    "Use natural predators like ladybugs to control aphids.",
    "Provide support for climbing plants early in their growth.",
    "Keep a garden journal to track successes and failures.",
    "Harvest vegetables at their peak ripeness for best flavor.",
  ];

  useEffect(() => {
    if (intervalRef.current) return; // Prevents duplicate execution in Strict Mode

    const randomIndex = Math.floor(Math.random() * recommendations.length);
    setRecommendation(recommendations[randomIndex]);

    return () => {
      if (intervalRef.current) clearTimeout(intervalRef.current);
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
      <div className="text-center max-w-md w-full px-4">
        {/* Farm Logo */}
        <div className="mb-6">
          <div className="mx-auto w-24 h-24 bg-green-600 rounded-full flex items-center justify-center">
            <i className="fas fa-seedling text-white text-4xl"></i>
          </div>
        </div>

        {/* Animated Spinner */}
        <div className="flex justify-center items-center mb-8">
          <div className="w-16 h-16 border-4 border-green-600 border-t-transparent rounded-full animate-spin"></div>
        </div>

        {/* Random Recommendation */}
        <div className="max-w-md mx-auto p-4 bg-white rounded-lg shadow-md border-l-4 border-green-500">
          <h3 className="text-lg font-semibold text-green-700 mb-2">
            Farming Tip:
          </h3>
          <p className="text-gray-700">{recommendation}</p>
        </div>
      </div>
    </div>
  );
};

export default LoadingPage;
