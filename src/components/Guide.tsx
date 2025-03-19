import React, { useState } from "react";

interface GuideSection {
  id: string;
  title: string;
  content: string;
}

const Guide: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>("getting-started");

  const guideSections: GuideSection[] = [
    {
      id: "getting-started",
      title: "Getting Started",
      content: `
        <h3 class="text-xl font-medium mb-3">Welcome to AgriTech</h3>
        <p class="mb-4">
          AgriTech provides powerful tools to help you manage your farm more efficiently.
          This guide will help you get started with our platform and make the most of its features.
        </p>
        <h4 class="text-lg font-medium mb-2">Quick Start Steps:</h4>
        <ol class="list-decimal pl-5 mb-4 space-y-2">
          <li>Complete your profile with your farm details</li>
          <li>Add your first crop using the "Select your crop" option</li>
          <li>Track your crop's progress in the Crop Tracker</li>
          <li>Use the Crop Predictor to get recommendations</li>
        </ol>
        <p>
          Our dashboard gives you a quick overview of your farm's performance and 
          key metrics to help you make informed decisions.
        </p>
      `,
    },
    {
      id: "crop-management",
      title: "Crop Management",
      content: `
        <h3 class="text-xl font-medium mb-3">Managing Your Crops</h3>
        <p class="mb-4">
          Effective crop management is essential for maximizing yields and profits.
          Our platform provides tools to help you through every stage of crop development.
        </p>
        <h4 class="text-lg font-medium mb-2">Key Features:</h4>
        <ul class="list-disc pl-5 mb-4 space-y-2">
          <li>
            <strong>Crop Selection:</strong> Choose from a database of crops with detailed information
            about growing conditions, seasons, and requirements.
          </li>
          <li>
            <strong>Growth Tracking:</strong> Monitor your crop's development from planting to harvest.
          </li>
          <li>
            <strong>Automated Alerts:</strong> Receive notifications about critical stages and actions needed.
          </li>
          <li>
            <strong>Yield Estimation:</strong> Get predictions on expected yields based on historical data.
          </li>
        </ul>
        <p>
          Regular monitoring and timely interventions can significantly improve your crop performance.
        </p>
      `,
    },
    {
      id: "pest-control",
      title: "Pest & Disease Control",
      content: `
        <h3 class="text-xl font-medium mb-3">Managing Pests and Diseases</h3>
        <p class="mb-4">
          Pests and diseases can significantly impact your crop yields. Early identification and
          appropriate management are crucial for preventing losses.
        </p>
        <h4 class="text-lg font-medium mb-2">Best Practices:</h4>
        <ul class="list-disc pl-5 mb-4 space-y-2">
          <li>Regular field inspections to detect issues early</li>
          <li>Use our weed identification tool to identify problematic weeds</li>
          <li>Implement integrated pest management (IPM) strategies</li>
          <li>Select appropriate pesticides based on our recommendations</li>
          <li>Follow recommended application rates and safety protocols</li>
        </ul>
        <p class="mb-4">
          Our system can help you identify common pests and diseases affecting your crops and provide
          treatment recommendations based on sustainable farming practices.
        </p>
        <div class="bg-yellow-50 p-4 rounded-md border-l-4 border-yellow-400">
          <h4 class="text-lg font-medium text-yellow-800 mb-2">Important Note</h4>
          <p class="text-yellow-800">
            Always follow local regulations regarding pesticide use and consider environmental 
            impacts when selecting treatment options.
          </p>
        </div>
      `,
    },
    {
      id: "cost-management",
      title: "Cost Management",
      content: `
        <h3 class="text-xl font-medium mb-3">Managing Farm Costs</h3>
        <p class="mb-4">
          Understanding and controlling your costs is essential for maintaining profitability.
          Our cost estimation tools help you plan and monitor expenses throughout the growing season.
        </p>
        <h4 class="text-lg font-medium mb-2">Cost Categories:</h4>
        <ul class="list-disc pl-5 mb-4 space-y-2">
          <li>
            <strong>Inputs:</strong> Seeds, fertilizers, pesticides, and other materials
          </li>
          <li>
            <strong>Labor:</strong> Workforce costs for planting, maintenance, and harvesting
          </li>
          <li>
            <strong>Equipment:</strong> Machinery costs, fuel, maintenance, and depreciation
          </li>
          <li>
            <strong>Water:</strong> Irrigation systems and water usage
          </li>
          <li>
            <strong>Land:</strong> Rent or mortgage payments for farm land
          </li>
        </ul>
        <p class="mb-4">
          By tracking these costs against your projected yields, you can calculate expected margins
          and make informed decisions about crop selection and resource allocation.
        </p>
        <div class="bg-green-50 p-4 rounded-md">
          <h4 class="text-lg font-medium text-green-800 mb-2">Pro Tip</h4>
          <p class="text-green-800">
            Use our cost estimation tool to compare different scenarios and optimize your 
            farming strategy for maximum profitability.
          </p>
        </div>
      `,
    },
  ];

  return (
    <div className="bg-white rounded-lg shadow-md">
      <div className="flex flex-col md:flex-row">
        <div className="md:w-1/4 bg-gray-50 p-4 md:p-6 border-b md:border-b-0 md:border-r border-gray-200">
          <h3 className="text-lg font-medium mb-4">Topics</h3>
          <nav className="flex flex-col space-y-1">
            {guideSections.map((section) => (
              <button
                key={section.id}
                className={`px-3 py-2 rounded-md text-left transition-colors ${
                  activeSection === section.id
                    ? "bg-green-100 text-green-800 font-medium"
                    : "hover:bg-gray-100"
                }`}
                onClick={() => setActiveSection(section.id)}
              >
                {section.title}
              </button>
            ))}
          </nav>
        </div>

        <div className="md:w-3/4 p-6">
          <h2 className="text-2xl font-semibold mb-6">Farming Guide</h2>

          <div
            className="prose max-w-none"
            dangerouslySetInnerHTML={{
              __html:
                guideSections.find((s) => s.id === activeSection)?.content ||
                "",
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default Guide;
