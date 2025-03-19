// import React from "react";
// import { CardInfo } from "../types/types";

// // Define CardProps for Cards component
// interface CardProps {
//   cards: CardInfo[];
// }

// const Cards: React.FC<CardProps> = ({ cards }) => {
//   const handleClick = (id: string) => {
//     console.log(`Card clicked: ${id}`);
//   };

//   return (
//     <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
//       {cards.map((card) => (
//         <div
//           key={card.id}
//           className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1 text-center cursor-pointer"
//           onClick={() => handleClick(card.id)}
//         >
//           <div className="text-green-600 text-4xl mb-4">
//             <i className={`fas fa-${card.icon}`}></i>
//           </div>
//           <h3 className="text-green-600 text-lg font-medium">{card.title}</h3>
//         </div>
//       ))}
//     </div>
//   );
// };

// const Dashboard: React.FC = () => {
//   const cards: CardInfo[] = [
//     { id: "CropPredForm", title: "Crop Predictor", icon: "seedling" },
//     { id: "weeds", title: "Weeds Identification", icon: "shield-alt" },
//     { id: "Pest Identification", title: "Pest Identification", icon: "clipboard-list" },
//   ];

//   const generateChartData = () => {
//     return Array(7)
//       .fill(0)
//       .map(() => Math.floor(Math.random() * 100));
//   };

//   return (
//     <div>
//       <h2 className="text-2xl font-semibold mb-6">Dashboard Overview</h2>

//       <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
//         {[
//           { label: "Total Crops", value: "14%", color: "green-500" },
//           { label: "Active Plans", value: "5%", color: "blue-500" },
//           { label: "Crop Yield (tonnes)", value: "85.2%", color: "yellow-500" },
//         ].map((item, index) => (
//           <div key={index} className="bg-white rounded-lg shadow-md p-5">
//             <div className="text-gray-500 text-sm mb-1">{item.label}</div>
//             <div className="text-2xl font-semibold mb-4">{item.value}</div>
//             <div className="h-24 bg-gray-100 rounded flex items-end p-2">
//               {generateChartData().map((h, i) => (
//                 <div
//                   key={i}
//                   className={`flex-1 bg-${item.color} rounded-t mx-px`}
//                   style={{ height: `${h}%` }}
//                 ></div>
//               ))}
//             </div>
//           </div>
//         ))}
//       </div>

//       <Cards cards={cards} />

//       <div className="bg-white rounded-lg shadow-md p-6">
//         <h3 className="text-lg font-medium text-gray-800 mb-4">
//           Recent Activities
//         </h3>
//         <div className="space-y-4">
//           {[
//             {
//               icon: "leaf",
//               text: "New crop added: Wheat",
//               time: "Just now",
//               color: "green-100",
//               iconColor: "green-600",
//             },
//             {
//               icon: "tint",
//               text: "Irrigation schedule updated",
//               time: "3 hours ago",
//               color: "blue-100",
//               iconColor: "blue-600",
//             },
//             {
//               icon: "exclamation-triangle",
//               text: "Pest alert: Aphids detected in Zone 2",
//               time: "Yesterday",
//               color: "red-100",
//               iconColor: "red-600",
//             },
//             {
//               icon: "sun",
//               text: "Weather alert: High temperatures expected",
//               time: "2 days ago",
//               color: "yellow-100",
//               iconColor: "yellow-600",
//             },
//           ].map((activity, index) => (
//             <div
//               key={index}
//               className="flex items-center py-2 border-b border-gray-100"
//             >
//               <div
//                 className={`w-10 h-10 rounded-full bg-${activity.color} flex items-center justify-center text-${activity.iconColor} mr-4`}
//               >
//                 <i className={`fas fa-${activity.icon}`}></i>
//               </div>
//               <div className="flex-1">
//                 <div className="text-sm text-gray-800 font-medium">
//                   {activity.text}
//                 </div>
//                 <div className="text-xs text-gray-500">{activity.time}</div>
//               </div>
//             </div>
//           ))}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Dashboard;

/*eslint-disable*/
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CardInfo } from "../types/types";

interface CardProps {
  cards: CardInfo[];
  onSelect: (id: string) => void;
}

const Cards: React.FC<CardProps> = ({ cards, onSelect }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {cards.map((card) => {
        const [isHovered, setIsHovered] = useState(false);
        return (
          <article
            key={card.id}
            className="grid place-items-center perspective-[1000px] cursor-pointer relative mt-20"
            onClick={() => onSelect(card.id)}
          >
            <div
              className={`w-45 h-28 rounded-2xl bg-white shadow-inner transition-transform duration-1000
                ${
                  isHovered
                    ? "translate-z-[100px] rotate-x-[75deg] shadow-xl"
                    : "shadow-inner"
                }`}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            />
            <nav
              className={`absolute w-8 text-center translate-x-[-80%] transition-transform duration-1000 inset-1/2
                ${
                  isHovered
                    ? "translate-y-[-90px] translate-z-[250px]"
                    : "translate-y-[-20px]"
                }`}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            >
              <i
                className={`fas fa-${card.icon} block text-2xl text-green-600`}
              ></i>
              <span className="block text-sm font-medium text-green-600">
                {card.title}
              </span>
            </nav>
            <p
              className={`absolute bottom-[-50px] w-full text-center text-gray-700 opacity-0 transition-opacity duration-500 ease-in-out
                ${isHovered ? "opacity-100 translate-y-[-10px]" : ""}`}
            >
              {card.description}
            </p>
          </article>
        );
      })}
    </div>
  );
};

const Dashboard: React.FC = () => {
  const navigate = useNavigate(); // Hook for navigation

  const cards: CardInfo[] = [
    {
      id: "crop-predictor",
      title: "Crop Predictor",
      icon: "seedling",
      description:
        "Predict the best crops to grow based on weather and soil conditions.",
    },
    {
      id: "weeds",
      title: "Weeds Identification",
      icon: "leaf",
      description:
        "Identify and classify weeds to enhance weed control strategies.",
    },
    {
      id: "pest-identification",
      title: "Pest Identification",
      icon: "bug",
      description:
        "Identify harmful pests affecting crops and get suitable control measures.",
    },
    {
      id: "crop-reqnut",
      title: "Crop Nutrient Requirements",
      icon: "flask",
      description:
        "Determine the necessary nutrients for optimal crop growth and yield.",
    },
    {
      id: "fertilizer",
      title: "Fertilizer Recommendation",
      icon: "tractor",
      description:
        "Get precise fertilizer recommendations based on soil and crop data.",
    },
    {
      id: "maize-pred",
      title: "Maize Disease Prediction",
      icon: "exclamation-triangle",
      description:
        "Detect and diagnose diseases in maize crops using image analysis.",
    },
    {
      id: "pest-pred",
      title: "Pest Control Recommendation",
      icon: "spray-can",
      description:
        "Find the best pesticide recommendations based on identified pests.",
    },
  ];
  const generateChartData = () => {
    return Array(7)
      .fill(0)
      .map(() => Math.floor(Math.random() * 100));
  };
  const colorMap: { [key: string]: string } = {
    "green-500": "bg-green-500",
    "pink-500": "bg-pink-500",
    "yellow-500": "bg-yellow-500",
  };

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-6">Dashboard Overview</h2>

      {/* Dashboard Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {[
          { label: "Total Crops", value: "14%", color: "green-500" },
          { label: "Active Plans", value: "5%", color: "pink-500" },
          { label: "Crop Yield (tonnes)", value: "85.2%", color: "yellow-500" },
        ].map((item, index) => (
          <div key={index} className="bg-white rounded-lg shadow-md p-5">
            <div className="text-gray-500 text-sm mb-1">{item.label}</div>
            <div className="text-2xl font-semibold mb-4">{item.value}</div>
            <div className="h-24 bg-gray-100 rounded flex items-end p-2">
              {generateChartData().map((h, i) => (
                <div
                  key={i}
                  className={`flex-1 ${colorMap[item.color]} rounded-t mx-px`}
                  style={{ height: `${h}%` }}
                ></div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Cards Navigation */}
      <div className="mb-20">
        <Cards cards={cards} onSelect={(id) => navigate(`/dashboard/${id}`)} />
      </div>

      {/* Recent Activities */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-medium text-gray-800 mb-4">
          Recent Activities
        </h3>
        <div className="space-y-4">
          {[
            { icon: "leaf", text: "New crop added: Wheat", time: "Just now" },
            {
              icon: "tint",
              text: "Irrigation schedule updated",
              time: "3 hours ago",
            },
            {
              icon: "exclamation-triangle",
              text: "Pest alert: Aphids detected in Zone 2",
              time: "Yesterday",
            },
            {
              icon: "sun",
              text: "Weather alert: High temperatures expected",
              time: "2 days ago",
            },
          ].map((activity, index) => (
            <div
              key={index}
              className="flex items-center py-2 border-b border-gray-100"
            >
              <div className="w-10 h-10 rounded-full flex items-center justify-center bg-gray-100 mr-4">
                <i className={`fas fa-${activity.icon}`}></i>
              </div>
              <div className="flex-1">
                <div className="text-sm text-gray-800 font-medium">
                  {activity.text}
                </div>
                <div className="text-xs text-gray-500">{activity.time}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
