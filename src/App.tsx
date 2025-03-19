// import React, { useState, Suspense, useEffect } from "react";
// import "@fortawesome/fontawesome-free/css/all.min.css";
// import {
//   BrowserRouter as Router,
//   Routes,
//   Route,
//   Navigate,
// } from "react-router-dom";
// import { NavItem } from "./types/types";
// import LoadingPage from "./components/LoadingPage";

// // Lazy load ALL components
// const Sidebar = React.lazy(() => import("./components/Sidebar"));
// const Header = React.lazy(() => import("./components/Header"));
// const Dashboard = React.lazy(() => import("./components/Dashboard"));
// const MyProfile = React.lazy(() => import("./components/MyProfile"));
// const SelectCrop = React.lazy(() => import("./components/SelectCrop"));
// const CropTracker = React.lazy(() => import("./components/CropTracker"));
// const Guide = React.lazy(() => import("./components/Guide"));
// const Logout = React.lazy(() => import("./components/Logout"));
// const CropPredForm = React.lazy(() => import("./components/CropPredForm"));
// const WeedPre = React.lazy(() => import("./components/WeedPred"));
// const PestPredictor = React.lazy(() => import("./components/PestPred"));

// const App: React.FC = () => {
//   const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
//   const [isLoading, setIsLoading] = useState<boolean>(true);

//   const navItems: NavItem[] = [
//     { id: "dashboard", title: "Dashboard", icon: "chart-line" },
//     { id: "profile", title: "My Profile", icon: "user" },
//     { id: "select-crop", title: "Select your crop", icon: "seedling" },
//     { id: "crop-tracker", title: "Crop Tracker", icon: "tasks" },
//     { id: "guide", title: "Guide", icon: "book" },
//     { id: "logout", title: "Logout", icon: "sign-out-alt" },
//   ];

//   useEffect(() => {
//     // Simulate initial loading time
//     const timer = setTimeout(() => {
//       setIsLoading(false);
//     }, 2000); // Show loading screen for 2 seconds

//     return () => clearTimeout(timer);
//   }, []);

//   if (isLoading) {
//     return <LoadingPage />;
//   }

//   return (
//     <Router>
//       <Suspense fallback={<LoadingPage />}>
//         <div className="flex h-screen bg-gray-100">
//           <Sidebar
//             navItems={navItems}
//             isOpen={sidebarOpen}
//             toggleSidebar={() => setSidebarOpen(!sidebarOpen)}
//           />
//           <div
//             className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${
//               sidebarOpen ? "md:ml-64" : "md:ml-20"
//             }`}
//           >
//             <Header />
//             <main className="flex-1 overflow-x-hidden overflow-y-auto p-4">
//               <Routes>
//                 <Route path="/" element={<Navigate to="/dashboard" />} />
//                 <Route path="/dashboard" element={<Dashboard />} />
//                 <Route
//                   path="/dashboard/crop-predictor"
//                   element={<CropPredForm />}
//                 />
//                 <Route path="/dashboard/weeds" element={<WeedPre />} />
//                 <Route
//                   path="/dashboard/pesticides"
//                   element={<PestPredictor />}
//                 />
//                 <Route path="/profile" element={<MyProfile />} />
//                 <Route path="/select-crop" element={<SelectCrop />} />
//                 <Route path="/crop-tracker" element={<CropTracker />} />
//                 <Route path="/guide" element={<Guide />} />
//                 <Route path="/logout" element={<Logout />} />
//               </Routes>
//             </main>
//           </div>
//         </div>
//       </Suspense>
//     </Router>
//   );
// };

// export default App;

import React, { useState, Suspense, useEffect } from "react";
import "@fortawesome/fontawesome-free/css/all.min.css";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { NavItem } from "./types/types";
import LoadingPage from "./components/LoadingPage";
import FertilizerRecommendation from "./components/FertRecom";
import MaizeDiseasePredictor from "./components/Mazie";
import PestRecommendation from "./components/Pest";

const Sidebar = React.lazy(() => import("./components/Sidebar"));
const Header = React.lazy(() => import("./components/Header"));
const Dashboard = React.lazy(() => import("./components/Dashboard"));
const MyProfile = React.lazy(() => import("./components/MyProfile"));
const SelectCrop = React.lazy(() => import("./components/SelectCrop"));
const CropTracker = React.lazy(() => import("./components/CropTracker"));
const Guide = React.lazy(() => import("./components/Guide"));
const Logout = React.lazy(() => import("./components/Logout"));
const CropPredForm = React.lazy(() => import("./components/CropPredForm"));
const WeedPre = React.lazy(() => import("./components/WeedPred"));
const PestPredictor = React.lazy(() => import("./components/PestPred"));
const CropReqNut = React.lazy(() => import("./components/CropReqNut"));

const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const navItems: NavItem[] = [
    { id: "dashboard", title: "Dashboard", icon: "chart-line" },
    { id: "profile", title: "My Profile", icon: "user" },
    { id: "select-crop", title: "Select your crop", icon: "seedling" },
    { id: "crop-tracker", title: "Crop Tracker", icon: "tasks" },
    { id: "guide", title: "Guide", icon: "book" },
    { id: "logout", title: "Logout", icon: "sign-out-alt" },
  ];

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000); // Ensure loading screen appears during initial load

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return <LoadingPage />;
  }

  return (
    <Router>
      <Suspense fallback={<LoadingPage />}>
        <div className="flex h-screen bg-gray-100">
          <div className="hidden md:block">
            <Sidebar
              navItems={navItems}
              isOpen={sidebarOpen}
              toggleSidebar={() => setSidebarOpen(!sidebarOpen)}
            />
          </div>
          <div
            className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${
              sidebarOpen ? "md:ml-64" : "md:ml-20"
            }`}
          >
            <Header />
            <main className="flex-1 overflow-x-hidden overflow-y-auto p-4">
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route
                  path="/dashboard/crop-predictor"
                  element={<CropPredForm />}
                />
                <Route path="/dashboard/weeds" element={<WeedPre />} />
                <Route
                  path="/dashboard/Pest Identification"
                  element={<PestPredictor />}
                />
                <Route
                  path="/dashboard/Fertilizer"
                  element={<FertilizerRecommendation />}
                />
                <Route
                  path="/dashboard/Mazie-Pred"
                  element={<MaizeDiseasePredictor />}
                />
                <Route
                  path="/dashboard/Pest-Pred"
                  element={<PestRecommendation />}
                />
                <Route path="/dashboard/crop-ReqNut" element={<CropReqNut />} />
                <Route path="/profile" element={<MyProfile />} />
                <Route path="/select-crop" element={<SelectCrop />} />
                <Route path="/crop-tracker" element={<CropTracker />} />
                <Route path="/guide" element={<Guide />} />
                <Route path="/logout" element={<Logout />} />
              </Routes>
            </main>
          </div>
        </div>
      </Suspense>
    </Router>
  );
};

export default App;
