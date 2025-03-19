import { StrictMode, useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

// Import FontAwesome library and specific icons
import { library } from "@fortawesome/fontawesome-svg-core";
import {
  faChartLine,
  faUser,
  faSeedling,
  faTasks,
  faBook,
  faSignOutAlt,
} from "@fortawesome/free-solid-svg-icons";

// Register the icons
library.add(faChartLine, faUser, faSeedling, faTasks, faBook, faSignOutAlt);

import LoadingPage from "./components/LoadingPage";
/*eslint-disable*/
const RootComponent: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 2500);
    return () => clearTimeout(timer);
  }, []);

  return isLoading ? <LoadingPage /> : <App />;
};

// Ensure "root" exists before rendering
const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(
    <StrictMode>
      <RootComponent />
    </StrictMode>
  );
} else {
  console.error("Root element not found.");
}
