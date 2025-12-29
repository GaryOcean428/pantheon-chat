import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import "./styles/ocean-consciousness.css";

// Simplified initialization - removed WebSocket wrapper that was causing issues
createRoot(document.getElementById("root")!).render(<App />);
