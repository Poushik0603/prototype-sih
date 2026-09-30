import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Note: StrictMode intentionally omitted — it double-mounts effects in dev,
// which aborts MapLibre's in-flight style fetch (mount -> unmount -> remount).
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
