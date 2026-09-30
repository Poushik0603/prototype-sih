import { useEffect, useState } from "react";
import MapView from "./components/MapView";
import RankedSidebar from "./components/RankedSidebar";
import Legend from "./components/Legend";
import AlertPanel from "./components/AlertPanel";
import { fetchRankedTerminals, RankedTerminal } from "./api";

export default function App() {
  const [terminals, setTerminals] = useState<RankedTerminal[]>([]);
  const [selected, setSelected] = useState<RankedTerminal | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchRankedTerminals(50)
      .then((data) => {
        setTerminals(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(String(err));
        setLoading(false);
      });
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">Cassandra AI</span>
          <span className="brand-tagline">Fraud &amp; Mule-Network Detection</span>
        </div>
        <div className="header-status">
          {loading
            ? "Loading terminals..."
            : error
            ? `Error: ${error}`
            : `${terminals.length} terminals ranked`}
        </div>
      </header>

      <div className="app-body">
        <RankedSidebar
          terminals={terminals}
          onSelect={setSelected}
          selectedId={selected?.terminal_id}
        />

        <div className="map-area">
          <MapView terminals={terminals} onSelect={setSelected} selectedId={selected?.terminal_id} />
          <Legend />
        </div>

        {selected && <AlertPanel terminal={selected} onClose={() => setSelected(null)} />}
      </div>
    </div>
  );
}
