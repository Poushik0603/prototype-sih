import { RankedTerminal } from "../api";
import { riskToColor } from "../riskColor";

interface Props {
  terminals: RankedTerminal[];
  onSelect: (t: RankedTerminal) => void;
  selectedId?: string;
}

export default function RankedSidebar({ terminals, onSelect, selectedId }: Props) {
  const top10 = terminals.slice(0, 10);

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>Top 10 Ranked Terminals</h2>
        <p className="sidebar-subtitle">Highest predicted mule-network risk</p>
      </div>
      <ul className="terminal-list">
        {top10.map((t) => (
          <li
            key={t.terminal_id}
            className={`terminal-item ${t.terminal_id === selectedId ? "selected" : ""}`}
            onClick={() => onSelect(t)}
          >
            <span className="rank-badge">#{t.rank}</span>
            <div className="terminal-info">
              <div className="terminal-id">{t.terminal_id}</div>
              <div className="terminal-meta">
                {t.type ?? "ATM"} &middot; {t.bank ?? "—"}
              </div>
            </div>
            <div className="risk-chip" style={{ background: riskToColor(t.risk_score) }}>
              {t.risk_score.toFixed(2)}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
