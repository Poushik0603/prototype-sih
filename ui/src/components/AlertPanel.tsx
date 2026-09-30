import { useState } from "react";
import { RankedTerminal, sendAlert } from "../api";
import { riskToColor, riskLabel } from "../riskColor";

interface Props {
  terminal: RankedTerminal | null;
  onClose: () => void;
}

export default function AlertPanel({ terminal, onClose }: Props) {
  const [sending, setSending] = useState(false);
  const [confirmation, setConfirmation] = useState<string | null>(null);

  if (!terminal) return null;

  const maxShap = Math.max(...terminal.top_shap_features.map((f) => f.value), 0.0001);

  const handleSendAlert = async () => {
    setSending(true);
    setConfirmation(null);
    try {
      const result = await sendAlert(terminal.terminal_id, "Duty Officer", terminal.risk_score);
      setConfirmation(result.detail ?? "Alert sent.");
    } catch (err) {
      setConfirmation("Failed to send alert (API unreachable).");
    } finally {
      setSending(false);
      setTimeout(() => setConfirmation(null), 5000);
    }
  };

  return (
    <div className="alert-panel">
      <div className="alert-panel-header">
        <div>
          <h2>{terminal.terminal_id}</h2>
          <p className="alert-panel-subtitle">
            {terminal.type ?? "ATM"} &middot; {terminal.bank ?? "—"}
          </p>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">
          &times;
        </button>
      </div>

      <div className="risk-summary">
        <div className="risk-score-block">
          <div className="risk-score-value" style={{ color: riskToColor(terminal.risk_score) }}>
            {terminal.risk_score.toFixed(2)}
          </div>
          <div className="risk-score-label">
            {riskLabel(terminal.risk_score)} risk &middot; rank #{terminal.rank}
          </div>
        </div>
        <div className="risk-meta">
          <div>
            <span className="meta-label">Baseline score</span>
            <span className="meta-value">{terminal.baseline_score.toFixed(2)}</span>
          </div>
          <div>
            <span className="meta-label">Window</span>
            <span className="meta-value">
              {new Date(terminal.window_start).toLocaleString()}
            </span>
          </div>
          <div>
            <span className="meta-label">Location</span>
            <span className="meta-value">
              {terminal.lat.toFixed(4)}, {terminal.lon.toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      <div className="shap-section">
        <h3>Top contributing factors (SHAP)</h3>
        <div className="shap-bars">
          {terminal.top_shap_features.map((f) => (
            <div className="shap-row" key={f.feature}>
              <span className="shap-feature-name">{f.feature}</span>
              <div className="shap-bar-track">
                <div
                  className="shap-bar-fill"
                  style={{ width: `${(f.value / maxShap) * 100}%` }}
                />
              </div>
              <span className="shap-value">{f.value.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </div>

      <button className="send-alert-btn" onClick={handleSendAlert} disabled={sending}>
        {sending ? "Sending..." : "Send Alert to Officer"}
      </button>

      {confirmation && <div className="toast">{confirmation}</div>}
    </div>
  );
}
