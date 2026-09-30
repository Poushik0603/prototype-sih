export interface ShapFeature {
  feature: string;
  value: number;
}

export interface RankedTerminal {
  terminal_id: string;
  lat: number;
  lon: number;
  window_start: string;
  risk_score: number;
  rank: number;
  top_shap_features: ShapFeature[];
  baseline_score: number;
  bank?: string;
  type?: string;
}

const API_BASE = "http://127.0.0.1:8000";

export async function fetchRankedTerminals(limit = 50): Promise<RankedTerminal[]> {
  const res = await fetch(`${API_BASE}/api/terminals/ranked?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch ranked terminals: ${res.status}`);
  return res.json();
}

export async function fetchTerminalDetail(terminalId: string): Promise<RankedTerminal> {
  const res = await fetch(`${API_BASE}/api/terminals/${terminalId}`);
  if (!res.ok) throw new Error(`Failed to fetch terminal ${terminalId}: ${res.status}`);
  return res.json();
}

export async function sendAlert(terminalId: string, officer: string, riskScore: number) {
  const res = await fetch(`${API_BASE}/api/alerts/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ terminal_id: terminalId, officer, risk_score: riskScore }),
  });
  if (!res.ok) throw new Error(`Failed to send alert: ${res.status}`);
  return res.json();
}
