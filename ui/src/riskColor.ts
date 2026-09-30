// Risk gradient: green (low) -> yellow (mid) -> red (high), using the
// status-palette steps (good / warning / critical) from the design system.
// Color is never the sole carrier of meaning: every marker also encodes
// risk via size, and every list/panel shows the numeric score alongside it.

const GOOD = [12, 163, 12]; // #0ca30c
const WARNING = [250, 178, 25]; // #fab219
const CRITICAL = [208, 59, 59]; // #d03b3b

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function mix(c1: number[], c2: number[], t: number) {
  return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)];
}

export function riskToColor(score: number): string {
  const s = Math.max(0, Math.min(1, score));
  let rgb: number[];
  if (s < 0.5) {
    rgb = mix(GOOD, WARNING, s / 0.5);
  } else {
    rgb = mix(WARNING, CRITICAL, (s - 0.5) / 0.5);
  }
  return `rgb(${Math.round(rgb[0])}, ${Math.round(rgb[1])}, ${Math.round(rgb[2])})`;
}

export function riskToRadius(score: number): number {
  // 6px (low) -> 18px (high)
  return 6 + Math.max(0, Math.min(1, score)) * 12;
}

export function riskLabel(score: number): string {
  if (score >= 0.66) return "High";
  if (score >= 0.33) return "Medium";
  return "Low";
}
