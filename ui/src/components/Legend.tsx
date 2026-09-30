export default function Legend() {
  return (
    <div className="legend">
      <div className="legend-title">Risk score</div>
      <div className="legend-gradient" />
      <div className="legend-labels">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>
      <div className="legend-note">Marker size also scales with risk</div>
    </div>
  );
}
