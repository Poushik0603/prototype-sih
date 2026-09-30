import { useEffect, useRef } from "react";
import maplibregl, { Map as MLMap, Marker } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { RankedTerminal } from "../api";
import { riskToColor, riskToRadius } from "../riskColor";

const STYLE_URL = "https://tiles.openfreemap.org/styles/liberty";

interface Props {
  terminals: RankedTerminal[];
  onSelect: (t: RankedTerminal) => void;
  selectedId?: string;
}

export default function MapView({ terminals, onSelect, selectedId }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MLMap | null>(null);
  const markersRef = useRef<Map<string, Marker>>(new Map());

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const center: [number, number] =
      terminals.length > 0
        ? [terminals[0].lon, terminals[0].lat]
        : [77.5946, 12.9716];

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: STYLE_URL,
      center,
      zoom: 11,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const existing = markersRef.current;
    const nextIds = new Set(terminals.map((t) => t.terminal_id));

    // remove stale markers
    for (const [id, marker] of existing) {
      if (!nextIds.has(id)) {
        marker.remove();
        existing.delete(id);
      }
    }

    terminals.forEach((t) => {
      const radius = riskToRadius(t.risk_score);
      const color = riskToColor(t.risk_score);
      const isSelected = t.terminal_id === selectedId;

      let marker = existing.get(t.terminal_id);
      if (!marker) {
        const el = document.createElement("div");
        el.className = "risk-marker";
        el.addEventListener("click", (e) => {
          e.stopPropagation();
          onSelect(t);
        });
        marker = new maplibregl.Marker({ element: el })
          .setLngLat([t.lon, t.lat])
          .addTo(map);
        existing.set(t.terminal_id, marker);
      }

      const el = marker.getElement();
      el.style.width = `${radius * 2}px`;
      el.style.height = `${radius * 2}px`;
      el.style.borderRadius = "50%";
      el.style.background = color;
      el.style.border = isSelected ? "3px solid #0b0b0b" : "2px solid rgba(255,255,255,0.85)";
      el.style.boxShadow = isSelected
        ? "0 0 0 4px rgba(11,11,11,0.15)"
        : "0 1px 3px rgba(0,0,0,0.35)";
      el.style.cursor = "pointer";
      el.title = `${t.terminal_id} — risk ${t.risk_score.toFixed(2)}`;
    });
  }, [terminals, selectedId, onSelect]);

  return <div ref={containerRef} className="map-container" />;
}
