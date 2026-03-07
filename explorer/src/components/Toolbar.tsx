import { useCallback, useState, useRef, useEffect } from "react";
import { Play, Share2, Loader2, HelpCircle, ChevronDown, Sun, Moon } from "lucide-react";
import LZString from "lz-string";
import type { WasmStatus } from "../hooks/useUccWasm";
import type { Theme } from "../hooks/useTheme";

interface Props {
  wasmStatus: WasmStatus;
  source: string;
  optimize: boolean;
  onOptimizeChange: (v: boolean) => void;
  onSimulate: () => void;
  simulating: boolean;
  shots: number;
  onShotsChange: (v: number) => void;
  onTourOpen: () => void;
  theme: Theme;
  onThemeToggle: () => void;
}

const MAX_URL_LENGTH = 8000;

export function Toolbar({
  wasmStatus,
  source,
  optimize,
  onOptimizeChange,
  onSimulate,
  simulating,
  shots,
  onShotsChange,
  onTourOpen,
  theme,
  onThemeToggle,
}: Props) {
  const [copied, setCopied] = useState(false);
  const [shotsOpen, setShotsOpen] = useState(false);
  const shotsRef = useRef<HTMLDivElement>(null);

  const buildShareUrl = useCallback(() => {
    const compressed = LZString.compressToEncodedURIComponent(source);
    return `${window.location.origin}${window.location.pathname}?code=${compressed}`;
  }, [source]);

  const handleShare = async () => {
    const url = buildShareUrl();
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      window.prompt("Copy this URL:", url);
    }
  };

  // Close shots dropdown on outside click
  useEffect(() => {
    if (!shotsOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (shotsRef.current && !shotsRef.current.contains(e.target as Node)) {
        setShotsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [shotsOpen]);

  // Approximate: source length correlates with URL length; exact check at share time
  const canShare = source.length <= MAX_URL_LENGTH;
  const canSimulate = wasmStatus === "ready" && !simulating;

  return (
    <div className="toolbar">
      <div className="toolbar-left">
        <span className="toolbar-title">UCC Compiler Explorer</span>
        <button
          className="toolbar-btn toolbar-btn-tour"
          onClick={onTourOpen}
          title="Guided tour"
        >
          <HelpCircle size={14} />
          Tour
        </button>
        <span className={`wasm-status wasm-status-${wasmStatus}`}>
          {wasmStatus === "loading" && (
            <><Loader2 size={12} className="spin" /> Loading Wasm...</>
          )}
          {wasmStatus === "ready" && "Wasm ready"}
          {wasmStatus === "error" && "Wasm failed to load"}
        </span>
      </div>
      <div className="toolbar-right">
        <label className="toolbar-toggle">
          <input
            type="checkbox"
            checked={optimize}
            onChange={(e) => onOptimizeChange(e.target.checked)}
          />
          Optimize
        </label>

        {/* Simulate button group with shots dropdown */}
        <div className="simulate-group" ref={shotsRef}>
          <button
            className="toolbar-btn toolbar-btn-primary simulate-main"
            onClick={onSimulate}
            disabled={!canSimulate}
            title={`Run ${shots.toLocaleString()}-shot Monte Carlo simulation`}
          >
            {simulating ? <Loader2 size={14} className="spin" /> : <Play size={14} />}
            Simulate
          </button>
          <button
            className="toolbar-btn toolbar-btn-primary simulate-toggle"
            onClick={() => setShotsOpen((v) => !v)}
            disabled={!canSimulate}
            title="Configure shots"
          >
            <ChevronDown size={12} />
          </button>
          {shotsOpen && (
            <div className="shots-dropdown">
              <label className="shots-label">
                Shots
                <input
                  type="number"
                  className="shots-input"
                  value={shots}
                  min={1}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (v > 0) onShotsChange(v);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      setShotsOpen(false);
                      onSimulate();
                    }
                  }}
                />
              </label>
              <div className="shots-presets">
                {[1000, 10000, 100000].map((n) => (
                  <button
                    key={n}
                    className={`shots-preset ${shots === n ? "shots-preset-active" : ""}`}
                    onClick={() => onShotsChange(n)}
                  >
                    {n.toLocaleString()}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <button
          className="toolbar-btn"
          onClick={handleShare}
          disabled={!canShare}
          title={canShare ? "Copy shareable link" : "Circuit too long to share via URL"}
        >
          <Share2 size={14} />
          {copied ? "Copied!" : "Share"}
        </button>

        <button
          className="toolbar-btn toolbar-btn-theme"
          onClick={onThemeToggle}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
        </button>
      </div>
    </div>
  );
}
