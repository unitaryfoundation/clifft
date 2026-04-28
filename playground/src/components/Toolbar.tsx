import { useCallback, useMemo, useState, useRef, useEffect } from "react";
import {
  Play,
  Share2,
  Loader2,
  HelpCircle,
  ChevronDown,
  Sun,
  Moon,
  Settings2,
  Bookmark,
  Clock,
  Trash2,
  Columns2,
  Download,
  X,
} from "lucide-react";
import LZString from "lz-string";
import markLight from "@docs/assets/logos/clifft-mark-light.png";
import markDark from "@docs/assets/logos/clifft-mark-dark.png";
import type { WasmStatus, PassConfig } from "../hooks/useClifftWasm";
import type { Theme } from "../hooks/useTheme";
import type { PassInfo, SourceOrigin } from "../types";
import type { SavedCircuit } from "../hooks/useCircuitStorage";

interface Props {
  wasmStatus: WasmStatus;
  source: string;
  passConfig: PassConfig;
  onPassConfigChange: (config: PassConfig) => void;
  availablePasses: PassInfo[];
  onSimulate: () => void;
  simulating: boolean;
  shots: number;
  onShotsChange: (v: number) => void;
  onTourOpen: () => void;
  theme: Theme;
  onThemeToggle: () => void;
  diffView: boolean;
  onDiffViewChange: (v: boolean) => void;
  savedCircuits: SavedCircuit[];
  onSaveCircuit: (name: string, source: string) => void;
  onLoadCircuit: (source: string) => void;
  onLoadFromUrl: (source: string, url: string) => void;
  onDeleteCircuit: (id: string) => void;
  sourceOrigin: SourceOrigin;
}

const SAFE_URL_LENGTH = 8000;
const MAX_URL_LENGTH = 32000;

export function Toolbar({
  wasmStatus,
  source,
  passConfig,
  onPassConfigChange,
  availablePasses,
  onSimulate,
  simulating,
  shots,
  onShotsChange,
  onTourOpen,
  theme,
  onThemeToggle,
  diffView,
  onDiffViewChange,
  savedCircuits,
  onSaveCircuit,
  onLoadCircuit,
  onLoadFromUrl,
  onDeleteCircuit,
  sourceOrigin,
}: Props) {
  const [copied, setCopied] = useState(false);
  const [shotsOpen, setShotsOpen] = useState(false);
  const [passesOpen, setPassesOpen] = useState(false);
  const [recentsOpen, setRecentsOpen] = useState(false);
  const [loadOpen, setLoadOpen] = useState(false);
  const [loadUrl, setLoadUrl] = useState("");
  const [loadBusy, setLoadBusy] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadHelpOpen, setLoadHelpOpen] = useState(false);
  const shotsRef = useRef<HTMLDivElement>(null);
  const passesRef = useRef<HTMLDivElement>(null);
  const recentsRef = useRef<HTMLDivElement>(null);
  const loadInputRef = useRef<HTMLInputElement>(null);

  const buildShareUrl = useCallback(() => {
    if (sourceOrigin) {
      return `${window.location.origin}${window.location.pathname}?url=${encodeURIComponent(sourceOrigin.value)}`;
    }
    const compressed = LZString.compressToEncodedURIComponent(source);
    return `${window.location.origin}${window.location.pathname}?code=${compressed}`;
  }, [source, sourceOrigin]);

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

  const handleSave = () => {
    const name = window.prompt(
      "Save circuit as:",
      `Circuit ${new Date().toLocaleString()}`,
    );
    if (name) {
      onSaveCircuit(name, source);
    }
  };

  // Close dropdowns on outside click
  useEffect(() => {
    if (!shotsOpen && !passesOpen && !recentsOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (shotsOpen && shotsRef.current && !shotsRef.current.contains(e.target as Node)) {
        setShotsOpen(false);
      }
      if (passesOpen && passesRef.current && !passesRef.current.contains(e.target as Node)) {
        setPassesOpen(false);
      }
      if (recentsOpen && recentsRef.current && !recentsRef.current.contains(e.target as Node)) {
        setRecentsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [shotsOpen, passesOpen, recentsOpen]);

  const togglePass = (name: string, kind: "hir" | "bytecode") => {
    const key = kind === "hir" ? "hir" : "bc";
    const current = passConfig[key];
    const next = current.includes(name)
      ? current.filter((n) => n !== name)
      : [...current, name];
    onPassConfigChange({ ...passConfig, [key]: next });
  };

  const selectDefaults = () => {
    const hir = availablePasses
      .filter((p) => p.kind === "hir" && p.default)
      .map((p) => p.name);
    const bc = availablePasses
      .filter((p) => p.kind === "bytecode" && p.default)
      .map((p) => p.name);
    onPassConfigChange({ hir, bc });
  };

  const deselectAll = () => {
    onPassConfigChange({ hir: [], bc: [] });
  };

  const hirPasses = availablePasses.filter((p) => p.kind === "hir");
  const bcPasses = availablePasses.filter((p) => p.kind === "bytecode");
  const totalEnabled = passConfig.hir.length + passConfig.bc.length;

  const shareUrl = useMemo(() => buildShareUrl(), [buildShareUrl]);
  // When sharing via ?url=, the link size doesn't depend on the source length.
  const canShare = sourceOrigin !== null || shareUrl.length <= MAX_URL_LENGTH;
  const shareWarning =
    !sourceOrigin &&
    shareUrl.length > SAFE_URL_LENGTH &&
    shareUrl.length <= MAX_URL_LENGTH;
  const canSimulate = wasmStatus === "ready" && !simulating;

  // Auto-focus URL input on open
  useEffect(() => {
    if (loadOpen) {
      loadInputRef.current?.focus();
    }
  }, [loadOpen]);

  const closeLoadModal = useCallback(() => {
    setLoadOpen(false);
    setLoadError(null);
    setLoadBusy(false);
    setLoadHelpOpen(false);
  }, []);

  const submitLoad = useCallback(async () => {
    const url = loadUrl.trim();
    if (!url) {
      setLoadError("Enter a URL.");
      return;
    }
    setLoadBusy(true);
    setLoadError(null);
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Fetch returned ${resp.status}`);
      const text = await resp.text();
      onLoadFromUrl(text, url);
      setLoadUrl("");
      closeLoadModal();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLoadError(`Failed to load: ${msg}`);
      setLoadBusy(false);
    }
  }, [loadUrl, onLoadFromUrl, closeLoadModal]);

  return (
    <div className="toolbar">
      <div className="toolbar-left">
        <img
          src={theme === "dark" ? markDark : markLight}
          alt=""
          className="toolbar-mark"
          aria-hidden="true"
        />
        <span className="toolbar-title">Clifft Playground</span>
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
        {/* Configure Passes */}
        <div className="passes-group" ref={passesRef}>
          <button
            className="toolbar-btn"
            onClick={() => setPassesOpen((v) => !v)}
            title="Configure optimization passes"
          >
            <Settings2 size={14} />
            Passes ({totalEnabled})
          </button>
          {passesOpen && (
            <div className="passes-dropdown">
              <div className="passes-actions">
                <button className="passes-action" onClick={selectDefaults}>
                  Defaults
                </button>
                <button className="passes-action" onClick={deselectAll}>
                  None
                </button>
              </div>
              {hirPasses.length > 0 && (
                <>
                  <div className="passes-section-label">HIR Passes</div>
                  {hirPasses.map((p) => (
                    <label key={p.name} className="passes-item">
                      <input
                        type="checkbox"
                        checked={passConfig.hir.includes(p.name)}
                        onChange={() => togglePass(p.name, "hir")}
                      />
                      <span>{p.name}</span>
                    </label>
                  ))}
                </>
              )}
              {bcPasses.length > 0 && (
                <>
                  <div className="passes-section-label">Bytecode Passes</div>
                  {bcPasses.map((p) => (
                    <label key={p.name} className="passes-item">
                      <input
                        type="checkbox"
                        checked={passConfig.bc.includes(p.name)}
                        onChange={() => togglePass(p.name, "bytecode")}
                      />
                      <span>{p.name}</span>
                    </label>
                  ))}
                </>
              )}
            </div>
          )}
        </div>

        {/* Diff View toggle */}
        <label className="toolbar-toggle" title="Show diff between baseline and optimized">
          <input
            type="checkbox"
            checked={diffView}
            onChange={(e) => onDiffViewChange(e.target.checked)}
          />
          <Columns2 size={12} />
          Diff
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

        {/* Save circuit */}
        <button
          className="toolbar-btn"
          onClick={handleSave}
          title="Save circuit to local storage"
        >
          <Bookmark size={14} />
          Save
        </button>

        {/* Load from URL */}
        <button
          className="toolbar-btn"
          onClick={() => setLoadOpen(true)}
          title="Load a circuit from a CORS-enabled URL"
        >
          <Download size={14} />
          Load
        </button>

        {/* Recents dropdown */}
        <div className="recents-group" ref={recentsRef}>
          <button
            className="toolbar-btn"
            onClick={() => setRecentsOpen((v) => !v)}
            title="Recent saved circuits"
            disabled={savedCircuits.length === 0}
          >
            <Clock size={14} />
            Recents
            {savedCircuits.length > 0 && (
              <span className="recents-badge">{savedCircuits.length}</span>
            )}
          </button>
          {recentsOpen && savedCircuits.length > 0 && (
            <div className="recents-dropdown">
              {savedCircuits.map((c) => (
                <div key={c.id} className="recents-item">
                  <button
                    className="recents-item-name"
                    onClick={() => {
                      onLoadCircuit(c.source);
                      setRecentsOpen(false);
                    }}
                    title={c.source.slice(0, 100)}
                  >
                    <span className="recents-item-label">{c.name}</span>
                    <span className="recents-item-date">
                      {new Date(c.timestamp).toLocaleDateString()}
                    </span>
                  </button>
                  <button
                    className="recents-item-delete"
                    onClick={() => onDeleteCircuit(c.id)}
                    title="Delete"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <button
          className={`toolbar-btn${shareWarning ? " toolbar-btn-warn" : ""}`}
          onClick={handleShare}
          disabled={!canShare}
          title={
            !canShare
              ? "Circuit too large to share via URL. Host it somewhere CORS-enabled and use Load to paste its URL."
              : shareWarning
                ? "URL is long and may not work in all browsers. Host the circuit externally and use Load to paste its URL."
                : sourceOrigin
                  ? "Copy shareable link (pointing at the loaded URL)"
                  : "Copy shareable link"
          }
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

      {loadOpen && (
        <div
          className="load-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Load circuit from URL"
          onKeyDown={(e) => {
            if (e.key === "Escape") closeLoadModal();
          }}
        >
          <div className="load-backdrop" onClick={closeLoadModal} />
          <div className="load-card">
            <button
              className="load-close"
              onClick={closeLoadModal}
              title="Close"
              aria-label="Close"
            >
              <X size={16} />
            </button>
            <div className="load-title">
              Load circuit from URL
              <button
                className="load-help-btn"
                onClick={() => setLoadHelpOpen((v) => !v)}
                title="What kind of URL?"
                aria-label="Help"
              >
                <HelpCircle size={14} />
              </button>
            </div>
            <div className="load-desc">
              Paste a direct URL to a raw <code>.stim</code> file. The host must
              allow CORS. After loading, the Share link will point to this URL
              until you edit the circuit.
            </div>
            {loadHelpOpen && (
              <div className="load-help">
                <strong>Works with any CORS-enabled raw text URL</strong>, for
                example:
                <ul>
                  <li>
                    GitHub Gist raw file:
                    {" "}<code>https://gist.githubusercontent.com/&lt;user&gt;/&lt;id&gt;/raw/&lt;name&gt;.stim</code>
                  </li>
                  <li>
                    GitHub repo raw file:
                    {" "}<code>https://raw.githubusercontent.com/&lt;user&gt;/&lt;repo&gt;/&lt;ref&gt;/&lt;path&gt;.stim</code>
                  </li>
                  <li>Any public static host that serves plain text with CORS.</li>
                </ul>
                Not a playground share URL &mdash; use a URL that points
                directly at the circuit file itself.
              </div>
            )}
            <input
              ref={loadInputRef}
              type="url"
              className="load-input"
              placeholder="https://example.com/circuit.stim"
              value={loadUrl}
              onChange={(e) => setLoadUrl(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !loadBusy) {
                  e.preventDefault();
                  void submitLoad();
                }
              }}
              disabled={loadBusy}
            />
            {loadError && <div className="load-error">{loadError}</div>}
            <div className="load-actions">
              <button
                className="toolbar-btn"
                onClick={closeLoadModal}
                disabled={loadBusy}
              >
                Cancel
              </button>
              <button
                className="toolbar-btn toolbar-btn-primary"
                onClick={() => void submitLoad()}
                disabled={loadBusy || !loadUrl.trim()}
              >
                {loadBusy ? <Loader2 size={14} className="spin" /> : <Download size={14} />}
                Load
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
