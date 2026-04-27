import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Allotment } from "allotment";
import "allotment/dist/style.css";
import Editor, { DiffEditor, type BeforeMount, type OnMount } from "@monaco-editor/react";
import type { editor as monacoEditor } from "monaco-editor";
import LZString from "lz-string";
import { useClifftWasm } from "./hooks/useClifftWasm";
import type { PassConfig } from "./hooks/useClifftWasm";
import { useTheme } from "./hooks/useTheme";
import { useCircuitStorage, saveDraft, loadDraft } from "./hooks/useCircuitStorage";
import { Toolbar } from "./components/Toolbar";
import { KHistoryChart } from "./components/KHistoryChart";
import { HistogramChart } from "./components/HistogramChart";
import { ExpValTable } from "./components/ExpValTable";
import { GuidedTour } from "./components/GuidedTour";
import { registerLanguages } from "./languages";
import { isCompileSuccess, isSimulateSuccess } from "./types";
import type { CompileResult, CompileSuccess, SimulateResult, SourceOrigin } from "./types";

const DEFAULT_SOURCE = `H 0
CNOT 0 1
T 0
T 1
M 0
M 1
`;

const DEBOUNCE_MS = 200;
const DEFAULT_SHOTS = 10000;
const TOUR_SEEN_KEY = "clifft-tour-seen";

function getInitialSource(): string {
  const params = new URLSearchParams(window.location.search);
  const code = params.get("code");
  if (code) {
    try {
      const decoded = LZString.decompressFromEncodedURIComponent(code);
      if (decoded) return decoded;
    } catch {
      // ignore bad encoded data
    }
  }
  // For ?url= we return the default/draft immediately and load async
  // in a useEffect (see useRemoteCircuit below).
  const draft = loadDraft();
  if (draft) return draft;
  return DEFAULT_SOURCE;
}

type RemoteLoadState =
  | { status: "idle" }
  | { status: "loading"; label: string }
  | { status: "error"; message: string };

async function fetchUrl(url: string): Promise<string> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Fetch returned ${resp.status}`);
  return resp.text();
}

function getInitialRemoteState(): RemoteLoadState {
  const params = new URLSearchParams(window.location.search);
  // ?code= takes precedence — don't fetch remote if inline code is present
  if (params.get("code")) return { status: "idle" };
  if (params.get("url")) {
    return { status: "loading", label: "Loading circuit from URL..." };
  }
  return { status: "idle" };
}

function useRemoteCircuit(
  onLoad: (source: string, url: string) => void,
  userHasEditedRef: React.RefObject<boolean>,
) {
  const [state, setState] = useState<RemoteLoadState>(getInitialRemoteState);
  const ranRef = useRef(false);

  useEffect(() => {
    if (ranRef.current) return;
    const params = new URLSearchParams(window.location.search);
    // ?code= takes precedence over remote params
    if (params.get("code")) return;
    const url = params.get("url");
    if (!url) return;

    ranRef.current = true;
    fetchUrl(url)
      .then((content) => {
        setState({ status: "idle" });
        // Don't clobber user edits or manual circuit selections that
        // happened while the fetch was in flight.
        if (!userHasEditedRef.current) {
          onLoad(content, url);
        }
      })
      .catch((err) => setState({ status: "error", message: `Failed to load URL: ${err.message}` }));
  }, [onLoad, userHasEditedRef]);

  return state;
}

// Decoration style for highlighted lines
const HIGHLIGHT_CLASS = "source-map-highlight";

export default function App() {
  const { status: wasmStatus, compile, compileBaseline, simulate, availablePasses } = useClifftWasm();
  const { theme, toggle: toggleTheme, chartColors } = useTheme();
  const { saved, saveCircuit, deleteCircuit } = useCircuitStorage();
  const monacoTheme = theme === "dark" ? "vs-dark" : "vs";
  const rulerColor = theme === "dark" ? "#4fc3f7" : "#0277bd";

  const [source, setSourceRaw] = useState(getInitialSource);
  const [sourceOrigin, setSourceOrigin] = useState<SourceOrigin>(null);
  const userHasEditedRef = useRef(false);
  // When we push text into the Monaco editor programmatically, we skip
  // the resulting onChange so it doesn't clear sourceOrigin.
  const skipNextOnChangeRef = useRef(false);
  const setSource = useCallback((v: string) => {
    if (skipNextOnChangeRef.current) {
      skipNextOnChangeRef.current = false;
      return;
    }
    userHasEditedRef.current = true;
    setSourceRaw(v);
    setSourceOrigin(null);
  }, []);
  const defaultPassConfig = useMemo<PassConfig>(() => {
    if (availablePasses.length === 0) return { hir: [], bc: [] };
    return {
      hir: availablePasses.filter((p) => p.kind === "hir" && p.default).map((p) => p.name),
      bc: availablePasses.filter((p) => p.kind === "bytecode" && p.default).map((p) => p.name),
    };
  }, [availablePasses]);
  const [userPassConfig, setUserPassConfig] = useState<PassConfig | null>(null);
  const passConfig = userPassConfig ?? defaultPassConfig;
  const setPassConfig = useCallback((config: PassConfig) => setUserPassConfig(config), []);
  const passConfigReady = availablePasses.length > 0;
  const [compileResult, setCompileResult] = useState<CompileResult | null>(null);
  const [baselineResult, setBaselineResult] = useState<CompileResult | null>(null);
  const [diffView, setDiffViewRaw] = useState(false);
  const [simResult, setSimResult] = useState<SimulateResult | null>(null);
  const [simElapsedMs, setSimElapsedMs] = useState<number | null>(null);
  const [simulating, setSimulating] = useState(false);
  const [simSource, setSimSource] = useState<string | null>(null);
  const simStale = simResult !== null && simSource !== null && source !== simSource;
  const [simTab, setSimTab] = useState<"measurements" | "exp_vals">("measurements");
  const [shots, setShots] = useState(DEFAULT_SHOTS);
  const [cursorSourceLine, setCursorSourceLine] = useState<number | null>(null);
  const [tourOpen, setTourOpen] = useState(
    () => !localStorage.getItem(TOUR_SEEN_KEY),
  );

  // Editor refs
  const sourceEditorRef = useRef<monacoEditor.IStandaloneCodeEditor | null>(null);
  const hirEditorRef = useRef<monacoEditor.IStandaloneCodeEditor | null>(null);
  const bcEditorRef = useRef<monacoEditor.IStandaloneCodeEditor | null>(null);

  // Decoration collections
  const hirDecosRef = useRef<monacoEditor.IEditorDecorationsCollection | null>(null);
  const bcDecosRef = useRef<monacoEditor.IEditorDecorationsCollection | null>(null);
  const bcPcDecosRef = useRef<monacoEditor.IEditorDecorationsCollection | null>(null);
  const sourceDecosRef = useRef<monacoEditor.IEditorDecorationsCollection | null>(null);

  // Ref to avoid stale closures in editor cursor handlers
  const compileResultRef = useRef<CompileResult | null>(null);

  // Derived: which bytecode PC maps to the cursor source line (for k-history highlight)
  const highlightPC = useMemo(() => {
    if (!compileResult || !isCompileSuccess(compileResult) || cursorSourceLine === null)
      return null;
    const idx = compileResult.bytecode_source_map.findIndex((lines: number[]) =>
      lines.includes(cursorSourceLine),
    );
    return idx >= 0 ? idx : null;
  }, [cursorSourceLine, compileResult]);

  // --- Debounced compilation ---
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Callback: toggle diff view and eagerly compile baseline when turning on
  const setDiffView = useCallback(
    (on: boolean) => {
      setDiffViewRaw(on);
      if (on && wasmStatus === "ready") {
        setBaselineResult(compileBaseline(source));
      }
    },
    [wasmStatus, compileBaseline, source],
  );

  useEffect(() => {
    if (wasmStatus !== "ready" || !passConfigReady) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      const result = compile(source, passConfig);
      setCompileResult(result);

      // Save draft on successful compile
      if (result && isCompileSuccess(result)) {
        saveDraft(source);
      }

      // Lazy baseline: only compile when diff view is active
      if (diffView) {
        setBaselineResult(compileBaseline(source));
      }
    }, DEBOUNCE_MS);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [source, passConfig, passConfigReady, wasmStatus, compile, compileBaseline, diffView]);

  // Keep ref in sync for cursor handlers that close over mount-time state
  useEffect(() => {
    compileResultRef.current = compileResult;
  }, [compileResult]);

  // Register custom languages once before any editor mounts
  const handleBeforeMount: BeforeMount = useCallback((monaco) => {
    registerLanguages(monaco);
  }, []);

  // Update read-only editor text while preserving cursor/selection
  const updateReadOnlyEditor = useCallback(
    (editor: monacoEditor.IStandaloneCodeEditor | null, value: string) => {
      if (!editor) return;
      const sel = editor.getSelection();
      editor.setValue(value);
      if (sel) editor.setSelection(sel);
    },
    [],
  );

  // --- Update read-only editors when compile result changes ---
  useEffect(() => {
    // Skip editor updates when diff view is active (DiffEditor manages its own content)
    if (diffView) return;
    if (!compileResult || !isCompileSuccess(compileResult)) {
      const errorText = compileResult?.error ? `; Error: ${compileResult.error}` : "";
      updateReadOnlyEditor(hirEditorRef.current, errorText);
      updateReadOnlyEditor(bcEditorRef.current, errorText);
      return;
    }
    updateReadOnlyEditor(hirEditorRef.current, compileResult.hir_ops.join("\n"));
    updateReadOnlyEditor(bcEditorRef.current, compileResult.bytecode.join("\n"));
  }, [compileResult, updateReadOnlyEditor, diffView]);

  // --- Bidirectional highlighting (only in non-diff mode) ---
  useEffect(() => {
    if (diffView) {
      hirDecosRef.current?.clear();
      bcDecosRef.current?.clear();
      bcPcDecosRef.current?.clear();
      sourceDecosRef.current?.clear();
      return;
    }
    if (!compileResult || !isCompileSuccess(compileResult) || cursorSourceLine === null) {
      hirDecosRef.current?.clear();
      bcDecosRef.current?.clear();
      bcPcDecosRef.current?.clear();
      sourceDecosRef.current?.clear();
      return;
    }

    const srcLine = cursorSourceLine;

    // Find HIR ops that map to this source line
    const hirLines: number[] = [];
    compileResult.hir_source_map.forEach((lines: number[], i: number) => {
      if (lines.includes(srcLine)) hirLines.push(i + 1); // Monaco is 1-indexed
    });

    // Find bytecode instructions that map to this source line
    const bcLines: number[] = [];
    compileResult.bytecode_source_map.forEach((lines: number[], i: number) => {
      if (lines.includes(srcLine)) bcLines.push(i + 1);
    });

    // Highlight the source line itself
    sourceDecosRef.current?.set([
      {
        range: { startLineNumber: srcLine, startColumn: 1, endLineNumber: srcLine, endColumn: 1 },
        options: {
          isWholeLine: true,
          className: HIGHLIGHT_CLASS,
          overviewRuler: { color: rulerColor, position: 2 },
        },
      },
    ]);

    hirDecosRef.current?.set(
      hirLines.map((ln) => ({
        range: { startLineNumber: ln, startColumn: 1, endLineNumber: ln, endColumn: 1 },
        options: {
          isWholeLine: true,
          className: HIGHLIGHT_CLASS,
          overviewRuler: { color: rulerColor, position: 2 },
        },
      })),
    );

    bcDecosRef.current?.set(
      bcLines.map((ln) => ({
        range: { startLineNumber: ln, startColumn: 1, endLineNumber: ln, endColumn: 1 },
        options: {
          isWholeLine: true,
          className: HIGHLIGHT_CLASS,
          overviewRuler: { color: rulerColor, position: 2 },
        },
      })),
    );

    if (bcLines.length > 0) {
      bcPcDecosRef.current?.set([
        {
          range: {
            startLineNumber: bcLines[0],
            startColumn: 1,
            endLineNumber: bcLines[0],
            endColumn: 1,
          },
          options: {
            glyphMarginClassName: "pc-glyph-marker",
          },
        },
      ]);
    } else {
      bcPcDecosRef.current?.clear();
    }
  }, [cursorSourceLine, compileResult, rulerColor, diffView]);

  // --- Editor mount handlers ---
  const onSourceMount: OnMount = (editor) => {
    sourceEditorRef.current = editor;
    sourceDecosRef.current = editor.createDecorationsCollection();
    editor.onDidChangeCursorPosition((e) => {
      setCursorSourceLine(e.position.lineNumber);
    });
  };

  const onHirMount: OnMount = (editor) => {
    hirEditorRef.current = editor;
    hirDecosRef.current = editor.createDecorationsCollection();
    editor.onDidChangeCursorPosition((e) => {
      const cr = compileResultRef.current;
      if (!cr || !isCompileSuccess(cr)) return;
      const hirIdx = e.position.lineNumber - 1;
      const srcLines = cr.hir_source_map[hirIdx];
      const validLine = srcLines?.find((l: number) => l >= 1);
      if (validLine !== undefined) {
        setCursorSourceLine(validLine);
      }
    });
  };

  const onBcMount: OnMount = (editor) => {
    bcEditorRef.current = editor;
    bcDecosRef.current = editor.createDecorationsCollection();
    bcPcDecosRef.current = editor.createDecorationsCollection();
    editor.onDidChangeCursorPosition((e) => {
      const cr = compileResultRef.current;
      if (!cr || !isCompileSuccess(cr)) return;
      const bcIdx = e.position.lineNumber - 1;
      const srcLines = cr.bytecode_source_map[bcIdx];
      const validLine = srcLines?.find((l: number) => l >= 1);
      if (validLine !== undefined) {
        setCursorSourceLine(validLine);
      }
    });
  };

  // --- Simulation ---
  const handleSimulate = useCallback(() => {
    setSimulating(true);
    requestAnimationFrame(() => {
      const t0 = performance.now();
      const result = simulate(source, shots, passConfig);
      const elapsed = performance.now() - t0;
      setSimResult(result);
      setSimElapsedMs(elapsed);
      setSimSource(source);
      setSimulating(false);
      // Auto-switch tab if result has expectation values
      if (result && isSimulateSuccess(result) && result.exp_vals && result.exp_vals.length > 0) {
        setSimTab("exp_vals");
      } else {
        setSimTab("measurements");
      }
    });
  }, [source, shots, passConfig, simulate]);

  const handleTourClose = useCallback(() => {
    setTourOpen(false);
    localStorage.setItem(TOUR_SEEN_KEY, "1");
  }, []);

  // Unified entry point for non-edit source updates. origin=null means
  // "this content has no remote URL to share back" (e.g. Recents). A
  // non-null origin makes Share emit a `?url=` link until the user edits.
  const loadWithOrigin = useCallback((circuitSource: string, origin: SourceOrigin) => {
    userHasEditedRef.current = true;
    setSourceRaw(circuitSource);
    setSourceOrigin(origin);
    if (sourceEditorRef.current) {
      skipNextOnChangeRef.current = true;
      sourceEditorRef.current.setValue(circuitSource);
    }
  }, []);

  const handleLoadCircuit = useCallback(
    (circuitSource: string) => loadWithOrigin(circuitSource, null),
    [loadWithOrigin],
  );
  const handleLoadFromUrl = useCallback(
    (circuitSource: string, url: string) =>
      loadWithOrigin(circuitSource, { kind: "url", value: url }),
    [loadWithOrigin],
  );
  const handleRemoteLoad = useCallback(
    (circuitSource: string, url: string) =>
      loadWithOrigin(circuitSource, { kind: "url", value: url }),
    [loadWithOrigin],
  );

  const remoteState = useRemoteCircuit(handleRemoteLoad, userHasEditedRef);

  // --- Stats bar ---
  const stats: CompileSuccess | null =
    compileResult && isCompileSuccess(compileResult) ? compileResult : null;
  const baselineStats: CompileSuccess | null =
    baselineResult && isCompileSuccess(baselineResult) ? baselineResult : null;

  // Format a stat with optional before->after arrow (only in diff mode)
  const formatStat = (label: string, optimized: number, baseline: number | undefined) => {
    if (diffView && baseline !== undefined && baseline !== optimized) {
      return (
        <span key={label}>
          {label}: {baseline} {"\u2192"} {optimized}
        </span>
      );
    }
    return (
      <span key={label}>
        {label}: {optimized}
      </span>
    );
  };

  // Diff content for DiffEditor
  const hirBaseline = baselineStats ? baselineStats.hir_ops.join("\n") : "";
  const hirOptimized = stats ? stats.hir_ops.join("\n") : "";
  const bcBaseline = baselineStats ? baselineStats.bytecode.join("\n") : "";
  const bcOptimized = stats ? stats.bytecode.join("\n") : "";

  return (
    <div className="app">
      {remoteState.status === "loading" && (
        <div className="remote-load-banner">Loading: {remoteState.label}</div>
      )}
      {remoteState.status === "error" && (
        <div className="remote-load-banner remote-load-error">{remoteState.message}</div>
      )}
      <Toolbar
        wasmStatus={wasmStatus}
        source={source}
        passConfig={passConfig}
        onPassConfigChange={setPassConfig}
        availablePasses={availablePasses}
        onSimulate={handleSimulate}
        simulating={simulating}
        shots={shots}
        onShotsChange={setShots}
        onTourOpen={() => setTourOpen(true)}
        theme={theme}
        onThemeToggle={toggleTheme}
        diffView={diffView}
        onDiffViewChange={setDiffView}
        savedCircuits={saved}
        onSaveCircuit={saveCircuit}
        onLoadCircuit={handleLoadCircuit}
        onLoadFromUrl={handleLoadFromUrl}
        onDeleteCircuit={deleteCircuit}
        sourceOrigin={sourceOrigin}
      />

      {stats && (
        <div className="stats-bar">
          {formatStat("Qubits", stats.num_qubits, baselineStats?.num_qubits)}
          {formatStat("Peak k", stats.peak_rank, baselineStats?.peak_rank)}
          {formatStat("T gates", stats.num_t_gates, baselineStats?.num_t_gates)}
          {formatStat("Measurements", stats.num_measurements, baselineStats?.num_measurements)}
          {formatStat("HIR ops", stats.hir_ops.length, baselineStats?.hir_ops.length)}
          {formatStat("Bytecode", stats.bytecode.length, baselineStats?.bytecode.length)}
        </div>
      )}

      <div className="workspace">
        <Allotment vertical defaultSizes={[65, 35]}>
          {/* Top: 3-column editors */}
          <Allotment.Pane>
            <Allotment defaultSizes={[34, 33, 33]}>
              <Allotment.Pane>
                <div className="editor-pane">
                  <div className="editor-label">Source (.stim)</div>
                  <Editor
                    defaultLanguage="stim"
                    defaultValue={source}
                    theme={monacoTheme}
                    onChange={(v) => setSource(v ?? "")}
                    beforeMount={handleBeforeMount}
                    onMount={onSourceMount}
                    options={{
                      minimap: { enabled: false },
                      fontSize: 13,
                      lineNumbers: "on",
                      scrollBeyondLastLine: false,
                      wordWrap: "off",
                      automaticLayout: true,
                    }}
                  />
                </div>
              </Allotment.Pane>
              <Allotment.Pane>
                <div className="editor-pane">
                  <div className="editor-label">
                    HIR (Heisenberg IR)
                    {diffView && <span className="editor-label-badge">DIFF</span>}
                  </div>
                  {diffView ? (
                    <DiffEditor
                      original={hirBaseline}
                      modified={hirOptimized}
                      language="clifft-hir"
                      theme={monacoTheme}
                      beforeMount={handleBeforeMount}
                      options={{
                        readOnly: true,
                        minimap: { enabled: false },
                        fontSize: 13,
                        scrollBeyondLastLine: false,
                        automaticLayout: true,
                        renderSideBySide: false,
                      }}
                    />
                  ) : (
                    <Editor
                      defaultLanguage="clifft-hir"
                      defaultValue=""
                      theme={monacoTheme}
                      beforeMount={handleBeforeMount}
                      onMount={onHirMount}
                      options={{
                        readOnly: true,
                        minimap: { enabled: false },
                        fontSize: 13,
                        lineNumbers: "on",
                        scrollBeyondLastLine: false,
                        wordWrap: "off",
                        automaticLayout: true,
                        domReadOnly: true,
                      }}
                    />
                  )}
                </div>
              </Allotment.Pane>
              <Allotment.Pane>
                <div className="editor-pane">
                  <div className="editor-label">
                    VM Bytecode
                    {diffView && <span className="editor-label-badge">DIFF</span>}
                  </div>
                  {diffView ? (
                    <DiffEditor
                      original={bcBaseline}
                      modified={bcOptimized}
                      language="clifft-bytecode"
                      theme={monacoTheme}
                      beforeMount={handleBeforeMount}
                      options={{
                        readOnly: true,
                        minimap: { enabled: false },
                        fontSize: 13,
                        scrollBeyondLastLine: false,
                        automaticLayout: true,
                        renderSideBySide: false,
                      }}
                    />
                  ) : (
                    <Editor
                      defaultLanguage="clifft-bytecode"
                      defaultValue=""
                      theme={monacoTheme}
                      beforeMount={handleBeforeMount}
                      onMount={onBcMount}
                      options={{
                        readOnly: true,
                        minimap: { enabled: false },
                        fontSize: 13,
                        lineNumbers: "on",
                        scrollBeyondLastLine: false,
                        wordWrap: "off",
                        automaticLayout: true,
                        domReadOnly: true,
                        glyphMargin: true,
                      }}
                    />
                  )}
                </div>
              </Allotment.Pane>
            </Allotment>
          </Allotment.Pane>

          {/* Bottom: k-history chart + simulation results */}
          <Allotment.Pane>
            <Allotment defaultSizes={[50, 50]}>
              <Allotment.Pane>
                <div className="chart-pane">
                  <div className="chart-label">Active Dimensions (k) Timeline</div>
                  <div className="chart-container">
                    <KHistoryChart
                      history={stats?.active_k_history ?? []}
                      baselineHistory={
                        diffView && baselineStats
                          ? baselineStats.active_k_history
                          : undefined
                      }
                      highlightPC={highlightPC}
                      colors={chartColors}
                    />
                  </div>
                </div>
              </Allotment.Pane>
              <Allotment.Pane>
                <div className="chart-pane">
                  <div className="chart-label sim-tabs">
                    <button
                      className={`sim-tab${simTab === "measurements" ? " sim-tab-active" : ""}`}
                      onClick={() => setSimTab("measurements")}
                    >
                      Measurements
                    </button>
                    {simResult && isSimulateSuccess(simResult) && simResult.exp_vals && simResult.exp_vals.length > 0 && (
                      <button
                        className={`sim-tab${simTab === "exp_vals" ? " sim-tab-active" : ""}`}
                        onClick={() => setSimTab("exp_vals")}
                      >
                        Expectation Values
                      </button>
                    )}
                    {simResult && isSimulateSuccess(simResult) && (
                      <span className="chart-label-detail">
                        {" "}({simResult.shots.toLocaleString()} shots)
                      </span>
                    )}
                    {simStale && (
                      <span className="sim-stale-label">stale</span>
                    )}
                  </div>
                  <div className="chart-container">
                    {simTab === "exp_vals" && simResult && isSimulateSuccess(simResult) && simResult.exp_vals && simResult.exp_vals.length > 0 ? (
                      <ExpValTable expVals={simResult.exp_vals} shots={simResult.shots} elapsedMs={simElapsedMs} />
                    ) : (
                      <HistogramChart result={simResult} elapsedMs={simElapsedMs} colors={chartColors} />
                    )}
                  </div>
                </div>
              </Allotment.Pane>
            </Allotment>
          </Allotment.Pane>
        </Allotment>
      </div>
      {tourOpen && <GuidedTour onClose={handleTourClose} />}
    </div>
  );
}
