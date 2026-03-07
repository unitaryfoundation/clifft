import { useState, useCallback, useRef, useEffect, type KeyboardEvent } from "react";
import { X, ChevronLeft, ChevronRight } from "lucide-react";

interface TourStep {
  title: string;
  content: string;
  target?: string; // CSS selector to highlight
}

const STEPS: TourStep[] = [
  {
    title: "Welcome to the UCC Compiler Explorer",
    content:
      "This tool lets you write quantum circuits in Stim format and see, " +
      "in real time, how the UCC compiler transforms them through its " +
      "pipeline. Let's walk through each panel.",
  },
  {
    title: "Source Editor (left)",
    content:
      "Write your circuit here using Stim syntax: gate names like H, CNOT, " +
      "T, S, M followed by qubit indices. The compiler runs automatically " +
      "as you type (200ms debounce).\n\n" +
      "Try it: delete a line or add new gates and watch the other panels update.",
    target: ".editor-pane:nth-child(1)",
  },
  {
    title: "Heisenberg IR (middle)",
    content:
      "UCC's front-end absorbs Clifford gates (H, CNOT, S, CZ, SWAP...) into a " +
      "Heisenberg frame tracked via Stim's tableau algebra. This means Cliffords " +
      "vanish from the IR entirely!\n\n" +
      "What remains are non-Clifford ops: T gates appear as phase rotations on " +
      "Pauli products (e.g. 'T +X0*Z1'), and measurements show their effective " +
      "Pauli observable (e.g. 'MEASURE +X0 -> rec[0]').\n\n" +
      "Fewer HIR ops than source lines = the compiler is doing its job.",
    target: ".editor-pane:nth-child(2)",
  },
  {
    title: "VM Bytecode (right)",
    content:
      "The back-end compresses multi-qubit Pauli products down to 1- and 2-qubit " +
      "operations on virtual axes using geometric decomposition. The output is a " +
      "flat stream of RISC-style 32-byte instructions.\n\n" +
      "OP_FRAME_* ops manipulate the Heisenberg tracking frame. OP_ARRAY_* ops " +
      "touch the Schrodinger state vector. OP_EXPAND grows the active subspace. " +
      "OP_PHASE_T applies T rotations. OP_MEAS_* performs measurements.",
    target: ".editor-pane:nth-child(3)",
  },
  {
    title: "Source Map Highlighting",
    content:
      "Click on any line in any editor. The corresponding lines in the other " +
      "two editors light up with blue highlights. Colored ticks appear in each " +
      "editor's scrollbar showing where the related code is.\n\n" +
      "This bidirectional source map lets you trace exactly how each source " +
      "instruction flows through compilation.",
  },
  {
    title: "Active Dimensions (k) Timeline",
    content:
      "This chart shows how the active subspace dimension 'k' changes at each " +
      "bytecode instruction. k starts at 0 and grows when OP_EXPAND adds a " +
      "qubit to the Schrodinger state.\n\n" +
      "The red dashed line marks k=20, the browser memory limit (~16 MB). " +
      "Circuits that stay below this line can be simulated in-browser.\n\n" +
      "The yellow dashed line follows your cursor position in the bytecode editor.",
    target: ".chart-pane:nth-child(1)",
  },
  {
    title: "Simulation",
    content:
      "Click Simulate to run a Monte Carlo simulation entirely in your browser " +
      "via WebAssembly. You can configure the number of shots by clicking the " +
      "arrow next to the Simulate button.\n\n" +
      "The histogram shows measurement outcome probabilities. Hover bars for " +
      "exact counts. Timing stats appear below the chart.",
    target: ".chart-pane:nth-child(2)",
  },
  {
    title: "Sharing & Options",
    content:
      "The Share button compresses your circuit into a URL you can send to anyone. " +
      "They'll see your exact circuit when they open the link.\n\n" +
      "The Optimize toggle controls whether the compiler's peephole pass manager " +
      "runs. Turn it off to see the unoptimized bytecode.",
  },
];

interface Props {
  onClose: () => void;
}

export function GuidedTour({ onClose }: Props) {
  const [step, setStep] = useState(0);
  const current = STEPS[step];

  const handlePrev = useCallback(() => setStep((s) => Math.max(0, s - 1)), []);
  const handleNext = useCallback(() => {
    if (step < STEPS.length - 1) {
      setStep((s) => s + 1);
    } else {
      onClose();
    }
  }, [step, onClose]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight" || e.key === "Enter") handleNext();
      if (e.key === "ArrowLeft") handlePrev();
    },
    [onClose, handleNext, handlePrev],
  );

  const overlayRef = useRef<HTMLDivElement>(null);

  // Auto-focus on mount so keyboard events work immediately
  useEffect(() => {
    overlayRef.current?.focus();
  }, []);

  return (
    <div
      className="tour-overlay"
      onKeyDown={handleKeyDown}
      tabIndex={-1}
      role="dialog"
      aria-modal="true"
      aria-label="Guided tour"
      ref={overlayRef}
    >
      <div className="tour-backdrop" onClick={onClose} />
      <div className="tour-card">
        <button className="tour-close" onClick={onClose} title="Close tour">
          <X size={16} />
        </button>
        <div className="tour-step-indicator">
          {step + 1} / {STEPS.length}
        </div>
        <h3 className="tour-title">{current.title}</h3>
        <div className="tour-content">
          {current.content.split("\n\n").map((para, i) => (
            <p key={i}>{para}</p>
          ))}
        </div>
        <div className="tour-nav">
          <button
            className="tour-nav-btn"
            onClick={handlePrev}
            disabled={step === 0}
          >
            <ChevronLeft size={14} /> Back
          </button>
          <div className="tour-dots">
            {STEPS.map((_, i) => (
              <button
                key={i}
                className={`tour-dot ${i === step ? "tour-dot-active" : ""}`}
                onClick={() => setStep(i)}
                aria-label={`Go to step ${i + 1}`}
              />
            ))}
          </div>
          <button className="tour-nav-btn tour-nav-btn-primary" onClick={handleNext}>
            {step === STEPS.length - 1 ? "Finish" : "Next"} <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
