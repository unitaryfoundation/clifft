import {
  useState,
  useCallback,
  useRef,
  useEffect,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import { X, ChevronLeft, ChevronRight } from "lucide-react";

interface TourStep {
  title: string;
  content: string;
  target?: string; // CSS selector to highlight
}

/** Render inline `code` spans within a text string. */
function renderInlineCode(text: string): ReactNode[] {
  const parts = text.split(/(`[^`]+`)/);
  return parts.map((part, i) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={i} className="tour-code">
          {part.slice(1, -1)}
        </code>
      );
    }
    return part;
  });
}

/**
 * Render tour content with basic markdown support:
 * - Paragraphs separated by blank lines
 * - Bullet lists (lines starting with "- ")
 * - Inline `code` spans
 */
function renderTourContent(content: string): ReactNode[] {
  const blocks = content.split("\n\n");
  return blocks.map((block, bi) => {
    const lines = block.split("\n");
    const isList = lines.every((l) => l.startsWith("- "));
    if (isList) {
      return (
        <ul key={bi} className="tour-list">
          {lines.map((line, li) => (
            <li key={li}>{renderInlineCode(line.slice(2))}</li>
          ))}
        </ul>
      );
    }
    return <p key={bi}>{renderInlineCode(block)}</p>;
  });
}

const STEPS: TourStep[] = [
  {
    title: "Welcome to the Clifft Playground",
    content:
      "Write a quantum circuit in Stim format and see how Clifft compiles it. " +
      "The panels show the original circuit, the Heisenberg IR, the VM bytecode, " +
      "and a small in-browser simulation.",
  },
  {
    title: "Source Editor (left)",
    content:
      "Write your circuit here using Stim syntax: gate names like `H`, `CNOT`, " +
      "`T`, `S`, and `M`, followed by qubit indices.\n\n" +
      "The compiler updates automatically as you type. Try deleting a line or " +
      "adding a gate and watch the other panels change.",
    target: ".editor-pane:nth-child(1)",
  },
  {
    title: "Heisenberg IR (middle)",
    content:
      "The front-end absorbs Clifford gates such as `H`, `CNOT`, `S`, `CZ`, and " +
      "`SWAP` into a Clifford frame. Those gates disappear from the IR.\n\n" +
      "What remains are the operations that need runtime work: non-Clifford gates, " +
      "measurements, detectors, observables, and noise. For example, a `T` gate " +
      "appears as a phase rotation on a Pauli product, and a measurement appears " +
      "as the effective Pauli observable being measured.\n\n" +
      "If the IR has fewer operations than the source circuit, the compiler is " +
      "doing useful work.",
    target: ".editor-pane:nth-child(2)",
  },
  {
    title: "VM Bytecode (right)",
    content:
      "The back-end lowers the IR into the instructions executed by Clifft's " +
      "virtual machine.\n\n" +
      "Some instructions only update the runtime Pauli frame, which is cheap. " +
      "Others touch the active state vector, which is where most simulation cost " +
      "comes from.\n\n" +
      "Useful patterns to look for:\n" +
      "- `OP_FRAME_*` updates the runtime Pauli frame\n" +
      "- `OP_ARRAY_*` touches the active state vector\n" +
      "- `OP_EXPAND` grows the active subspace\n" +
      "- `OP_ARRAY_T` applies a T rotation\n" +
      "- `OP_MEAS_*` performs a measurement",
    target: ".editor-pane:nth-child(3)",
  },
  {
    title: "Source Map Highlighting",
    content:
      "Click any line in any editor to highlight the related lines in the other " +
      "panels. Colored ticks in the scrollbars show where those related lines are.\n\n" +
      "This lets you trace how a source instruction changes as it moves through " +
      "the compiler.",
  },
  {
    title: "Active Dimension Timeline",
    content:
      "This chart tracks the active dimension `k` across the bytecode. Clifft's " +
      "active state vector has size `2^k`, so keeping `k` small is the key to " +
      "fast simulation.\n\n" +
      "`k` starts at 0 and grows when `OP_EXPAND` adds a qubit to the active state. " +
      "Measurements can reduce `k` again.\n\n" +
      "The red dashed line marks the browser memory limit. The yellow dashed line " +
      "follows your cursor position in the bytecode editor.",
    target: ".chart-pane:nth-child(1)",
  },
  {
    title: "Simulation",
    content:
      "Click `Simulate` to run a Monte Carlo simulation entirely in your browser " +
      "using WebAssembly. Use the arrow next to `Simulate` to choose the number " +
      "of shots.\n\n" +
      "The histogram shows measurement outcome probabilities. Hover over a bar " +
      "to see exact counts. Timing stats appear below the chart.",
    target: ".chart-pane:nth-child(2)",
  },
  {
    title: "Sharing & Options",
    content:
      "Use `Share` to create a URL for the current circuit.\n\n" +
      "For larger circuits, use `Load` to import a public `.stim` file by URL. " +
      "The file must be readable by the browser, so raw GitHub URLs or public " +
      "Gist raw URLs work well. Until you edit the loaded circuit, `Share` will " +
      "create a compact link that points back to that file.\n\n" +
      "`Save` stores circuits in your browser, and `Recents` shows circuits you " +
      "have saved.\n\n" +
      "`Passes` lets you toggle HIR and bytecode optimization passes. `Diff` shows " +
      "the unoptimized and optimized outputs side by side.",
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
        <div className="tour-content">{renderTourContent(current.content)}</div>
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
