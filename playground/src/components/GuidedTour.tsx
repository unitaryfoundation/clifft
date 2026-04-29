import { useEffect } from "react";
import { driver } from "driver.js";
import "driver.js/dist/driver.css";

interface TourStep {
  title: string;
  content: string;
  target?: string;
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
    target: '[data-tour="source"]',
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
    target: '[data-tour="hir"]',
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
    target: '[data-tour="bytecode"]',
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
    target: '[data-tour="active-dim"]',
  },
  {
    title: "Simulation",
    content:
      "Click `Simulate` to run a Monte Carlo simulation entirely in your browser " +
      "using WebAssembly. Use the arrow next to `Simulate` to choose the number " +
      "of shots.\n\n" +
      "The histogram shows measurement outcome probabilities. Hover over a bar " +
      "to see exact counts. Timing stats appear below the chart.",
    target: '[data-tour="histogram"]',
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
    target: '[data-tour="actions"]',
  },
];

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function inlineCode(text: string): string {
  return escapeHtml(text).replace(
    /`([^`]+)`/g,
    '<code class="tour-code">$1</code>',
  );
}

function contentToHtml(content: string): string {
  const blocks = content.split("\n\n");
  return blocks
    .map((block) => {
      const lines = block.split("\n");
      const isList = lines.every((l) => l.startsWith("- "));
      if (isList) {
        const items = lines
          .map((l) => `<li>${inlineCode(l.slice(2))}</li>`)
          .join("");
        return `<ul class="tour-list">${items}</ul>`;
      }
      return `<p>${inlineCode(block)}</p>`;
    })
    .join("");
}

interface Props {
  onClose: () => void;
}

export function GuidedTour({ onClose }: Props) {
  useEffect(() => {
    const driverObj = driver({
      showProgress: true,
      progressText: "Step {{current}} of {{total}}",
      nextBtnText: "Next →",
      prevBtnText: "← Back",
      doneBtnText: "Finish",
      animate: true,
      smoothScroll: true,
      stagePadding: 6,
      stageRadius: 6,
      onDestroyed: onClose,
      steps: STEPS.map((step) => ({
        element: step.target,
        popover: {
          title: step.title,
          description: contentToHtml(step.content),
        },
      })),
    });
    driverObj.drive();
    return () => {
      driverObj.destroy();
    };
  }, [onClose]);

  return null;
}
