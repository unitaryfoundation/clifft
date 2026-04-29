import { useEffect } from "react";
import { driver } from "driver.js";
import "driver.js/dist/driver.css";

interface TourStep {
  title: string;
  html: string;
  target?: string;
}

const STEPS: TourStep[] = [
  {
    title: "Welcome to the Clifft Playground",
    html: `
      <p>Write a quantum circuit in Stim format and see how Clifft compiles
      it. The panels show the original circuit, the Heisenberg IR, the VM
      bytecode, and a small in-browser simulation.</p>
    `,
  },
  {
    title: "Source Editor (left)",
    target: '[data-tour="source"]',
    html: `
      <p>Write your circuit here using Stim syntax: gate names like
      <code>H</code>, <code>CNOT</code>, <code>T</code>, <code>S</code>, and
      <code>M</code>, followed by qubit indices.</p>
      <p>The compiler updates automatically as you type. Try deleting a line
      or adding a gate and watch the other panels change.</p>
    `,
  },
  {
    title: "Heisenberg IR (middle)",
    target: '[data-tour="hir"]',
    html: `
      <p>The front-end absorbs Clifford gates such as <code>H</code>,
      <code>CNOT</code>, <code>S</code>, <code>CZ</code>, and <code>SWAP</code>
      into a Clifford frame. Those gates disappear from the IR.</p>
      <p>What remains are the operations that need runtime work: non-Clifford
      gates, measurements, detectors, observables, and noise. For example, a
      <code>T</code> gate appears as a phase rotation on a Pauli product, and
      a measurement appears as the effective Pauli observable being
      measured.</p>
      <p>If the IR has fewer operations than the source circuit, the compiler
      is doing useful work.</p>
    `,
  },
  {
    title: "VM Bytecode (right)",
    target: '[data-tour="bytecode"]',
    html: `
      <p>The back-end lowers the IR into the instructions executed by
      Clifft's virtual machine.</p>
      <p>Some instructions only update the runtime Pauli frame, which is
      cheap. Others touch the active state vector, which is where most
      simulation cost comes from.</p>
      <p>Useful patterns to look for:</p>
      <ul>
        <li><code>OP_FRAME_*</code> updates the runtime Pauli frame</li>
        <li><code>OP_ARRAY_*</code> touches the active state vector</li>
        <li><code>OP_EXPAND</code> grows the active subspace</li>
        <li><code>OP_ARRAY_T</code> applies a T rotation</li>
        <li><code>OP_MEAS_*</code> performs a measurement</li>
      </ul>
    `,
  },
  {
    title: "Source Map Highlighting",
    html: `
      <p>Click any line in any editor to highlight the related lines in the
      other panels. Colored ticks in the scrollbars show where those related
      lines are.</p>
      <p>This lets you trace how a source instruction changes as it moves
      through the compiler.</p>
    `,
  },
  {
    title: "Active Dimension Timeline",
    target: '[data-tour="active-dim"]',
    html: `
      <p>This chart tracks the active dimension <code>k</code> across the
      bytecode. Clifft's active state vector has size <code>2^k</code>, so
      keeping <code>k</code> small is the key to fast simulation.</p>
      <p><code>k</code> starts at 0 and grows when <code>OP_EXPAND</code>
      adds a qubit to the active state. Measurements can reduce <code>k</code>
      again.</p>
      <p>The red dashed line marks the browser memory limit. The yellow
      dashed line follows your cursor position in the bytecode editor.</p>
    `,
  },
  {
    title: "Simulation",
    target: '[data-tour="histogram"]',
    html: `
      <p>Click <code>Simulate</code> to run a Monte Carlo simulation entirely
      in your browser using WebAssembly. Use the arrow next to
      <code>Simulate</code> to choose the number of shots.</p>
      <p>The histogram shows measurement outcome probabilities. Hover over a
      bar to see exact counts. Timing stats appear below the chart.</p>
    `,
  },
  {
    title: "Sharing & Options",
    target: '[data-tour="actions"]',
    html: `
      <p>Use <code>Share</code> to create a URL for the current circuit.</p>
      <p>For larger circuits, use <code>Load</code> to import a public
      <code>.stim</code> file by URL. The file must be readable by the
      browser, so raw GitHub URLs or public Gist raw URLs work well. Until
      you edit the loaded circuit, <code>Share</code> will create a compact
      link that points back to that file.</p>
      <p><code>Save</code> stores circuits in your browser, and
      <code>Recents</code> shows circuits you have saved.</p>
      <p><code>Passes</code> lets you toggle HIR and bytecode optimization
      passes. <code>Diff</code> shows the unoptimized and optimized outputs
      side by side.</p>
    `,
  },
];

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
          description: step.html,
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
