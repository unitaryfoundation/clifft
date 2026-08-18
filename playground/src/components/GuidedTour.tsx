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
      it. The panels show the original circuit, the Heisenberg IR, the
      symbolic sampling plan, and a small in-browser simulation.</p>
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
    title: "Sampling Plan (right)",
    target: '[data-tour="lowered"]',
    html: `
      <p>The planner turns the HIR into semantic actions over stabilizer
      coordinates and Boolean symbols. This is the portable representation
      used by Clifft's symbolic-coordinate backend. The uppercase names, like
      <code>ROTATE_ACTIVE</code>, <code>PROMOTE_DORMANT</code>, and
      <code>MEASURE_ACTIVE</code>, are the planner actions themselves &mdash;
      hover over one to see an inline description of what it does.</p>
      <p>Pauli factors such as <code>X0</code> and <code>Z1</code> refer to
      active symbolic coordinates, not physical qubits. The <code>w&lt;k&gt;</code>
      prefix on each line shows the active width before and, when it changes,
      after the action; a trailing <code>passes=n</code> estimates how many
      dense-state traversals the action costs, and lines without it touch no
      dense state at all.</p>
      <p>Expressions like <code>s0^s1^s3</code> are XORs of per-shot Boolean
      symbols &mdash; noise draws, measurement branches, and the like. Long
      expressions may be truncated as <code>...(+N)</code> in this panel; the
      full expression still exists in the underlying plan.</p>
    `,
  },
  {
    title: "Source Map Highlighting",
    html: `
      <p>Outside diff view, click any line in any editor to highlight the related lines in the
      other panels. Colored ticks in the scrollbars show where those related
      lines are.</p>
      <p>This lets you trace how a source instruction changes as it moves
      through the compiler.</p>
    `,
  },
  {
    title: "Active Width Timeline",
    target: '[data-tour="active-dim"]',
    html: `
      <p>This chart tracks the active width <code>k</code> across the
      sampling plan. Clifft's active state vector has size <code>2^k</code>, so
      keeping <code>k</code> small is the key to fast simulation.</p>
      <p><code>k</code> grows when a dormant coordinate is promoted into the
      active state. Measurements can reduce <code>k</code> again.</p>
      <p>The red dashed line marks the browser memory limit. The yellow
      dashed line follows your source or plan selection.</p>
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
      <p><code>Passes</code> lets you toggle HIR optimization passes.
      <code>Diff</code> shows the unoptimized and optimized outputs
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
