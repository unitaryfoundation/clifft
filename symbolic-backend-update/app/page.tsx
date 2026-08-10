import type { CSSProperties, ReactNode } from "react";

type MethodKey = "legacy" | "symbolic" | "symftSingle" | "symftBatch";

type CircuitResult = {
  name: string;
  detail: string;
  k: number;
  values: Record<MethodKey, number>;
  note?: string;
};

const methods: Array<{ key: MethodKey; label: string; short: string }> = [
  { key: "legacy", label: "Clifft legacy", short: "Legacy" },
  { key: "symbolic", label: "Clifft symbolic", short: "Symbolic" },
  { key: "symftSingle", label: "SymFT single-shot", short: "SymFT single" },
  { key: "symftBatch", label: "SymFT batched", short: "SymFT batch" },
];

const circuits: CircuitResult[] = [
  {
    name: "Surface d7 r7",
    detail: "Clifford-heavy QEC",
    k: 0,
    values: { legacy: 1, symbolic: 1.31, symftSingle: 0.59, symftBatch: 9.2 },
  },
  {
    name: "Cultivation d3",
    detail: "Small active state",
    k: 4,
    values: { legacy: 1, symbolic: 1.35, symftSingle: 0.93, symftBatch: 3.22 },
  },
  {
    name: "Cultivation d5",
    detail: "Moderate active state",
    k: 10,
    values: { legacy: 1, symbolic: 1.9, symftSingle: 1.67, symftBatch: 2.27 },
  },
  {
    name: "Distillation",
    detail: "Output-heavy, k=5",
    k: 5,
    values: { legacy: 1, symbolic: 5.09, symftSingle: 2.74, symftBatch: 16.8 },
  },
  {
    name: "Coherent d3 r3",
    detail: "Measurement-rich",
    k: 7,
    values: { legacy: 1, symbolic: 1.07, symftSingle: 1.04, symftBatch: 1.43 },
    note: "Cross-project result is directional because detector references differ.",
  },
  {
    name: "Coherent d5 r1",
    detail: "Large active state",
    k: 12,
    values: { legacy: 1, symbolic: 0.59, symftSingle: 2.21, symftBatch: 1.98 },
  },
  {
    name: "Coherent d5 r5",
    detail: "Weakly coupled, high k",
    k: 22,
    values: { legacy: 1, symbolic: 2.21, symftSingle: 34.36, symftBatch: 34.34 },
    note: "Most shots reject early; treat this as order-of-magnitude evidence.",
  },
  {
    name: "Controlled k12 / L512",
    detail: "Dense-state control",
    k: 12,
    values: { legacy: 1, symbolic: 1.66, symftSingle: 0.56, symftBatch: 0.62 },
  },
];

const minRatio = 0.5;
const maxRatio = 40;

function ratioPosition(value: number) {
  const low = Math.log2(minRatio);
  const high = Math.log2(maxRatio);
  return ((Math.log2(value) - low) / (high - low)) * 100;
}

function RatioBar({ method, value }: { method: MethodKey; value: number }) {
  const position = Math.max(2, Math.min(100, ratioPosition(value)));
  const style = { "--bar-width": `${position}%` } as CSSProperties;
  return (
    <div className="ratio-row">
      <span className="ratio-method">{methods.find((item) => item.key === method)?.short}</span>
      <div className="ratio-track" aria-hidden="true">
        <span className="ratio-baseline" />
        <span className={`ratio-fill ratio-fill-${method}`} style={style} />
      </div>
      <strong>{value.toFixed(value >= 10 ? 1 : 2)}x</strong>
    </div>
  );
}

function QuestionHeader({ number, children }: { number: string; children: ReactNode }) {
  return (
    <div className="question-heading">
      <span className="question-number">{number}</span>
      <h2>{children}</h2>
    </div>
  );
}

export default function Home() {
  return (
    <main>
      <header className="hero" id="top">
        <nav className="topbar" aria-label="Page navigation">
          <a className="brand" href="#top" aria-label="Clifft symbolic backend update">
            <span className="brand-mark">C</span>
            <span>Clifft / symbolic backend</span>
          </a>
          <span className="date-label">Team briefing - August 2026</span>
        </nav>

        <div className="hero-copy">
          <p className="eyebrow">One week into the CPU backend refactor</p>
          <h1>Symbolic Clifft is already competitive with the legacy backend.</h1>
          <p className="hero-summary">
            The new backend now matches or beats legacy Clifft on seven of eight measured
            workloads. It also clarifies where SymFT&apos;s remaining advantages come from - and
            which ideas are worth bringing back to Clifft.
          </p>
        </div>

        <div className="headline-metrics" aria-label="Headline performance results">
          <article>
            <span className="metric-value">7 / 8</span>
            <span className="metric-label">workloads at or above legacy throughput</span>
          </article>
          <article>
            <span className="metric-value">5.09x</span>
            <span className="metric-label">legacy throughput on distillation</span>
          </article>
          <article>
            <span className="metric-value">1.66x</span>
            <span className="metric-label">legacy throughput on the k=12 dense control</span>
          </article>
          <article className="metric-watch">
            <span className="metric-value">0.59x</span>
            <span className="metric-label">legacy on coherent d5 r1 - the main CPU gap</span>
          </article>
        </div>
      </header>

      <div className="page-shell">
        <aside className="contents" aria-label="Briefing contents">
          <p>Questions</p>
          <ol>
            <li><a href="#results">Headline results</a></li>
            <li><a href="#techniques">SymFT techniques</a></li>
            <li><a href="#drivers">Performance drivers</a></li>
            <li><a href="#progress">Progress and next steps</a></li>
            <li><a href="#svm">Why no SVM?</a></li>
            <li><a href="#gpu">What about GPU?</a></li>
          </ol>
        </aside>

        <div className="briefing">
          <section className="faq-section" id="results">
            <QuestionHeader number="01">What are the headline results?</QuestionHeader>
            <p className="section-lead">
              Sampling-only throughput on one pinned EPYC core with AVX-512. Every value is
              normalized to the legacy backend on the same circuit; higher is better. The bar
              scale is logarithmic so both moderate and large differences remain visible.
            </p>

            <div className="legend" aria-label="Performance chart legend">
              {methods.map((method) => (
                <span key={method.key} className={`legend-${method.key}`}>
                  <i />{method.label}
                </span>
              ))}
            </div>

            <div className="scale-labels" aria-hidden="true">
              {[0.5, 1, 2, 4, 8, 16, 32].map((value) => (
                <span key={value} style={{ left: `${ratioPosition(value)}%` }}>{value}x</span>
              ))}
            </div>

            <div className="results-grid">
              {circuits.map((circuit) => (
                <article className="result-card" key={circuit.name}>
                  <div className="result-title">
                    <div>
                      <h3>{circuit.name}</h3>
                      <p>{circuit.detail}</p>
                    </div>
                    <span>max k = {circuit.k}</span>
                  </div>
                  <div className="ratio-bars">
                    {methods.map((method) => (
                      <RatioBar key={method.key} method={method.key} value={circuit.values[method.key]} />
                    ))}
                  </div>
                  {circuit.note && <p className="result-note">{circuit.note}</p>}
                </article>
              ))}
            </div>

            <div className="compile-callout">
              <div className="callout-icon">!</div>
              <div>
                <h3>Sampling is no longer the whole story.</h3>
                <p>
                  Symbolic compilation is still expensive on output-heavy circuits: 460 ms vs
                  8 ms on surface d7 r7, and 596 ms vs 5 ms on cultivation d5. Compilation is a
                  separate optimization track from the execution results above.
                </p>
              </div>
            </div>
          </section>

          <section className="faq-section" id="techniques">
            <QuestionHeader number="02">What techniques did SymFT introduce?</QuestionHeader>
            <p className="section-lead">
              SymFT changed both the mathematical decomposition and the way the resulting work
              is executed. The symbolic Clifft refactor adopts the first three ideas directly.
            </p>

            <div className="architecture-grid">
              <article className="pipeline legacy-pipeline">
                <p className="pipeline-label">Legacy Clifft</p>
                <h3>Localize, then execute</h3>
                <div className="pipeline-steps">
                  <span>Pauli localization</span><b>-&gt;</b>
                  <span>Flat SVM bytecode</span><b>-&gt;</b>
                  <span>Localized Clifford and dense-state operations per shot</span>
                </div>
                <p>
                  Predictable instructions and strong specialized kernels, but localization can
                  create additional dense transformations.
                </p>
              </article>

              <article className="pipeline symbolic-pipeline">
                <p className="pipeline-label">Symbolic Clifft / SymFT style</p>
                <h3>Factor once, execute direct actions</h3>
                <div className="pipeline-steps">
                  <span>Symbolic Clifford-Pauli frame</span><b>-&gt;</b>
                  <span>Shared active-coordinate plan</span><b>-&gt;</b>
                  <span>Signs plus direct Pauli and measurement kernels per shot</span>
                </div>
                <p>
                  The planner resolves topology once. Runtime kernels act directly on the small
                  active state without tableau evolution or Pauli localization.
                </p>
              </article>
            </div>

            <div className="technique-strip">
              <article><span>Core method</span><strong>Symbolic signs</strong><p>Noise, feedback, and branches become compact Boolean expressions.</p></article>
              <article><span>Core method</span><strong>Adaptive coordinates</strong><p>Only non-stabilizer degrees of freedom occupy the dense state.</p></article>
              <article><span>Core method</span><strong>Direct kernels</strong><p>Multi-coordinate Paulis execute without first localizing to one axis.</p></article>
              <article><span>Throughput layer</span><strong>Batching and components</strong><p>Many small shots can run together; separable states can stay factored.</p></article>
            </div>
          </section>

          <section className="faq-section" id="drivers">
            <QuestionHeader number="03">What actually drives SymFT&apos;s performance?</QuestionHeader>
            <p className="section-lead">
              The ablations show there is no single answer. Different mechanisms win in
              different active-width regimes.
            </p>

            <div className="driver-grid">
              <article className="driver-card driver-stream">
                <span className="driver-kicker">Low k / long stream</span>
                <h3>Remove per-shot Clifford work, then batch.</h3>
                <p>
                  At k=0, SymFT moves from 0.59x legacy in single-shot mode to 9.20x when batched.
                  Distillation moves from 2.74x to 16.80x.
                </p>
              </article>
              <article className="driver-card driver-dense">
                <span className="driver-kicker">Large dense state</span>
                <h3>Memory layout and kernels dominate.</h3>
                <p>
                  Cross-shot packing fades as 2^k work takes over. On the controlled k=12 case,
                  symbolic Clifft reaches 1.66x legacy while batched SymFT is 0.62x.
                </p>
              </article>
              <article className="driver-card driver-components">
                <span className="driver-kicker">Weakly coupled high k</span>
                <h3>Product components can change the problem.</h3>
                <p>
                  SymFT reaches about 34x legacy on coherent d5 r5 with batch size one. That is
                  factorization, not cross-shot packing.
                </p>
              </article>
              <article className="driver-card driver-width">
                <span className="driver-kicker">Measurement-rich circuits</span>
                <h3>Reclaim active width as soon as possible.</h3>
                <p>
                  Deterministic measurement pivots and planner simplifications reduce the number
                  of coefficients touched by every later dense action.
                </p>
              </article>
            </div>
          </section>

          <section className="faq-section" id="progress">
            <QuestionHeader number="04">What has the refactor done so far, and what is next?</QuestionHeader>
            <p className="section-lead">
              The new path was built alongside legacy so it could reuse the same Python oracle
              tests while remaining easy to compare and remove if the architecture failed. It
              now has broad CPU feature parity and a much clearer performance profile.
            </p>

            <div className="progress-list">
              <article><span>Foundation</span><h3>End-to-end symbolic execution</h3><p>Noise, feedback, resets, detectors, instruments, trajectories, expectation values, exact queries, and importance sampling.</p></article>
              <article><span>#281</span><h3>Affine expression registers</h3><p>Replaced repeated expression scans with incremental CSR propagation.</p></article>
              <article><span>#282 - #284</span><h3>Scalar kernels and fusion</h3><p>Specialized direct rotations, fused constant runs, and added AVX-512 U4 execution.</p></article>
              <article><span>#288</span><h3>Direct AVX-512 rotations</h3><p>Closed most moderate-k gaps without adding packed-shot execution.</p></article>
            </div>

            <div className="next-panel">
              <div>
                <p className="panel-label">Next on CPU</p>
                <h3>Reduce dense passes before adding more machinery.</h3>
                <ul>
                  <li>Measure same-Pauli and dynamic-sign fusion opportunities.</li>
                  <li>Finish low-pivot and pivot-four direct kernels where evidence supports them.</li>
                  <li>Improve measurement probability and collapse kernels.</li>
                  <li>Fix planner and continuation compilation cost.</li>
                  <li>Add AVX2, then revisit OpenMP, batching, and product components.</li>
                </ul>
              </div>
              <div className="deprecation-card">
                <span>Will legacy go away?</span>
                <strong>That is the goal.</strong>
                <p>
                  Keep the public Python API, replace the implementation underneath it, and retain
                  legacy only as an oracle until correctness, portability, compilation, and the
                  remaining user-facing surfaces are ready.
                </p>
              </div>
            </div>
          </section>

          <section className="faq-section short-answer" id="svm">
            <QuestionHeader number="05">Why do we not have a new SVM? Will we later?</QuestionHeader>
            <div className="answer-card">
              <p>
                We might add one later, but the SVM paradigm itself was not a measured driver of
                performance. A compact bytecode could still help with inspection, stable backend
                boundaries, or device lowering. We will revisit it after the meaningful execution
                and compiler optimizations land.
              </p>
            </div>
          </section>

          <section className="faq-section short-answer" id="gpu">
            <QuestionHeader number="06">What does this mean for a GPU backend?</QuestionHeader>
            <div className="answer-card gpu-answer">
              <p>
                The immutable symbolic plan now looks like the better starting boundary, but it
                should not simply be copied to a GPU. Device execution will likely want aggressive
                shot batching, different state layouts, and its own lowered program. The earlier
                GPU/SVM explorations remain useful evidence; the backend design itself is worth
                revisiting around the symbolic architecture.
              </p>
            </div>
          </section>

          <footer>
            <p>Sources and deeper detail</p>
            <div>
              <a href="https://github.com/unitaryfoundation/clifft/blob/bc-issue-280-status-update/SYMBOLIC_BACKEND_RESULTS.md">Four-way benchmark results</a>
              <a href="https://github.com/unitaryfoundation/clifft/issues/280">Clifft performance investigation</a>
              <a href="https://github.com/unitaryfoundation/arxiv-stim-clifft-monitor/issues/9">SymFT ablation study</a>
              <a href="https://github.com/unitaryfoundation/clifft/discussions/236">Symbolic backend discussion</a>
            </div>
            <small>
              Measurements use one pinned AMD EPYC 9554P core, one thread, Release builds, and
              AVX-512. Sampling time excludes compilation. PR #288 is treated as the current
              symbolic backend for this briefing.
            </small>
          </footer>
        </div>
      </div>
    </main>
  );
}
