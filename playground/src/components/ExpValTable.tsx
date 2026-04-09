import type { ExpValEntry } from "../types";

interface Props {
  expVals: ExpValEntry[];
  shots: number;
  elapsedMs: number | null;
}

function formatTiming(elapsedMs: number, shots: number): string {
  const totalUs = elapsedMs * 1000;
  const usPerShot = totalUs / shots;
  if (elapsedMs < 1000) {
    return `${elapsedMs.toFixed(1)} ms total | ${usPerShot.toFixed(1)} us/shot`;
  }
  return `${(elapsedMs / 1000).toFixed(2)} s total | ${usPerShot.toFixed(1)} us/shot`;
}

function meanColor(mean: number): string {
  const abs = Math.abs(mean);
  if (abs > 0.9) return "var(--accent-green)";
  if (abs > 0.5) return "var(--accent)";
  return "var(--text-dim)";
}

export function ExpValTable({ expVals, shots, elapsedMs }: Props) {
  return (
    <div className="exp-val-wrapper">
      <div className="exp-val-table-scroll">
        <table className="exp-val-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Pauli</th>
              <th>Line</th>
              <th style={{ textAlign: "right" }}>{"\u27E8"}P{"\u27E9"}</th>
              <th style={{ textAlign: "right"}}>{"\u00B1"} {"\u03C3"}</th>
            </tr>
          </thead>
          <tbody>
            {expVals.map((ev, i) => (
              <tr key={i}>
                <td className="exp-val-idx">{i}</td>
                <td className="exp-val-label">
                  <code>{ev.label ?? `exp[${i}]`}</code>
                </td>
                <td className="exp-val-line">{ev.line ?? ""}</td>
                <td className="exp-val-mean" style={{ color: meanColor(ev.mean) }}>
                  {ev.mean >= 0 ? "+" : ""}{ev.mean.toFixed(4)}
                </td>
                <td className="exp-val-std">{ev.std.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {elapsedMs !== null && (
        <div className="histogram-timing">
          {formatTiming(elapsedMs, shots)}
        </div>
      )}
    </div>
  );
}
