import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { isSimulateSuccess } from "../types";
import type { SimulateResult } from "../types";
import type { ChartColors } from "../hooks/useTheme";

interface Props {
  result: SimulateResult | null;
  elapsedMs: number | null;
  colors: ChartColors;
}

const BAR_COLORS = [
  "#4fc3f7",
  "#81c784",
  "#ffb74d",
  "#ce93d8",
  "#ef5350",
  "#ffd54f",
  "#4db6ac",
  "#f06292",
];

function formatTiming(elapsedMs: number, shots: number): string {
  const totalUs = elapsedMs * 1000;
  const usPerShot = totalUs / shots;
  if (elapsedMs < 1000) {
    return `${elapsedMs.toFixed(1)} ms total | ${usPerShot.toFixed(1)} us/shot`;
  }
  return `${(elapsedMs / 1000).toFixed(2)} s total | ${usPerShot.toFixed(1)} us/shot`;
}

export function HistogramChart({ result, elapsedMs, colors }: Props) {
  if (!result) {
    return <div className="chart-placeholder">Click Simulate to run</div>;
  }

  if (!isSimulateSuccess(result)) {
    return (
      <div className="chart-error">
        <span className="error-icon">&#x26a0;</span>
        {result.error === "MemoryLimitExceeded" ? (
          <>
            <strong>Circuit too large for browser simulation.</strong>
            <br />
            Peak rank exceeds 20 qubits (~16 MB). Use the native Python CLI for
            larger circuits.
          </>
        ) : (
          result.error
        )}
      </div>
    );
  }

  if (result.num_measurements === 0) {
    return <div className="chart-placeholder">No measurements in circuit</div>;
  }

  const entries = Object.entries(result.histogram)
    .map(([bitstring, count]) => ({
      bitstring,
      count,
      probability: count / result.shots,
    }))
    .sort((a, b) => a.bitstring.localeCompare(b.bitstring));

  return (
    <div className="histogram-wrapper">
      <div className="histogram-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={entries} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
            <XAxis
              dataKey="bitstring"
              stroke={colors.axis}
              fontSize={entries.length > 16 ? 9 : 11}
              angle={entries.length > 8 ? -45 : 0}
              textAnchor={entries.length > 8 ? "end" : "middle"}
              height={entries.length > 8 ? 60 : 30}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              stroke={colors.axis}
              fontSize={11}
            />
            <Tooltip
              contentStyle={{ background: colors.tooltipBg, border: `1px solid ${colors.tooltipBorder}`, fontSize: 12 }}
              formatter={(value, _name, item) => {
                const v = Number(value);
                const count = (item as { payload: { count: number } }).payload.count;
                return [
                  `${(v * 100).toFixed(1)}% (${count}/${result.shots})`,
                  "Probability",
                ];
              }}
            />
            <Bar dataKey="probability" isAnimationActive={false}>
              {entries.map((_, i) => (
                <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {elapsedMs !== null && (
        <div className="histogram-timing">
          {formatTiming(elapsedMs, result.shots)}
        </div>
      )}
    </div>
  );
}
