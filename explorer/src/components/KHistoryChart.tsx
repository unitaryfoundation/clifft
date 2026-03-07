import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { ChartColors } from "../hooks/useTheme";

interface Props {
  history: number[];
  highlightPC: number | null;
  colors: ChartColors;
}

const MEMORY_LIMIT_K = 20;

export function KHistoryChart({ history, highlightPC, colors }: Props) {
  if (history.length === 0) {
    return <div className="chart-placeholder">No bytecode yet</div>;
  }

  const data = history.map((k, i) => ({ pc: i, k }));
  const maxK = Math.max(...history, MEMORY_LIMIT_K + 2);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
        <XAxis
          dataKey="pc"
          stroke={colors.axis}
          fontSize={11}
          label={{ value: "Bytecode PC", position: "insideBottom", offset: -12, fill: colors.axis, fontSize: 11 }}
        />
        <YAxis
          domain={[0, maxK]}
          stroke={colors.axis}
          fontSize={11}
          label={{ value: "Active k", angle: -90, position: "insideLeft", fill: colors.axis, fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ background: colors.tooltipBg, border: `1px solid ${colors.tooltipBorder}`, fontSize: 12 }}
          labelFormatter={(pc) => `PC: ${pc}`}
          formatter={(value) => [`k = ${value}`, "Active Dimensions"]}
        />
        <Area
          type="stepAfter"
          dataKey="k"
          stroke={colors.accent}
          fill={colors.accentFill}
          fillOpacity={0.15}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <ReferenceLine
          y={MEMORY_LIMIT_K}
          stroke={colors.error}
          strokeDasharray="6 3"
          label={{
            value: "Browser Memory Limit (~16MB)",
            position: "right",
            fill: colors.error,
            fontSize: 10,
          }}
        />
        {highlightPC !== null && (
          <ReferenceLine x={highlightPC} stroke="#ffd54f" strokeDasharray="3 3" />
        )}
      </AreaChart>
    </ResponsiveContainer>
  );
}
