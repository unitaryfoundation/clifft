import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Line,
} from "recharts";
import type { ChartColors } from "../hooks/useTheme";

interface Props {
  history: number[];
  baselineHistory?: number[];
  highlightPC: number | null;
  colors: ChartColors;
}

const MEMORY_LIMIT_K = 20;

export function KHistoryChart({ history, baselineHistory, highlightPC, colors }: Props) {
  if (history.length === 0) {
    return <div className="chart-placeholder">No bytecode yet</div>;
  }

  // Build data array; baseline may have different length so pad with undefined
  const maxLen = Math.max(history.length, baselineHistory?.length ?? 0);
  const data = Array.from({ length: maxLen }, (_, i) => ({
    pc: i,
    k: i < history.length ? history[i] : undefined,
    baseline: baselineHistory && i < baselineHistory.length ? baselineHistory[i] : undefined,
  }));

  const allVals = [...history, ...(baselineHistory ?? [])];
  const maxK = Math.max(...allVals, MEMORY_LIMIT_K + 2);

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
          labelStyle={{ color: colors.tooltipText }}
          itemStyle={{ color: colors.tooltipText }}
          labelFormatter={(pc) => `PC: ${pc}`}
        />
        {baselineHistory && baselineHistory.length > 0 && (
          <Line
            type="stepAfter"
            dataKey="baseline"
            stroke={colors.axis}
            strokeWidth={1.5}
            strokeDasharray="6 3"
            dot={false}
            isAnimationActive={false}
            name="Baseline k"
            connectNulls={false}
          />
        )}
        <Area
          type="stepAfter"
          dataKey="k"
          stroke={colors.accent}
          fill={colors.accentFill}
          fillOpacity={0.15}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          name="Optimized k"
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
