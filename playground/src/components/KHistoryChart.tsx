import { memo, useMemo } from "react";
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

const MEMORY_LIMIT_K = 24;
const CHART_MARGIN = { top: 8, right: 16, bottom: 24, left: 8 };

type ChartDatum = { pc: number; k?: number; baseline?: number };

interface BaseChartProps {
  data: ChartDatum[];
  maxK: number;
  hasBaseline: boolean;
  colors: ChartColors;
}

// Heavy chart body. Re-renders only when the underlying data, baseline,
// or theme colors change -- not on cursor moves. Without the memo the
// recharts measurement pass walks every axis tick / text element on
// every click and forces tens of thousands of synchronous layouts at
// large circuit sizes.
const BaseChart = memo(function BaseChart({ data, maxK, hasBaseline, colors }: BaseChartProps) {
  return (
    <AreaChart data={data} margin={CHART_MARGIN}>
      <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
      <XAxis
        dataKey="pc"
        stroke={colors.axis}
        fontSize={11}
        label={{
          value: "Bytecode PC",
          position: "insideBottom",
          offset: -12,
          fill: colors.axis,
          fontSize: 11,
        }}
      />
      <YAxis
        domain={[0, maxK]}
        stroke={colors.axis}
        fontSize={11}
        label={{
          value: "Active k",
          angle: -90,
          position: "insideLeft",
          fill: colors.axis,
          fontSize: 11,
        }}
      />
      <Tooltip
        contentStyle={{
          background: colors.tooltipBg,
          border: `1px solid ${colors.tooltipBorder}`,
          fontSize: 12,
        }}
        labelStyle={{ color: colors.tooltipText }}
        itemStyle={{ color: colors.tooltipText }}
        labelFormatter={(pc) => `PC: ${pc}`}
      />
      {hasBaseline && (
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
          value: "Browser Memory Limit (~256 MB)",
          position: "right",
          fill: colors.error,
          fontSize: 10,
        }}
      />
    </AreaChart>
  );
});

export function KHistoryChart({ history, baselineHistory, highlightPC, colors }: Props) {
  // Build the data array once per (history, baselineHistory) pair.
  const chartData = useMemo(() => {
    const maxLen = Math.max(history.length, baselineHistory?.length ?? 0);
    const data: ChartDatum[] = Array.from({ length: maxLen }, (_, i) => ({
      pc: i,
      k: i < history.length ? history[i] : undefined,
      baseline:
        baselineHistory && i < baselineHistory.length ? baselineHistory[i] : undefined,
    }));
    const allVals = [...history, ...(baselineHistory ?? [])];
    const maxK = Math.max(...allVals, MEMORY_LIMIT_K + 2);
    return { data, maxK };
  }, [history, baselineHistory]);

  if (history.length === 0) {
    return <div className="chart-placeholder">No bytecode yet</div>;
  }

  const hasBaseline = !!baselineHistory && baselineHistory.length > 0;

  // Cursor annotation. Replaces the prior in-chart ReferenceLine: pulling
  // the highlightPC out of the AreaChart subtree means recharts no longer
  // re-measures every axis tick on click, which is the dominant cost in
  // the playground click latency at large circuit sizes.
  const cursorK =
    highlightPC !== null && highlightPC >= 0 && highlightPC < history.length
      ? history[highlightPC]
      : null;

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <ResponsiveContainer width="100%" height="100%">
        <BaseChart
          data={chartData.data}
          maxK={chartData.maxK}
          hasBaseline={hasBaseline}
          colors={colors}
        />
      </ResponsiveContainer>
      {highlightPC !== null && (
        <div
          style={{
            position: "absolute",
            top: 4,
            right: 16,
            padding: "2px 6px",
            borderRadius: 3,
            background: colors.tooltipBg,
            border: `1px solid ${colors.tooltipBorder}`,
            color: colors.tooltipText,
            fontSize: 11,
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            pointerEvents: "none",
          }}
        >
          {`PC ${highlightPC}${cursorK !== undefined && cursorK !== null ? ` · k=${cursorK}` : ""}`}
        </div>
      )}
    </div>
  );
}
