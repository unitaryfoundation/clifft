import { createContext, memo, useContext, useMemo } from "react";
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
  usePlotArea,
  useXAxisScale,
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

// Cursor highlight delivered through context rather than as a prop on the
// memoized chart, so changes don't invalidate BaseChart's React.memo.
// HighlightLine consumes the context and re-renders independently when
// the cursor moves; the chart body stays untouched.
const HighlightPCContext = createContext<number | null>(null);

const HighlightLine = memo(function HighlightLine() {
  const highlightPC = useContext(HighlightPCContext);
  const plotArea = usePlotArea();
  const xScale = useXAxisScale();

  if (highlightPC === null || !plotArea || !xScale) return null;
  const raw = xScale(highlightPC);
  const x = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isFinite(x)) return null;
  if (x < plotArea.x || x > plotArea.x + plotArea.width) return null;

  return (
    <line
      x1={x}
      x2={x}
      y1={plotArea.y}
      y2={plotArea.y + plotArea.height}
      stroke="#ffd54f"
      strokeDasharray="3 3"
      pointerEvents="none"
    />
  );
});

interface BaseChartProps {
  data: ChartDatum[];
  maxK: number;
  hasBaseline: boolean;
  colors: ChartColors;
}

// Heavy chart body. Memoized so the recharts axis/area measurement pass
// only re-runs when the underlying data, baseline flag, or theme colors
// change -- not on cursor moves. The cursor highlight is rendered by
// HighlightLine, which subscribes to HighlightPCContext and lives inside
// AreaChart so it can pull the chart's plot area + x scale from the
// recharts hooks.
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
      <HighlightLine />
    </AreaChart>
  );
});

export function KHistoryChart({ history, baselineHistory, highlightPC, colors }: Props) {
  // Build the data array and y-axis upper bound once per
  // (history, baselineHistory) pair. Single-pass scan avoids the
  // [...history, ...baselineHistory] allocation and the Math.max(...)
  // function-argument spread, which can hit V8's argument count limit
  // on multi-tens-of-thousands of entries.
  const chartData = useMemo(() => {
    const maxLen = Math.max(history.length, baselineHistory?.length ?? 0);
    let maxVal = MEMORY_LIMIT_K + 2;
    const data: ChartDatum[] = new Array(maxLen);
    for (let i = 0; i < maxLen; i++) {
      const k = i < history.length ? history[i] : undefined;
      const baseline =
        baselineHistory && i < baselineHistory.length ? baselineHistory[i] : undefined;
      if (k !== undefined && k > maxVal) maxVal = k;
      if (baseline !== undefined && baseline > maxVal) maxVal = baseline;
      data[i] = { pc: i, k, baseline };
    }
    return { data, maxK: maxVal };
  }, [history, baselineHistory]);

  if (history.length === 0) {
    return <div className="chart-placeholder">No bytecode yet</div>;
  }

  const hasBaseline = !!baselineHistory && baselineHistory.length > 0;

  return (
    <HighlightPCContext.Provider value={highlightPC}>
      <ResponsiveContainer width="100%" height="100%">
        <BaseChart
          data={chartData.data}
          maxK={chartData.maxK}
          hasBaseline={hasBaseline}
          colors={colors}
        />
      </ResponsiveContainer>
    </HighlightPCContext.Provider>
  );
}
