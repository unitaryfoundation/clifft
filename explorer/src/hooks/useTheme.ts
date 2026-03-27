import { useState, useEffect, useCallback, useMemo } from "react";

export type Theme = "light" | "dark";

export interface ChartColors {
  grid: string;
  axis: string;
  tooltipBg: string;
  tooltipBorder: string;
  tooltipText: string;
  accent: string;
  accentFill: string;
  error: string;
}

const DARK_CHART: ChartColors = {
  grid: "#333",
  axis: "#888",
  tooltipBg: "#252526",
  tooltipBorder: "#444",
  tooltipText: "#e0e0e0",
  accent: "#4fc3f7",
  accentFill: "#4fc3f7",
  error: "#ef5350",
};

const LIGHT_CHART: ChartColors = {
  grid: "#d0d0d0",
  axis: "#666",
  tooltipBg: "#ffffff",
  tooltipBorder: "#d0d0d0",
  tooltipText: "#333333",
  accent: "#0277bd",
  accentFill: "#0277bd",
  error: "#c62828",
};

const STORAGE_KEY = "ucc-theme";

function getSystemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function getInitialTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return getSystemTheme();
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getInitialTheme);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    localStorage.setItem(STORAGE_KEY, t);
  }, []);

  const toggle = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  // Apply data-theme attribute to <html> for CSS variable switching
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // Listen for system theme changes (only when no explicit user preference)
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = (e: MediaQueryListEvent) => {
      if (!localStorage.getItem(STORAGE_KEY)) {
        setThemeState(e.matches ? "dark" : "light");
      }
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const chartColors = useMemo(
    () => (theme === "dark" ? DARK_CHART : LIGHT_CHART),
    [theme],
  );

  return { theme, toggle, chartColors } as const;
}
