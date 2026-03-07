import { useEffect, useRef, useState, useCallback } from "react";
import type { CompileResult, SimulateResult, UccModule } from "../types";

declare function createUccModule(opts?: {
  locateFile?: (path: string) => string;
}): Promise<UccModule>;

export type WasmStatus = "loading" | "ready" | "error";

// Shared promise ensures only one module instantiation across StrictMode
// double-mounts and concurrent hook instances.
let modulePromise: Promise<UccModule> | null = null;

function loadModule(): Promise<UccModule> {
  if (modulePromise) return modulePromise;

  modulePromise = new Promise<UccModule>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${import.meta.env.BASE_URL}ucc_wasm.js`;
    script.async = true;
    script.setAttribute("data-ucc-wasm", "true");
    script.onload = () => {
      const base = import.meta.env.BASE_URL;
      createUccModule({
        locateFile: (path: string) => `${base}${path}`,
      }).then(resolve, reject);
    };
    script.onerror = () => reject(new Error("Failed to load ucc_wasm.js"));
    document.head.appendChild(script);
  });

  return modulePromise;
}

export function useUccWasm() {
  const moduleRef = useRef<UccModule | null>(null);
  const [status, setStatus] = useState<WasmStatus>("loading");

  useEffect(() => {
    let cancelled = false;

    loadModule()
      .then((mod) => {
        if (!cancelled) {
          moduleRef.current = mod;
          setStatus("ready");
        }
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const compile = useCallback(
    (source: string, optimize: boolean): CompileResult | null => {
      if (!moduleRef.current) return null;
      try {
        const json = moduleRef.current.compile_to_json(source, optimize);
        return JSON.parse(json) as CompileResult;
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
    },
    [],
  );

  const simulate = useCallback(
    (source: string, shots: number, optimize: boolean): SimulateResult | null => {
      if (!moduleRef.current) return null;
      try {
        const json = moduleRef.current.simulate_wasm(source, shots, optimize);
        return JSON.parse(json) as SimulateResult;
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
    },
    [],
  );

  return { status, compile, simulate };
}
