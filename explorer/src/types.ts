export interface PassInfo {
  name: string;
  kind: "hir" | "bytecode";
  default: boolean;
}

export interface CompileSuccess {
  error?: undefined;
  num_qubits: number;
  peak_rank: number;
  num_measurements: number;
  num_t_gates: number;
  hir_ops: string[];
  bytecode: string[];
  hir_source_map: number[][];
  bytecode_source_map: number[][];
  active_k_history: number[];
}

export interface CompileError {
  error: string;
}

export type CompileResult = CompileSuccess | CompileError;

export function isCompileSuccess(r: CompileResult): r is CompileSuccess {
  return !('error' in r) || r.error === undefined;
}

export interface ExpValEntry {
  label?: string;
  line?: number;
  mean: number;
  std: number;
}

export interface SimulateSuccess {
  error?: undefined;
  histogram: Record<string, number>;
  shots: number;
  num_measurements: number;
  exp_vals?: ExpValEntry[];
}

export interface SimulateError {
  error: string;
}

export type SimulateResult = SimulateSuccess | SimulateError;

export function isSimulateSuccess(r: SimulateResult): r is SimulateSuccess {
  return !('error' in r) || r.error === undefined;
}

export interface UccModule {
  get_available_passes: () => string;
  compile_to_json: (source: string, passes_json: string) => string;
  simulate_wasm: (source: string, shots: number, passes_json: string) => string;
}
