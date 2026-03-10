// Smoke test for the UCC Wasm module.
// Run via: just test-wasm

import assert from "node:assert/strict";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const createModule = require("../../build-wasm/ucc_wasm.js");

const mod = await createModule();

// --- compile_to_json ---
const json = mod.compile_to_json("H 0\nT 0\nM 0", true);
const result = JSON.parse(json);

console.log("compile_to_json result:");
console.log("  num_qubits:", result.num_qubits);
console.log("  peak_rank:", result.peak_rank);
console.log("  hir_ops:", result.hir_ops.length, "ops");
console.log("  bytecode:", result.bytecode.length, "instructions");
console.log("  active_k_history:", result.active_k_history);
console.log("  hir_source_map:", result.hir_source_map);
console.log("  bytecode_source_map sample:", result.bytecode_source_map.slice(0, 3));

assert.equal(result.error, undefined, "Expected no error");
assert.equal(result.num_qubits, 1, "Expected 1 qubit");
assert.ok(result.peak_rank >= 1, "Expected peak_rank >= 1");
assert.ok(result.hir_ops.length > 0, "Expected HIR ops");
assert.ok(result.bytecode.length > 0, "Expected bytecode");
assert.equal(
    result.active_k_history.length,
    result.bytecode.length,
    "k_history parallel to bytecode"
);
assert.equal(
    result.bytecode_source_map.length,
    result.bytecode.length,
    "source_map parallel to bytecode"
);

// --- optimize flag toggle ---
// T T = S; peephole fusion should reduce 2 T ops to 1 S op
const unoptJson = mod.compile_to_json("T 0\nT 0\nM 0", false);
const unopt = JSON.parse(unoptJson);
const optJson = mod.compile_to_json("T 0\nT 0\nM 0", true);
const opt = JSON.parse(optJson);
console.log("\nOptimize toggle:");
console.log("  unoptimized HIR ops:", unopt.hir_ops.length);
console.log("  optimized HIR ops:", opt.hir_ops.length);
assert.ok(
    unopt.hir_ops.length > opt.hir_ops.length,
    "Optimized should have fewer ops (T+T fused to S)"
);

// --- simulate_wasm ---
const simJson = mod.simulate_wasm("H 0\nM 0", 1000, true);
const simResult = JSON.parse(simJson);

console.log("\nsimulate_wasm result:");
console.log("  histogram:", simResult.histogram);
console.log("  shots:", simResult.shots);

assert.equal(simResult.error, undefined, "Expected no simulation error");
assert.equal(simResult.shots, 1000, "Expected 1000 shots");
assert.equal(simResult.num_measurements, 1, "Expected 1 measurement");
const total = Object.values(simResult.histogram).reduce((a, b) => a + b, 0);
assert.equal(total, 1000, "Histogram counts should sum to shots");
// H|0> should produce roughly 50/50 distribution
const count0 = simResult.histogram["0"] || 0;
const count1 = simResult.histogram["1"] || 0;
assert.ok(count0 >= 350 && count0 <= 650, `Expected ~500 zeros, got ${count0}`);
assert.ok(count1 >= 350 && count1 <= 650, `Expected ~500 ones, got ${count1}`);

// --- no-measurement circuit returns consistent schema ---
const noMeasJson = mod.simulate_wasm("H 0", 100, true);
const noMeasResult = JSON.parse(noMeasJson);
console.log("\nNo-measurement test:", noMeasResult);
assert.equal(noMeasResult.shots, 100, "Expected shots in no-measurement result");
assert.equal(noMeasResult.num_measurements, 0, "Expected 0 measurements");
assert.deepEqual(noMeasResult.histogram, {}, "Expected empty histogram");

// --- memory limit guard ---
const bigLines = [];
for (let i = 0; i < 25; i++) bigLines.push(`H ${i}`);
for (let i = 0; i < 25; i++) bigLines.push(`T ${i}`);
bigLines.push("M 0");
const bigJson = mod.simulate_wasm(bigLines.join("\n"), 10, true);
const bigResult = JSON.parse(bigJson);
console.log("\nMemory limit test:", bigResult.error);
assert.equal(bigResult.error, "MemoryLimitExceeded", "Expected MemoryLimitExceeded");

// --- shots limit guard ---
const tooManyJson = mod.simulate_wasm("H 0\nM 0", 200000, true);
const tooManyResult = JSON.parse(tooManyJson);
console.log("Shots limit test:", tooManyResult.error);
assert.ok(tooManyResult.error.startsWith("ShotsLimitExceeded"), "Expected ShotsLimitExceeded");

// --- parse error ---
const errJson = mod.compile_to_json("INVALID_GATE 0", true);
const errResult = JSON.parse(errJson);
console.log("Parse error test:", errResult.error ? "caught" : "MISSING");
assert.ok(errResult.error, "Expected parse error");

console.log("\nAll Wasm smoke tests passed.");
