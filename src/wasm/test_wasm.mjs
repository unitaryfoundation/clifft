// Smoke test for the Clifft Wasm module.
// Run via: just test-wasm

import assert from "node:assert/strict";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const createModule = require("../../build-wasm/clifft_wasm.js");

const mod = await createModule();

// Default passes config (empty string = use defaults)
const DEFAULTS = "";
// No passes at all
const NO_PASSES = JSON.stringify({ hir: [] });

// --- get_available_passes ---
const passesJson = mod.get_available_passes();
const passes = JSON.parse(passesJson);
console.log("get_available_passes:", passes.length, "passes");
assert.ok(passes.length >= 4, "Expected at least the current four HIR passes");
const names = passes.map((p) => p.name);
assert.ok(names.includes("PeepholeFusionPass"), "Missing PeepholeFusionPass");
assert.ok(names.includes("StatevectorSqueezePass"), "Missing StatevectorSqueezePass");
assert.ok(names.includes("RemoveNoisePass"), "Missing RemoveNoisePass");
// Check schema
for (const p of passes) {
    assert.ok(typeof p.name === "string");
    assert.equal(p.kind, "hir");
    assert.ok(typeof p.default === "boolean");
}

// --- compile_to_json with defaults ---
const json = mod.compile_to_json("H 0\nT 0\nM 0", DEFAULTS);
const result = JSON.parse(json);

console.log("compile_to_json result:");
console.log("  num_qubits:", result.num_qubits);
console.log("  max_active_width:", result.max_active_width);
console.log("  hir_ops:", result.hir_ops.length, "ops");
console.log("  sampling_plan:", result.sampling_plan.length, "actions");
console.log("  wasm_program:", result.wasm_program.length, "actions");
console.log("  active_width_history:", result.active_width_history);
console.log("  hir_source_map:", result.hir_source_map);
console.log("  sampling_plan_source_map sample:", result.sampling_plan_source_map.slice(0, 3));

assert.equal(result.error, undefined, "Expected no error");
assert.equal(result.num_qubits, 1, "Expected 1 qubit");
assert.ok(result.max_active_width >= 0, "Expected max_active_width >= 0");
assert.ok(result.hir_ops.length > 0, "Expected HIR ops");
assert.ok(result.sampling_plan.length > 0, "Expected sampling plan actions");
assert.ok(result.wasm_program.length > 0, "Expected WASM program actions");
assert.equal(
    result.active_width_history.length,
    result.sampling_plan.length,
    "active-width history parallel to sampling plan"
);
assert.equal(
    result.sampling_plan_source_map.length,
    result.sampling_plan.length,
    "source map parallel to sampling plan"
);
assert.equal(result.wasm_program_source_map.length, result.wasm_program.length);
assert.equal(result.wasm_program_plan_ranges.length, result.wasm_program.length);
assert.ok(
    result.sampling_plan_source_map.some((lines) => lines.includes(3)),
    "Expected plan provenance for the measurement source line"
);

// --- optimize toggle via pass config ---
// T T = S; peephole fusion should reduce 2 T ops to 1 S op
const unoptJson = mod.compile_to_json("T 0\nT 0\nM 0", NO_PASSES);
const unopt = JSON.parse(unoptJson);
const optJson = mod.compile_to_json("T 0\nT 0\nM 0", DEFAULTS);
const opt = JSON.parse(optJson);
console.log("\nOptimize toggle:");
console.log("  unoptimized HIR ops:", unopt.hir_ops.length);
console.log("  optimized HIR ops:", opt.hir_ops.length);
assert.ok(
    unopt.hir_ops.length > opt.hir_ops.length,
    "Optimized should have fewer ops (T+T fused to S)"
);

// --- selective passes ---
const hirOnlyJson = mod.compile_to_json(
    "T 0\nT 0\nM 0",
    JSON.stringify({ hir: ["PeepholeFusionPass"] })
);
const hirOnly = JSON.parse(hirOnlyJson);
console.log("\nSelective passes (HIR only):");
console.log("  HIR ops:", hirOnly.hir_ops.length);
assert.ok(hirOnly.hir_ops.length <= opt.hir_ops.length, "HIR-only should still fuse T+T");

const unknownPassConfig = JSON.parse(
    mod.compile_to_json("M 0", JSON.stringify({ hir: [], unknown: [] }))
);
assert.match(
    unknownPassConfig.error,
    /Unknown pass configuration key/,
    "Unknown pass configuration keys should fail explicitly"
);

// --- executable provenance across lowering fusion ---
const fusedSource = "H 0\nH 1\nT 0\nT 1\nT 0\nT 1\nT 0\nM 0";
const fused = JSON.parse(mod.compile_to_json(fusedSource, NO_PASSES));
const fusedAction = fused.wasm_program_plan_ranges.find((range) => range.end - range.begin > 1);
assert.ok(fusedAction, "Expected one WASM action to cover a fused plan range");
const fusedIndex = fused.wasm_program_plan_ranges.indexOf(fusedAction);
assert.ok(
    fused.wasm_program_source_map[fusedIndex].length > 1,
    "Expected fused WASM action to retain source lines"
);
assert.ok(
    fused.wasm_program.every((action) => !action.includes("OP_")),
    "Expected prepared symbolic actions"
);

// --- simulate_wasm ---
const simJson = mod.simulate_wasm("H 0\nM 0", 1000, DEFAULTS);
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

// --- symbolic presampled-noise execution ---
const noisyResult = JSON.parse(mod.simulate_wasm("X_ERROR(1) 0\nM 0", 100, DEFAULTS));
assert.deepEqual(noisyResult.histogram, { "1": 100 }, "Certain X error should flip every shot");

// --- EXP_VAL expectation value probes ---
const evJson = mod.simulate_wasm("H 0\nEXP_VAL X0 Z0", 1000, DEFAULTS);
const evResult = JSON.parse(evJson);
console.log("\nEXP_VAL test:");
console.log("  exp_vals:", evResult.exp_vals);
assert.equal(evResult.exp_vals.length, 2, "Expected 2 exp_val entries");
// <X> on |+> = +1 (deterministic)
assert.ok(Math.abs(evResult.exp_vals[0].mean - 1.0) < 0.01, `Expected <X>=+1, got ${evResult.exp_vals[0].mean}`);
// <Z> on |+> = 0
assert.ok(Math.abs(evResult.exp_vals[1].mean) < 0.1, `Expected <Z>=0, got ${evResult.exp_vals[1].mean}`);
// Check labels from source text
assert.equal(evResult.exp_vals[0].label, "X0", "Expected label X0");
assert.equal(evResult.exp_vals[1].label, "Z0", "Expected label Z0");
assert.equal(evResult.exp_vals[0].line, 2, "Expected line 2");
assert.equal(evResult.exp_vals[1].line, 2, "Expected line 2 (same line)");
// std for deterministic <X>=+1 should be 0
assert.ok(evResult.exp_vals[0].std < 0.01, `Expected std~0, got ${evResult.exp_vals[0].std}`);

// --- EXP_VAL-only circuit (no measurements) returns exp_vals ---
const evOnlyJson = mod.simulate_wasm("EXP_VAL Z0", 100, DEFAULTS);
const evOnlyResult = JSON.parse(evOnlyJson);
console.log("EXP_VAL-only test:", { exp_vals: evOnlyResult.exp_vals, num_measurements: evOnlyResult.num_measurements });
assert.equal(evOnlyResult.num_measurements, 0, "Expected 0 measurements");
assert.equal(evOnlyResult.exp_vals.length, 1, "Expected 1 exp_val");
assert.ok(Math.abs(evOnlyResult.exp_vals[0].mean - 1.0) < 1e-10, "Expected <Z>=+1 on |0>");

// --- no-measurement circuit returns consistent schema ---
const noMeasJson = mod.simulate_wasm("H 0", 100, DEFAULTS);
const noMeasResult = JSON.parse(noMeasJson);
console.log("\nNo-measurement test:", noMeasResult);
assert.equal(noMeasResult.shots, 100, "Expected shots in no-measurement result");
assert.equal(noMeasResult.num_measurements, 0, "Expected 0 measurements");
assert.deepEqual(noMeasResult.histogram, {}, "Expected empty histogram");

// --- memory limit guard ---
const bigLines = [];
for (let i = 0; i < 30; i++) bigLines.push(`H ${i}`);
for (let i = 0; i < 30; i++) bigLines.push(`T ${i}`);
bigLines.push("M 0");
const bigJson = mod.simulate_wasm(bigLines.join("\n"), 10, DEFAULTS);
const bigResult = JSON.parse(bigJson);
console.log("\nMemory limit test:", bigResult.error);
assert.equal(bigResult.error, "MemoryLimitExceeded", "Expected MemoryLimitExceeded");

// --- shots limit guard ---
const tooManyJson = mod.simulate_wasm("H 0\nM 0", 200000, DEFAULTS);
const tooManyResult = JSON.parse(tooManyJson);
console.log("Shots limit test:", tooManyResult.error);
assert.ok(tooManyResult.error.startsWith("ShotsLimitExceeded"), "Expected ShotsLimitExceeded");

// --- parse error ---
const errJson = mod.compile_to_json("INVALID_GATE 0", DEFAULTS);
const errResult = JSON.parse(errJson);
console.log("Parse error test:", errResult.error ? "caught" : "MISSING");
assert.ok(errResult.error, "Expected parse error");

console.log("\nAll Wasm smoke tests passed.");
