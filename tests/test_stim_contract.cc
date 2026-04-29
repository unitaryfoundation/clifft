// Stim API Contract Tests
//
// These tests verify our assumptions about Stim's TableauSimulator and Tableau APIs.
// If Stim changes its semantics in a future version, these tests will catch it early
// rather than causing mysterious failures in the Front-End.
//
// Key assumptions documented here:
// 1. TableauSimulator.inv_state tracks the INVERSE tableau (U_dag)
// 2. inv_state.zs[q] gives Heisenberg-rewound Z: U_dag Z_q U
// 3. inv_state.xs[q] gives Heisenberg-rewound X: U_dag X_q U
// 4. PauliString masks can be extracted via .xs.u64[0] and .zs.u64[0]
// 5. Tableau.then() composes tableaux correctly
// 6. Tableau.inverse() computes the inverse correctly

#include "stim.h"

#include <catch2/catch_test_macros.hpp>
#include <random>
#include <utility>

TEST_CASE("Stim contract: TableauSimulator Heisenberg rewinding", "[stim][contract]") {
    // TableauSimulator.inv_state tracks the inverse tableau
    // inv_state.zs[q] gives the Heisenberg-rewound Z observable: U_dag Z_q U
    // inv_state.xs[q] gives the Heisenberg-rewound X observable: U_dag X_q U

    SECTION("Initial state: Z rewound is Z") {
        std::mt19937_64 rng(42);
        stim::TableauSimulator<64> sim(std::move(rng), 2);

        // Initially no gates applied, so U = I
        // inv_state.zs[0] should be Z0
        auto z0 = sim.inv_state.zs[0];
        REQUIRE(z0.xs[0] == false);
        REQUIRE(z0.zs[0] == true);
        REQUIRE(z0.xs[1] == false);
        REQUIRE(z0.zs[1] == false);
    }

    SECTION("After H: Z rewound is X") {
        std::mt19937_64 rng(42);
        stim::TableauSimulator<64> sim(std::move(rng), 2);

        stim::Circuit circuit;
        circuit.safe_append_u("H", {0});
        sim.safe_do_circuit(circuit);

        // H_dag Z H = X (H is self-inverse)
        auto z0 = sim.inv_state.zs[0];
        REQUIRE(z0.xs[0] == true);   // X component
        REQUIRE(z0.zs[0] == false);  // No Z component
    }

    SECTION("After H; S: Z rewound is X (S commutes with Z)") {
        std::mt19937_64 rng(42);
        stim::TableauSimulator<64> sim(std::move(rng), 2);

        stim::Circuit circuit;
        circuit.safe_append_u("H", {0});
        circuit.safe_append_u("S", {0});
        sim.safe_do_circuit(circuit);

        // (SH)_dag Z (SH) = H_dag S_dag Z S H = H_dag Z H = X
        // (Because S commutes with Z)
        auto z0 = sim.inv_state.zs[0];
        REQUIRE(z0.xs[0] == true);   // X component
        REQUIRE(z0.zs[0] == false);  // No Z component
    }

    SECTION("After H; S: X rewound is +Y") {
        std::mt19937_64 rng(42);
        stim::TableauSimulator<64> sim(std::move(rng), 2);

        stim::Circuit circuit;
        circuit.safe_append_u("H", {0});
        circuit.safe_append_u("S", {0});
        sim.safe_do_circuit(circuit);

        // U = S @ H, so U_dag X U = H_dag S_dag X S H = +Y
        // (Verified numerically: S_dag X S = Y, then H_dag Y H = Y)
        auto x0 = sim.inv_state.xs[0];
        // Y has xs=true, zs=true, positive sign
        REQUIRE(x0.xs[0] == true);
        REQUIRE(x0.zs[0] == true);
        REQUIRE(x0.sign == false);  // Positive phase
    }

    SECTION("After CX: Z1 rewound has Z0 component (CNOT propagates Z)") {
        std::mt19937_64 rng(42);
        stim::TableauSimulator<64> sim(std::move(rng), 2);

        stim::Circuit circuit;
        circuit.safe_append_u("CX", {0, 1});  // Control=0, Target=1
        sim.safe_do_circuit(circuit);

        // CX_dag Z_target CX = Z_control * Z_target
        auto z1 = sim.inv_state.zs[1];
        REQUIRE(z1.zs[0] == true);  // Z on qubit 0 (propagated from target)
        REQUIRE(z1.zs[1] == true);  // Z on qubit 1
    }

    SECTION("After H; CX: creates Bell-like Z propagation") {
        std::mt19937_64 rng(42);
        stim::TableauSimulator<64> sim(std::move(rng), 2);

        stim::Circuit circuit;
        circuit.safe_append_u("H", {0});
        circuit.safe_append_u("CX", {0, 1});
        sim.safe_do_circuit(circuit);

        // Circuit: H 0, CX 0 1
        // This creates the Bell state |00> + |11>
        // Z1 rewound: (CX H)_dag Z1 (CX H) = H_dag CX_dag Z1 CX H = H_dag (Z0 Z1) H = X0 Z1
        auto z1 = sim.inv_state.zs[1];
        REQUIRE(z1.xs[0] == true);  // X on qubit 0
        REQUIRE(z1.zs[1] == true);  // Z on qubit 1
    }
}

TEST_CASE("Stim contract: mask extraction for HIR", "[stim][contract]") {
    // The Front-End needs to extract uint64_t masks from PauliString

    std::mt19937_64 rng(42);
    stim::TableauSimulator<64> sim(std::move(rng), 4);

    stim::Circuit circuit;
    circuit.safe_append_u("H", {0});
    circuit.safe_append_u("H", {1});
    circuit.safe_append_u("CX", {0, 2});
    circuit.safe_append_u("CX", {1, 3});
    sim.safe_do_circuit(circuit);

    // Z2 rewound should be X0 Z2
    auto z2 = sim.inv_state.zs[2];
    uint64_t x_mask = z2.xs.u64[0];
    uint64_t z_mask = z2.zs.u64[0];

    REQUIRE(x_mask == 0b0001);  // X on qubit 0
    REQUIRE(z_mask == 0b0100);  // Z on qubit 2
}

TEST_CASE("Stim contract: Tableau composition", "[stim][contract]") {
    // t1.then(t2) applies t1 first, then t2.

    stim::Tableau<64> t1(2);
    stim::Tableau<64> t2(2);

    // Apply H to t1, S to t2
    t1.inplace_scatter_append(stim::GATE_DATA.at("H").tableau<64>(), {0});
    t2.inplace_scatter_append(stim::GATE_DATA.at("S").tableau<64>(), {0});

    // Compose: t1.then(t2) applies t1 first, then t2
    auto composed = t1.then(t2);
    REQUIRE(composed.num_qubits == 2);

    // t1.then(t2) means apply t1 first, then t2, so U = S * H.
    // The forward tableau stores U P U_dag, so composed.zs[0] is
    //   (SH) Z (SH)_dag = S (H Z H_dag) S_dag = S X S_dag = Y.
    auto z0 = composed.zs[0];
    REQUIRE(z0.xs[0] == true);  // Y has X component
    REQUIRE(z0.zs[0] == true);  // Y has Z component
}

TEST_CASE("Stim contract: Tableau inverse", "[stim][contract]") {
    // Verify Tableau::inverse() round-trips: fwd.then(inv) = identity.

    std::mt19937_64 rng(42);
    stim::TableauSimulator<64> sim(std::move(rng), 2);

    stim::Circuit circuit;
    circuit.safe_append_u("H", {0});
    circuit.safe_append_u("CX", {0, 1});
    sim.safe_do_circuit(circuit);

    // Get the inverse tableau
    auto inv = sim.inv_state;

    // Compute its inverse (which should be the forward tableau)
    auto fwd = inv.inverse();

    // fwd.then(inv) should be identity
    auto should_be_identity = fwd.then(inv);

    // Check identity: xs[0] should be +X only on qubit 0
    REQUIRE(should_be_identity.xs[0].xs[0] == true);
    REQUIRE(should_be_identity.xs[0].zs[0] == false);
    REQUIRE(should_be_identity.xs[0].xs[1] == false);

    // Check identity: zs[0] should be +Z only on qubit 0
    REQUIRE(should_be_identity.zs[0].xs[0] == false);
    REQUIRE(should_be_identity.zs[0].zs[0] == true);
    REQUIRE(should_be_identity.zs[0].zs[1] == false);
}
