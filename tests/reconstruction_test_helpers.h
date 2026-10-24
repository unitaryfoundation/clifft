#pragma once

#include "clifft/optimizer/global_reconstruction.h"

#include "dense_circuit_oracle.h"

#include <map>
#include <string>
#include <vector>

namespace clifft::test {

inline std::vector<std::string> reconstruction_law_cases() {
    return {
        "",
        "DETECTOR\nOBSERVABLE_INCLUDE(2)\n",
        "H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "H 0 1\nT 0\nT_DAG 1\nCX 0 1\nS 0\nMPP Y0*X1\nM 0 1\nDETECTOR rec[-3] rec[-1]\n",
        "H 0\nCX 0 1\nT 0\nR 0\nH 0\nT_DAG 0\nM 0 1\nDETECTOR rec[-2] rec[-1]\n",
        "H 0\nT 0\nMRX 0\nRY 1\nCX 0 1\nT_DAG 1\nMRY 1\nM 0 1\n",
        "H 0\nT 0\nM(0.2) !0\nCX rec[-1] 1\nH 1\nT 1\nMX 1\nDETECTOR rec[-1] rec[-2]\n",
        "H 0\nPAULI_CHANNEL_1(0.1,0.2,0.3) 0\nT 0\nMX 0\nDETECTOR rec[-1]\n",
        "H 0 1\nT 0\nDEPOLARIZE2(0.4) 0 1\nT_DAG 1\nMPP !X0*Y1\nM 0 1\n",
        "H 0 1\nT 0\nPAULI_CHANNEL_2(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0,0,0,0,0) 0 "
        "1\nT_DAG 1\nMPP X0*Y1\nM 0 1\n",
        "H 0 1\nT 0\nE(0.2) X0 Y1\nELSE_CORRELATED_ERROR(0.3) Z0\nT_DAG 1\nMX 0 1\nDETECTOR "
        "rec[-1] rec[-2]\n",
        "H 0\nT 0\nM 0\nDETECTOR rec[-1]\nREADOUT_NOISE(0.25) rec[-1]\nCX rec[-1] 1\nM "
        "1\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "H 0\nT 0\nMX 0\nREPEAT 2 {\nCZ rec[-1] 0\nT_DAG 0\nMX 0\n}\nDETECTOR rec[-3] rec[-1]\n",
    };
}

inline std::map<uint64_t, double> output_law(const ExactPlanDistribution& distribution,
                                             const ReconstructionOutputLayout& layout) {
    std::map<uint64_t, double> law;
    for (const auto& e : distribution.kept_outcomes) {
        const uint64_t parities =
            e.outcome.detectors | (e.outcome.observables << layout.num_detectors);
        const uint64_t records = layout.num_records ? e.outcome.visible_records : 0;
        law[records | (parities << layout.num_records)] += e.probability;
    }
    return law;
}

inline void require_same_law(std::map<uint64_t, double> actual,
                             const std::map<uint64_t, double>& expected) {
    for (const auto& [key, p] : expected) {
        REQUIRE_THAT(actual[key], Catch::Matchers::WithinAbs(p, 2e-12));
        actual.erase(key);
    }
    for (const auto& [key, p] : actual)
        REQUIRE_THAT(p, Catch::Matchers::WithinAbs(0, 2e-12));
}

}  // namespace clifft::test
