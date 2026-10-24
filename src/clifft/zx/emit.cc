#include "clifft/zx/emit.h"

#include "clifft/util/numeric.h"
#include "clifft/zx/bits.h"

#include <cmath>
#include <numbers>
#include <utility>

namespace clifft::zx {
namespace {

constexpr uint32_t kNone = ~uint32_t{0};

// The emitted form of one PBC op, before the line layout is final.
struct Pending {
    const PbcOp* op = nullptr;
    uint32_t offset = 0;      // the group's first line
    Bits sign_items;          // basis items whose sign lines carry Z
    bool needs_zero = false;  // a measurement of a constant
};

}  // namespace

std::optional<HirModule> emit_hir(const HirModule& input, const SingleCopyDiagram& diagram,
                                  const PbcProgram& program,
                                  const std::vector<std::vector<uint32_t>>& orders,
                                  std::string* reason) {
    auto reject = [&](std::string why) -> std::optional<HirModule> {
        if (reason) {
            *reason = std::move(why);
        }
        return std::nullopt;
    };

    // Noise bits as columns of one global vector.
    std::vector<uint32_t> column(diagram.params.size(), kNone);
    uint32_t num_columns = 0;
    for (uint32_t p = 0; p < diagram.params.size(); ++p) {
        if (diagram.params[p].kind == DiagramParam::Kind::Noise) {
            column[p] = num_columns++;
        }
    }
    std::vector<uint32_t> group_of_site(input.noise_sites.size(), kNone);
    std::vector<uint32_t> readout_groups;
    std::vector<bool> readout_in_diagram(input.readout_noise.size(), false);
    for (uint32_t g = 0; g < diagram.noise.size(); ++g) {
        const NoiseGroup& group = diagram.noise[g];
        if (group.source == NoiseGroup::Source::Readout) {
            readout_groups.push_back(g);
            readout_in_diagram[group.source_index] = true;
            continue;
        }
        if (group_of_site[group.source_index] != kNone) {
            return reject("a noise site is applied more than once");
        }
        group_of_site[group.source_index] = g;
    }

    // The sign basis: each noise-dependent sign part that is not yet a
    // combination of earlier ones becomes a new basis item and sign line.
    size_t num_ops = 0;
    for (const PbcGroup& g : program.groups) {
        num_ops += g.ops.size();
    }
    EchelonBasis basis(num_columns, num_ops);
    std::vector<uint32_t> line_of_item(num_ops, kNone);
    std::vector<Bits> sign_vectors;  // over columns, by sign line
    std::vector<Pending> pending;
    pending.reserve(num_ops);
    std::vector<std::vector<size_t>> pending_of_group(program.groups.size());
    uint32_t offset = 0;
    bool needs_zero = false;
    for (size_t gi = 0; gi < program.groups.size(); ++gi) {
        const PbcGroup& g = program.groups[gi];
        for (uint32_t j : orders[gi]) {
            const PbcOp& op = g.ops[j];
            Bits noise(num_columns);
            for (size_t k = 0; k < g.noise.size(); ++k) {
                if (op.sign.test(1 + k)) {
                    noise.flip(column[g.noise[k]]);
                }
            }
            Bits items(num_ops);
            if (noise.any()) {
                std::optional<Bits> combination = basis.solve(noise);
                if (!combination) {
                    const size_t item = pending.size();
                    line_of_item[item] = static_cast<uint32_t>(sign_vectors.size());
                    sign_vectors.push_back(noise);
                    (void)basis.insert(noise, Bits::single(num_ops, item));
                    combination = basis.solve(noise);
                }
                items = std::move(*combination);
            }
            const bool body = op.x.any() || op.z.any();
            Pending p{.op = &op, .offset = offset, .sign_items = std::move(items)};
            p.needs_zero = !body && !p.sign_items.any() && op.kind == PbcOp::Kind::Measurement;
            needs_zero |= p.needs_zero;
            pending_of_group[gi].push_back(pending.size());
            pending.push_back(std::move(p));
        }
        offset += g.num_lines;
    }
    const uint32_t zero_line = offset;
    const uint32_t first_sign_line = offset + (needs_zero ? 1 : 0);
    const uint32_t num_qubits = first_sign_line + static_cast<uint32_t>(sign_vectors.size());

    size_t num_channels = readout_groups.size();
    for (const NoiseSite& site : input.noise_sites) {
        num_channels += site.channels.size();
    }
    HirModule out(num_qubits, num_ops, num_channels);
    out.num_measurements = input.num_measurements;
    out.num_detectors = input.num_detectors;
    out.num_observables = input.num_observables;
    out.detector_targets = input.detector_targets;
    out.observable_targets = input.observable_targets;
    out.readout_noise = input.readout_noise;
    out.neglect_instrument_damping = input.neglect_instrument_damping;
    out.postselection = input.postselection;

    // A channel flips each sign line whose parity its pattern changes.
    auto channel_mask = [&](const NoiseGroup* group, size_t outcome) {
        const PauliMaskHandle h = out.claim_empty_noise_channel_mask();
        if (!group) {
            return h;  // a site no record sees
        }
        Bits fired(num_columns);
        const uint64_t pattern = group->outcomes[outcome].pattern;
        for (size_t k = 0; k < group->bits.size(); ++k) {
            if ((pattern >> k) & 1) {
                fired.set(column[group->bits[k]]);
            }
        }
        MutablePauliMaskView mask = out.noise_channel_masks.mut_at(h);
        for (uint32_t l = 0; l < sign_vectors.size(); ++l) {
            if (sign_vectors[l].parity_with(fired)) {
                mask.x().bit_set(first_sign_line + l, true);
            }
        }
        return h;
    };
    // Sites keep their index, order and probabilities.
    out.noise_sites.reserve(input.noise_sites.size() + readout_groups.size());
    for (uint32_t s = 0; s < input.noise_sites.size(); ++s) {
        const NoiseSite& site = input.noise_sites[s];
        const NoiseGroup* group =
            group_of_site[s] != kNone ? &diagram.noise[group_of_site[s]] : nullptr;
        NoiseSite rebuilt{.total_probability = site.total_probability, .channels = {}};
        for (size_t c = 0; c < site.channels.size(); ++c) {
            rebuilt.channels.push_back(
                {.mask = channel_mask(group, c + 1), .prob = site.channels[c].prob});
        }
        out.noise_sites.push_back(std::move(rebuilt));
    }
    std::vector<NoiseSiteIdx> readout_sites;
    for (uint32_t g : readout_groups) {
        const NoiseGroup& group = diagram.noise[g];
        const double p = group.outcomes[1].probability;
        readout_sites.push_back(NoiseSiteIdx{static_cast<uint32_t>(out.noise_sites.size())});
        out.noise_sites.push_back(
            {.total_probability = p, .channels = {{.mask = channel_mask(&group, 1), .prob = p}}});
    }

    // Noise first, in its original order: every op's sign reads all of it.
    std::vector<std::vector<uint32_t>> readout_of_record(input.num_measurements +
                                                         input.num_hidden_measurements);
    std::vector<HeisenbergOp> classical;  // detectors and observables
    for (const HeisenbergOp& op : input.ops) {
        switch (op.op_type()) {
            case OpType::NOISE:
                out.append_noise(op.noise_site_idx());
                break;
            case OpType::READOUT_NOISE: {
                const auto k = static_cast<uint32_t>(op.readout_noise_idx());
                if (readout_in_diagram[k]) {
                    break;
                }
                const uint32_t r = input.readout_noise[k].meas_idx;
                if (r >= input.num_measurements) {
                    return reject("readout noise flips a hidden record");
                }
                readout_of_record[r].push_back(k);
                break;
            }
            case OpType::DETECTOR:
            case OpType::OBSERVABLE:
                classical.push_back(op);
                break;
            default:
                break;
        }
    }

    for (NoiseSiteIdx site : readout_sites) {
        out.append_noise(site);
    }

    uint32_t num_hidden = 0;
    for (size_t gi = 0; gi < program.groups.size(); ++gi) {
        for (size_t idx : pending_of_group[gi]) {
            const Pending& p = pending[idx];
            const PbcOp& op = *p.op;
            auto fill = [&](MutablePauliMaskView slot) {
                for (size_t l = 0; l < op.x.size(); ++l) {
                    slot.x().bit_set(p.offset + static_cast<uint32_t>(l), op.x.test(l));
                    slot.z().bit_set(p.offset + static_cast<uint32_t>(l), op.z.test(l));
                }
                for (size_t item : p.sign_items.ones()) {
                    slot.z().bit_set(first_sign_line + line_of_item[item], true);
                }
                if (p.needs_zero) {
                    slot.z().bit_set(zero_line, true);
                }
                slot.set_sign(op.sign.test(0));
            };
            if (op.kind == PbcOp::Kind::Rotation) {
                if (!op.x.any() && !op.z.any()) {
                    continue;  // a phase, global within each noise trajectory
                }
                // The reader leaves only the non-Clifford remainder of each
                // phase, so alpha lies in [-1/4, 1/4] and T is its only
                // named case.
                const double alpha = 2.0 * op.theta / std::numbers::pi;
                if (std::abs(std::abs(alpha) - 0.25) < kRotationCanonicalizationTolerance) {
                    out.append_tgate(alpha < 0.0, fill);
                } else {
                    out.append_phase_rotation(alpha, fill);
                }
                continue;
            }
            if (op.record < 0) {
                const MeasRecordIdx slot{input.num_measurements + num_hidden++};
                out.append_measure(slot, fill).set_hidden(true);
                continue;
            }
            const auto r = static_cast<uint32_t>(op.record);
            out.append_measure(MeasRecordIdx{r}, fill);
            for (uint32_t k : readout_of_record[r]) {
                out.append_readout_noise(ReadoutNoiseIdx{k});
            }
        }
    }
    out.num_hidden_measurements = num_hidden;
    out.ops.insert(out.ops.end(), classical.begin(), classical.end());
    if (!input.source_map.empty()) {
        out.source_map.assign(out.ops.size(), {});
    }
    return out;
}

}  // namespace clifft::zx
