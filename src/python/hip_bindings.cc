#include "clifft/frontend/hir.h"
#include "clifft/sampling/hip/executable_plan.h"
#include "clifft/sampling/hip/sampler.h"
#include "clifft/sampling/planner.h"

#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

template <typename T>
nb::ndarray<nb::numpy, T, nb::c_contig> vec_to_numpy(std::vector<T> values,
                                                     std::initializer_list<size_t> shape) {
    auto owner_ptr = std::make_unique<std::vector<T>>(std::move(values));
    T* data = owner_ptr->data();
    nb::capsule owner(owner_ptr.release(),
                      [](void* pointer) noexcept { delete static_cast<std::vector<T>*>(pointer); });
    return nb::ndarray<nb::numpy, T, nb::c_contig>(data, shape, owner);
}

nb::object rows_to_python(clifft::sampling::SamplingResult result, uint32_t rows,
                          const clifft::sampling::hip::Sampler& sampler) {
    auto measurements =
        vec_to_numpy(std::move(result.measurements), {rows, sampler.num_visible_records()});
    auto detectors = vec_to_numpy(std::move(result.detectors), {rows, sampler.num_detectors()});
    auto observables =
        vec_to_numpy(std::move(result.observables), {rows, sampler.num_observables()});
    auto exp_vals = vec_to_numpy(std::move(result.exp_vals), {rows, sampler.num_exp_vals()});
    nb::object cls = nb::module_::import_("clifft._sample_result").attr("SampleResult");
    return cls(measurements, detectors, observables, nb::none(), nb::none(), nb::none(), nb::none(),
               exp_vals);
}

nb::object survivors_to_python(clifft::sampling::SamplingSurvivorResult result,
                               const clifft::sampling::hip::Sampler& sampler, bool keep_records) {
    const uint32_t rows = keep_records ? result.passed_shots : 0;
    auto measurements =
        vec_to_numpy(std::move(result.measurements), {rows, sampler.num_visible_records()});
    auto detectors = vec_to_numpy(std::move(result.detectors), {rows, sampler.num_detectors()});
    auto observables =
        vec_to_numpy(std::move(result.observables), {rows, sampler.num_observables()});
    auto exp_vals = vec_to_numpy(std::move(result.exp_vals), {rows, sampler.num_exp_vals()});
    const size_t observable_count = result.observable_ones.size();
    auto observable_ones = vec_to_numpy(std::move(result.observable_ones), {observable_count});
    nb::object cls = nb::module_::import_("clifft._sample_result").attr("SampleResult");
    return cls(measurements, detectors, observables, result.total_shots, result.passed_shots,
               result.logical_errors, observable_ones, exp_vals);
}

}  // namespace

NB_MODULE(_clifft_hip, module) {
    using clifft::sampling::hip::CoefficientPrecision;
    using clifft::sampling::hip::ExecutablePlan;
    using clifft::sampling::hip::Sampler;

    module.doc() = "Experimental Clifft HIP backend";

    nb::enum_<CoefficientPrecision>(module, "CoefficientPrecision")
        .value("FP64", CoefficientPrecision::FP64)
        .value("FP32", CoefficientPrecision::FP32);

    nb::class_<ExecutablePlan>(module, "Program")
        .def_prop_ro("peak_active_width", &ExecutablePlan::peak_active_width)
        .def_prop_ro("num_actions", &ExecutablePlan::num_actions)
        .def_prop_ro("num_measurements", &ExecutablePlan::num_visible_records)
        .def_prop_ro("num_records", &ExecutablePlan::num_records)
        .def_prop_ro("num_detectors", &ExecutablePlan::num_detectors)
        .def_prop_ro("num_observables", &ExecutablePlan::num_observables)
        .def_prop_ro("num_exp_vals", &ExecutablePlan::num_exp_vals)
        .def_prop_ro("has_postselection", &ExecutablePlan::has_postselection)
        .def_prop_ro("packed_bytes", &ExecutablePlan::packed_bytes)
        .def("inspect", &ExecutablePlan::inspect)
        .def("__repr__", [](const ExecutablePlan& program) {
            return "Program(" + std::to_string(program.num_actions()) +
                   " actions, peak_active_width=" + std::to_string(program.peak_active_width()) +
                   ")";
        });

    module.def(
        "lower",
        [](const clifft::HirModule& hir, std::vector<uint8_t> postselection_mask,
           std::vector<uint8_t> expected_detectors, std::vector<uint8_t> expected_observables) {
            nb::gil_scoped_release release;
            return ExecutablePlan(clifft::sampling::plan_sampling(
                hir, {postselection_mask, expected_detectors, expected_observables}));
        },
        nb::arg("hir"), nb::arg("postselection_mask") = std::vector<uint8_t>{},
        nb::arg("expected_detectors") = std::vector<uint8_t>{},
        nb::arg("expected_observables") = std::vector<uint8_t>{});

    nb::class_<Sampler>(module, "Sampler")
        .def(nb::init<const ExecutablePlan&, CoefficientPrecision, uint32_t>(), nb::arg("program"),
             nb::arg("coefficient_precision") = CoefficientPrecision::FP64,
             nb::arg("max_batch_shots") = clifft::sampling::hip::kDefaultMaxBatchShots)
        .def_prop_ro("coefficient_precision",
                     [](const Sampler& sampler) { return sampler.coefficient_precision(); })
        .def_prop_ro("max_batch_shots",
                     [](const Sampler& sampler) { return sampler.max_batch_shots(); })
        .def_prop_ro("allocated_device_bytes",
                     [](const Sampler& sampler) { return sampler.allocated_device_bytes(); })
        .def(
            "sample",
            [](Sampler& sampler, uint32_t shots, std::optional<uint64_t> seed,
               uint32_t block_size) {
                clifft::sampling::SamplingResult result;
                {
                    nb::gil_scoped_release release;
                    result = sampler.sample(shots, seed, block_size);
                }
                return rows_to_python(std::move(result), shots, sampler);
            },
            nb::arg("shots"), nb::arg("seed") = nb::none(),
            nb::arg("block_size") = clifft::sampling::hip::kDefaultBlockSize)
        .def(
            "sample_survivors",
            [](Sampler& sampler, uint32_t shots, bool keep_records, std::optional<uint64_t> seed,
               uint32_t block_size) {
                clifft::sampling::SamplingSurvivorResult result;
                {
                    nb::gil_scoped_release release;
                    result = sampler.sample_survivors(shots, keep_records, seed, block_size);
                }
                return survivors_to_python(std::move(result), sampler, keep_records);
            },
            nb::arg("shots"), nb::arg("keep_records") = false, nb::arg("seed") = nb::none(),
            nb::arg("block_size") = clifft::sampling::hip::kDefaultBlockSize)
        .def(
            "replay_shot",
            [](Sampler& sampler, std::vector<uint8_t> forced_records) {
                clifft::sampling::hip::ReplayResult result;
                {
                    nb::gil_scoped_release release;
                    result = sampler.replay_shot(forced_records);
                }
                nb::dict output;
                output["reachable"] = result.reachable;
                output["survived"] = result.survived;
                output["log_probability"] = result.log_probability;
                output["outputs"] =
                    rows_to_python(std::move(result.outputs),
                                   result.reachable && result.survived ? 1 : 0, sampler);
                return output;
            },
            nb::arg("forced_records"));

    module.def("is_available", &clifft::sampling::hip::is_available);
    module.def("backend_info", &clifft::sampling::hip::backend_info);
}
