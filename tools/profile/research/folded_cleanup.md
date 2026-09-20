# Folded block cleanup

The research prototype is checkpointed at `50fc27df` on
`codex/cultivation-circuit-study`. The cleanup is isolated on
`codex/folded-block-cleanup`. Historical reports and their measurements describe
the checkpoint; they have not been rewritten as measurements of this refactor.

## Scope and intended production interface

The eventual interface is an opt-in compiler specialization: accept an ordinary
physical circuit, identify a supported folded region, certify its signed-code
boundary and record/noise effects, and substitute a precompiled kernel. Users
would continue to sample through Clifft's usual APIs. Unmatched regions would
follow ordinary compilation. Recognition must examine actual operations and
boundary proofs, without requiring generator stage labels.

This cleanup remains a research implementation. Its compiler still consumes the
reconstructed MSC generator's stage labels, and its standalone runner combines
Clifft prefix execution with the specialized blocks. It does not install an HIR
pass or add a public sampling option.

Core integration needs a deliberate extension to the sampling plan and executor:
the existing action set assumes dense active coefficients, and instrument
continuations preserve that representation. A specialized block needs an explicit
planned boundary and state transfer, shared record and noise indexing, and tests
through the ordinary sampling API. The instrument trap API is not a general
representation-switching hook. Replay, fixed-fault sampling, final-state queries
and non-scalar backends need explicit supported behavior before the option is
advertised as transparent.

## Simplifications

- A nonrecursive `FoldedBlock` executes one folded region and optionally its
  terminal continuation. `Protocol` owns the concrete sequence of blocks and
  growth; an intermediate f5 block has no dummy terminal measurement.
- `sequence.txt` explicitly selects the supported sequence. File existence no
  longer changes execution. Bundle versions are required; regenerate research
  bundles with the updated compiler instead of retaining compatibility branches.
- The contraction kernel takes caller-owned inputs. Fixture parsing and fixture
  normalization checks live in `sample_folded_checks.cpp`.
- The compiler emits compact gathers directly. The native loader only validates
  dimensions and addresses. The expanded NumPy representation remains a reference
  evaluator, and the older bound-plan format remains in the independent profiling
  tool that exercises it.
- Growth compilation lives in `folded_growth.py`; the audit owns physical Stim
  replay. Growth table dimensions and record offsets are compiler-supplied.
- Block boundaries use physical logical X/Z axes. Contraction coset conversions
  belong inside the folded block, rather than in the growth adapter.

No block registry or extensible gadget framework is introduced. The two Clifford
continuations keep their existing execution algorithms: sequential CSS extraction
and growth have different useful table structures today. A common transfer IR
should wait for a second concrete use that actually reduces code. Bounds checks
and boundary certification remain, even when existing fixtures do not exercise
every rejection.

## Validation

All 25 regression tests pass: complete physical f3/f5/f7 histories with both
f7 prefix modes, native terminal checks against independent dense projectors,
folded logical Born probabilities, compact and irregular contraction tables,
and rejection of malformed table addresses and dimensions. Reused directories
with stale growth files are also covered.

Twelve f7 histories at p=0.001, seed 9134, match the checkpoint exactly, including
all physical fault selections, visible and hidden records, detector bits, logical
output and conditional log probability. The standalone growth audit also passes
36 physical Stim replays across logical X/Y/Z inputs and p=0, 0.001 and 0.03.

The bounded timing comparison uses the same PR-471-enabled core build with
unbounded scheduling, GCC 13.3 Release, one worker pinned to CPU 0, p=0.001 and
seed 9134. Each process performs 16 warmups and three batches: 16 full attempts
per batch or 64 early-rejected attempts. Acceptance and logical-failure counts
match in every batch. These short runs check for regressions, not error rates or
small performance differences.

| Measurement | Checkpoint | Cleanup |
| --- | ---: | ---: |
| Full attempt median | 193.84 ms | 194.84 ms |
| Early-rejected attempt median | 34.89 ms | 34.73 ms |
| Peak process RSS, full attempts | 39.89 MiB | 40.28 MiB |
| Peak process RSS, early rejection | 40.15 MiB | 40.22 MiB |
| Native setup, full run | 0.279 s | 0.096 s |
| Serialized contraction tables | 36,380,187 bytes | 10,331,041 bytes |

Resident contraction lookup and workspace payloads are unchanged: 9,023,524 and
2,570,720 bytes respectively. The smaller files reduce loading work; the Python
reference planner still materializes expanded gathers during compilation.

- [Raw comparison commands, batches, memory and executable hashes](folded_cleanup_benchmark.json)
- [Identical histories and growth replay results](folded_cleanup_validation.json)
