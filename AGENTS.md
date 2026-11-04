# Clifft Agent Instructions

This file records repository-specific constraints and recurring review patterns
that are easy to miss from the code alone. Keep design details in their
canonical docs and use `just --list` for routine build and test commands:

- `README.md`: high-level project overview and quick links
- `docs/guide/`: user-facing task guides (motivate *why* and *when* to use a
  feature before options and limits)
- `docs/theory/`: mathematical and architectural explanations
- `docs/reference/`: API, gate, and optimization-pass reference
- `docs/development/`: contributor workflow (`contributing.md`), test strategy
  (`testing.md`), and hardware backend notes

## Project Planning

Use the [Clifft development project](https://github.com/orgs/unitaryfoundation/projects/35)
and its README for current priorities and the planning workflow.

- When asked to manage planning, read the live project and relevant issue/PR
  discussions. Apply the maintainer's agreed priorities and verify saved changes.
- Keep priority and order in project fields, and goals, evidence, and acceptance
  criteria in issues. Avoid duplicating the backlog in repository documentation.
- Check existing membership and linked PRs before adding items. Keep bugs in
  their owning track with the `bug` label; prioritize them by impact and evidence.

## Source of Truth and Architecture Changes

Use `src/clifft/`, `tests/`, `docs/`, and `README.md` as the source of truth.
Inspect the current implementation and tests before resolving architectural
ambiguity.

If a proposed change contradicts an architectural invariant or the current
architecture cannot support it:

1. Stop instead of implementing a workaround or rewriting architecture-facing
   documentation to justify the change.
2. Explain the discrepancy and propose an explicit architectural change.
3. Wait for human confirmation before implementation.

## Architectural Invariants

- **Allocation-free hot execution:** Executors must preallocate coefficient,
  record, symbolic-state, and scratch storage before entering hot dispatch or
  kernels. The sanctioned exception is an explicit trap, continuation, or online
  activation boundary, where storage may grow but never shrink before dispatch
  resumes. No allocation is allowed inside an ordinary dispatch loop or kernel.
- **Exception-free hot execution:** Validate external and compiler-produced
  inputs before entering hot execution. Use assertions rather than exceptions
  for internal preconditions in per-shot reset, ordinary dispatch, and kernels.
  Throwing validation remains appropriate during construction, planning,
  lowering, and explicit trap, continuation, or online activation boundaries.
- **Deterministic RNG:** Prefer `Xoshiro256PlusPlus::next_double()` in
  `src/clifft/util/xoshiro.h` for uniform draws in `[0, 1)`. Do not use
  `std::uniform_real_distribution`; its output is implementation-defined.
- **No runtime topology planning in compiled executors:** The compiler or planner
  must precompute coordinate changes, Pauli pairings, phase behavior,
  active-width transitions, and symbolic dependencies. The executor may apply
  compiler-precomputed multi-coordinate active-Pauli actions directly, but
  runtime code must not perform tableau evolution, commutation analysis,
  localization, or dependency discovery. The explicitly selected
  noncomputational online backend is a separate mode: it may evolve its
  Clifford frame and active coordinates at runtime. Its ordinary dispatch
  remains allocation-free and exception-free; coefficient storage may grow only
  after suspension at an explicit activation boundary.

## Code Simplicity, Prose, and Test Rules

- **Simplify before finishing:** Review the full diff in hindsight. Search for
  and reuse existing helpers before adding new ones; inline trivial
  single-caller bridge helpers, collapse temporary adapter structs or flags, and
  fold single-function bridge files into their owning module. Keep abstractions
  that clarify nontrivial logic, group related state, serve a distinct caller,
  or provide compile-time isolation (such as ISA-specific kernel translation
  units).
- **Replace, do not shim:** When lifting a limitation or replacing a
  compilation or execution path, delete the superseded validation guard,
  write-only field, legacy path, error test, and stale comment in the same
  change.
- **No hand-maintained lists that drift:** Derive lists from the authoritative
  registry or use exhaustive `switch` statements so the compiler catches new
  enum values. Do not enumerate pass lists, gate subsets, or default tables in
  docstrings and comments where they can go stale.
- **Keep temporary investigation artifacts out of the repository:** Put probe
  dumps, ad hoc benchmark drivers, task-specific timing reports, release
  evidence, task plans, and task-specific design notes in the issue or PR.
  Maintained documentation assets, including supporting benchmark data, belong
  with the documentation.
- **Plain prose and established terminology:** Write comments, docstrings, error
  messages, and docs in direct English grounded in standard quantum-circuit
  concepts and Clifft's existing terms (such as *barriers* rather than *fences*,
  and *statevector* rather than *state ray*). Avoid internal specification
  jargon (such as *"executable contract"* or *"supported scientific workflow"*),
  terms that clash with QEC or CS terminology, and defensive *"what this is
  not"* disclaimers.
- **Standalone, present-tense comments:** Keep source files ASCII-only (use
  `pi`, `|0>`, and `Schrodinger`). Give files and non-trivial types a concise
  overview comment (with a short example or formula when helpful), add brief
  guideposts in dense functions, and name non-obvious numeric constants; omit
  comments that restate the code. Write in the present tense without PR diaries,
  bug-fix defenses, or issue/PR numbers in comments or test names.
- **Right-sized documentation:** Give each topic one canonical home and link
  rather than duplicate. Open user-guide sections by motivating *why* a user
  would care in 1-2 sentences, preserve limitations and tradeoffs that help
  users choose correctly, fold long circuits or tables into collapsible blocks,
  and omit internal implementation catalogs, uncommitted APIs, or
  *"stable"*/*"experimental"* labels outside `clifft.experimental`. In
  performance and benchmark docs, explain workload behavior and why performance
  changed; leave collection dates, candidate tags, dependency lock details, and
  harness audit rules in the linked benchmark repository or PR description.
  Update existing sections instead of appending per-release notes, then read the
  whole revised page for flow and consistent detail.
- **Test behavior and invariants, not prose:** Test mathematical, statistical,
  and structural invariants (cross-checking unitaries against Qiskit Aer and
  Clifford noise against Stim), and ensure each test exercises the behavior it
  claims to cover. Optimization-specific tests must demonstrate that the
  intended rewrite fires. Keep validation machinery proportional to regression
  risk and maintenance cost; reuse existing harnesses and prefer focused checks
  over new instrumentation or redundant test matrices. Catch2 `TEST_CASE` names
  must avoid `[]`, `()`, and `,`; never substring-match English prose in
  docs/comments or assert that deleted helpers stay absent. Follow
  [Writing Tests](docs/development/contributing.md#writing-tests).

## Acknowledgments and References

Reserve `docs/acknowledgments.md` (and the `README.md` summary) for substantive
external research or software influences, not minor internal or incidental
items. Each acknowledgment should complete the sentence "Clifft benefited from
this work through..." with a concrete algorithm, implementation, design
influence, validation method, or reused artifact. Distinguish direct use from
inspiration, link the relevant paper, software, or other source, and preserve
citations beside the technical discussion. A paper is not required; the specific
contribution is.

## Git and AI-Assisted Contribution Rules

- Never commit directly to `main`; use a feature branch.
- Keep commits atomic and use conventional prefixes such as `feat:`, `fix:`,
  `test:`, and `docs:`.
- Python is the external API; C++ interfaces have no cross-version compatibility
  guarantee.
- Mark any backward-incompatible Python API change, including removals and
  renames, with a `!` after the type (`feat(api)!:`) and a `BREAKING CHANGE:`
  footer describing the migration. Signature, return-type, default, and
  behavioral changes count. The changelog surfaces both markers; without them
  the entry reads as an ordinary feature.
- Run `uv run --frozen --only-group dev pre-commit run --all-files --show-diff-on-failure`
  before every commit, in addition to the tests appropriate for the change.
- Include an `Assisted-by:` trailer identifying the agent that actually
  assisted. Each agent must use its own name, model, and provider address; do
  not copy a named example literally. Format:

  ```text
  Assisted-by: AGENT_NAME (MODEL_NAME) <PROVIDER_NOREPLY_EMAIL>
  ```
