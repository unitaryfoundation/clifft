# Planning and Triage

The [Clifft development project](https://github.com/orgs/unitaryfoundation/projects/35)
holds the current work queues and release plans. Issues hold the goals, evidence,
acceptance criteria, and decisions; pull requests hold the implementation and
review. This guide describes how to maintain the project without duplicating its
backlog in documentation.

Public visibility lets anyone read the project. Editing requires project write
or admin access. Contributors can report problems and propose changes through
repository issues and pull requests without project write access.

## Tracks and Bugs

Tracks are ongoing areas of work that continue across releases:

| Track | Scope |
| --- | --- |
| Compiler and correctness | Compiler transformations, semantic correctness, and their validation |
| CPU throughput | CPU execution performance and reducing work per shot |
| Leakage and loss | Noncomputational states and their simulation workflows |
| New simulation capabilities | New supported workloads and simulation approaches |
| Verification and infrastructure | Formal verification, build and test infrastructure, and backend validation |

Choose the track responsible for the deliverable. For work spanning several
tracks, choose its primary track and split independently useful deliverables
when needed.

Bugs stay in their owning track and use the existing `bug` label. For example,
a leakage bug belongs in Leakage and loss; a CPU throughput regression belongs
in CPU throughput. The bug report template already applies this label.
Filter the project by `label:bug -status:Done` to see open bug work across tracks.

Prioritize bugs by impact and evidence. Incorrect simulation results and
regressions that block users can interrupt planned work; the `bug` label alone
does not make every report urgent. Other labels, such as `performance`, `GPU`,
and `documentation`, describe characteristics that may span tracks. Do not
duplicate Track or Priority values as labels.

## Planning Fields

| Field | Meaning |
| --- | --- |
| Track | The area responsible for the work |
| Priority | Planning horizon: Now, Next, Later, or Parked |
| Order | Preferred sequence within a track; lower numbers come first |
| Status | Execution progress: Todo, In progress, In review, or Done |
| Milestone | Intended repository release target |

**Now** means active implementation or review. **Next** is queued work for the
next available capacity. **Later** is useful work beyond the current focus.
**Parked** needs new evidence, capacity, or a concrete consumer before resuming.
These values describe scheduling, not bug severity.

Keep the active set small. Space Order values by ten so work can be inserted
between existing items. Preferred order is not a prerequisite: record issue
dependencies only for actual blockers. In review does not mean approved, and
passing CI alone does not establish that work is complete.

Use milestones to scope the next one or two releases. Research can remain
unscheduled and conclude with a documented proceed, defer, or reject decision.
Update the Upcoming releases view's milestone filter when release targets change.

## Intake and Routine Maintenance

Configure the project's **Workflows > Auto-add to project** for
`unitaryfoundation/clifft` with the filter `is:issue is:open`. This catches
matching issues when they are created or updated; enabling it does not backfill
existing issues. Import any missing backlog separately. Auto-add does not choose
Track, Priority, or Order. See
[GitHub's auto-add documentation](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/adding-items-automatically).

1. Review unclassified arrivals in the **Inbox** view, filtered by
   `no:track -status:Done`. Assign Track and Priority together, and give queued
   work an Order. Leave Milestone unset until there is a release commitment.
2. Use **Tracks** to review the order within each track, **Current work** for
   active implementation and review, and **Upcoming releases** for release scope.
3. Prefer an issue as the planning item, with its implementation PR linked.
   Add a standalone PR manually when it represents an independent deliverable.
   Avoid counting the same deliverable twice; an umbrella issue and a PR for one
   bounded increment can both be useful.
4. Update Status as work progresses. Use completion workflows for closed issues
   and merged PRs, and check that the full issue scope is satisfied before
   closing an umbrella issue. The project also has an Auto-close issue workflow:
   moving an issue to Done can close the issue itself.
5. When work finishes, revisit the next item in the affected track. Adjust the
   queue and release scope when evidence or available capacity changes.

## Maintenance by AI Agents

When asked to manage the project, read the live project fields and relevant issue
and PR discussions before making changes. Use the latest evidence and the
maintainer's agreed priorities; a chat summary or this guide is not a current
backlog. Apply authorized updates and report material changes and unresolved
decisions.

Check for existing project membership before adding an item. Preserve links
between issues and PRs, and verify saved field values after edits. If an API
listing unexpectedly appears empty, check individual item membership or the
browser before rebuilding the project.

Keep planning decisions in project fields and technical rationale in issues.
Update this guide when the workflow changes, rather than copying a snapshot of
priorities into repository files. AI assistance does not replace the review
requirements in the [contribution guide](contributing.md#ai-assisted-contributions).
