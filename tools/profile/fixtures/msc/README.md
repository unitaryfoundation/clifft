# Original MSC controls

These two files are byte-for-byte copies of the immutable benchmark corpus at
`/tmp/clifft-bench/workloads/circuits/`, originally copied from
`unitaryfoundation/clifft-paper` commit
`db7dc9f13a2c2854690e92390c779048a1ac1400`:

| Local file | Original source path | SHA-256 |
| --- | --- | --- |
| `msc_d3_inject_cultivate_p1e-3.stim` | `qec_bench/circuits/cultivation_d3.stim` | `90a7d841e003e5ee38137cd9a3eb6529bb552e49c424bc6b0932a27d97cdb41f` |
| `msc_d5_inject_cultivate_p1e-3.stim` | `qec_bench/circuits/cultivation_d5.stim` | `c2b4566917bd9bf27a5705284dac02700ef0dcc7c03c91066670db376d633a6d` |

The accompanying Apache-2.0 license is copied from the same corpus. These are
color-code cultivation controls, distinct from the regular-surface f7
reconstruction. They retain the original physical noise, measurement records,
feedforward, detector declarations, and terminal observable. No one-round
memory inputs were added.
