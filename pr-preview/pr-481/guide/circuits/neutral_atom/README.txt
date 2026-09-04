Neutral-atom tutorial circuits
==============================

These circuits are translations of the final Shor schedules used for Figure 9
of:

  Rich Rines et al., "Demonstration of a Logical Architecture Uniting Motion
  and In-Place Entanglement," arXiv:2509.13247.
  https://arxiv.org/abs/2509.13247

The original schedules and noise model are in the public supplementary
artifact:

  https://zenodo.org/records/17137995
  archive: shor_cdcx_mhc.zip
  archive MD5: 9441a5407c19ba7366e041a2241b8b4c
  source code license: Apache-2.0

The checked-in circuits use the artifact's alpha=1 noise parameters. Physical
atom motion has already been resolved into atom-wire relabeling, so no quantum
SWAP represents a move. The tutorial script derives the remaining noise-sweep
points from these nominal circuits in memory.
