# Action census: compiled SamplingPlan programs (current main)

clifft 0.8.1.dev23+ge3c76df23, default passes, `clifft.compile()` ->
`Program.inspect_action()` walk. Same circuit corpus as the legacy-bytecode
census (research/gpu/opcode_census.md on the clifft-research branch); the
legacy numbers are quoted there and are NOT comparable action-for-action,
only work-share-for-work-share.

## Shape metrics

dense% = share of actions touching the 2^w coefficient array (the rest are
frame/record actions, effectively free on any backend). rot/meas = rotation
actions (direct + fused) per active measurement. prom/meas = width-raising
actions per width-lowering one. mean w = coefficient-visit-weighted.

| workload | peak w | actions | dense% | rot/meas | prom/meas | mean w |
|---|---|---|---|---|---|---|
| rand_cliffT_n20_d40_t05 | 15 | 42 | 88% | 0.5 | 15/15 | 13.6 |
| rand_cliffT_n20_d40_t15 | 20 | 99 | 100% | 3.0 | 20/20 | 19.5 |
| hidden_shift_np8_tl2 | 16 | 48 | 100% | 1.0 | 16/16 | 14.8 |
| qaoa_ring_n16_p3 | 15 | 112 | 99% | 5.4 | 15/15 | 14.9 |
| iqp_n028_t1_cz30 | 14 | 56 | 75% | 1.0 | 14/14 | 12.5 |
| hidden_ccz_t4 | 12 | 50 | 100% | 2.2 | 12/12 | 10.9 |
| conveyor_r16_w24 | 11 | 768 | 74% | 1.1 | 185/185 | 8.4 |
| surface_d5_r6 | 0 | 169 | 0% | inf | 0/0 | 0.0 |

## Share of visit-weighted dense work by action class

| workload | ROTATE | FUSED_ROTATION | DYNAMIC_FUSED_ROTATION | PROMOTE | MEASURE_ACTIVE |
|---|---|---|---|---|---|
| rand_cliffT_n20_d40_t05 | 5% | - | - | 32% | 63% |
| rand_cliffT_n20_d40_t15 | 82% | 0% | - | 6% | 12% |
| hidden_shift_np8_tl2 | 25% | - | - | 25% | 50% |
| qaoa_ring_n16_p3 | 92% | - | - | 3% | 5% |
| iqp_n028_t1_cz30 | 15% | - | - | 28% | 57% |
| hidden_ccz_t4 | 34% | 11% | - | 18% | 37% |
| conveyor_r16_w24 | 22% | - | - | 26% | 52% |
| surface_d5_r6 | - | - | - | - | - |

## On-chip residency: share of dense visits inside each width band

Bytes per coefficient: 16 (split f64). Bands ignore reduction scratch, so
they are upper bounds on shared-memory/LDS eligibility.

| workload | peak w | w<=11 (64KB LDS) | w<=13 (227KB smem) |
|---|---|---|---|
| rand_cliffT_n20_d40_t05 | 15 | 9% | 37% |
| rand_cliffT_n20_d40_t15 | 20 | 0% | 0% |
| hidden_shift_np8_tl2 | 16 | 4% | 16% |
| qaoa_ring_n16_p3 | 15 | 1% | 3% |
| iqp_n028_t1_cz30 | 14 | 19% | 72% |
| hidden_ccz_t4 | 12 | 54% | 100% |
| conveyor_r16_w24 | 11 | 100% | 100% |
| surface_d5_r6 | 0 | 0% | 0% |

## Width profile of dense work (share of visits at each w, top 5)

- **rand_cliffT_n20_d40_t05**: w=14: 32%, w=15: 32%, w=13: 20%, w=12: 8%, w=11: 4%
- **rand_cliffT_n20_d40_t15**: w=20: 75%, w=18: 10%, w=19: 9%, w=17: 3%, w=16: 2%
- **hidden_shift_np8_tl2**: w=16: 38%, w=15: 31%, w=14: 16%, w=13: 8%, w=12: 4%
- **qaoa_ring_n16_p3**: w=15: 93%, w=14: 3%, w=13: 2%, w=12: 1%, w=11: 0%
- **iqp_n028_t1_cz30**: w=13: 35%, w=14: 28%, w=12: 18%, w=11: 9%, w=10: 5%
- **hidden_ccz_t4**: w=12: 46%, w=11: 28%, w=9: 10%, w=10: 9%, w=8: 3%
- **conveyor_r16_w24**: w=9: 33%, w=8: 24%, w=10: 20%, w=7: 11%, w=6: 5%
- **surface_d5_r6**: no dense work (frame-only program)

## Measurement and rotation kernel selections

- **rand_cliffT_n20_d40_t05**: meas [scalar: 15], rotate [scalar: 7]
- **rand_cliffT_n20_d40_t15**: meas [scalar: 20], rotate [scalar: 58]
- **hidden_shift_np8_tl2**: meas [scalar: 16], rotate [scalar: 16]
- **qaoa_ring_n16_p3**: meas [scalar: 15], rotate [scalar: 81]
- **iqp_n028_t1_cz30**: meas [scalar: 14], rotate [scalar: 14]
- **hidden_ccz_t4**: meas [scalar: 12], rotate [scalar: 22]
- **conveyor_r16_w24**: meas [scalar: 185], rotate [scalar: 199]
