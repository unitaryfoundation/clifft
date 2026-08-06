# Opcode census: real compiled programs vs the microbench schedule

clifft 0.5.1.dev37+gb8c9758b8.d20260806, default optimization passes.

## Shape metrics

dense% = share of instructions touching the 2^k array (rest are frame-only,
free on any backend). gates/meas = dense gate ops per active measurement
(synthetic layer hard-codes 3k+2 = 50 at k=16). exp/meas = rank-raising ops
per rank-lowering op (1/1 in the synthetic layer). mean-k = amp-weighted.

| workload | peak k | instrs | dense% | gates/meas | exp/meas | mean k |
|---|---|---|---|---|---|---|
| rand_cliffT_n20_d40_t05 | 15 | 543 | 25% | 7.1 | 15/15 | 13.8 |
| rand_cliffT_n20_d40_t15 | 20 | 621 | 56% | 15.3 | 20/20 | 19.6 |
| hidden_shift_np8_tl2 | 16 | 78 | 76% | 1.7 | 16/16 | 14.7 |
| qaoa_ring_n16_p3 | 15 | 225 | 77% | 9.6 | 15/15 | 14.9 |
| iqp_n028_t1_cz30 | 14 | 663 | 23% | 8.9 | 14/14 | 12.6 |
| hidden_ccz_t4 | 12 | 148 | 76% | 7.3 | 12/12 | 11.3 |
| conveyor_r16_w24 | 11 | 2550 | 38% | 3.3 | 185/185 | 8.5 |
| surface_d5_r6 | 0 | 967 | 0% | inf | 0/0 | 0.0 |
| synthetic k=16 L=4 | 16 | 208 | 100% | 50.0 | 4/4 | 16.0 |

## Share of amplitude-weighted dense work by op class

| workload | H | T | CZ | CNOT | U2 | U4 | EXPAND | EXPAND_T | MEAS_DIAG | MEAS_INTERFERE |
|---|---|---|---|---|---|---|---|---|---|---|
| rand_cliffT_n20_d40_t05 | 0% | 3% | 4% | 48% | 2% | – | – | 14% | 0% | 28% |
| rand_cliffT_n20_d40_t15 | 10% | 12% | 7% | 48% | 14% | 1% | – | 3% | 0% | 5% |
| hidden_shift_np8_tl2 | 0% | 0% | 2% | – | – | 24% | – | 24% | – | 49% |
| qaoa_ring_n16_p3 | – | 2% | 11% | 5% | 52% | 21% | – | 2% | – | 5% |
| iqp_n028_t1_cz30 | 8% | 4% | 5% | 41% | – | – | – | 14% | – | 28% |
| hidden_ccz_t4 | 12% | 26% | 1% | 25% | – | 6% | – | 10% | – | 20% |
| conveyor_r16_w24 | 17% | 9% | 5% | 9% | – | – | – | 20% | – | 40% |
| surface_d5_r6 | – | – | – | – | – | – | – | – | – | – |
| synthetic k=16 L=4 | 44% | 22% | 1% | 20% | 3% | 3% | 3% | – | 5% | – |

## k profile of dense work (share of amps touched at each k, top 5)

- **rand_cliffT_n20_d40_t05**: k=15: 44%, k=14: 24%, k=13: 18%, k=12: 6%, k=11: 3%
- **rand_cliffT_n20_d40_t15**: k=20: 82%, k=18: 8%, k=19: 6%, k=17: 2%, k=16: 1%
- **hidden_shift_np8_tl2**: k=16: 37%, k=15: 32%, k=14: 15%, k=13: 8%, k=12: 4%
- **qaoa_ring_n16_p3**: k=15: 93%, k=14: 3%, k=13: 2%, k=12: 1%, k=11: 0%
- **iqp_n028_t1_cz30**: k=14: 37%, k=13: 25%, k=12: 20%, k=11: 8%, k=10: 7%
- **hidden_ccz_t4**: k=12: 67%, k=11: 14%, k=9: 10%, k=10: 5%, k=8: 2%
- **conveyor_r16_w24**: k=9: 34%, k=8: 23%, k=10: 21%, k=7: 10%, k=6: 5%
- **surface_d5_r6**: no dense work (frame-only program)
- **synthetic k=16 L=4**: k=16: 97%, k=15: 3%
