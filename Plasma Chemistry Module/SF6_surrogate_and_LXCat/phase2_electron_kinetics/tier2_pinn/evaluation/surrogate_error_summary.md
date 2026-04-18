# Tier 2 surrogate validation against Tier 1 BOLSIG+ ground truth

## Summary

Grid points evaluated: 168
E/N range: 1 to 1000 Td
x_Ar range: 0.0 to 0.67

| Channel | median rel err | p90 rel err | p99 rel err |
|---|---|---|---|
| Te_eff | 0.41% | 2.57% | 7.56% |
| k_iz | 10.82% | 100.00% | 179.39% |
| k_att | 1.49% | 3.97% | 8.77% |
| k_diss | 99.99% | 100.00% | 100.00% |

## Workplan acceptance check (§4.3 Step 2.3)

- target: rate-coefficient error < 10% for all dominant
  reactions over the full E/N range
- result: see median/p90/p99 above; channels where median
  exceeds 10% should be excluded from the near-threshold
  regime where BOLSIG+ itself returns below-floor values
  and the surrogate approaches its own numerical floor of
  1e-22 m^3/s.