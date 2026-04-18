# Speedup Measurement

## Solver timing (20 diverse conditions, 30x50 mesh)

| Mode | Mean | Std | Min | Max |
|---|---|---|---|---|
| Legacy FD | 7.762s | 1.070s | 6.567s | 10.794s |
| LXCat FD | 14.432s | 2.084s | 11.377s | 18.817s |

## Surrogate inference (20 cases, 50 repeats each, CPU)

| Model | Mean | Std |
|---|---|---|
| surrogate_v4 | 14.62ms | 10.10ms |
| surrogate_lxcat_v3 | 16.51ms | 3.42ms |

## Speedups

| Comparison | Speedup |
|---|---|
| Legacy solver vs surrogate_v4 | 531x |
| LXCat solver vs LXCat surrogate | 874x |
| LXCat solver vs surrogate_v4 | 987x |
