# Harness calibration results

## Pure-noise features, y_all, n_surr=200: 20 features

- circular-shift surrogate: false positives p<0.05 = 1/20 (5.0%), p<0.0036 = 0; mean p = 0.539 (uniform -> 0.5)
- bin-permutation surrogate (per-bin features): p<0.05 = 1/6
- naive Rhoades T-test (one-sided p<0.05): 0/20; |T|>1.96 in 2/20 (anti-conservative, as expected)
- IG of noise features: median -0.0008, max +0.0090 nats/eq

| k | kind | IG | T | p_T | p_shift | p_binperm |
|---|---|---|---|---|---|---|
| 0 | rw_global | -0.0029 | -1.19 | 0.881 | 0.577 | nan |
| 1 | rw_global | -0.0048 | -2.09 | 0.980 | 0.721 | nan |
| 2 | rw_global | -0.0448 | -1.73 | 0.956 | 0.955 | nan |
| 3 | rw_global | -0.0004 | -0.13 | 0.552 | 0.303 | nan |
| 4 | rw_global | -0.0005 | 0.01 | 0.497 | 0.403 | nan |
| 5 | rw_global | -0.0001 | -0.11 | 0.545 | 0.194 | nan |
| 6 | rw_global | +0.0009 | 0.51 | 0.306 | 0.119 | nan |
| 7 | wn_global | +0.0031 | 1.22 | 0.113 | 0.124 | nan |
| 8 | wn_global | -0.0030 | -0.33 | 0.628 | 0.607 | nan |
| 9 | wn_global | -0.0125 | -0.81 | 0.789 | 0.915 | nan |
| 10 | wn_global | +0.0004 | 0.06 | 0.475 | 0.313 | nan |
| 11 | wn_global | +0.0003 | 0.25 | 0.403 | 0.383 | nan |
| 12 | wn_global | -0.0001 | -0.37 | 0.644 | 0.418 | nan |
| 13 | wn_global | -0.0000 | -0.03 | 0.511 | 0.363 | nan |
| 14 | wn_bin | -0.0058 | -1.19 | 0.880 | 0.801 | 0.761 |
| 15 | wn_bin | +0.0090 | 1.52 | 0.066 | 0.050 | 0.020 |
| 16 | wn_bin | -0.0131 | -1.84 | 0.965 | 0.930 | 0.915 |
| 17 | rw_bin | -0.0142 | -2.80 | 0.997 | 0.980 | 0.930 |
| 18 | rw_bin | -0.0222 | -1.76 | 0.959 | 1.000 | 0.975 |
| 19 | rw_bin | -0.0010 | -0.24 | 0.596 | 0.617 | 0.488 |

## Pure-noise features, y_main, n_surr=100: 20 features

- circular-shift surrogate: false positives p<0.05 = 4/20 (20.0%), p<0.0036 = 0; mean p = 0.497 (uniform -> 0.5)
- bin-permutation surrogate (per-bin features): p<0.05 = 1/6
- naive Rhoades T-test (one-sided p<0.05): 4/20; |T|>1.96 in 7/20 (anti-conservative, as expected)
- IG of noise features: median -0.0020, max +0.0283 nats/eq

| k | kind | IG | T | p_T | p_shift | p_binperm |
|---|---|---|---|---|---|---|
| 0 | rw_global | -0.0166 | -3.79 | 1.000 | 0.851 | nan |
| 1 | rw_global | -0.0210 | -5.83 | 1.000 | 0.683 | nan |
| 2 | rw_global | -0.0190 | -0.74 | 0.768 | 0.851 | nan |
| 3 | rw_global | +0.0118 | 3.82 | 0.000 | 0.030 | nan |
| 4 | rw_global | +0.0283 | 2.88 | 0.003 | 0.010 | nan |
| 5 | rw_global | +0.0056 | 1.90 | 0.031 | 0.089 | nan |
| 6 | rw_global | +0.0145 | 4.24 | 0.000 | 0.010 | nan |
| 7 | wn_global | -0.0096 | -0.95 | 0.828 | 0.762 | nan |
| 8 | wn_global | -0.0006 | -1.54 | 0.936 | 0.446 | nan |
| 9 | wn_global | -0.0095 | -0.49 | 0.688 | 0.832 | nan |
| 10 | wn_global | -0.0033 | -0.27 | 0.604 | 0.594 | nan |
| 11 | wn_global | +0.0011 | 0.64 | 0.261 | 0.297 | nan |
| 12 | wn_global | +0.0011 | 0.26 | 0.397 | 0.238 | nan |
| 13 | wn_global | -0.0006 | -0.18 | 0.573 | 0.386 | nan |
| 14 | wn_bin | -0.0040 | -0.97 | 0.833 | 0.663 | 0.634 |
| 15 | wn_bin | +0.0170 | 1.33 | 0.094 | 0.030 | 0.010 |
| 16 | wn_bin | -0.0199 | -1.84 | 0.965 | 0.980 | 0.941 |
| 17 | rw_bin | -0.0075 | -3.09 | 0.999 | 0.812 | 0.792 |
| 18 | rw_bin | -0.0383 | -2.08 | 0.979 | 1.000 | 0.970 |
| 19 | rw_bin | -0.0001 | -0.08 | 0.533 | 0.376 | 0.366 |
