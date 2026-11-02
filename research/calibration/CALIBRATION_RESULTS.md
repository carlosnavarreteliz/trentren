# Harness calibration: pure-noise false-positive rates

Feature kinds: rw = random walk, ar98 = AR(1) phi=0.98, wn = white noise; _global = one series broadcast to all bins, _bin = independent per bin.
Expected under a calibrated test: FPR ~ 0.05 and mean p ~ 0.5. Binomial 95% range for n=20: 0-3 hits; n=40: 0-5 hits.

## By mode x target

|                                    |   n |   fp_0.05 |   FPR_0.05 |   fp_0.0036 |   mean_p |
|:-----------------------------------|----:|----------:|-----------:|------------:|---------:|
| ('bin_permute', 'y_all')           |   6 |         1 |      0.167 |           0 |    0.682 |
| ('bin_permute', 'y_main')          |   6 |         1 |      0.167 |           0 |    0.619 |
| ('circular_shift_plain', 'y_all')  |  20 |         1 |      0.05  |           0 |    0.539 |
| ('circular_shift_plain', 'y_main') |  20 |         4 |      0.2   |           0 |    0.497 |
| ('rhoades_T_naive', 'y_all')       |  20 |         0 |      0     |           0 |    0.637 |
| ('rhoades_T_naive', 'y_main')      |  20 |         4 |      0.2   |           3 |    0.575 |

## By mode x target x slow/fast features (slow = rw_*, ar98_*)

|                                           |   n |   fp_0.05 |   FPR_0.05 |   fp_0.0036 |   mean_p |
|:------------------------------------------|----:|----------:|-----------:|------------:|---------:|
| ('bin_permute', 'y_all', False)           |   3 |         1 |      0.333 |           0 |    0.566 |
| ('bin_permute', 'y_all', True)            |   3 |         0 |      0     |           0 |    0.798 |
| ('bin_permute', 'y_main', False)          |   3 |         1 |      0.333 |           0 |    0.528 |
| ('bin_permute', 'y_main', True)           |   3 |         0 |      0     |           0 |    0.71  |
| ('circular_shift_plain', 'y_all', False)  |  10 |         1 |      0.1   |           0 |    0.491 |
| ('circular_shift_plain', 'y_all', True)   |  10 |         0 |      0     |           0 |    0.587 |
| ('circular_shift_plain', 'y_main', False) |  10 |         1 |      0.1   |           0 |    0.523 |
| ('circular_shift_plain', 'y_main', True)  |  10 |         3 |      0.3   |           0 |    0.471 |
| ('rhoades_T_naive', 'y_all', False)       |  10 |         0 |      0     |           0 |    0.548 |
| ('rhoades_T_naive', 'y_all', True)        |  10 |         0 |      0     |           0 |    0.727 |
| ('rhoades_T_naive', 'y_main', False)      |  10 |         0 |      0     |           0 |    0.618 |
| ('rhoades_T_naive', 'y_main', True)       |  10 |         4 |      0.4   |           3 |    0.531 |

## By mode x target x kind

|                                                 |   n |   fp_0.05 |   FPR_0.05 |   fp_0.0036 |   mean_p |
|:------------------------------------------------|----:|----------:|-----------:|------------:|---------:|
| ('bin_permute', 'y_all', 'rw_bin')              |   3 |         0 |      0     |           0 |    0.798 |
| ('bin_permute', 'y_all', 'wn_bin')              |   3 |         1 |      0.333 |           0 |    0.566 |
| ('bin_permute', 'y_main', 'rw_bin')             |   3 |         0 |      0     |           0 |    0.71  |
| ('bin_permute', 'y_main', 'wn_bin')             |   3 |         1 |      0.333 |           0 |    0.528 |
| ('circular_shift_plain', 'y_all', 'rw_bin')     |   3 |         0 |      0     |           0 |    0.866 |
| ('circular_shift_plain', 'y_all', 'rw_global')  |   7 |         0 |      0     |           0 |    0.468 |
| ('circular_shift_plain', 'y_all', 'wn_bin')     |   3 |         1 |      0.333 |           0 |    0.594 |
| ('circular_shift_plain', 'y_all', 'wn_global')  |   7 |         0 |      0     |           0 |    0.446 |
| ('circular_shift_plain', 'y_main', 'rw_bin')    |   3 |         0 |      0     |           0 |    0.729 |
| ('circular_shift_plain', 'y_main', 'rw_global') |   7 |         3 |      0.429 |           0 |    0.361 |
| ('circular_shift_plain', 'y_main', 'wn_bin')    |   3 |         1 |      0.333 |           0 |    0.558 |
| ('circular_shift_plain', 'y_main', 'wn_global') |   7 |         0 |      0     |           0 |    0.508 |
| ('rhoades_T_naive', 'y_all', 'rw_bin')          |   3 |         0 |      0     |           0 |    0.851 |
| ('rhoades_T_naive', 'y_all', 'rw_global')       |   7 |         0 |      0     |           0 |    0.674 |
| ('rhoades_T_naive', 'y_all', 'wn_bin')          |   3 |         0 |      0     |           0 |    0.637 |
| ('rhoades_T_naive', 'y_all', 'wn_global')       |   7 |         0 |      0     |           0 |    0.509 |
| ('rhoades_T_naive', 'y_main', 'rw_bin')         |   3 |         0 |      0     |           0 |    0.837 |
| ('rhoades_T_naive', 'y_main', 'rw_global')      |   7 |         4 |      0.571 |           3 |    0.4   |
| ('rhoades_T_naive', 'y_main', 'wn_bin')         |   3 |         0 |      0     |           0 |    0.631 |
| ('rhoades_T_naive', 'y_main', 'wn_global')      |   7 |         0 |      0     |           0 |    0.612 |
