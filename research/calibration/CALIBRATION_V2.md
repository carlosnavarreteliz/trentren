# Harness calibration: pure-noise false-positive rates

**Status: final (all runs finished).** Rows loaded: 180.

Feature kinds: rw = random walk, ar98 = AR(1) phi=0.98, wn = white noise; _global = one series broadcast to all bins, _bin = independent per bin.
Expected under a calibrated test: FPR ~ 0.05 and mean p ~ 0.5. Binomial 95% range for n=20: 0-3 hits; n=40: 0-5 hits.

## By mode x target

|                                         |   n |   fp_0.05 |   FPR_0.05 |   fp_0.0036 |   mean_p |
|:----------------------------------------|----:|----------:|-----------:|------------:|---------:|
| ('bin_permute', 'y_all')                |  22 |         3 |      0.136 |           0 |    0.573 |
| ('bin_permute', 'y_main')               |  16 |         1 |      0.062 |           0 |    0.555 |
| ('circular_shift_endmatched', 'y_all')  |  20 |         0 |      0     |           0 |    0.475 |
| ('circular_shift_endmatched', 'y_main') |  20 |         0 |      0     |           0 |    0.5   |
| ('circular_shift_plain', 'y_all')       |  20 |         1 |      0.05  |           0 |    0.539 |
| ('circular_shift_plain', 'y_main')      |  20 |         4 |      0.2   |           0 |    0.497 |
| ('phase_randomize', 'y_all')            |  60 |         5 |      0.083 |           0 |    0.444 |
| ('phase_randomize', 'y_main')           |  60 |         0 |      0     |           0 |    0.478 |
| ('rhoades_T_naive', 'y_all')            |  40 |         0 |      0     |           0 |    0.637 |
| ('rhoades_T_naive', 'y_main')           |  20 |         4 |      0.2   |           3 |    0.575 |

## By mode x target x slow/fast features (slow = rw_*, ar98_*)

|                                               |   n |   fp_0.05 |   FPR_0.05 |   fp_0.0036 |   mean_p |
|:----------------------------------------------|----:|----------:|-----------:|------------:|---------:|
| ('bin_permute', 'y_all', False)               |  11 |         2 |      0.182 |           0 |    0.542 |
| ('bin_permute', 'y_all', True)                |  11 |         1 |      0.091 |           0 |    0.604 |
| ('bin_permute', 'y_main', False)              |   8 |         1 |      0.125 |           0 |    0.608 |
| ('bin_permute', 'y_main', True)               |   8 |         0 |      0     |           0 |    0.502 |
| ('circular_shift_endmatched', 'y_all', False) |  10 |         0 |      0     |           0 |    0.483 |
| ('circular_shift_endmatched', 'y_all', True)  |  10 |         0 |      0     |           0 |    0.467 |
| ('circular_shift_endmatched', 'y_main', True) |  20 |         0 |      0     |           0 |    0.5   |
| ('circular_shift_plain', 'y_all', False)      |  10 |         1 |      0.1   |           0 |    0.491 |
| ('circular_shift_plain', 'y_all', True)       |  10 |         0 |      0     |           0 |    0.587 |
| ('circular_shift_plain', 'y_main', False)     |  10 |         1 |      0.1   |           0 |    0.523 |
| ('circular_shift_plain', 'y_main', True)      |  10 |         3 |      0.3   |           0 |    0.471 |
| ('phase_randomize', 'y_all', False)           |  25 |         2 |      0.08  |           0 |    0.49  |
| ('phase_randomize', 'y_all', True)            |  35 |         3 |      0.086 |           0 |    0.411 |
| ('phase_randomize', 'y_main', False)          |  25 |         0 |      0     |           0 |    0.521 |
| ('phase_randomize', 'y_main', True)           |  35 |         0 |      0     |           0 |    0.447 |
| ('rhoades_T_naive', 'y_all', False)           |  20 |         0 |      0     |           0 |    0.548 |
| ('rhoades_T_naive', 'y_all', True)            |  20 |         0 |      0     |           0 |    0.727 |
| ('rhoades_T_naive', 'y_main', False)          |  10 |         0 |      0     |           0 |    0.618 |
| ('rhoades_T_naive', 'y_main', True)           |  10 |         4 |      0.4   |           3 |    0.531 |

## By mode x target x kind

|                                                      |   n |   fp_0.05 |   FPR_0.05 |   fp_0.0036 |   mean_p |
|:-----------------------------------------------------|----:|----------:|-----------:|------------:|---------:|
| ('bin_permute', 'y_all', 'rw_bin')                   |  11 |         1 |      0.091 |           0 |    0.604 |
| ('bin_permute', 'y_all', 'wn_bin')                   |  11 |         2 |      0.182 |           0 |    0.542 |
| ('bin_permute', 'y_main', 'rw_bin')                  |   8 |         0 |      0     |           0 |    0.502 |
| ('bin_permute', 'y_main', 'wn_bin')                  |   8 |         1 |      0.125 |           0 |    0.608 |
| ('circular_shift_endmatched', 'y_all', 'rw_bin')     |   3 |         0 |      0     |           0 |    0.601 |
| ('circular_shift_endmatched', 'y_all', 'rw_global')  |   7 |         0 |      0     |           0 |    0.41  |
| ('circular_shift_endmatched', 'y_all', 'wn_bin')     |   3 |         0 |      0     |           0 |    0.614 |
| ('circular_shift_endmatched', 'y_all', 'wn_global')  |   7 |         0 |      0     |           0 |    0.427 |
| ('circular_shift_endmatched', 'y_main', 'rw_global') |  20 |         0 |      0     |           0 |    0.5   |
| ('circular_shift_plain', 'y_all', 'rw_bin')          |   3 |         0 |      0     |           0 |    0.866 |
| ('circular_shift_plain', 'y_all', 'rw_global')       |   7 |         0 |      0     |           0 |    0.468 |
| ('circular_shift_plain', 'y_all', 'wn_bin')          |   3 |         1 |      0.333 |           0 |    0.594 |
| ('circular_shift_plain', 'y_all', 'wn_global')       |   7 |         0 |      0     |           0 |    0.446 |
| ('circular_shift_plain', 'y_main', 'rw_bin')         |   3 |         0 |      0     |           0 |    0.729 |
| ('circular_shift_plain', 'y_main', 'rw_global')      |   7 |         3 |      0.429 |           0 |    0.361 |
| ('circular_shift_plain', 'y_main', 'wn_bin')         |   3 |         1 |      0.333 |           0 |    0.558 |
| ('circular_shift_plain', 'y_main', 'wn_global')      |   7 |         0 |      0     |           0 |    0.508 |
| ('phase_randomize', 'y_all', 'ar98_global')          |  10 |         0 |      0     |           0 |    0.402 |
| ('phase_randomize', 'y_all', 'rw_bin')               |   5 |         1 |      0.2   |           0 |    0.253 |
| ('phase_randomize', 'y_all', 'rw_global')            |  20 |         2 |      0.1   |           0 |    0.455 |
| ('phase_randomize', 'y_all', 'wn_bin')               |   5 |         0 |      0     |           0 |    0.503 |
| ('phase_randomize', 'y_all', 'wn_global')            |  20 |         2 |      0.1   |           0 |    0.486 |
| ('phase_randomize', 'y_main', 'ar98_global')         |  10 |         0 |      0     |           0 |    0.371 |
| ('phase_randomize', 'y_main', 'rw_bin')              |   5 |         0 |      0     |           0 |    0.3   |
| ('phase_randomize', 'y_main', 'rw_global')           |  20 |         0 |      0     |           0 |    0.522 |
| ('phase_randomize', 'y_main', 'wn_bin')              |   5 |         0 |      0     |           0 |    0.612 |
| ('phase_randomize', 'y_main', 'wn_global')           |  20 |         0 |      0     |           0 |    0.498 |
| ('rhoades_T_naive', 'y_all', 'rw_bin')               |   6 |         0 |      0     |           0 |    0.851 |
| ('rhoades_T_naive', 'y_all', 'rw_global')            |  14 |         0 |      0     |           0 |    0.674 |
| ('rhoades_T_naive', 'y_all', 'wn_bin')               |   6 |         0 |      0     |           0 |    0.637 |
| ('rhoades_T_naive', 'y_all', 'wn_global')            |  14 |         0 |      0     |           0 |    0.509 |
| ('rhoades_T_naive', 'y_main', 'rw_bin')              |   3 |         0 |      0     |           0 |    0.837 |
| ('rhoades_T_naive', 'y_main', 'rw_global')           |   7 |         4 |      0.571 |           3 |    0.4   |
| ('rhoades_T_naive', 'y_main', 'wn_bin')              |   3 |         0 |      0     |           0 |    0.631 |
| ('rhoades_T_naive', 'y_main', 'wn_global')           |   7 |         0 |      0     |           0 |    0.612 |
