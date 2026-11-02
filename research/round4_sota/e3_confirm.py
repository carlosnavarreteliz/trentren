"""E3 confirmacion: surrogate n=1000 para p estable."""
import sys
sys.path.insert(0, "research")
sys.path.insert(0, "research/round4_sota")
import harness as H
from e3_robust import build_X


def main():
    X = build_X()
    p50 = H.get_panel(step_days=1, horizon_days=1, target_mag=5.0)
    res = H.evaluate(X, target="y_all", model="logit", panel=p50)
    sur = H.surrogate_test(X, n=1000, mode="phase_randomize", target="y_all", panel=p50)
    print(f"confirmacion: {H.summarize(res, sur)}")


if __name__ == "__main__":
    sys.exit(main())
