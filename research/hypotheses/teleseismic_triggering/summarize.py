"""Compile results/*.json into a markdown table (stdout)."""
import json, glob
rows = []
for fn in sorted(glob.glob("/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering/results/h*.json")):
    d = json.load(open(fn)); r = d["evaluate"]
    ps = d.get("sur_event_shift", {}); ph = d.get("sur_phase", {}); bp = d.get("sur_bin_permute", {})
    bb = d["bootstrap_full_vs_base"]
    rows.append(f"| {d['hyp']} | {d['target']} | s{d['step']}/h{d['horizon']} M{d['mag']} {int(d['dist_km'])}km {d['model']} | "
                f"{r['ig_per_event']:+.4f} [{bb['ci_lo']:+.3f},{bb['ci_hi']:+.3f}] | {r['ig_full_vs_etas']:+.3f} | "
                f"{ps.get('p_value', float('nan')):.3f} (n={ps.get('n','-')}) | {ph.get('p_value', float('nan')):.3f} | {bp.get('p_value', float('nan')):.3f} | "
                f"{r['base']['roc_auc']:.3f}/{r['full']['roc_auc']:.3f} | {r['base']['molchan_ass']:.3f}/{r['full']['molchan_ass']:.3f} | {r['n_test_events']} | "
                f"{r.get('coef_extra', r.get('importance_extra'))} |")
print("| hyp | target | panel | IG full-base [year CI] | IG full-vs-ETAS | p event-shift | p phase | p bin-perm | AUC base/full | ASS base/full | N test | coef |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
