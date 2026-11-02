#!/usr/bin/env python3
"""
Generates publication-quality scientific figures (300 DPI) for the Chilean Earthquake
Forecasting Study and copies them to both results/figures/ and the artifact directory.
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
ARTIFACT_DIR = "/Users/cnavarreteliz/.gemini/antigravity/brain/28396880-7a37-4637-adee-69acb26da68a"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def save_and_copy(fig, filename: str):
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    art_path = os.path.join(ARTIFACT_DIR, filename)
    shutil.copy2(out_path, art_path)
    print(f"Saved {out_path} and copied to {art_path}")


def plot_fig1_tectonic_and_catalog(df_raw: pd.DataFrame, df_feat: pd.DataFrame, live_df: pd.DataFrame):
    fig = plt.figure(figsize=(15, 9.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.15, 1.0, 1.15], height_ratios=[1.1, 1.0], wspace=0.28, hspace=0.32)

    # Panel A (Left, spanning 2 rows): Map of Chilean Subduction Zone Seismicity & Trench
    ax_map = fig.add_subplot(gs[:, 0])
    sub = df_feat.sample(min(12000, len(df_feat)), random_state=42)
    regime_colors = {
        0: ("#00a8cc", "Outer-Rise / Trench"),
        1: ("#e63946", "Megathrust Interface (0–55 km)"),
        2: ("#f4a261", "Shallow Andean Crustal"),
        3: ("#457b9d", "Intermediate Intraslab (55–180 km)"),
        4: ("#1d3557", "Deep Slab (>180 km)"),
    }
    for reg_id, (col, label) in regime_colors.items():
        m = sub["regime"] == reg_id
        if np.any(m):
            ax_map.scatter(
                sub.loc[m, "longitude"], sub.loc[m, "latitude"],
                s=np.clip((sub.loc[m, "mag"] - 3.5) ** 2.2 * 2.5, 2, 60),
                c=col, alpha=0.45, edgecolors="none", label=label
            )

    # Plot Peru-Chile Trench axis
    lat_prof = np.linspace(-55.0, -17.0, 100)
    from src.physics_features import TRENCH_PROFILE
    prof = TRENCH_PROFILE[np.argsort(TRENCH_PROFILE[:, 0])]
    trench_lon = np.interp(lat_prof, prof[:, 0], prof[:, 1])
    ax_map.plot(trench_lon, lat_prof, color="#111111", linestyle="--", linewidth=2.0, label="Peru–Chile Trench Axis")

    # Highlight M >= 7.8 historical megathrust mainshocks
    giants = df_raw[df_raw["mag"] >= 7.8].sort_values("mag", ascending=False)
    ax_map.scatter(
        giants["longitude"], giants["latitude"],
        s=(giants["mag"] - 6.5) ** 3.2 * 18, c="#ffbe0b", edgecolors="#000000",
        marker="*", linewidth=1.0, zorder=10, label="Historical Megathrust (M ≥ 7.8)"
    )
    for _, row in giants.head(8).iterrows():
        yr = str(row["time"])[:4]
        ax_map.annotate(
            f"{yr} (M{row['mag']:.1f})",
            xy=(row["longitude"], row["latitude"]),
            xytext=(row["longitude"] + 0.8, row["latitude"] + 0.3),
            fontsize=7.5, fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.15", fc="#fff3b0", ec="#333333", alpha=0.85),
        )

    ax_map.set_xlim(-77.2, -66.0)
    ax_map.set_ylim(-56.0, -17.0)
    ax_map.set_xlabel("Longitude (°W)")
    ax_map.set_ylabel("Latitude (°S)")
    ax_map.set_title("A. 3D Subduction Seismicity & Megathrust Ruptures (1900–2026)")
    ax_map.grid(True, linestyle=":", alpha=0.4)
    ax_map.legend(loc="lower left", framealpha=0.92, fontsize=7.8)

    # Panel B (Top Middle): Along-Strike Geodetic Locking & Slip Deficit Profile
    ax_lock = fig.add_subplot(gs[0, 1])
    lat_bins = np.linspace(-46.0, -17.5, 40)
    df_recent = df_feat[df_feat["year_frac"] >= 2020.0]
    bin_idx = np.digitize(df_recent["latitude"], lat_bins)
    lat_centers, lock_vals, slip_vals, bpos_vals = [], [], [], []
    for b in range(1, len(lat_bins)):
        m = bin_idx == b
        if np.sum(m) >= 5:
            lat_centers.append(0.5 * (lat_bins[b - 1] + lat_bins[b]))
            lock_vals.append(df_recent.loc[m, "locking_chi"].mean())
            slip_vals.append(df_recent.loc[m, "tectonic_slip_deficit_m"].mean())
            bpos_vals.append(df_recent.loc[m, "b_pos_100"].mean())

    ax_lock.plot(lock_vals, lat_centers, color="#d62828", linewidth=2.2, label="Locking Degree χ (Métois et al.)")
    ax_lock.fill_betweenx(lat_centers, 0, lock_vals, color="#d62828", alpha=0.18)
    ax_lock_top = ax_lock.twiny()
    ax_lock_top.plot(slip_vals, lat_centers, color="#1d3557", linewidth=2.0, linestyle="-.", label="Slip Deficit (m)")
    ax_lock.set_ylim(-46.5, -17.0)
    ax_lock.set_xlim(0.2, 1.0)
    ax_lock_top.set_xlim(0.0, 7.5)
    ax_lock.set_xlabel("Interseismic Locking χ [0–1]", color="#d62828")
    ax_lock_top.set_xlabel("Accumulated Slip Deficit (m)", color="#1d3557")
    ax_lock.set_ylabel("Latitude (°S)")
    ax_lock.set_title("B. Geodetic Locking & Elastic Slip Deficit", pad=24)
    ax_lock.grid(True, linestyle=":", alpha=0.4)

    # Highlight key seismic gaps on Panel B
    for gap_name, gap_lat in [("1868/1877 North Gap", -19.0), ("1922 Atacama Gap", -27.0), ("1730/1985 Central Gap", -33.2), ("1960 Valdivia Segment", -41.0)]:
        ax_lock.axhline(gap_lat, color="#555555", linestyle=":", linewidth=0.9)
        ax_lock.text(0.23, gap_lat + 0.4, gap_name, fontsize=7.5, fontweight="bold", color="#222222")

    # Panel C (Top Right): Catalog Non-Stationarity vs. Mc>=4.0 Homogeneity & b-Positive Stability
    ax_cat = fig.add_subplot(gs[0, 2])
    df_raw["year"] = pd.to_datetime(df_raw["time"], utc=True, format="ISO8601").dt.year
    yr_all = df_raw[df_raw["year"] >= 1975].groupby("year").size()
    yr_m40 = df_raw[(df_raw["year"] >= 1975) & (df_raw["mag"] >= 4.0)].groupby("year").size()
    ax_cat.plot(yr_all.index, yr_all.values, color="#8d99ae", linewidth=1.6, linestyle="--", label="Raw Catalog (M ≥ 2.0, Reporting Shift)")
    ax_cat.plot(yr_m40.index, yr_m40.values, color="#2b2d42", linewidth=2.2, label="Homogenized Catalog (M ≥ 4.0)")
    ax_cat.axvspan(1995, 2008, color="#ffb703", alpha=0.2, label="1995–2008 Microseismicity Surge")
    ax_cat.set_xlabel("Year")
    ax_cat.set_ylabel("Annual Earthquake Count")
    ax_cat.set_title("C. Catalog Completeness & Reporting Non-Stationarity")
    ax_cat.legend(loc="upper left", fontsize=7.8)
    ax_cat.grid(True, linestyle=":", alpha=0.4)

    # Panel D (Bottom Middle & Right): Aki-Utsu vs. van der Elst b-Positive Estimator under STAI
    ax_b = fig.add_subplot(gs[1, 1:])
    df_b = df_feat[(df_feat["year_frac"] >= 2009.5) & (df_feat["year_frac"] <= 2016.5)].copy()
    df_b_roll = df_b.set_index("time")[["b_aki_50", "b_pos_50"]].rolling("30D").mean()
    ax_b.plot(df_b_roll.index, df_b_roll["b_aki_50"], color="#e76f51", linewidth=1.3, alpha=0.85, label="Classical Aki–Utsu MLE b-value (Biased by STAI)")
    ax_b.plot(df_b_roll.index, df_b_roll["b_pos_50"], color="#2a9d8f", linewidth=1.9, label="Laplace b-Positive (b+) Estimator (van der Elst 2021, STAI-Invariant)")
    for dt_str, lbl in [("2010-02-27", "2010 Maule M8.8"), ("2014-04-01", "2014 Iquique M8.2"), ("2015-09-16", "2015 Illapel M8.3")]:
        ts = pd.Timestamp(dt_str, tz="UTC")
        ax_b.axvline(ts, color="#d62828", linestyle="--", linewidth=1.5)
        ax_b.text(ts, 1.48, f"  {lbl}", color="#d62828", fontweight="bold", fontsize=8.5)
    ax_b.set_ylim(0.55, 1.58)
    ax_b.set_ylabel("Estimated b-value")
    ax_b.set_xlabel("Date (2009–2016 Megathrust Sequence Era)")
    ax_b.set_title("D. Differential Stress Tracking: Classical b-value vs. Bias-Free b-Positive (b+) Estimator")
    ax_b.legend(loc="lower right", fontsize=8.5)
    ax_b.grid(True, linestyle=":", alpha=0.4)

    save_and_copy(fig, "fig1_chile_tectonic_map_and_locking.png")


def plot_fig2_csep_benchmarks(bm: dict, preds_data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    plt.subplots_adjust(wspace=0.25, hspace=0.32)

    model_styles = {
        "M1: Poisson-GR (CSEP Baseline)": ("#8d99ae", ":", 1.6),
        "M2: Classical-ETAS (Ogata 1998)": ("#457b9d", "--", 2.0),
        "M3: Reyes-Panakkat-ML (b-value+RF)": ("#e9c46a", "-.", 1.8),
        "M4: Neural-Point-Process (RECAST)": ("#f4a261", "-.", 1.8),
        "M5: Anisotropic-Slab-ETAS (3D Slab2)": ("#2a9d8f", "-", 2.1),
        "M6: PI-ST-Fusion (Ours)": ("#d62828", "-", 2.8),
    }

    # Panel A: Molchan Error Diagram on Task 3 (M >= 5.5 in 7d) / Task 1 (M >= 5.5 in 14d)
    ax_mol = axes[0, 0]
    t1 = preds_data["Task3_M55_7d"]
    y_te = np.array(t1["y_te"])
    from src.models_and_evaluation import compute_molchan_curve_and_ass
    ax_mol.plot([0, 1], [1, 0], color="#333333", linestyle="--", linewidth=1.2, label="Random Guess (ASS = 0.500)")
    for m_name, (col, ls, lw) in model_styles.items():
        if m_name in t1["preds_te"]:
            p_arr = np.array(t1["preds_te"][m_name])
            tau, nu, ass, _, _ = compute_molchan_curve_and_ass(y_te, p_arr)
            short_n = m_name.split(" (")[0]
            ax_mol.plot(tau, nu, color=col, linestyle=ls, linewidth=lw, label=f"{short_n} (ASS = {ass:.3f})")
    ax_mol.set_xlim(0, 1)
    ax_mol.set_ylim(0, 1)
    ax_mol.set_xlabel("Space-Time Alarm Fraction (τ)")
    ax_mol.set_ylabel("Miss Rate / Unpredicted Fraction (ν)")
    ax_mol.set_title("A. CSEP Molchan Error Diagram (M ≥ 5.5, Blind Test 2018–2026)")
    ax_mol.legend(loc="upper right", fontsize=8.0)
    ax_mol.grid(True, linestyle=":", alpha=0.4)

    # Panel B: ROC Curves on Pre-Mainshock Nucleation Regime (No M>=5.5 in past 7d)
    ax_roc = axes[0, 1]
    pre_mask = np.array(t1["is_pre_mainshock"], dtype=bool)
    y_pre = y_te[pre_mask]
    ax_roc.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1.2)
    for m_name, (col, ls, lw) in model_styles.items():
        if m_name in t1["preds_te"]:
            p_arr = np.array(t1["preds_te"][m_name])[pre_mask]
            fpr, tpr, _ = roc_curve(y_pre, p_arr)
            auc_val = roc_auc_score(y_pre, p_arr)
            short_n = m_name.split(" (")[0]
            ax_roc.plot(fpr, tpr, color=col, linestyle=ls, linewidth=lw, label=f"{short_n} (AUC = {auc_val:.3f})")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (Hit Rate)")
    ax_roc.set_title("B. Pre-Mainshock Nucleation ROC (Declustered Regime 2018–2026)")
    ax_roc.legend(loc="lower right", fontsize=8.0)
    ax_roc.grid(True, linestyle=":", alpha=0.4)

    # Panel C: Multi-Task ROC-AUC Comparison Across All 6 Tasks
    ax_bar = axes[1, 0]
    task_labels = [
        "T1: M≥5.5 (14d)\nFull Catalog",
        "T1: M≥5.5 (14d)\nPre-Mainshock",
        "T2: M≥6.0 (14d)\nFull Catalog",
        "T3: M≥5.5 (7d)\nFull Catalog",
        "T3: M≥5.5 (7d)\nPre-Mainshock",
        "T5: Foreshock\nEscalation",
    ]
    t_specs = [
        ("Task1_M55_14d", "full_test_catalog"),
        ("Task1_M55_14d", "pre_mainshock_nucleation_test"),
        ("Task2_M60_14d", "full_test_catalog"),
        ("Task3_M55_7d", "full_test_catalog"),
        ("Task3_M55_7d", "pre_mainshock_nucleation_test"),
        ("Task5_Foreshock_Escalation_14d", "pre_mainshock_nucleation_test"),
    ]
    x = np.arange(len(task_labels))
    width = 0.13
    for idx, (m_name, (col, _, _)) in enumerate(model_styles.items()):
        vals = [bm[tk][mode][m_name]["roc_auc"] for tk, mode in t_specs]
        short_n = m_name.split(" (")[0]
        ax_bar.bar(x + (idx - 2.5) * width, vals, width, label=short_n, color=col, edgecolor="#222222", linewidth=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(task_labels, fontsize=8.0)
    ax_bar.set_ylim(0.45, 0.73)
    ax_bar.axhline(0.50, color="#333333", linestyle=":", linewidth=1.0)
    ax_bar.set_ylabel("Out-of-Sample ROC-AUC (2018–2026)")
    ax_bar.set_title("C. Benchmark Performance Across All Prediction Tasks")
    ax_bar.legend(loc="upper left", ncol=2, fontsize=7.5)
    ax_bar.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Panel D: CSEP Information Gain per Earthquake (IG_PE in nats/event over Poisson & ETAS)
    ax_ig = axes[1, 1]
    models_comp = [
        "M2: Classical-ETAS (Ogata 1998)",
        "M3: Reyes-Panakkat-ML (b-value+RF)",
        "M4: Neural-Point-Process (RECAST)",
        "M5: Anisotropic-Slab-ETAS (3D Slab2)",
        "M6: PI-ST-Fusion (Ours)",
    ]
    ig_t1_pre = [bm["Task1_M55_14d"]["pre_mainshock_nucleation_test"][m]["ig_pe_vs_poisson_nats"] for m in models_comp]
    ig_t3_pre = [bm["Task3_M55_7d"]["pre_mainshock_nucleation_test"][m]["ig_pe_vs_poisson_nats"] for m in models_comp]
    y_pos = np.arange(len(models_comp))
    h = 0.35
    ax_ig.barh(y_pos - h/2, ig_t1_pre, height=h, color="#457b9d", edgecolor="#222222", label="Task 1: M≥5.5 (14d Pre-Mainshock)")
    ax_ig.barh(y_pos + h/2, ig_t3_pre, height=h, color="#d62828", edgecolor="#222222", label="Task 3: M≥5.5 (7d Pre-Mainshock)")
    ax_ig.set_yticks(y_pos)
    ax_ig.set_yticklabels([m.split(" (")[0] for m in models_comp], fontsize=8.5)
    ax_ig.axvline(0.0, color="#222222", linewidth=1.0)
    ax_ig.set_xlabel("CSEP Information Gain per Target Earthquake (nats / event)")
    ax_ig.set_title("D. CSEP Information Gain Over Poisson Baseline (Pre-Mainshock)")
    ax_ig.legend(loc="lower right", fontsize=8.0)
    ax_ig.grid(True, axis="x", linestyle=":", alpha=0.4)

    save_and_copy(fig, "fig2_csep_benchmark_molchan_roc_pr.png")


def plot_fig3_case_studies(df_pred: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    cases = [
        (
            "A. 2014 Iquique Megathrust (Mw 8.2, Apr 1, 2014)",
            "2014-01-15", "2014-04-10", "2014-04-01T23:46:47Z",
            -21.8, -18.5, 8.2
        ),
        (
            "B. 2015 Illapel Megathrust (Mw 8.3, Sep 16, 2015)",
            "2015-07-01", "2015-09-25", "2015-09-16T22:54:32Z",
            -32.8, -29.5, 8.3
        ),
        (
            "C. 2017 Valparaíso Sequence (Mw 6.9, Apr 24, 2017)",
            "2017-03-01", "2017-05-02", "2017-04-24T21:38:30Z",
            -34.2, -32.0, 6.9
        ),
        (
            "D. 2024 San Pedro de Atacama / Calama (Mw 7.4, Jul 19, 2024 - Blind Test)",
            "2024-04-15", "2024-07-26", "2024-07-19T01:50:48Z",
            -25.0, -21.5, 7.4
        ),
    ]

    for idx, (title, t_start, t_end, t_main, lat_min, lat_max, m_main) in enumerate(cases):
        ax = axes[idx // 2, idx % 2]
        sub = df_pred[
            (df_pred["time"] >= pd.Timestamp(t_start, tz="UTC"))
            & (df_pred["time"] <= pd.Timestamp(t_end, tz="UTC"))
            & (df_pred["latitude"] >= lat_min)
            & (df_pred["latitude"] <= lat_max)
        ].copy()

        t_rup = pd.Timestamp(t_main)
        if len(sub) > 0:
            ax.plot(
                sub["time"], sub["pred_m55_14d_Poisson"] * 100.0,
                color="#8d99ae", linestyle=":", linewidth=1.4, label="M1: Poisson-GR Baseline"
            )
            ax.plot(
                sub["time"], sub["pred_m55_14d_ETAS"] * 100.0,
                color="#457b9d", linestyle="--", linewidth=1.7, label="M2: Classical-ETAS"
            )
            ax.plot(
                sub["time"], sub["pred_m55_14d_PI_ST_Fusion"] * 100.0,
                color="#d62828", linestyle="-", linewidth=2.3, label="M6: PI-ST-Fusion (Ours)"
            )

            # Plot observed earthquakes on secondary axis
            ax2 = ax.twinx()
            pre_ev = sub[sub["time"] < t_rup]
            ax2.scatter(
                pre_ev["time"], pre_ev["mag"],
                c="#2a9d8f", s=(pre_ev["mag"] - 3.6) ** 2.5 * 14, alpha=0.65,
                edgecolors="#1d3557", linewidth=0.5, label="Precursory / Foreshock Events"
            )
            ax2.scatter(
                [t_rup], [m_main], c="#ffbe0b", edgecolors="#000000",
                marker="*", s=260, zorder=15, label=f"Mainshock (Mw {m_main})"
            )
            ax2.set_ylim(3.8, 9.0)
            ax2.set_ylabel("Magnitude (Mw)", color="#2a9d8f")

        ax.axvline(t_rup, color="#d62828", linestyle="-.", linewidth=1.8)
        ax.set_ylabel("14-Day Hazard Probability P(M ≥ 5.5) [%]", color="#d62828")
        ax.set_title(title, fontsize=10.2)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(axis="x", rotation=20)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7.8)

    save_and_copy(fig, "fig3_precursory_case_studies_chile.png")


def plot_fig4_attribution_and_live_forecast(fi_df: pd.DataFrame, live_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.05, 1.15]})
    plt.subplots_adjust(wspace=0.32)

    # Panel A: Top 15 Physical Features in PI-ST-Fusion
    ax_fi = axes[0]
    top_fi = fi_df.head(15).iloc[::-1]
    pretty_names = {
        "etas_aniso_intensity": "Anisotropic Trench-ETAS Intensity",
        "etas_med_7d": "Medium-Term (7d) Subduction ETAS",
        "cumulative_moment_release_10yr": "10-Yr Cumulative Seismic Moment",
        "background_rate_mu_local": "Local Background Rate μ(x)",
        "delta_b_pos_25_100": "Differential Stress Drop Δb+ (25 vs 100)",
        "frac_interface_30d": "Megathrust Interface Fraction (0–55 km)",
        "tectonic_slip_deficit_m": "Locked Tectonic Slip Deficit (m)",
        "elastic_strain_index": "Interseismic Elastic Strain Index (χ·Δu)",
        "strike_rad": "Peru–Chile Trench Curvature / Strike",
        "b_pos_25": "Short-Term Laplace b-Positive (b+)",
        "mag": "Current Event Magnitude (Mw)",
        "z_rel_km": "Slab2 Relative Interface Depth (z - z_slab)",
        "benioff_accel_7d_30d": "Benioff Strain Acceleration (7d/30d)",
        "time_since_last_m60_local": "Recurrence Time Since Last M≥6.0",
        "zaliapin_log_eta": "Zaliapin–Ben-Zion Proximity (log η_ij)",
        "etas_interface_intensity": "Interface-Specific ETAS Intensity",
        "rate_ratio_7d_90d": "Seismic Rate Ratio (7d / 90d)",
    }
    labels = [pretty_names.get(f, f) for f in top_fi["feature"]]
    vals = top_fi["importance_full"].values / np.max(top_fi["importance_full"].values) * 100.0
    bars = ax_fi.barh(range(len(labels)), vals, color="#2a9d8f", edgecolor="#1d3557", height=0.68)
    for b_idx, f_key in enumerate(top_fi["feature"]):
        if "b_pos" in f_key or "slip_deficit" in f_key or "elastic" in f_key or "strike" in f_key or "aniso" in f_key:
            bars[b_idx].set_color("#e63946")
    ax_fi.set_yticks(range(len(labels)))
    ax_fi.set_yticklabels(labels, fontsize=8.5)
    ax_fi.set_xlabel("Normalized Physical Attribution Importance (%)")
    ax_fi.set_title("A. Physical Feature Attribution in PI-ST-Fusion\n(Red = Novel Subduction & b+ Stress Contributions)")
    ax_fi.grid(True, axis="x", linestyle=":", alpha=0.4)

    # Panel B: Live Operational Forecast for Chile (Oct - Nov 2026)
    ax_live = axes[1]
    zones = [z.split(" (")[0] for z in live_df["zone"]][::-1]
    p50 = live_df["prob_m50_30d"].values[::-1] * 100.0
    p55 = live_df["prob_m55_30d"].values[::-1] * 100.0
    p60 = live_df["prob_m60_30d"].values[::-1] * 100.0
    mmax = live_df["expected_mmax_30d"].values[::-1]

    y = np.arange(len(zones))
    h = 0.26
    ax_live.barh(y + h, p50, height=h, color="#457b9d", edgecolor="#1d3557", label="30-Day P(M ≥ 5.0)")
    ax_live.barh(y, p55, height=h, color="#f4a261", edgecolor="#1d3557", label="30-Day P(M ≥ 5.5)")
    ax_live.barh(y - h, p60, height=h, color="#d62828", edgecolor="#1d3557", label="30-Day P(M ≥ 6.0)")

    for i_z in range(len(zones)):
        ax_live.text(
            p50[i_z] + 0.8, y[i_z],
            f"E[Mmax]={mmax[i_z]:.2f} | P(≥5.5)={p55[i_z]:.1f}%",
            va="center", fontsize=8.0, fontweight="bold", color="#1d3557"
        )

    ax_live.set_yticks(y)
    ax_live.set_yticklabels(zones, fontsize=8.8, fontweight="bold")
    ax_live.set_xlim(0, max(p50) * 1.48)
    ax_live.set_xlabel("30-Day Earthquake Probability (%)")
    ax_live.set_title("B. Prospective Operational Seismic Forecast for Chile\n(30-Day Window Starting Oct 17, 2026)")
    ax_live.legend(loc="lower right", fontsize=8.5)
    ax_live.grid(True, axis="x", linestyle=":", alpha=0.4)

    save_and_copy(fig, "fig4_feature_attribution_and_live_forecast_2026.png")


def main():
    raw_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "chile_usgs_1900_2026.csv"))
    feat_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "chile_features_m40.csv"))
    feat_df["time"] = pd.to_datetime(feat_df["time"], utc=True, format="ISO8601")

    pred_df = pd.read_csv(os.path.join(BASE_DIR, "results", "catalog_with_predictions.csv"))
    pred_df["time"] = pd.to_datetime(pred_df["time"], utc=True, format="ISO8601")

    fi_df = pd.read_csv(os.path.join(BASE_DIR, "results", "feature_importance_full.csv"))
    live_df = pd.read_csv(os.path.join(BASE_DIR, "results", "live_chile_forecast_oct2026.csv"))

    with open(os.path.join(BASE_DIR, "results", "benchmark_metrics.json")) as f:
        bm = json.load(f)
    with open(os.path.join(BASE_DIR, "results", "test_predictions.json")) as f:
        preds_data = json.load(f)

    plot_fig1_tectonic_and_catalog(raw_df, feat_df, live_df)
    plot_fig2_csep_benchmarks(bm, preds_data)
    plot_fig3_case_studies(pred_df)
    plot_fig4_attribution_and_live_forecast(fi_df, live_df)


if __name__ == "__main__":
    main()
