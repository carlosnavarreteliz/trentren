#!/usr/bin/env python3
"""
Physics-Informed Spatiotemporal Feature Engineering for the Chilean Subduction Zone.

Implements strictly causal (zero lookahead leakage) physical and statistical seismology features:
1. Subduction Slab Geometry (Hayes et al., 2018 Slab2 approximation for Peru-Chile Trench):
   - Trench-perpendicular distance d_trench (km), along-strike distance s_along (km),
     local trench strike azimuth, slab interface depth z_slab, residual depth z_rel,
     and tectonic regime classification (Outer-Rise, Megathrust Interface, Intermediate
     Intraslab, Deep, Crustal).
2. Geodetic Interseismic Locking & Elastic Rebound Slip/Moment Deficit (Métois et al., 2016):
   - Local plate convergence rate v_conv (mm/yr), locking degree chi(lat), time since last
     major/giant rupture in segment, accumulated tectonic moment deficit since 1900,
     and Locking-Gradient Migration (migration of seismicity from transitional zones into
     locked asperities, Ruiz et al., 2014; Métois et al., 2016).
3. Anisotropic Multi-Scale Spatiotemporal ETAS Intensity & Omori Deviation Anomaly:
   - Decomposed into along-strike (L_parallel) and cross-strike (L_perp) triggering kernels,
     depth-specific (interface vs. intraslab) triggering contributions, and the Omori Decay
     Deviation Residual (ratio of short-term observed rate to expected Omori decay).
4. Bias-Free Differential Stress Proxy via b-Positive (b+) Estimator (van der Elst, 2021 JGR)
   & Foreshock Traffic Light System (Gulia & Wiemer, 2019 Nature) + Sub-Threshold Fusion (Xiong et al., 2026):
   - Laplace positive-difference b+ over N=25, 50, 100, 250 local events (both M>=4.0 homogenized
     and full M>=2.5 sub-threshold catalog), Aki-Utsu MLE b-value, STAI bias (b_Aki - b+),
     and temporal differential stress drop Delta_b+.
5. Cross-Depth Slab-Pull to Megathrust Stress Transfer (Bouchon et al., 2016 Nature Geoscience):
   - Ratio and acceleration of intermediate-depth intraslab seismicity (55-180 km) relative
     to shallow interplate seismicity (0-55 km), and down-dip / up-dip hypocentral migration.
6. Zaliapin-Ben-Zion Space-Time-Energy Nearest-Neighbor Proximity (Zaliapin & Ben-Zion, 2013):
   - Rescaled time T_ij, rescaled distance R_ij, proximity eta_ij = T_ij * R_ij,
     spatial Shannon entropy H_spatial, and Accelerating Moment Release (AMR / Benioff strain).
"""

import os
import json
import numpy as np
import pandas as pd

# Reference Peru-Chile Trench coordinates (Latitude -> Trench Longitude in deg, Slab Dip in deg, Locking chi)
# Derived from USGS Slab2 (Hayes et al., 2018) and Métois et al. (2016, GJI)
TRENCH_PROFILE = np.array([
    # lat, trench_lon, dip_deg, locking_chi, v_conv_mm_yr
    [-17.0, -71.8, 19.0, 0.55, 65.0],
    [-18.5, -71.6, 20.0, 0.68, 65.5],
    [-20.0, -71.3, 21.0, 0.80, 66.0],
    [-21.5, -71.3, 21.5, 0.72, 66.0],
    [-23.0, -71.4, 22.0, 0.75, 66.0],
    [-25.0, -71.5, 19.0, 0.70, 66.0],
    [-27.0, -71.8, 16.0, 0.78, 66.0],
    [-29.0, -72.1, 14.0, 0.62, 66.0],
    [-30.5, -72.4, 13.5, 0.70, 66.0],
    [-32.0, -72.6, 15.5, 0.78, 66.0],
    [-33.5, -72.8, 18.0, 0.88, 66.0],
    [-35.0, -73.4, 19.0, 0.84, 65.5],
    [-37.0, -74.0, 19.5, 0.82, 65.5],
    [-39.0, -74.6, 20.0, 0.85, 65.0],
    [-41.5, -75.0, 20.0, 0.80, 65.0],
    [-44.0, -75.4, 20.5, 0.75, 64.5],
    [-46.5, -75.8, 22.0, 0.50, 45.0],
    [-50.0, -76.2, 24.0, 0.35, 22.0],
    [-56.0, -75.5, 25.0, 0.30, 20.0],
])

# Historical giant megathrust ruptures (prior to modern instrumental catalog or major resets)
HISTORICAL_GIANTS = [
    # (year, lat_min, lat_max, mag)
    (1730.5, -34.5, -31.5, 9.1),
    (1835.1, -38.5, -35.0, 8.5),
    (1868.6, -19.5, -16.5, 8.8),
    (1877.4, -24.0, -19.5, 8.8),
    (1906.6, -34.5, -32.0, 8.2),
    (1922.9, -29.0, -25.5, 8.5),
    (1939.1, -37.5, -35.5, 8.3),
    (1943.3, -32.2, -30.0, 8.1),
    (1960.4, -46.5, -37.5, 9.5),
]


def interpolate_trench_properties(lats: np.ndarray, lons: np.ndarray, depths: np.ndarray):
    """
    Computes intrinsic subduction coordinates for each hypocenter.
    """
    prof = TRENCH_PROFILE[np.argsort(TRENCH_PROFILE[:, 0])]
    p_lat = prof[:, 0]
    p_lon = prof[:, 1]
    p_dip = prof[:, 2]
    p_chi = prof[:, 3]
    p_vel = prof[:, 4]

    trench_lon = np.interp(lats, p_lat, p_lon)
    dip_deg = np.interp(lats, p_lat, p_dip)
    locking_chi = np.interp(lats, p_lat, p_chi)
    v_conv = np.interp(lats, p_lat, p_vel)

    dlon_dlat = np.interp(lats, p_lat, np.gradient(p_lon, p_lat))
    cos_lat = np.cos(np.radians(lats))
    strike_rad = np.arctan2(dlon_dlat * cos_lat, 1.0)

    d_east_km = (lons - trench_lon) * 111.19 * cos_lat
    d_trench_km = d_east_km * np.cos(strike_rad)
    s_along_km = (-17.0 - lats) * 111.19

    dip_rad = np.radians(dip_deg)
    d_pos = np.maximum(d_trench_km, 0.0)
    z_slab_km = 6.0 + d_pos * np.tan(dip_rad) + 0.0008 * (d_pos ** 2)
    z_rel_km = depths - z_slab_km

    regime = np.ones(len(lats), dtype=np.int32)
    regime[d_trench_km < 0.0] = 0
    regime[(depths > 55.0) & (depths <= 180.0)] = 3
    regime[depths > 180.0] = 4
    regime[(d_trench_km > 80.0) & (depths <= 40.0) & (z_rel_km < -15.0)] = 2

    return {
        "trench_lon": trench_lon,
        "d_trench_km": d_trench_km,
        "s_along_km": s_along_km,
        "dip_deg": dip_deg,
        "z_slab_km": z_slab_km,
        "z_rel_km": z_rel_km,
        "locking_chi": locking_chi,
        "v_conv": v_conv,
        "strike_rad": strike_rad,
        "regime": regime,
    }


def compute_b_positive(mags: np.ndarray, dmc: float = 0.2) -> float:
    """
    Nicholas van der Elst (2021, JGR Solid Earth) b-positive estimator.
    """
    if len(mags) < 6:
        return 1.0
    diffs = np.diff(mags)
    pos_diffs = diffs[diffs >= dmc]
    if len(pos_diffs) < 4:
        mc = np.min(mags)
        mean_m = np.mean(mags)
        if mean_m <= mc + 0.05:
            return 1.0
        return float(np.clip(np.log10(np.e) / (mean_m - (mc - 0.05)), 0.4, 2.2))
    mean_diff = np.mean(pos_diffs)
    if mean_diff <= dmc + 0.02:
        return 1.5
    b_pos = np.log10(np.e) / (mean_diff - dmc + 0.05)
    return float(np.clip(b_pos, 0.35, 2.2))


def compute_b_aki_utsu(mags: np.ndarray) -> float:
    """Standard Aki (1965) / Utsu (1965) Maximum Likelihood b-value estimator."""
    if len(mags) < 6:
        return 1.0
    mc = np.min(mags)
    mean_m = np.mean(mags)
    denom = mean_m - (mc - 0.05)
    if denom <= 0.05:
        return 1.0
    return float(np.clip(np.log10(np.e) / denom, 0.35, 2.2))


def load_full_and_homogeneous_catalogs(csv_path: str, mc_cutoff: float = 4.0):
    """
    Loads both the full catalog (M >= 2.0, 54,855 events) for sub-threshold b-positive
    & microseismic coalescence (Xiong et al., 2026) and the homogenized catalog (M >= 4.0).
    """
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df.dropna(subset=["time", "latitude", "longitude", "mag"]).copy()
    df["depth"] = df["depth"].fillna(33.0).clip(lower=0.0, upper=650.0)
    df = df.sort_values("time").reset_index(drop=True)

    t0 = pd.Timestamp("1900-01-01T00:00:00Z")
    df["t_days"] = (df["time"] - t0).dt.total_seconds() / 86400.0
    df["year_frac"] = 1900.0 + df["t_days"] / 365.25

    slab_props = interpolate_trench_properties(
        df["latitude"].values, df["longitude"].values, df["depth"].values
    )
    for k, v in slab_props.items():
        df[k] = v

    df["benioff_strain"] = np.power(10.0, 0.75 * df["mag"].values)
    df["moment_proxy"] = np.power(10.0, 1.5 * (df["mag"].values - 7.0))

    df_cat = df[df["mag"] >= mc_cutoff].copy().reset_index(drop=True)
    return df, df_cat


def extract_spatiotemporal_features(df_full: pd.DataFrame, df_cat: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts strictly causal (t_i < t_current) physics-informed features for every event
    in the homogenized catalog (M >= 4.0), utilizing both M >= 4.0 and sub-threshold M >= 2.5
    histories.
    """
    n = len(df_cat)
    t_days = df_cat["t_days"].values
    years = df_cat["year_frac"].values
    lats = df_cat["latitude"].values
    lons = df_cat["longitude"].values
    depths = df_cat["depth"].values
    mags = df_cat["mag"].values
    s_along = df_cat["s_along_km"].values
    d_trench = df_cat["d_trench_km"].values
    z_rel = df_cat["z_rel_km"].values
    regime = df_cat["regime"].values
    locking = df_cat["locking_chi"].values
    v_conv = df_cat["v_conv"].values
    benioff = df_cat["benioff_strain"].values
    moment_proxy = df_cat["moment_proxy"].values

    cos_lat = np.cos(np.radians(lats))
    y_km = lats * 111.19
    x_km = lons * 111.19 * cos_lat

    # Full catalog arrays for sub-threshold b+ (Xiong et al., 2026 Fusion)
    f_t_days = df_full["t_days"].values
    f_lats = df_full["latitude"].values
    f_lons = df_full["longitude"].values
    f_mags = df_full["mag"].values
    f_y_km = f_lats * 111.19
    f_x_km = f_lons * 111.19 * np.cos(np.radians(f_lats))

    feats = {
        # 1. Classical & Anisotropic Subduction ETAS intensities
        "etas_iso_intensity": np.zeros(n),
        "etas_aniso_intensity": np.zeros(n),
        "etas_short_1d": np.zeros(n),
        "etas_med_7d": np.zeros(n),
        "etas_long_30d": np.zeros(n),
        "etas_interface_intensity": np.zeros(n),
        "etas_intraslab_intensity": np.zeros(n),
        "background_rate_mu_local": np.zeros(n),
        "omori_deviation_residual": np.zeros(n),

        # 2. Local Activity Counts & Rates across Scales (R <= 150 km)
        "count_local_1d": np.zeros(n),
        "count_local_7d": np.zeros(n),
        "count_local_30d": np.zeros(n),
        "count_local_90d": np.zeros(n),
        "count_local_365d": np.zeros(n),
        "rate_ratio_1d_30d": np.zeros(n),
        "rate_ratio_7d_90d": np.zeros(n),
        "rate_ratio_30d_365d": np.zeros(n),

        # 3. Magnitude & Energy Statistics
        "max_mag_local_7d": np.zeros(n),
        "max_mag_local_30d": np.zeros(n),
        "max_mag_local_365d": np.zeros(n),
        "mean_mag_local_50": np.zeros(n),
        "std_mag_local_50": np.zeros(n),
        "benioff_rate_7d": np.zeros(n),
        "benioff_rate_30d": np.zeros(n),
        "benioff_accel_7d_30d": np.zeros(n),
        "time_for_last_20_events_days": np.zeros(n),
        "time_since_last_m50_local": np.zeros(n),
        "time_since_last_m60_local": np.zeros(n),
        "time_since_last_m70_local": np.zeros(n),

        # 4. Differential Stress & Gulia-Wiemer / van der Elst b-Positive Dynamics
        "b_aki_50": np.zeros(n),
        "b_pos_25": np.zeros(n),
        "b_pos_50": np.zeros(n),
        "b_pos_100": np.zeros(n),
        "b_pos_250": np.zeros(n),
        "delta_b_pos_25_100": np.zeros(n),
        "delta_b_pos_50_250": np.zeros(n),
        "stai_bias_aki_minus_bpos": np.zeros(n),
        "gulia_wiemer_tl_score": np.zeros(n),
        "mag_deficit_bath": np.zeros(n),
        "b_pos_subthresh_40": np.zeros(n),
        "delta_b_pos_subthresh": np.zeros(n),

        # 5. 3D Subduction Cross-Depth Coupling & Asperity Migration
        "frac_interface_30d": np.zeros(n),
        "frac_intraslab_30d": np.zeros(n),
        "intraslab_to_interface_ratio_30d": np.zeros(n),
        "intraslab_rate_change_14d_90d": np.zeros(n),
        "mean_depth_local_30d": np.zeros(n),
        "depth_migration_vel_km_day": np.zeros(n),
        "along_strike_spread_km_30d": np.zeros(n),
        "trench_migration_km": np.zeros(n),
        "locking_gradient_migration": np.zeros(n),

        # 6. Geodetic Locking & Elastic Rebound Slip-Deficit Memory (Métois et al., 2016)
        "years_since_giant_rupture": np.zeros(n),
        "tectonic_slip_deficit_m": np.zeros(n),
        "elastic_strain_index": np.zeros(n),
        "cumulative_moment_release_10yr": np.zeros(n),

        # 7. Zaliapin-Ben-Zion Space-Time-Energy Nearest-Neighbor & Coalescence
        "zaliapin_log_eta": np.zeros(n),
        "zaliapin_log_T": np.zeros(n),
        "zaliapin_log_R": np.zeros(n),
        "spatial_entropy_30d": np.zeros(n),
        "interevent_dt_log_sec": np.zeros(n),
        "interevent_dist_km": np.zeros(n),
    }

    mc = 4.0
    alpha_etas = 1.85
    c_etas = 0.015
    p_etas = 1.12
    q_etas = 1.65

    print(f"Extracting physics-informed spatiotemporal features for {n} events...")
    for i in range(n):
        ti = t_days[i]
        yi_year = years[i]
        xi, yi = x_km[i], y_km[i]
        si, di = s_along[i], d_trench[i]
        lati = lats[i]

        idx_10yr = np.searchsorted(t_days[:i], ti - 3650.0, side="left")

        # 1. Elastic Rebound & Time since giant rupture
        last_giant_yr = 1868.0
        for g_yr, g_lat_min, g_lat_max, g_mag in HISTORICAL_GIANTS:
            if g_yr < yi_year and (g_lat_min - 0.5 <= lati <= g_lat_max + 0.5):
                if g_yr > last_giant_yr:
                    last_giant_yr = g_yr
        if i > 0:
            m8_mask = (mags[:i] >= 8.0) & (np.abs(lats[:i] - lati) <= 2.2)
            if np.any(m8_mask):
                last_cat_m8 = np.max(years[:i][m8_mask])
                if last_cat_m8 > last_giant_yr:
                    last_giant_yr = last_cat_m8

        yrs_since_giant = max(0.1, yi_year - last_giant_yr)
        feats["years_since_giant_rupture"][i] = yrs_since_giant
        slip_def_m = yrs_since_giant * (v_conv[i] / 1000.0) * locking[i]
        feats["tectonic_slip_deficit_m"][i] = slip_def_m
        feats["elastic_strain_index"][i] = np.log1p(slip_def_m) * locking[i]

        if i == 0:
            for k_b in ["b_aki_50", "b_pos_25", "b_pos_50", "b_pos_100", "b_pos_250", "b_pos_subthresh_40"]:
                feats[k_b][i] = 1.0
            continue

        dt_prev = max(1e-5, ti - t_days[i - 1])
        feats["interevent_dt_log_sec"][i] = np.log10(dt_prev * 86400.0)
        feats["interevent_dist_km"][i] = np.hypot(xi - x_km[i - 1], yi - y_km[i - 1])

        sl = slice(idx_10yr, i)
        dt_sl = ti - t_days[sl]
        dx_sl = xi - x_km[sl]
        dy_sl = yi - y_km[sl]
        dist_sq = dx_sl * dx_sl + dy_sl * dy_sl
        dist_km = np.sqrt(dist_sq)
        m_sl = mags[sl]
        reg_sl = regime[sl]
        dep_sl = depths[sl]
        ben_sl = benioff[sl]
        mom_sl = moment_proxy[sl]
        lock_sl = locking[sl]

        local_mask = dist_km <= 150.0
        n_local_10yr = np.sum(local_mask)
        span_days = max(30.0, min(3650.0, ti - t_days[0]))
        feats["background_rate_mu_local"][i] = (n_local_10yr + 1.0) / span_days

        # Sub-threshold b-positive from full catalog (past 180 days, R <= 150 km)
        f_idx_end = np.searchsorted(f_t_days, ti, side="left")
        f_idx_start = np.searchsorted(f_t_days[:f_idx_end], ti - 180.0, side="left")
        if f_idx_end > f_idx_start + 8:
            f_sl = slice(f_idx_start, f_idx_end)
            f_dist = np.hypot(xi - f_x_km[f_sl], yi - f_y_km[f_sl])
            f_loc_m = f_mags[f_sl][f_dist <= 150.0]
            if len(f_loc_m) >= 8:
                bp_sub_40 = compute_b_positive(f_loc_m[-40:], dmc=0.2)
                bp_sub_150 = compute_b_positive(f_loc_m[-150:], dmc=0.2)
                feats["b_pos_subthresh_40"][i] = bp_sub_40
                feats["delta_b_pos_subthresh"][i] = bp_sub_40 - bp_sub_150
            else:
                feats["b_pos_subthresh_40"][i] = 1.0
        else:
            feats["b_pos_subthresh_40"][i] = 1.0

        if np.any(local_mask):
            dt_loc = dt_sl[local_mask]
            m_loc = m_sl[local_mask]
            dep_loc = dep_sl[local_mask]
            reg_loc = reg_sl[local_mask]
            ben_loc = ben_sl[local_mask]
            mom_loc = mom_sl[local_mask]
            lock_loc = lock_sl[local_mask]
            s_loc = s_along[sl][local_mask]
            d_tr_loc = d_trench[sl][local_mask]

            feats["cumulative_moment_release_10yr"][i] = np.log1p(np.sum(mom_loc))

            m5_m = m_loc >= 5.0
            m6_m = m_loc >= 6.0
            m7_m = m_loc >= 7.0
            feats["time_since_last_m50_local"][i] = np.min(dt_loc[m5_m]) if np.any(m5_m) else 3650.0
            feats["time_since_last_m60_local"][i] = np.min(dt_loc[m6_m]) if np.any(m6_m) else 3650.0
            feats["time_since_last_m70_local"][i] = np.min(dt_loc[m7_m]) if np.any(m7_m) else 3650.0

            feats["time_for_last_20_events_days"][i] = dt_loc[-20] if len(dt_loc) >= 20 else span_days

            w1 = dt_loc <= 1.0
            w7 = dt_loc <= 7.0
            w14 = dt_loc <= 14.0
            w30 = dt_loc <= 30.0
            w90 = dt_loc <= 90.0
            w365 = dt_loc <= 365.25

            c1 = np.sum(w1)
            c7 = np.sum(w7)
            c30 = np.sum(w30)
            c90 = np.sum(w90)
            c365 = np.sum(w365)

            feats["count_local_1d"][i] = c1
            feats["count_local_7d"][i] = c7
            feats["count_local_30d"][i] = c30
            feats["count_local_90d"][i] = c90
            feats["count_local_365d"][i] = c365

            feats["rate_ratio_1d_30d"][i] = (c1 + 0.1) / ((c30 / 30.0) + 0.1)
            feats["rate_ratio_7d_90d"][i] = (c7 / 7.0 + 0.05) / ((c90 / 90.0) + 0.05)
            feats["rate_ratio_30d_365d"][i] = (c30 / 30.0 + 0.02) / ((c365 / 365.25) + 0.02)

            feats["max_mag_local_7d"][i] = np.max(m_loc[w7]) if c7 > 0 else 4.0
            feats["max_mag_local_30d"][i] = np.max(m_loc[w30]) if c30 > 0 else 4.0
            feats["max_mag_local_365d"][i] = np.max(m_loc[w365]) if c365 > 0 else 4.0

            b7 = np.sum(ben_loc[w7]) / 7.0
            b30 = np.sum(ben_loc[w30]) / 30.0
            feats["benioff_rate_7d"][i] = np.log1p(b7)
            feats["benioff_rate_30d"][i] = np.log1p(b30)
            feats["benioff_accel_7d_30d"][i] = (b7 + 1.0) / (b30 + 1.0)

            m_recent_25 = m_loc[-25:]
            m_recent_50 = m_loc[-50:]
            m_recent_100 = m_loc[-100:]
            m_recent_250 = m_loc[-250:]

            feats["mean_mag_local_50"][i] = np.mean(m_recent_50)
            feats["std_mag_local_50"][i] = np.std(m_recent_50) if len(m_recent_50) > 2 else 0.3

            b_aki = compute_b_aki_utsu(m_recent_50)
            b_p25 = compute_b_positive(m_recent_25, dmc=0.2)
            b_p50 = compute_b_positive(m_recent_50, dmc=0.2)
            b_p100 = compute_b_positive(m_recent_100, dmc=0.2)
            b_p250 = compute_b_positive(m_recent_250, dmc=0.2)

            feats["b_aki_50"][i] = b_aki
            feats["b_pos_25"][i] = b_p25
            feats["b_pos_50"][i] = b_p50
            feats["b_pos_100"][i] = b_p100
            feats["b_pos_250"][i] = b_p250
            feats["delta_b_pos_25_100"][i] = b_p25 - b_p100
            feats["delta_b_pos_50_250"][i] = b_p50 - b_p250
            feats["stai_bias_aki_minus_bpos"][i] = b_aki - b_p50

            gw_drop = (b_p250 - b_p25) / max(0.5, b_p250)
            feats["gulia_wiemer_tl_score"][i] = gw_drop * np.log1p(c7)

            if c30 >= 2:
                m_sorted = np.sort(m_loc[w30])
                feats["mag_deficit_bath"][i] = 1.2 - (m_sorted[-1] - m_sorted[-2])

            if c30 > 0:
                reg30 = reg_loc[w30]
                n_inter30 = np.sum(reg30 == 1)
                n_slab30 = np.sum(reg30 == 3)
                feats["frac_interface_30d"][i] = n_inter30 / c30
                feats["frac_intraslab_30d"][i] = n_slab30 / c30
                feats["intraslab_to_interface_ratio_30d"][i] = (n_slab30 + 0.5) / (n_inter30 + 0.5)
                feats["mean_depth_local_30d"][i] = np.mean(dep_loc[w30])
                feats["along_strike_spread_km_30d"][i] = np.std(s_loc[w30]) if c30 >= 3 else 0.0

                w_old = w30 & (~w7)
                if np.any(w7) and np.any(w_old):
                    feats["depth_migration_vel_km_day"][i] = np.mean(dep_loc[w7]) - np.mean(dep_loc[w_old])
                    feats["trench_migration_km"][i] = np.mean(d_tr_loc[w7]) - np.mean(d_tr_loc[w_old])
                    feats["locking_gradient_migration"][i] = np.mean(lock_loc[w7]) - np.mean(lock_loc[w_old])

                if c30 >= 5:
                    bins_s = np.floor((s_loc[w30] - si + 150.0) / 75.0).clip(0, 3).astype(int)
                    bins_d = np.floor((d_tr_loc[w30] - di + 150.0) / 75.0).clip(0, 3).astype(int)
                    cell_ids = bins_s * 4 + bins_d
                    _, counts_cell = np.unique(cell_ids, return_counts=True)
                    probs = counts_cell / np.sum(counts_cell)
                    feats["spatial_entropy_30d"][i] = -np.sum(probs * np.log(probs + 1e-12))

            if c90 > 0:
                n_slab14 = np.sum((reg_loc == 3) & w14) / 14.0
                n_slab90 = np.sum((reg_loc == 3) & w90) / 90.0
                feats["intraslab_rate_change_14d_90d"][i] = (n_slab14 + 0.02) / (n_slab90 + 0.02)
        else:
            feats["time_since_last_m50_local"][i] = 3650.0
            feats["time_since_last_m60_local"][i] = 3650.0
            feats["time_since_last_m70_local"][i] = 3650.0
            feats["time_for_last_20_events_days"][i] = span_days
            for k_b in ["b_aki_50", "b_pos_25", "b_pos_50", "b_pos_100", "b_pos_250"]:
                feats[k_b][i] = 1.0

        sl_etas_mask = (dt_sl <= 365.25) | (m_sl >= 6.5)
        if np.any(sl_etas_mask):
            dt_e = np.maximum(1e-4, dt_sl[sl_etas_mask])
            m_e = m_sl[sl_etas_mask]
            dist_sq_e = dist_sq[sl_etas_mask]
            ds_along_e = s_along[sl][sl_etas_mask] - si
            dd_trench_e = d_trench[sl][sl_etas_mask] - di
            reg_e = reg_sl[sl_etas_mask]

            prod = np.exp(alpha_etas * (m_e - mc))
            omori = np.power(dt_e + c_etas, -p_etas)
            scale_m = 64.0 * np.exp(0.85 * (m_e - mc))
            sp_iso = np.power(dist_sq_e / scale_m + 1.0, -q_etas)

            aniso_dist_sq = (ds_along_e / 1.6) ** 2 + (dd_trench_e * 1.4) ** 2
            sp_aniso = np.power(aniso_dist_sq / scale_m + 1.0, -q_etas)

            contrib_iso = prod * omori * sp_iso
            contrib_aniso = prod * omori * sp_aniso

            feats["etas_iso_intensity"][i] = np.log1p(np.sum(contrib_iso))
            feats["etas_aniso_intensity"][i] = np.log1p(np.sum(contrib_aniso))
            feats["etas_short_1d"][i] = np.log1p(np.sum(contrib_aniso[dt_e <= 1.0]))
            feats["etas_med_7d"][i] = np.log1p(np.sum(contrib_aniso[dt_e <= 7.0]))
            feats["etas_long_30d"][i] = np.log1p(np.sum(contrib_aniso[dt_e <= 30.0]))
            feats["etas_interface_intensity"][i] = np.log1p(np.sum(contrib_aniso[reg_e == 1]))
            feats["etas_intraslab_intensity"][i] = np.log1p(np.sum(contrib_aniso[reg_e == 3]))

            # Omori Deviation Residual: log ratio of short-term (1d) activity vs expected from 7-30d decay
            expected_from_med = feats["etas_med_7d"][i] - feats["etas_short_1d"][i]
            feats["omori_deviation_residual"][i] = feats["etas_short_1d"][i] - 0.5 * expected_from_med

            r_e = np.maximum(0.5, np.sqrt(dist_sq_e))
            b_ref = 1.0
            T_ij = dt_e * np.power(10.0, -0.5 * b_ref * m_e)
            R_ij = np.power(r_e, 1.6) * np.power(10.0, -0.5 * b_ref * m_e)
            eta_ij = T_ij * R_ij
            min_idx = np.argmin(eta_ij)
            feats["zaliapin_log_eta"][i] = np.log10(max(1e-12, eta_ij[min_idx]))
            feats["zaliapin_log_T"][i] = np.log10(max(1e-12, T_ij[min_idx]))
            feats["zaliapin_log_R"][i] = np.log10(max(1e-12, R_ij[min_idx]))

    for k, arr in feats.items():
        df_cat[k] = arr

    print("Computing forward ground-truth targets for evaluation horizons (7d, 14d, 30d)...")
    y_m50_7d = np.zeros(n, dtype=np.int32)
    y_m55_7d = np.zeros(n, dtype=np.int32)
    y_m60_7d = np.zeros(n, dtype=np.int32)
    y_m50_14d = np.zeros(n, dtype=np.int32)
    y_m55_14d = np.zeros(n, dtype=np.int32)
    y_m60_14d = np.zeros(n, dtype=np.int32)
    y_m55_30d = np.zeros(n, dtype=np.int32)
    y_m60_30d = np.zeros(n, dtype=np.int32)
    max_mag_14d = np.full(n, 4.0, dtype=np.float64)
    max_mag_30d = np.full(n, 4.0, dtype=np.float64)
    y_foreshock_esc_14d = np.zeros(n, dtype=np.int32)

    for i in range(n):
        ti = t_days[i]
        xi, yi = x_km[i], y_km[i]
        mi = mags[i]
        idx_end = np.searchsorted(t_days, ti + 30.0, side="right")
        if idx_end <= i + 1:
            continue
        sl_f = slice(i + 1, idx_end)
        dt_f = t_days[sl_f] - ti
        dist_f = np.hypot(x_km[sl_f] - xi, y_km[sl_f] - yi)
        loc_f = dist_f <= 150.0
        if not np.any(loc_f):
            continue
        dt_loc_f = dt_f[loc_f]
        m_loc_f = mags[sl_f][loc_f]

        m30_max = np.max(m_loc_f)
        max_mag_30d[i] = m30_max
        if m30_max >= 5.5:
            y_m55_30d[i] = 1
        if m30_max >= 6.0:
            y_m60_30d[i] = 1

        f14 = dt_loc_f <= 14.0
        if np.any(f14):
            m14_max = np.max(m_loc_f[f14])
            max_mag_14d[i] = m14_max
            if m14_max >= 5.0:
                y_m50_14d[i] = 1
            if m14_max >= 5.5:
                y_m55_14d[i] = 1
            if m14_max >= 6.0:
                y_m60_14d[i] = 1
            if m14_max >= 5.5 and m14_max > max(mi, df_cat["max_mag_local_7d"].values[i]):
                y_foreshock_esc_14d[i] = 1

        f7 = dt_loc_f <= 7.0
        if np.any(f7):
            m7_max = np.max(m_loc_f[f7])
            if m7_max >= 5.0:
                y_m50_7d[i] = 1
            if m7_max >= 5.5:
                y_m55_7d[i] = 1
            if m7_max >= 6.0:
                y_m60_7d[i] = 1

    df_cat["y_m50_7d"] = y_m50_7d
    df_cat["y_m55_7d"] = y_m55_7d
    df_cat["y_m60_7d"] = y_m60_7d
    df_cat["y_m50_14d"] = y_m50_14d
    df_cat["y_m55_14d"] = y_m55_14d
    df_cat["y_m60_14d"] = y_m60_14d
    df_cat["y_m55_30d"] = y_m55_30d
    df_cat["y_m60_30d"] = y_m60_30d
    df_cat["max_mag_14d"] = max_mag_14d
    df_cat["max_mag_30d"] = max_mag_30d
    df_cat["y_foreshock_esc_14d"] = y_foreshock_esc_14d

    df_cat["is_pre_mainshock_regime"] = (df_cat["time_since_last_m50_local"] >= 7.0) | (
        df_cat["max_mag_local_7d"] < 5.5
    )

    return df_cat


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, "data", "raw", "chile_usgs_1900_2026.csv")
    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)

    df_full, df_cat = load_full_and_homogeneous_catalogs(raw_csv, mc_cutoff=4.0)
    df_feat = extract_spatiotemporal_features(df_full, df_cat)

    out_csv = os.path.join(proc_dir, "chile_features_m40.csv")
    df_feat.to_csv(out_csv, index=False)
    print(f"Saved processed feature dataset ({len(df_feat)} events, {df_feat.shape[1]} columns) to {out_csv}")

