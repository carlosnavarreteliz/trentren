"""Subduction margin polylines (trench traces, hand-digitized, coarse ~50 km) and event -> (margin, along-strike) mapping.

Each polyline is a list of (lon, lat). Longitudes may run past +-180 (Aleutians use < -180, Tonga > 180); event
longitudes are wrapped to within 180 deg of the polyline's mean longitude before projection.
Chile itself is NOT here: Chile uses the harness box and latitude bins (lat -46..-17, lon -78..-66).
"""
import numpy as np

MARGINS = {
    "PER": [(-72.6, -17.5), (-74.0, -16.5), (-75.8, -15.0), (-77.5, -13.0), (-78.8, -11.0), (-80.0, -9.0),
            (-80.9, -7.0), (-81.6, -5.0), (-81.5, -3.0), (-81.0, -1.0), (-80.4, 1.0), (-79.4, 2.5), (-78.6, 4.0),
            (-78.2, 5.5)],
    "MEX": [(-83.0, 7.3), (-84.5, 8.3), (-86.0, 9.6), (-87.5, 10.9), (-89.0, 12.0), (-90.5, 12.9), (-92.5, 13.7),
            (-94.5, 14.6), (-96.5, 15.2), (-98.5, 15.6), (-100.5, 16.3), (-102.5, 17.3), (-104.5, 18.5),
            (-105.8, 20.0), (-106.3, 21.0)],
    "CAS": [(-125.3, 40.3), (-125.1, 42.0), (-125.2, 44.0), (-125.1, 46.0), (-125.6, 47.5), (-126.6, 48.8),
            (-128.0, 50.0)],
    "ALA": [(-145.0, 59.5), (-148.0, 58.6), (-151.0, 57.0), (-154.0, 55.8), (-157.0, 54.7), (-160.0, 53.9),
            (-163.0, 53.2), (-166.0, 52.4), (-169.0, 51.6), (-172.0, 51.0), (-176.0, 50.5), (-180.0, 50.3),
            (-184.0, 50.6), (-188.0, 51.5), (-192.0, 53.0), (-194.5, 54.3)],
    "KAM": [(164.0, 55.8), (162.8, 54.0), (161.0, 52.0), (158.5, 50.0), (155.8, 48.0), (153.0, 46.2), (150.0, 44.4),
            (147.0, 42.6), (145.0, 41.5), (144.0, 40.6)],
    "JPN": [(146.0, 42.2), (144.6, 41.0), (144.3, 39.5), (143.9, 38.0), (143.3, 36.5), (142.5, 35.0), (142.0, 34.0), (141.6, 33.0)],
    "IBM": [(142.6, 35.2), (142.0, 34.0), (142.2, 32.0), (142.5, 30.0), (142.8, 28.0), (143.0, 26.0), (143.5, 24.5), (145.0, 22.5),
            (146.7, 20.0), (147.5, 17.0), (147.5, 14.5), (146.0, 12.0), (143.5, 11.2), (141.5, 11.5)],
    "NRY": [(138.6, 34.4), (137.5, 33.6), (136.2, 32.9), (135.0, 32.6), (133.8, 32.2), (132.8, 31.5), (132.3, 30.5),
            (131.7, 29.2), (130.8, 27.8), (129.6, 25.9), (128.0, 24.9), (126.3, 24.0), (124.5, 23.3), (123.0, 23.0)],
    "PHT": [(126.0, 14.5), (126.6, 12.5), (127.0, 10.5), (127.0, 8.5), (126.9, 6.5), (126.8, 4.5), (127.0, 3.0)],
    "MNL": [(120.3, 22.0), (119.6, 20.0), (119.2, 18.0), (119.3, 16.0), (119.8, 14.0), (120.3, 13.0)],
    "SUN": [(93.0, 15.5), (92.4, 13.0), (92.4, 10.5), (92.9, 8.0), (93.8, 6.0), (95.0, 4.2), (96.3, 2.0), (97.6, 0.0),
            (99.0, -2.0), (100.6, -4.2), (102.4, -6.3), (104.5, -8.2), (107.0, -9.6), (109.5, -10.3), (112.0, -10.7),
            (114.5, -11.0), (117.0, -11.2), (119.5, -11.0)],
    "SOL": [(147.5, -7.0), (149.5, -7.1), (151.5, -6.8), (153.2, -5.8), (154.6, -6.4), (155.8, -7.6), (157.5, -9.0),
            (159.5, -10.5), (161.5, -11.3), (163.0, -11.5)],
    "VAN": [(165.0, -10.0), (165.8, -12.0), (166.2, -14.0), (166.6, -16.0), (167.3, -18.0), (168.2, -20.0),
            (169.5, -21.8), (171.0, -22.6), (172.5, -22.8)],
    "TON": [(187.0, -15.2), (186.6, -17.0), (186.4, -19.0), (185.8, -21.0), (185.0, -23.0), (184.4, -25.0),
            (183.7, -27.5), (183.0, -30.0), (182.4, -32.5), (181.7, -35.0), (181.0, -36.8), (179.6, -38.5),
            (178.6, -40.0), (177.5, -41.5)],
    "HEL": [(19.8, 38.5), (20.5, 37.0), (21.5, 36.0), (22.8, 35.2), (24.5, 34.6), (26.0, 34.6), (27.5, 35.3),
            (28.8, 36.0)],
}
KM_PER_DEG = 111.195
SEAWARD_KM, LANDWARD_KM = 300.0, 700.0
# Nankai-Ryukyu: the auto rule is contaminated by the Izu-Bonin deep slab south of its eastern end; landward = NW.
SIGN_OVERRIDE = {"NRY": -1.0}
CHILE_BOX = (-46.0, -17.0, -78.0, -66.0)


def project(lon, lat, poly):
    """Project points on a polyline. Returns (s_km along strike, d_km signed perpendicular distance (left of travel
    direction positive), inside flag (projection within the polyline extent)."""
    P = np.asarray(poly, float)
    lon0 = P[:, 0].mean()
    lonw = (np.asarray(lon, float) - lon0 + 180.0) % 360.0 - 180.0 + lon0
    lat = np.asarray(lat, float)
    n = len(lonw)
    best = np.full(n, np.inf)
    s_out = np.full(n, np.nan)
    d_out = np.full(n, np.nan)
    inside = np.zeros(n, bool)
    cum = 0.0
    nseg = len(P) - 1
    for i in range(nseg):
        (x1, y1), (x2, y2) = P[i], P[i + 1]
        cl = np.cos(np.radians(0.5 * (y1 + y2)))
        ax, ay = (x2 - x1) * cl * KM_PER_DEG, (y2 - y1) * KM_PER_DEG
        L = np.hypot(ax, ay)
        px, py = (lonw - x1) * cl * KM_PER_DEG, (lat - y1) * KM_PER_DEG
        t = (px * ax + py * ay) / L ** 2
        tc = np.clip(t, 0, 1)
        dx, dy = px - tc * ax, py - tc * ay
        dist = np.hypot(dx, dy)
        cross = (ax * py - ay * px) / L          # >0 left of travel direction
        upd = dist < best
        best = np.where(upd, dist, best)
        s_out = np.where(upd, cum + tc * L, s_out)
        d_out = np.where(upd, np.where(cross >= 0, 1.0, -1.0) * dist, d_out)
        ok = np.ones(n, bool)
        if i == 0:
            ok &= t >= 0
        if i == nseg - 1:
            ok &= t <= 1
        inside = np.where(upd, ok, inside)
        cum += L
    return s_out, d_out, inside, cum


def polyline_length_km(poly):
    return project(np.array([poly[0][0]]), np.array([poly[0][1]]), poly)[3]


def in_chile_box(lon, lat):
    la0, la1, lo0, lo1 = CHILE_BOX
    return (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)


def assign(lon, lat, depth):
    """Assign events to margins. Returns (margin code array ('' = none), s_km, d_km landward-positive)."""
    lon, lat, depth = map(lambda a: np.asarray(a, float), (lon, lat, depth))
    n = len(lon)
    code = np.array([""] * n, dtype=object)
    s_best = np.full(n, np.nan)
    d_best = np.full(n, np.nan)
    absd = np.full(n, np.inf)
    chile = in_chile_box(lon, lat)
    signs = {}
    for m, poly in MARGINS.items():
        s, d, inside, L = project(lon, lat, poly)
        near = inside & (np.abs(d) < 800) & (depth > 100)
        sg = np.sign(np.median(d[near])) if near.sum() > 20 else 1.0
        sg = SIGN_OVERRIDE.get(m, sg)
        signs[m] = sg
        dl = d * sg                                   # landward positive
        ok = inside & (dl >= -SEAWARD_KM) & (dl <= LANDWARD_KM) & ~chile
        upd = ok & (np.abs(dl) < absd)
        code = np.where(upd, m, code)
        s_best = np.where(upd, s, s_best)
        d_best = np.where(upd, dl, d_best)
        absd = np.where(upd, np.abs(dl), absd)
    return code, s_best, d_best, signs
