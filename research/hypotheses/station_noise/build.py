import sys, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/station_noise")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import features as F
D = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/station_noise/"
t = time.time()
z = F.build_site_z(D + "noise_mode_long.parquet", D + "channels_region.txt")
z.to_parquet(D + "site_z.parquet")
print("built", z.shape, time.time() - t)
print(z.groupby(["band", "dense"]).z.describe())
s = z[~z.dense].groupby(["site", "band"]).agg(lat=("lat", "first"), d0=("date", "min"), d1=("date", "max"), n=("z", "size"))
print(s.xs("hf", level="band").sort_values("lat").to_string())
