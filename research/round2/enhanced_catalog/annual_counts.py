"""Annual counts per ML band of the IPOC catalog (network-change / level-shift check). Usage: python -I annual_counts.py <ipoc.txt>"""
import sys; sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog")
import ec_load, pandas as pd
c = ec_load.load_ipoc(sys.argv[1]); y = c.time.dt.year
t = pd.DataFrame({"ML<2.5": (c.mag < 2.5).groupby(y).sum(), "2.5-3": ((c.mag >= 2.5) & (c.mag < 3)).groupby(y).sum(),
                  "ML>=3": (c.mag >= 3).groupby(y).sum(), "ML>=4.5": (c.mag >= 4.5).groupby(y).sum()})
print(t.to_string())
