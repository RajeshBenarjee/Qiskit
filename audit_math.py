import json, numpy as np

with open('data/processed/events/north_andhra_2026/affected_zones.geojson', 'r', encoding='utf-8') as f:
    zones = json.load(f)['features']
with open('data/processed/events/north_andhra_2026/affected_shelters.geojson', 'r', encoding='utf-8') as f:
    shelters = json.load(f)['features']

sz = [
    next(z for z in zones if 'kalingapatnam' in z['properties']['name'].lower()),
    next(z for z in zones if 'bandaruvani' in z['properties']['name'].lower()),
    next(z for z in zones if 'vatsavalasa' in z['properties']['name'].lower())
]
ss = [
    next(s for s in shelters if 'calingapatnam' in s['properties']['name'].lower()),
    next(s for s in shelters if 'bandravanipeta' in s['properties']['name'].lower())
]

speed_kph = 35.0
speed_mps = speed_kph * 1000.0 / 3600.0

print('Zone Data:')
for i, z in enumerate(sz):
    p = z['properties']
    print(f"  Z{i+1}: {p['name']} | DistToLandfall: {p.get('dist_to_landfall_km')} km | StormRisk: {p.get('dynamic_storm_risk')}")

print('\nDetailed Link Computations:')
for i, z in enumerate(sz):
    pz = (z['properties']['utm_easting'], z['properties']['utm_northing'])
    risk = z['properties']['dynamic_storm_risk']
    for j, s in enumerate(ss):
        ps = (s['properties']['utm_easting'], s['properties']['utm_northing'])
        dist_m = np.hypot(pz[0] - ps[0], pz[1] - ps[1])
        t_sec = dist_m / speed_mps
        t_min = t_sec / 60.0
        risk_weight = 1.0 + 2.0 * risk
        cost = round(t_min * risk_weight, 2)
        print(f"  Z{i+1} ({z['properties']['name']:16s}) -> S{j+1} ({s['properties']['name']:16s}): "
              f"Dist = {dist_m:7.1f} m | RawTime = {t_min:5.2f} min ({t_sec:5.1f} s) | "
              f"RiskWeight = {risk_weight:4.2f} (1 + 2*{risk:.2f}) | Cost C_ij = {cost:5.2f}")
