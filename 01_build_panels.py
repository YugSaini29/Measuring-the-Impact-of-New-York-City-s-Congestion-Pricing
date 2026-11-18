#!/usr/bin/env python3
"""
Group 7 - NYC congestion pricing: step 1 (download -> zone flags -> clean -> panels)

Stages (run in this order, or use `all`):
  download  fetch TLC Parquet files + zone lookup + zone shapefile
  zones     flag every taxi zone as cbd / buffer / outside / unknown
  panels    clean each trip file, write zone-hour panels + audit table
  summary   small CSVs you can paste/upload back for discussion

Install:  pip install duckdb pandas pyarrow geopandas shapely pyproj

Examples:
  python 01_build_panels.py all
  python 01_build_panels.py download --services yellow green      # skip the big HVFHV files
  python 01_build_panels.py zones --buffer-ft 1500
  python 01_build_panels.py panels --months 2025-02 2024-02

Row tiers (nothing is silently discarded; every row is kept and tagged)
  tier 0 = fails basic checks (pickup outside the file's month, dropoff <= pickup, null zone)
  tier 1 = passes basic checks but fails quality filters (duration, miles, base fare, speed)
  tier 2 = passes everything
  Summary CSVs stack three versions in a `dataset` column:
    raw    = all rows            counts = tier >= 1 (use for trip-count hypotheses)
    strict = tier 2 only         (use for fare / speed / duration outcomes)

Design notes
  * TLC timestamps are already NYC local wall-clock time (no timezone stored),
    so nothing is converted; the DST hour is simply not distinguishable.
  * Rows are NOT dropped for unknown zones (264/265) - they get region 'unknown'.
  * cbd_congestion_fee does not exist in 2024 files; it is treated as 0 there
    (the audit table records whether the column was present).
  * Taxi 'charge_ex_tip' = total_amount - tip_amount; HVFHV 'charge_ex_tip' =
    base fare + tolls + bcf + sales tax + congestion surcharge + airport fee + cbd fee.
"""
import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

import duckdb

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MISC = "https://d37ci6vzurychx.cloudfront.net/misc"
DEFAULT_MONTHS = ["2024-02", "2024-03", "2024-10", "2024-11", "2024-12",
                  "2025-01", "2025-02", "2025-03"]
ALL_SERVICES = ["yellow", "green", "fhvhv"]
AIRPORT_IDS = {1, 132, 138}  # EWR, JFK, LaGuardia

# Column names per service (lower-case; matched case-insensitively against the file)
SPEC = {
    "yellow": dict(pu_ts="tpep_pickup_datetime", do_ts="tpep_dropoff_datetime",
                   pu="pulocationid", do="dolocationid", miles="trip_distance",
                   base="fare_amount", tip="tip_amount", total="total_amount",
                   cbd="cbd_congestion_fee"),
    "green": dict(pu_ts="lpep_pickup_datetime", do_ts="lpep_dropoff_datetime",
                  pu="pulocationid", do="dolocationid", miles="trip_distance",
                  base="fare_amount", tip="tip_amount", total="total_amount",
                  cbd="cbd_congestion_fee"),
    "fhvhv": dict(pu_ts="pickup_datetime", do_ts="dropoff_datetime",
                  pu="pulocationid", do="dolocationid", miles="trip_miles",
                  base="base_passenger_fare", cbd="cbd_congestion_fee",
                  req="request_datetime", driver_pay="driver_pay"),
}


# ----------------------------------------------------------------------------- helpers
def month_range(ym):
    y, m = map(int, ym.split("-"))
    start = f"{y:04d}-{m:02d}-01"
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return start, f"{ny:04d}-{nm:02d}-01"


def trip_path(data_dir, svc, ym):
    return Path(data_dir) / f"{svc}_tripdata_{ym}.parquet"


def fetch(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have  {dest.name}")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dest)
            print(f"  got   {dest.name}  ({dest.stat().st_size/1e6:.0f} MB)")
            return True
        except Exception as e:  # noqa
            print(f"  retry {dest.name} ({attempt+1}/3): {e}")
            time.sleep(3)
    print(f"  FAILED {url}  -> download it manually from the TLC page into {dest.parent}")
    return False


# ----------------------------------------------------------------------------- stage: download
def stage_download(a):
    d = Path(a.data_dir)
    d.mkdir(parents=True, exist_ok=True)
    fetch(f"{MISC}/taxi_zone_lookup.csv", d / "taxi_zone_lookup.csv")
    fetch(f"{MISC}/taxi_zones.zip", d / "taxi_zones.zip")
    for ym in a.months:
        for svc in a.services:
            fetch(f"{BASE}/{svc}_tripdata_{ym}.parquet", trip_path(d, svc, ym))


# ----------------------------------------------------------------------------- stage: zones
def stage_zones(a):
    import geopandas as gpd
    import pandas as pd
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lookup = pd.read_csv(d / "taxi_zone_lookup.csv")
    lookup.columns = [c.lower() for c in lookup.columns]
    lookup = lookup.rename(columns={"locationid": "zone_id"})

    # Extract the shapefile zip ourselves (the zip:// path form breaks on Windows)
    import zipfile
    zpath = d / "taxi_zones.zip"
    if not zipfile.is_zipfile(zpath):
        sys.exit(f"{zpath} is not a valid zip (download probably failed). Delete it and rerun "
                 f"`download`, or fetch 'Taxi Zone Shapefile' from the TLC page into {d}.")
    shp_dir = d / "taxi_zones_shp"
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(shp_dir)
    shp = next(shp_dir.rglob("*.shp"), None)
    if shp is None:
        sys.exit(f"No .shp file found inside {zpath}")
    gdf = gpd.read_file(shp).to_crs(2263)  # NY Long Island, feet
    gdf = gdf.rename(columns={"LocationID": "zone_id"})
    gdf = gdf.dissolve(by="zone_id").reset_index()[["zone_id", "geometry"]]  # some ids have 2 polygons

    # --- 60th Street line, approximated in the Manhattan street grid -------------
    # Anchor: Columbus Circle (59th St). 60th St is ~260 ft north along the avenue
    # direction. The grid is rotated ~29 deg east of true north.
    t = Transformer.from_crs(4326, 2263, always_xy=True)
    x59, y59 = t.transform(-73.9819, 40.7681)
    th = math.radians(29.0)
    n = (math.sin(th), math.cos(th))          # along avenues (grid north)
    s = (math.cos(th), -math.sin(th))         # along streets
    p0 = (x59 + 260 * n[0], y59 + 260 * n[1])
    L = 60000.0
    south = Polygon([
        (p0[0] - L * s[0], p0[1] - L * s[1]),
        (p0[0] + L * s[0], p0[1] + L * s[1]),
        (p0[0] + L * s[0] - L * n[0], p0[1] + L * s[1] - L * n[1]),
        (p0[0] - L * s[0] - L * n[0], p0[1] - L * s[1] - L * n[1]),
    ])

    df = lookup.merge(gdf, on="zone_id", how="left")
    df["cbd_frac"] = 0.0
    manh = df["borough"].eq("Manhattan") & df["geometry"].notna()
    df.loc[manh, "cbd_frac"] = df.loc[manh, "geometry"].apply(
        lambda g: g.intersection(south).area / g.area)

    df["region"] = "outside"
    df.loc[manh & (df["cbd_frac"] >= 0.5), "region"] = "cbd"
    df.loc[df["geometry"].isna(), "region"] = "unknown"   # 264 / 265 etc.

    cbd_union = unary_union(list(df.loc[df["region"] == "cbd", "geometry"]))
    df["dist_ft"] = df["geometry"].apply(lambda g: cbd_union.distance(g) if g is not None else float("nan"))
    df.loc[(df["region"] == "outside") & (df["dist_ft"] <= a.buffer_ft), "region"] = "buffer"
    df["straddles_60th"] = df["cbd_frac"].between(0.05, 0.95)
    df["is_airport"] = df["zone_id"].isin(AIRPORT_IDS)

    ov = out / "zone_overrides.csv"   # optional manual fixes: columns zone_id,region
    if ov.exists():
        o = pd.read_csv(ov)
        df = df.merge(o.rename(columns={"region": "region_override"}), on="zone_id", how="left")
        df["region"] = df["region_override"].fillna(df["region"])
        df = df.drop(columns="region_override")
        print(f"  applied {len(o)} overrides from {ov}")

    cols = ["zone_id", "borough", "zone", "service_zone", "cbd_frac", "dist_ft",
            "straddles_60th", "is_airport", "region"]
    df[cols].to_csv(out / "zone_flags.csv", index=False)
    print(df["region"].value_counts().to_string())
    print("\nZones straddling the 60th St line - CHECK THESE against the MTA map:")
    print(df.loc[df["straddles_60th"], ["zone_id", "zone", "cbd_frac", "region"]].to_string(index=False))
    print(f"\nWrote {out/'zone_flags.csv'}. Put corrections in {ov} (zone_id,region) and rerun.")


# ----------------------------------------------------------------------------- stage: panels
def connect(a):
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}")
    con.execute(f"SET memory_limit='{a.memory_limit}'")
    tmp = Path(a.out_dir) / "duck_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def load_zones(con, a):
    zf = Path(a.out_dir) / "zone_flags.csv"
    if not zf.exists():
        sys.exit("zone_flags.csv missing - run the `zones` stage first.")
    con.execute(f"CREATE OR REPLACE TABLE zones AS SELECT * FROM read_csv_auto('{zf}')")


def process_file(con, a, svc, ym):
    src = trip_path(a.data_dir, svc, ym)
    if not src.exists():
        print(f"  skip {svc} {ym}: file not found")
        return None
    panels = Path(a.out_dir) / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    existing = panels / f"pickups_{svc}_{ym}.parquet"
    if existing.exists() and not a.force:
        have = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{existing}')").fetchall()}
        if "tier" in have:
            print(f"  done already: {svc} {ym} (use --force to redo)")
            return None
        print(f"  {svc} {ym}: existing panel is the old filtered version -> rebuilding with tiers")

    sp = SPEC[svc]
    names = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src}')").fetchall()]
    low = {c.lower(): c for c in names}

    def col(key, default="NULL"):
        c = low.get(sp.get(key, "").lower()) if key in sp else None
        return f'"{c}"' if c else default

    def raw(name, default="0"):
        c = low.get(name)
        return f'COALESCE(CAST("{c}" AS DOUBLE), 0)' if c else default

    has_cbd = sp["cbd"] in low
    cbd = f'COALESCE(CAST("{low[sp["cbd"]]}" AS DOUBLE), 0)' if has_cbd else "0.0"

    if svc == "fhvhv":
        charge = " + ".join([raw("base_passenger_fare"), raw("tolls"), raw("bcf"), raw("sales_tax"),
                             raw("congestion_surcharge"), raw("airport_fee"), cbd])
        wait = (f"epoch(CAST({col('pu_ts')} AS TIMESTAMP)) - epoch(CAST({col('req')} AS TIMESTAMP))"
                if col("req") != "NULL" else "NULL")
    else:
        charge = f"{raw(sp['total'])} - {raw(sp['tip'])}"
        wait = "NULL"

    start, end = month_range(ym)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE t AS
        SELECT CAST({col('pu_ts')} AS TIMESTAMP) AS pu_ts,
               CAST({col('do_ts')} AS TIMESTAMP) AS do_ts,
               CAST({col('pu')} AS INTEGER) AS pu_zone,
               CAST({col('do')} AS INTEGER) AS do_zone,
               CAST({col('miles')} AS DOUBLE) AS miles,
               CAST({col('base')} AS DOUBLE) AS base,
               CAST(({charge}) AS DOUBLE) AS charge,
               {cbd} AS cbd,
               CAST({col('driver_pay')} AS DOUBLE) AS driver_pay,
               CAST(({wait}) AS DOUBLE) AS wait_sec
        FROM read_parquet('{src}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE c0 AS
        SELECT *,
               epoch(do_ts) - epoch(pu_ts) AS secs,
               NOT COALESCE(pu_ts >= TIMESTAMP '{start}' AND pu_ts < TIMESTAMP '{end}', FALSE) AS f_month,
               COALESCE(do_ts <= pu_ts, TRUE) AS f_time_order,
               COALESCE(epoch(do_ts) - epoch(pu_ts) NOT BETWEEN 60 AND 14400, TRUE) AS f_duration,
               COALESCE(miles <= 0 OR miles > 100, TRUE) AS f_miles,
               COALESCE(base <= 0, TRUE) AS f_base_fare,
               COALESCE(miles / NULLIF((epoch(do_ts) - epoch(pu_ts)) / 3600.0, 0) > 80, FALSE) AS f_speed,
               (pu_zone IS NULL OR do_zone IS NULL) AS f_null_zone
        FROM t
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE c AS
        SELECT *, CASE WHEN f_month OR f_time_order OR f_null_zone THEN 0
                       WHEN f_duration OR f_miles OR f_base_fare OR f_speed THEN 1
                       ELSE 2 END AS tier
        FROM c0
    """)

    # ---- audit row
    dup = "NULL"
    if a.dupes:
        dup = ("(SELECT count(*) - count(DISTINCT (pu_ts, do_ts, pu_zone, do_zone, miles, base)) FROM t)")
    aud = con.execute(f"""
        SELECT '{svc}' AS service, '{ym}' AS ym, count(*) AS rows_raw,
               sum(f_month::INT) AS drop_outside_month, sum(f_time_order::INT) AS drop_time_order,
               sum(f_duration::INT) AS drop_duration, sum(f_miles::INT) AS drop_miles,
               sum(f_base_fare::INT) AS drop_base_fare, sum(f_speed::INT) AS drop_speed,
               sum((tier >= 1)::INT) AS rows_counts_set,
               sum((tier = 2)::INT) AS rows_clean,
               sum(f_null_zone::INT) AS null_zone_rows,
               sum((pu_zone IN (264,265))::INT) AS unknown_pu_zone_rows,
               {has_cbd} AS has_cbd_column,
               sum((cbd > 0)::INT) AS rows_with_cbd_fee,
               min(pu_ts) AS min_pickup, max(pu_ts) AS max_pickup,
               {dup} AS duplicate_like_rows
        FROM c
    """).df()

    # ---- panels (clean rows only)
    con.execute(f"""
        COPY (SELECT '{svc}' AS service, CAST(pu_ts AS DATE) AS date, hour(pu_ts) AS hour,
                     pu_zone AS zone, tier, count(*) AS trips, sum(miles) AS miles, sum(secs) AS secs,
                     sum(base) AS base_fare, sum(charge) AS charge_ex_tip, sum(cbd) AS cbd_fee,
                     sum((cbd > 0)::INT) AS n_fee_trips, sum(driver_pay) AS driver_pay,
                     sum(CASE WHEN wait_sec BETWEEN 0 AND 3600 THEN wait_sec END) AS wait_sec,
                     count(CASE WHEN wait_sec BETWEEN 0 AND 3600 THEN 1 END) AS n_wait
              FROM c GROUP BY 2, 3, 4, 5)
        TO '{panels / f"pickups_{svc}_{ym}.parquet"}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (SELECT '{svc}' AS service, CAST(do_ts AS DATE) AS date, hour(do_ts) AS hour,
                     do_zone AS zone, tier, count(*) AS trips
              FROM c GROUP BY 2, 3, 4, 5)
        TO '{panels / f"dropoffs_{svc}_{ym}.parquet"}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (SELECT '{svc}' AS service, CAST(pu_ts AS DATE) AS date, hour(pu_ts) AS hour,
                     tier, COALESCE(zp.region, 'unknown') AS pu_region, COALESCE(zd.region, 'unknown') AS do_region,
                     count(*) AS trips, sum(base) AS base_fare, sum(cbd) AS cbd_fee, sum(miles) AS miles
              FROM c
              LEFT JOIN zones zp ON c.pu_zone = zp.zone_id
              LEFT JOIN zones zd ON c.do_zone = zd.zone_id
              GROUP BY 2, 3, 4, 5, 6)
        TO '{panels / f"flows_{svc}_{ym}.parquet"}' (FORMAT PARQUET)
    """)
    r = aud.iloc[0]
    print(f"  {svc} {ym}: raw {int(r.rows_raw):,} | counts-set {int(r.rows_counts_set):,} | strict {int(r.rows_clean):,} "
          f"(cbd col: {bool(r.has_cbd_column)}, fee>0 rows: {int(r.rows_with_cbd_fee):,})")
    con.execute("DROP TABLE IF EXISTS c0")
    con.execute("DROP TABLE IF EXISTS c")
    con.execute("DROP TABLE IF EXISTS t")
    return aud


def stage_panels(a):
    import pandas as pd
    con = connect(a)
    load_zones(con, a)
    out = Path(a.out_dir)
    audits = []
    for ym in a.months:
        for svc in a.services:
            r = process_file(con, a, svc, ym)
            if r is not None:
                audits.append(r)
    if audits:
        new = pd.concat(audits)
        f = out / "audit.csv"
        if f.exists():
            old = pd.read_csv(f)
            key = new[["service", "ym"]].apply(tuple, axis=1)
            old = old[~old[["service", "ym"]].apply(tuple, axis=1).isin(set(key))]
            new = pd.concat([old, new])
        new.sort_values(["service", "ym"]).to_csv(f, index=False)
        print(f"Wrote {f}")


# ----------------------------------------------------------------------------- stage: summary
def stage_summary(a):
    con = connect(a)
    load_zones(con, a)
    out = Path(a.out_dir)
    pk = out / "panels" / "pickups_*.parquet"
    fl = out / "panels" / "flows_*.parquet"
    ds = "CROSS JOIN (VALUES ('raw', 0), ('counts', 1), ('strict', 2)) d(dataset, min_tier)"
    con.execute(f"""
        COPY (
          SELECT d.dataset, p.service, strftime(p.date, '%Y-%m') AS ym, COALESCE(z.region, 'unknown') AS region,
                 CASE WHEN dayofweek(p.date) IN (0, 6) THEN 'weekend' ELSE 'weekday' END AS daytype,
                 p.hour, count(DISTINCT p.date) AS days, sum(p.trips) AS trips,
                 sum(p.trips) / NULLIF(count(DISTINCT p.date), 0) AS trips_per_day,
                 sum(p.secs) / NULLIF(sum(p.miles), 0) AS sec_per_mile,
                 sum(p.base_fare) / sum(p.trips) AS avg_base_fare,
                 sum(p.charge_ex_tip) / sum(p.trips) AS avg_charge_ex_tip,
                 sum(p.cbd_fee) / sum(p.trips) AS avg_cbd_fee,
                 sum(p.n_fee_trips) * 1.0 / sum(p.trips) AS share_trips_with_fee
          FROM read_parquet('{pk}') p {ds}
          LEFT JOIN zones z ON p.zone = z.zone_id
          WHERE p.tier >= d.min_tier
          GROUP BY ALL ORDER BY 1, 2, 3, 4, 5, 6
        ) TO '{out / "summary_pickups_by_region_hour.csv"}' (HEADER, DELIMITER ',')
    """)
    con.execute(f"""
        COPY (
          SELECT d.dataset, service, strftime(date, '%Y-%m') AS ym, pu_region, do_region,
                 count(DISTINCT date) AS days, sum(trips) AS trips,
                 sum(trips) / NULLIF(count(DISTINCT date), 0) AS trips_per_day
          FROM read_parquet('{fl}') f {ds}
          WHERE f.tier >= d.min_tier
          GROUP BY ALL ORDER BY 1, 2, 3, 4, 5
        ) TO '{out / "summary_flows_region_pairs.csv"}' (HEADER, DELIMITER ',')
    """)
    con.execute(f"""
        COPY (
          SELECT d.dataset, service, date, COALESCE(z.region,'unknown') AS region, sum(trips) AS trips,
                 sum(base_fare) AS base_fare, sum(cbd_fee) AS cbd_fee
          FROM read_parquet('{pk}') p {ds}
          LEFT JOIN zones z ON p.zone = z.zone_id
          WHERE p.tier >= d.min_tier
          GROUP BY ALL ORDER BY 1, 2, 3, 4
        ) TO '{out / "summary_daily_by_region.csv"}' (HEADER, DELIMITER ',')
    """)
    # Zone-level daily panel (peak / off-peak) for the regressions: zone fixed effects, clustered SEs.
    # Peak = 5am-9pm weekdays, 9am-9pm weekends. trips_counts = tier>=1, *_strict = tier 2 only.
    con.execute(f"""
        COPY (
          SELECT p.service, p.date, p.zone, COALESCE(z.region, 'unknown') AS region,
                 COALESCE(z.is_airport, FALSE) AS is_airport,
                 CASE WHEN (dayofweek(p.date) BETWEEN 1 AND 5 AND p.hour BETWEEN 5 AND 20)
                        OR (dayofweek(p.date) IN (0, 6) AND p.hour BETWEEN 9 AND 20)
                      THEN 'peak' ELSE 'offpeak' END AS period,
                 sum(p.trips) AS trips_counts,
                 sum(CASE WHEN p.tier = 2 THEN p.trips END) AS trips_strict,
                 sum(CASE WHEN p.tier = 2 THEN p.secs END) AS secs_strict,
                 sum(CASE WHEN p.tier = 2 THEN p.miles END) AS miles_strict,
                 sum(CASE WHEN p.tier = 2 THEN p.base_fare END) AS base_fare_strict,
                 sum(CASE WHEN p.tier = 2 THEN p.cbd_fee END) AS cbd_fee_strict,
                 sum(CASE WHEN p.tier = 2 THEN p.n_fee_trips END) AS n_fee_trips_strict
          FROM read_parquet('{pk}') p LEFT JOIN zones z ON p.zone = z.zone_id
          WHERE p.tier >= 1
          GROUP BY ALL ORDER BY 1, 2, 3, 6
        ) TO '{out / "summary_zone_daily.csv"}' (HEADER, DELIMITER ',')
    """)
    print(f"Wrote 4 summary CSVs to {out}/ (3 stack dataset = raw / counts / strict; summary_zone_daily.csv is zone x day x peak). "
          f"If summary_zone_daily.csv is too big to upload, zip it.")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["download", "zones", "panels", "summary", "all"])
    p.add_argument("--months", nargs="+", default=DEFAULT_MONTHS, help="YYYY-MM list")
    p.add_argument("--services", nargs="+", default=ALL_SERVICES, choices=ALL_SERVICES)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--buffer-ft", type=float, default=1000.0,
                   help="outside zones within this distance of the CBD zones are 'buffer'")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory-limit", default="6GB")
    p.add_argument("--dupes", action="store_true", help="also count duplicate-looking rows (slow on HVFHV)")
    p.add_argument("--force", action="store_true", help="rebuild panels even if they exist")
    a = p.parse_args()

    stages = ["download", "zones", "panels", "summary"] if a.stage == "all" else [a.stage]
    for st in stages:
        print(f"== {st}")
        {"download": stage_download, "zones": stage_zones,
         "panels": stage_panels, "summary": stage_summary}[st](a)


if __name__ == "__main__":
    main()