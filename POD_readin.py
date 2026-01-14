from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    CartesianRepresentation, CartesianDifferential,
    ITRS, GCRS
)
from tudatpy.interface import spice

def read_sp3_pv(
        sp3_path: Iterable[str | Path],
        start_utc: datetime,
        end_utc: datetime,
        sat_id: str
) -> Tuple[List[datetime],List[Tuple[float,float,float]],List[Tuple[float,float,float]]]:
    
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("Should be UTC time with tzinfo")
    start_utc = start_utc.astimezone(timezone.utc)
    end_utc   = end_utc.astimezone(timezone.utc)
    if end_utc < start_utc:
        raise ValueError("end_utc should >= start_utc")
    
    sat_id =sat_id.strip()
    p_tag = "P" + sat_id  # "PL65"
    v_tag = "V" + sat_id  # "VL65"
    vel_scale = 0.1

    def parse_epoch(line: str) -> datetime:
        parts = line.strip().split()
        yy,mo,dd,hh,mm = map(int, parts[1:6])
        sec = float(parts[6])
        whole = int(sec)
        micro = int(round((sec-whole)*1e6))
        return datetime(yy,mo,dd,hh,mm,whole,micro,tzinfo=timezone.utc)
    
    buf: Dict[datetime, Dict[str, Tuple[float, float, float]]] = {}

    for path in map(Path, sp3_path):
        current_epoch: Optional[datetime] = None
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line:
                    continue
                if line.startswith("*"):
                    current_epoch = parse_epoch(line)
                    continue
                if current_epoch is None:
                    continue
                if current_epoch < start_utc or current_epoch > end_utc:
                    continue

                if line.startswith(p_tag) or line.startswith(v_tag):
                    parts = line.split()
                    tag = parts[0]
                    x,y,z = map(float, parts[1:4])

                    if any(abs(v) >= 999999.999999 for v in (x,y,z)):
                        continue
                    if current_epoch not in buf:
                        buf[current_epoch] = {}
                    if tag.startswith("P"):
                        buf[current_epoch]["r_m"] = (x*1000.0, y*1000.0, z*1000.0)  #km->m
                    else:
                        buf[current_epoch]["v_mps"] = (x * vel_scale, y * vel_scale, z * vel_scale) #dm/s->m/s

    epochs = sorted(t for t, d in buf.items() if ("r_m" in d and "v_mps" in d))
    times_utc = epochs
    r_itrs_m = [buf[t]["r_m"] for t in epochs]
    v_itrs_mps= [buf[t]["v_mps"] for t in epochs]
    return times_utc, r_itrs_m, v_itrs_mps
                    

def build_pod_from_sp3(
        sp3_paths,
        start_utc: datetime,
        end_utc: datetime,
        sat_id: str
):
    times_utc, r_itrs_m, v_itrs_mps = read_sp3_pv(sp3_paths, start_utc, end_utc, sat_id=sat_id)

    N = len(times_utc)
    t_gcrs = np.zeros(N)
    pos_gcrs = np.zeros((N,3))
    vel_gcrs = np.zeros((N,3))

    spice.load_standard_kernels()

    for k in range(N):
        t_ast = Time(times_utc[k], scale="utc")
        pos_itrs = CartesianRepresentation(
            r_itrs_m[k][0] * u.m,
            r_itrs_m[k][1] * u.m,
            r_itrs_m[k][2] * u.m
        )
        vel_itrs = CartesianDifferential(
            v_itrs_mps[k][0] * u.m/u.s,
            v_itrs_mps[k][1] * u.m/u.s,
            v_itrs_mps[k][2] * u.m/u.s
        )
        state_itrs = pos_itrs.with_differentials(vel_itrs) #state_itrs = (pos, vel)
        coord_itrs = ITRS(state_itrs,obstime=t_ast)

        #to GCRS
        coord_gcrs = coord_itrs.transform_to(GCRS(obstime=t_ast))
        r_gcrs = coord_gcrs.cartesian.xyz.to(u.m).value
        v_gcrs = coord_gcrs.velocity.d_xyz.to(u.m/u.s).value

        pos_gcrs[k,:] = r_gcrs
        vel_gcrs[k,:] = v_gcrs
        t_str = t_ast.utc.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        t_gcrs[k] = spice.convert_date_string_to_ephemeris_time(t_str)

    return t_gcrs, pos_gcrs, vel_gcrs


