from datetime import datetime, timezone
from POD_readin import build_pod_from_sp3, read_sp3_pv
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    CartesianRepresentation, CartesianDifferential,
    ITRS, GCRS
)
sp3_files = [
    "poddata/GFZOP_RSO_L65_G_20240218_220000_20240219_120000_v03.sp3",
    "poddata/GFZOP_RSO_L65_G_20240219_100000_20240220_000000_v03.sp3",
    "poddata/GFZOP_RSO_L65_G_20240219_220000_20240220_120000_v03.sp3",
]
start = datetime(2024, 2, 19, 00, 00, 0, tzinfo=timezone.utc)
end   = datetime(2024, 2, 19, 00, 3, 0, tzinfo=timezone.utc)

t_itrs, pos_itrs, vel_itrs = read_sp3_pv(
    sp3_files, start, end, sat_id="L65"
)
t_gcrs, pos_gcrs, vel_gcrs = build_pod_from_sp3(
    sp3_files, start, end, sat_id="L65"
)

t_itrs   = np.array(t_itrs)
pos_itrs = np.array(pos_itrs)
vel_itrs = np.array(vel_itrs)
print("ITRS POD_direct:",t_itrs.shape, pos_itrs.shape, vel_itrs.shape)
print("first_itrs:", t_itrs[0], pos_itrs[0], vel_itrs[0])

print("GCRS:",t_gcrs.shape, pos_gcrs.shape, vel_gcrs.shape)
print("first_gcrs:", t_gcrs[0], pos_gcrs[0], vel_gcrs[0])


N = len(t_itrs)
