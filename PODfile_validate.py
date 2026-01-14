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
end   = datetime(2024, 2, 19, 6,  0, 0, tzinfo=timezone.utc)

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
assert pos_itrs.shape == (N, 3)
assert vel_itrs.shape == (N, 3)
assert pos_gcrs.shape == (N, 3)
assert vel_gcrs.shape == (N, 3)

dr_gcrs_list = []
dv_gcrs_list = []
dr_round_list = []
dv_round_list = []

# also check norm consistency
dnr_list = []
dnv_list = []

for k in range(N):
    # Time object
    t_ast = Time(t_itrs[k], scale="utc")

    # Build ITRS state from your direct POD (already SI)
    rep = CartesianRepresentation(pos_itrs[k, 0] * u.m,
                                  pos_itrs[k, 1] * u.m,
                                  pos_itrs[k, 2] * u.m)
    diff = CartesianDifferential(vel_itrs[k, 0] * u.m/u.s,
                                 vel_itrs[k, 1] * u.m/u.s,
                                 vel_itrs[k, 2] * u.m/u.s)
    state_itrs = rep.with_differentials(diff)
    c_itrs = ITRS(state_itrs, obstime=t_ast)

    # Forward: ITRS -> GCRS (fresh)
    c_gcrs = c_itrs.transform_to(GCRS(obstime=t_ast))
    r_g = c_gcrs.cartesian.xyz.to(u.m).value
    v_g = c_gcrs.velocity.d_xyz.to(u.m/u.s).value

    # Compare "fresh" GCRS vs your stored GCRS
    dr_gcrs_list.append(r_g - pos_gcrs[k])
    dv_gcrs_list.append(v_g - vel_gcrs[k])

    # Round-trip: GCRS -> ITRS and compare back to input ITRS
    c_back = c_gcrs.transform_to(ITRS(obstime=t_ast))
    r_back = c_back.cartesian.xyz.to(u.m).value
    v_back = c_back.velocity.d_xyz.to(u.m/u.s).value
    dr_round_list.append(r_back - pos_itrs[k])
    dv_round_list.append(v_back - vel_itrs[k])

    # Norm consistency (should be ~0 within numerical tolerance)
    dnr_list.append(np.linalg.norm(r_g) - np.linalg.norm(pos_itrs[k]))
    dnv_list.append(np.linalg.norm(v_g) - np.linalg.norm(vel_itrs[k]))

dr_gcrs = np.array(dr_gcrs_list)
dv_gcrs = np.array(dv_gcrs_list)
dr_round = np.array(dr_round_list)
dv_round = np.array(dv_round_list)
dnr = np.array(dnr_list)
dnv = np.array(dnv_list)

def rms(x):
    return np.sqrt(np.mean(np.sum(x*x, axis=1)))

print("\n========== Frame Transform Validation ==========")

print("A) Fresh GCRS vs stored GCRS (should be ~0 if pipeline consistent)")
print("   max |dr| (m):  ", np.max(np.linalg.norm(dr_gcrs, axis=1)))
print("   rms |dr| (m):  ", rms(dr_gcrs))
print("   max |dv| (m/s):", np.max(np.linalg.norm(dv_gcrs, axis=1)))
print("   rms |dv| (m/s):", rms(dv_gcrs))

print("\nB) Round-trip ITRS -> GCRS -> ITRS (should return to original)")
print("   max |dr| (m):  ", np.max(np.linalg.norm(dr_round, axis=1)))
print("   rms |dr| (m):  ", rms(dr_round))
print("   max |dv| (m/s):", np.max(np.linalg.norm(dv_round, axis=1)))
print("   rms |dv| (m/s):", rms(dv_round))

print("\nC) Norm checks (not required, but sanity)")
print("   max | |r_gcrs|-|r_itrs| | (m):", np.max(np.abs(dnr)))
print("   max | |v_gcrs|-|v_itrs| | (m/s):", np.max(np.abs(dnv)))

# Optional: print a few worst cases
idxA = np.argmax(np.linalg.norm(dr_gcrs, axis=1))
idxB = np.argmax(np.linalg.norm(dr_round, axis=1))
print("\nWorst-case index (fresh-vs-stored):", idxA, "UTC:", t_itrs[idxA])
print("  dr (m):", dr_gcrs[idxA], " |dr|=", np.linalg.norm(dr_gcrs[idxA]))
print("  dv (m/s):", dv_gcrs[idxA], " |dv|=", np.linalg.norm(dv_gcrs[idxA]))

print("\nWorst-case index (round-trip):", idxB, "UTC:", t_itrs[idxB])
print("  dr (m):", dr_round[idxB], " |dr|=", np.linalg.norm(dr_round[idxB]))
print("  dv (m/s):", dv_round[idxB], " |dv|=", np.linalg.norm(dv_round[idxB]))

print("===============================================\n")