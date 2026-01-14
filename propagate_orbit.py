from datetime import datetime, timezone
import numpy as np
from tudatpy.astro import time_representation
from tudatpy.interface import spice

from POD_readin import build_pod_from_sp3
from dynamics_setup import make_bodies, make_propagator_settings, run_forward_simulation
from POD_readin import build_pod_from_sp3, read_sp3_pv

spice.load_standard_kernels()

#read in orbit ========================================================
#02-18 22:00 ──────────── 02-19 12:00
#                02-19 10:00 ──────────── 02-20 00:00
#                                02-19 22:00 ──────────── 02-20 12:00
sp3_files = [
    "poddata/GFZOP_RSO_L65_G_20240218_220000_20240219_120000_v03.sp3",
    "poddata/GFZOP_RSO_L65_G_20240219_100000_20240220_000000_v03.sp3",
    "poddata/GFZOP_RSO_L65_G_20240219_220000_20240220_120000_v03.sp3",
]
start = datetime(2024, 2, 19, 00, 00, 0, tzinfo=timezone.utc)
end   = datetime(2024, 2, 20, 1,  0, 0, tzinfo=timezone.utc)

t_itrs, pos_itrs, vel_itrs = read_sp3_pv(
    sp3_files, start, end, sat_id="L65")
t_gcrs, pos_gcrs, vel_gcrs = build_pod_from_sp3(
    sp3_files, start, end, sat_id="L65")
t_itrs   = np.array(t_itrs)
pos_itrs = np.array(pos_itrs)
vel_itrs = np.array(vel_itrs)
print("ITRS POD_direct:",t_itrs.shape, pos_itrs.shape, vel_itrs.shape)
print("first_itrs:", t_itrs[0], pos_itrs[0], vel_itrs[0])
print("GCRS:",t_gcrs.shape, pos_gcrs.shape, vel_gcrs.shape)
print("first_gcrs:", t_gcrs[0], pos_gcrs[0], vel_gcrs[0])
print("**************************************")

start_epoch = time_representation.date_time_components_to_epoch(
    start.year, start.month, start.day,
    start.hour, start.minute,
    start.second + start.microsecond * 1e-6
)
end_epoch = float(t_gcrs[-1]) + 1.0
initial_state = np.hstack((pos_gcrs[0], vel_gcrs[0]))
print("Start epoch (J2000 s):", start_epoch)
print("Initial state (m, m/s):", initial_state)
print("||r|| initial (km):", np.linalg.norm(pos_gcrs[0]) / 1e3)
print("||v|| initial (km/s):", np.linalg.norm(vel_gcrs[0]) / 1e3)
print("**************************************")

#dynamics_setup =====================================================
bodies = make_bodies(
    space_weather_file="sw19571001.txt",
    satellite_name="grace_fo",
    cd_guess=3.0,
)
propagator_settings = make_propagator_settings(
    bodies, initial_state, start_epoch, end_epoch, satellite_name="grace_fo"
)
state_history = run_forward_simulation(bodies, propagator_settings)

final_epoch  = list(state_history.keys())[-1]
final_state  = state_history[final_epoch]
print("Final epoch:", final_epoch)
print("||r|| (km):", np.linalg.norm(final_state[:3]) / 1e3)
print("||v|| (km/s):", np.linalg.norm(final_state[3:6]) / 1e3)

#np.savez("forward_state_history.npz",
#         t=np.array(list(state_history.keys())),
#         x=np.array(list(state_history.values())))