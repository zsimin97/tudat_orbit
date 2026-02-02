from datetime import datetime, timezone
import numpy as np
from tudatpy.astro import time_representation
from tudatpy.interface import spice

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
end   = datetime(2024, 2, 20, 00,  0, 0, tzinfo=timezone.utc)

# ONLY need INITIAL state
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

start_epoch = float(t_gcrs[0])
end_epoch = float(t_gcrs[-1]) 
initial_state = np.hstack((pos_gcrs[0], vel_gcrs[0]))
print("Observed Propagator initial")
print("Start epoch (J2000 s):", start_epoch)
print("Initial state (m, m/s):", initial_state)
print("||r|| initial (km):", np.linalg.norm(pos_gcrs[0]) / 1e3)
print("||v|| initial (km/s):", np.linalg.norm(vel_gcrs[0]) / 1e3)
print("**************************************")

#dynamics_setup =====================================================
bodies = make_bodies(
    space_weather_file="sw19571001.txt",
    satellite_name="grace_fo",
    cd_guess=1.99,
)

grav_rank= 120
propagator_settings = make_propagator_settings(
    bodies, initial_state, start_epoch, end_epoch, grav_rank, satellite_name="grace_fo"
)
state_history = run_forward_simulation(bodies, propagator_settings)

#validate============================================
prop_init = list(state_history.keys())[0]
prop_init_state  = state_history[prop_init]
print("Simulated Propagator initial")
print("Start epoch:", prop_init)
print("Start state (m, m/s):", state_history[prop_init])
print("||r|| (km):", np.linalg.norm(prop_init_state[:3]) / 1e3)
print("||v|| (km/s):", np.linalg.norm(prop_init_state[3:6]) / 1e3)
print("**************************************")

final_epoch  = list(state_history.keys())[-1]
final_state  = state_history[final_epoch]
print("Simulated Propagator")
print("Final epoch:", final_epoch)
print("Final state (m, m/s):", final_state)
print("||r|| (km):", np.linalg.norm(final_state[:3]) / 1e3)
print("||v|| (km/s):", np.linalg.norm(final_state[3:6]) / 1e3)
print("**************************************")
print("Observation POD")
print("Final epoch:", end_epoch)
print("Final state (m, m/s):", np.hstack((pos_gcrs[-1], vel_gcrs[-1])))
print("||r|| (km):", np.linalg.norm(pos_gcrs[-1]) / 1e3)
print("||v|| (km/s):", np.linalg.norm(vel_gcrs[-1]) / 1e3)

#np.savez("forward_state_history.npz",
#         t=np.array(list(state_history.keys())),
#         x=np.array(list(state_history.values())))

final_epoch = list(state_history.keys())[-1]
final_state_sim = np.array(state_history[final_epoch])
final_state_pod = np.hstack((pos_gcrs[-1], vel_gcrs[-1]))
delta_r_vec = final_state_sim[:3] - final_state_pod[:3]
dist_error = np.linalg.norm(delta_r_vec)

r_pod = final_state_pod[:3]
v_pod = final_state_pod[3:]
e_r = r_pod / np.linalg.norm(r_pod)                       # Radial
e_c = np.cross(r_pod, v_pod) / np.linalg.norm(np.cross(r_pod, v_pod)) # Cross-track
e_i = np.cross(e_c, e_r)             # In-track
# project
error_r = np.dot(delta_r_vec, e_r)
error_i = np.dot(delta_r_vec, e_i)
error_c = np.dot(delta_r_vec, e_c)
print("**************************************")
print("Observation POD Final State Check:")
print("Final Position Error (Total):", dist_error, "m")
print(f"Error (m): R: {error_r:.3f}, I: {error_i:.3f}, C: {error_c:.3f}")
print("**************************************")

# whole arc RMS
t_sim = np.array(list(state_history.keys()))
states_sim = np.array(list(state_history.values()))
errors_pos = []
errors_vel = []

for i in range(len(t_gcrs)):
    target_t = t_gcrs[i]
    idx = np.argmin(np.abs(t_sim - target_t))
    
    if np.abs(t_sim[idx] - target_t) < 0.1:
        diff_state = states_sim[idx] - np.hstack((pos_gcrs[i], vel_gcrs[i]))
        errors_pos.append(np.linalg.norm(diff_state[:3]))
        errors_vel.append(np.linalg.norm(diff_state[3:6]))

errors_pos = np.array(errors_pos)
errors_vel = np.array(errors_vel)
rms_pos = np.sqrt(np.mean(errors_pos**2))
rms_vel = np.sqrt(np.mean(errors_vel**2))
max_pos = np.max(errors_pos)

print("********** Full Arc RMS **********")
print(f"obs points: {len(errors_pos)}")
print(f"position RMS: {rms_pos:.4f} m")
print(f"velocity RMS: {rms_vel:.6f} m/s")
print(f"Max: {max_pos:.4f} m")
print("***************************************************")