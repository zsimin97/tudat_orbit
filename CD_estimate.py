from datetime import datetime, timezone
from POD_readin import build_pod_from_sp3, read_sp3_pv
import numpy as np
import inspect

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.astro import time_representation
from tudatpy import numerical_simulation
from tudatpy.math import interpolators as interp

from tudatpy import dynamics
from tudatpy.dynamics import environment_setup, propagation_setup, simulator

from tudatpy import estimation
from tudatpy.estimation import estimation_analysis
from tudatpy.estimation import observable_models_setup, observations,observations_setup
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation.observable_models_setup.model_settings import ObservableType
from tudatpy.estimation.observations_setup import observations_wrapper

spice.load_standard_kernels()

#read in orbit 
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
print("**************************************")
#set arc start==========================================================================
#window = 6
start_epoch = time_representation.date_time_components_to_epoch(
    start.year, start.month, start.day,
    start.hour, start.minute,
    start.second + start.microsecond * 1e-6
)
#end_epoch   = start_epoch + window * 60.0 *60.0
end_epoch = float(t_gcrs[-1]) + 1.0
initial_state = np.hstack((pos_gcrs[0], vel_gcrs[0]))
print("Start epoch (J2000 s):", start_epoch)
print("Initial state (m, m/s):", initial_state)
print("||r|| initial (km):", np.linalg.norm(pos_gcrs[0]) / 1e3)
print("||v|| initial (km/s):", np.linalg.norm(vel_gcrs[0]) / 1e3)
print("**************************************")

#set accelaration====================================================================
#Create default body settings
bodies_to_create = ["Earth", "Sun", "Moon"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation
)

#msis atmosphere
body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00(
    space_weather_file="sw19571001.txt",
    use_storm_conditions=True,
     use_anomalous_oxygen=True,
)

#satellite
satellite_name = "grace_fo"
body_settings.add_empty_settings(satellite_name)
body_settings.get(satellite_name).constant_mass = 600.0
reference_area = 2.0      
cd              = 3 #cd_value      
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
          reference_area,
          [cd, 0.0, 0.0]
    )
body_settings.get(satellite_name).aerodynamic_coefficient_settings = aero_coefficient_settings 
bodies = environment_setup.create_system_of_bodies(body_settings)

#acceleration setting
acc_settings_spacecraft = {
    "Earth": [
        propagation_setup.acceleration.spherical_harmonic_gravity(20,20),  
        propagation_setup.acceleration.aerodynamic(),         
        ],
    "Sun": [
        propagation_setup.acceleration.point_mass_gravity()
        ],
    "Moon": [
        propagation_setup.acceleration.point_mass_gravity()
        ]
}
acceleration_settings = {satellite_name: acc_settings_spacecraft}
central_bodies       = ["Earth"]
bodies_to_propagate  = [satellite_name]
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies
)

#propagation======================================================================
termination_settings = propagation_setup.propagator.time_termination(end_epoch)
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
    initial_time_step=60.0, coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    start_epoch,
    integrator_settings,
    termination_settings,
)

# simulation================================================================
simulator = simulator.create_dynamics_simulator(
   bodies, propagator_settings
)

state_history = simulator.state_history
final_epoch  = list(state_history.keys())[-1]
final_state  = state_history[final_epoch]
    
print("Final epoch:", end_epoch)
print("Final state simulated (m, m/s):", final_state)
print("||r|| final simulated (km):", np.linalg.norm(final_state[:3]) / 1e3)
print("||v|| final simulated (km/s):", np.linalg.norm(final_state[3:6]) / 1e3)

N = len(t_gcrs)
end_state = np.hstack((pos_gcrs[N-1], vel_gcrs[N-1]))
print("Final epoch:", t_gcrs[N-1])
print("Final state pod (m, m/s):", end_state)
print("||r|| final pod (km):", np.linalg.norm(pos_gcrs[N-1]) / 1e3)
print("||v|| final pod (km/s):", np.linalg.norm(vel_gcrs[N-1]) / 1e3)
print("*******************************************")

# observation=============================================================
link_ends = {links.observed_body: links.body_origin_link_end_id(satellite_name)}
link_definition = links.LinkDefinition(link_ends)
observation_settings_list = [
    model_settings.cartesian_position(link_definition),
    model_settings.cartesian_velocity(link_definition)
]
N = pos_gcrs.shape[0]
obs_times = [time_representation.Time(float(t)) for t in t_gcrs]
print(ObservableType, ObservableType.__module__)
print(type(obs_times[0]), obs_times[0])
pos_seq = [np.asarray(pos_gcrs[k, :], dtype=np.float64).reshape(3, 1)
           for k in range(pos_gcrs.shape[0])]
vel_seq = [np.asarray(vel_gcrs[k, :], dtype=np.float64).reshape(3, 1)
           for k in range(vel_gcrs.shape[0])]
existing_observations = {
    ObservableType.position_observable_type: (link_ends, (pos_seq, obs_times)),
    ObservableType.velocity_observable_type: (link_ends, (vel_seq, obs_times)),
}
observation_collection = observations_wrapper.set_existing_observations(
    existing_observations,
    links.observed_body
)


#estimator ========================================================
parameter_settings = dynamics.parameters_setup.initial_states(propagator_settings, bodies)
parameter_settings.append(dynamics.parameters_setup.constant_drag_coefficient(satellite_name))
parameter_to_estimate = dynamics.parameters_setup.create_parameter_set(
    parameter_settings, bodies)
print("param size:", parameter_to_estimate.parameter_vector.size)
print("param vector (initial):", parameter_to_estimate.parameter_vector)
print("Cd guess:", cd)  

estimator = estimation_analysis.Estimator(
    bodies,
    parameter_to_estimate,
    observation_settings_list,
    propagator_settings
)

#prior
sigma_r  = 0.01 # m  1 centimeter
sigma_v  = 0.01      # m/s  5-10centi/s
sigma_cd = 0.2       # Cd 
P0 = np.diag([sigma_r**2]*3 + [sigma_v**2]*3 + [sigma_cd**2])
invP0 = np.linalg.inv(P0)
estimation_input = estimation_analysis.EstimationInput(observation_collection, invP0)

#weight
sigma_pos = 1.0 # m
w = 1.0 / (sigma_pos**2)
observation_collection.set_constant_weight(w)

estimation_input.define_estimation_settings(
    reintegrate_variational_equations=True,
    reintegrate_equations_on_first_iteration=True
)

estimation_output = estimator.perform_estimation(estimation_input)
print("Estimated parameters:", estimation_output.final_parameters)

print(f'Estimated state and parameters:\n\n {parameter_to_estimate.parameter_vector}\n')
