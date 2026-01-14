from datetime import datetime, timezone
import numpy as np

from tudatpy.astro import time_representation
from tudatpy.estimation import estimation_analysis
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation.observable_models_setup.model_settings import ObservableType
from tudatpy.estimation.observations_setup import observations_wrapper
from tudatpy import dynamics

from POD_readin import build_pod_from_sp3
from dynamics_setup import make_bodies, make_propagator_settings, run_forward_simulation
from POD_readin import build_pod_from_sp3, read_sp3_pv

# POD / observations===========================================
sp3_files = [
    "poddata/GFZOP_RSO_L65_G_20240218_220000_20240219_120000_v03.sp3",
    "poddata/GFZOP_RSO_L65_G_20240219_100000_20240220_000000_v03.sp3",
    "poddata/GFZOP_RSO_L65_G_20240219_220000_20240220_120000_v03.sp3",
]
start = datetime(2024, 2, 19, 0, 0, 0, tzinfo=timezone.utc)
end   = datetime(2024, 2, 20, 1, 0, 0, tzinfo=timezone.utc)

t_gcrs, pos_gcrs, vel_gcrs = build_pod_from_sp3(sp3_files, start, end, sat_id="L65")
t_gcrs = np.asarray(t_gcrs)
pos_gcrs = np.asarray(pos_gcrs)
vel_gcrs = np.asarray(vel_gcrs)

obs_times = [time_representation.Time(float(t)) for t in t_gcrs]
pos_seq = [np.asarray(pos_gcrs[k, :], dtype=np.float64).reshape(3, 1) for k in range(pos_gcrs.shape[0])]
vel_seq = [np.asarray(vel_gcrs[k, :], dtype=np.float64).reshape(3, 1) for k in range(vel_gcrs.shape[0])]

satellite_name = "grace_fo"
link_ends = {links.observed_body: links.body_origin_link_end_id(satellite_name)}
link_definition = links.LinkDefinition(link_ends)

observation_settings_list = [
    model_settings.cartesian_position(link_definition),
    model_settings.cartesian_velocity(link_definition),
]

existing_observations = {
    ObservableType.position_observable_type: (link_ends, (pos_seq, obs_times)),
    ObservableType.velocity_observable_type: (link_ends, (vel_seq, obs_times)),
}
observation_collection = observations_wrapper.set_existing_observations(
    existing_observations,
    links.observed_body
)

# dynamics / propagator setup ==============================
start_epoch = time_representation.date_time_components_to_epoch(
    start.year, start.month, start.day,
    start.hour, start.minute,
    start.second + start.microsecond * 1e-6
)
end_epoch = float(t_gcrs[-1]) + 1.0
initial_state = np.hstack((pos_gcrs[0], vel_gcrs[0]))

cd_guess = 3.0
bodies = make_bodies(
    space_weather_file="sw19571001.txt",
    satellite_name=satellite_name,
    cd_guess=cd_guess,
)

propagator_settings = make_propagator_settings(
    bodies, initial_state, start_epoch, end_epoch, satellite_name=satellite_name
)

# --- estimation ---
parameter_settings = dynamics.parameters_setup.initial_states(propagator_settings, bodies)
parameter_settings.append(dynamics.parameters_setup.constant_drag_coefficient(satellite_name))
parameter_to_estimate = dynamics.parameters_setup.create_parameter_set(parameter_settings, bodies)

estimator = estimation_analysis.Estimator(
    bodies,
    parameter_to_estimate,
    observation_settings_list,
    propagator_settings
)

# prior
sigma_r  = 0.01
sigma_v  = 0.01
sigma_cd = 0.2
P0 = np.diag([sigma_r**2]*3 + [sigma_v**2]*3 + [sigma_cd**2])
invP0 = np.linalg.inv(P0)
estimation_input = estimation_analysis.EstimationInput(observation_collection, invP0)

# weight
sigma_pos = 1.0
w = 1.0 / (sigma_pos**2)
observation_collection.set_constant_weight(w)

estimation_input.define_estimation_settings(
    reintegrate_variational_equations=True,
    reintegrate_equations_on_first_iteration=True
)

estimation_output = estimator.perform_estimation(estimation_input)
print("Estimated parameters:", estimation_output.final_parameters)
print("Final parameter vector:\n", parameter_to_estimate.parameter_vector)