import numpy as np
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.astro import time_representation

# spice.load_standard_kernels()

def make_bodies(
    space_weather_file: str,
    satellite_name: str = "grace_fo",
    mass: float = 600.0,
    reference_area: float = 2.0,
    cd_guess: float = 3.0,
    use_storm_conditions: bool = True,
    use_anomalous_oxygen: bool = True,
):
    bodies_to_create = ["Earth", "Sun", "Moon"]
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation
    )

    body_settings.add_empty_settings(satellite_name)
    #msis atmosphere
    body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00(
        space_weather_file=space_weather_file,
        use_storm_conditions=use_storm_conditions,
        use_anomalous_oxygen=use_anomalous_oxygen,
    )
    
    #solar radiation pressure
    cr_guess = 1.3
    occulting_bodies = ["Earth"] 
    srp_coefficient_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area, cr_guess, occulting_bodies)
    body_settings.get(satellite_name).radiation_pressure_settings = {
        "Sun": srp_coefficient_settings}

    #satellite
    body_settings.get(satellite_name).constant_mass = mass
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
          reference_area,
          [cd_guess, 0.0, 0.0]
        )
    body_settings.get(satellite_name).aerodynamic_coefficient_settings = aero_coefficient_settings 
    bodies = environment_setup.create_system_of_bodies(body_settings)

    return bodies


#==============================================================
def make_propagator_settings(
    bodies,
    initial_state: np.ndarray,
    start_epoch: float,
    end_epoch: float,
    grav_rank: int = 20,
    satellite_name: str = "grace_fo",
):
      
    #acceleration  
    acc_settings_spacecraft = {
    "Earth": [
        propagation_setup.acceleration.spherical_harmonic_gravity(grav_rank,grav_rank),  
        propagation_setup.acceleration.aerodynamic(),         
        ],
    "Sun": [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.cannonball_radiation_pressure()
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
    
    #propagator
    termination_settings = propagation_setup.propagator.time_termination(end_epoch, terminate_exactly_on_final_condition = True)
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
    
    return propagator_settings

#true run =====================================================
def run_forward_simulation(bodies, propagator_settings):
    dyn_sim = simulator.create_dynamics_simulator(bodies, propagator_settings)
    return dyn_sim.state_history
