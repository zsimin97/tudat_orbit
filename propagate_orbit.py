import numpy as np
from dynamics_setup import make_propagator_settings, run_forward_simulation
from tudatpy.astro import time_representation
from tudatpy.estimation import estimation_analysis
from tudatpy.estimation.observable_models_setup import links, model_settings
from tudatpy.estimation.observable_models_setup.model_settings import ObservableType
from tudatpy.estimation.observations_setup import observations_wrapper
from tudatpy import dynamics
from tudatpy.dynamics import simulator

def propagate_orbit(
    start_epoch: float,
    end_epoch: float,
    bodies,
    initial_state: np.ndarray,
    initial_covariance,
    satellite_name="grace-fo",
    grav_rank: int = 20,
    step_size: float = 10.0
):

    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (6,):
        raise ValueError("initial_state must be a 6-element vector")

    # propagator
    propagator_settings = make_propagator_settings(
        bodies=bodies,
        initial_state=initial_state,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        grav_rank=grav_rank,
        step_size=step_size
    )

    parameter_settings = dynamics.parameters_setup.initial_states(propagator_settings, bodies)
    parameter_settings.append(dynamics.parameters_setup.constant_drag_coefficient(satellite_name))
    parameters_to_estimate = dynamics.parameters_setup.create_parameter_set(parameter_settings, bodies)

    variational_solver = simulator.create_variational_equations_solver(
        bodies,
        propagator_settings,
        parameters_to_estimate,
        simulate_dynamics_on_creation=True  
    )

    # propagation
    #state_history = run_forward_simulation(
    #    bodies=bodies,
    #    propagator_settings=propagator_settings,
    #)
    
    state_history = variational_solver.state_history
    stm_history = variational_solver.state_transition_matrix_history

    times = np.array(list(state_history.keys()))
    states = np.array(list(state_history.values()))
    n_steps = len(times)
    covariances = np.zeros((n_steps, 6, 6))
    covariances[0] = initial_covariance
    P = initial_covariance.copy()

    for i in range(1, n_steps):
        STM_i = stm_history[times[i]][:6, :6]
        STM_im1 = stm_history[times[i-1]][:6, :6]
        Phi = STM_i @ np.linalg.inv(STM_im1)
        
        # only Φ
        P = Phi @ P @ Phi.T  # no Q
        print(f"  Position RSS std: {np.sqrt(np.trace(P[:3,:3])):.2f} m")
        print(f"  Velocity RSS std: {np.sqrt(np.trace(P[3:,3:]))*1000:.2f} mm/s")
        covariances[i] = P
    
    print(f"Without SNC Complete")
    return times, states, covariances
    #return state_history

## SNC
def propagate_state_and_covariance(
    start_epoch,
    end_epoch,
    bodies,
    initial_state,
    initial_covariance,
    process_noise,  
    satellite_name="grace-fo",
    grav_rank=20,
    step_size=60.0
):
    
    #Returns:
    #    times: array of epoch times
    #    states: array of states (N x 6)
    #    covariances: array of covariances (N x 6 x 6)
    
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (6,):
        raise ValueError("initial_state must be a 6-element vector")
    print("Initial state (m, m/s):", initial_state)
    
    # propagator
    propagator_settings = make_propagator_settings(
        bodies=bodies,
        initial_state=initial_state,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        grav_rank=grav_rank,
        step_size=step_size,
        satellite_name=satellite_name,
    )

    parameter_settings = dynamics.parameters_setup.initial_states(propagator_settings, bodies)
    parameter_settings.append(dynamics.parameters_setup.constant_drag_coefficient(satellite_name))
    parameters_to_estimate = dynamics.parameters_setup.create_parameter_set(parameter_settings, bodies)

    variational_solver = simulator.create_variational_equations_solver(
        bodies,
        propagator_settings,
        parameters_to_estimate,
        simulate_dynamics_on_creation=True  
    )
    state_history = variational_solver.state_history
    stm_history = variational_solver.state_transition_matrix_history

    times = np.array(list(state_history.keys()))
    states = np.array(list(state_history.values()))
    n_steps = len(times)
    covariances = np.zeros((n_steps, 6, 6))
    covariances[0] = initial_covariance
    P = initial_covariance.copy()
    print(f"Propagating covariance for {n_steps} steps")

    for i in range(1, n_steps):
        print(f"\n===== Step {i}/{n_steps-1} =====")

        dt = times[i] - times[i-1]
        print(f"dt = {dt:.3f} seconds")

        # STM_i: t0 to ti, Φ(ti​,t0​)
        # STM_im1 : t0 to  ti-1, Φ(ti−1​,t0​)
        STM_i = stm_history[times[i]][:6, :6]  # translational state
        STM_im1 = stm_history[times[i-1]][:6, :6]
        #print("STM_i:")
        #print(STM_i)
        #print("STM_im1:")
        #print(STM_im1)     
        #print(f"STM_i norm: {np.linalg.norm(STM_i):.6f}")
        #print(f"STM_im1 norm: {np.linalg.norm(STM_im1):.6f}")
        
        # Phi : ti-1 to ti
        # Φ(ti​,ti−1​)=Φ(ti​,t0​)Φ(ti−1​,t0​)−1
        Phi = STM_i @ np.linalg.inv(STM_im1)
        #print("Phi:")
        #print(Phi)
        #print(f"Phi norm: {np.linalg.norm(Phi):.6f}")
        print(f"Phi det: {np.linalg.det(Phi):.6f}")

        Q = compute_process_noise_matrix(dt, process_noise)
        #print("Q:")
        #print(Q)
        print(f"Q trace: {np.trace(Q):.2e}")
        print(f"Q[0,0] (position): {Q[0,0]:.2e}")
        print(f"Q[3,3] (velocity): {Q[3,3]:.2e}")

        print(f"P (before update):")
        #print("P:")
        #print(P)
        print(f"  Position variancwoxiane trace: {np.trace(P[:3,:3]):.2e}")
        print(f"  Velocity variance trace: {np.trace(P[3:,3:]):.2e}")
        print(f"  Position RSS std: {np.sqrt(np.trace(P[:3,:3])):.2f} m")
        print(f"  Velocity RSS std: {np.sqrt(np.trace(P[3:,3:]))*1000:.2f} mm/s")

        P = Phi @ P @ Phi.T + Q
        print(f"P (after update):")
        #print("P:")
        #print(P)
        print(f"  Position variance trace: {np.trace(P[:3,:3]):.2e}")
        print(f"  Velocity variance trace: {np.trace(P[3:,3:]):.2e}")
        print(f"  Position RSS std: {np.sqrt(np.trace(P[:3,:3])):.2f} m")
        print(f"  Velocity RSS std: {np.sqrt(np.trace(P[3:,3:]))*1000:.2f} mm/s")

        covariances[i] = P
        print(f"Stored covariance at index {i}")

        if i % 1000 == 0:
            pos_rss = np.sqrt(np.trace(P[:3, :3]))
            print(f"  Step {i}/{n_steps}: Position RSS std = {pos_rss:.2f} m")

        print(f"Total steps propagated: {n_steps-1}")
    
    return times, states, covariances
    #return state_history, stm_history



def compute_process_noise_matrix(dt, process_noise):

    #Q = [dt³/3·I₃,  dt²/2·I₃] σ_a²
    #    [dt²/2·I₃,  dt·I₃   ]
    
    #Args:
    #    dt: time step (seconds)
    #    process_noise: process noise acceleration (m/s²)
    
    #Returns:
    #    Q: 6x6 process noise covariance matrix

    Q = np.zeros((6, 6))
    sigma_sq = process_noise**2
    
    # Position-Position block
    Q[0:3, 0:3] = (dt**3 / 3) * sigma_sq * np.eye(3)
    
    # Position-Velocity cross terms
    Q[0:3, 3:6] = (dt**2 / 2) * sigma_sq * np.eye(3)
    Q[3:6, 0:3] = (dt**2 / 2) * sigma_sq * np.eye(3)
    
    # Velocity-Velocity block
    Q[3:6, 3:6] = dt * sigma_sq * np.eye(3)
    
    return Q