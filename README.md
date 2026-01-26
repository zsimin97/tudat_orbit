# tudat_orbit for TudatPy-based scripts


## poddata
- The current POD data is from GFZ RSO version, SP3 format.
- L65 for GRACE-FO 1.
- coordinate: CTS, Conventional Terrestrial System (belong to ITRS, ECEF, Earth-Centered, Earth-Fixed)  
- time line: yyyy mm dd hh mm 0.000
- POD data: position(km) and velocity(dm/s), not orbit elements.


## scripts

- `POD_readin.py`
   read in POD data and transformat coordinate
   - read_sp3_pv: read in given time position(r) and velocity(v) from the SP3 file 
      - input: data path; start_time; end_time; satellite_id
      - output(Tuple|List): time_epoch; r_itrs(m); v_itrs(m/s);  
   - build_pod_from_sp3: coorinate transformation ITRF93 -> J2000 
      - input(Tuple|List, read_sp3_pv output): time_epoch; r_itrs(m); v_itrs(m/s); 
      - output:t_gcrs, pos_gcrs, vel_gcrs

- `POD_validate.py` validate coordinate transformation

- `dynamics_setup.py`
   Defines reusable functions to construct the dynamical system used in Tudat
   - make_bodies: return bodies
   - make_propagator_settings: return propagator_settings
   - run_forward_simulation: return dyn_sim.state_history

- `propagate_orbit.py`
   forward orbit propagation using Tudat, not def subroutine
   - workflow:
      - Read POD data and construct initial state `POD_readin.py`
      - Set up dynamical environment and force models `dynamics_setup.py`
      - Propagate orbit forward in time `dynamics_setup.py`
      - Output propagated state history for diagnostics

- `estimate_cd.py`
   - estimate_cd_and_state
      - input: sp3_files, start_time, end_time, cd_guess, sat_id
      - output: initial_vector, final_vector (vector: initial states (6), Cd (1))

- `CD_estimate.py` No longer in use, split into dynamics_setup.py and estimate_cd.py.
   
