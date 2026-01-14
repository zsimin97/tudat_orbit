#tudat_orbit for TudatPy-based scripts

The current POD data is from GFZ RSO version, SP3 format.
L65 for GRACE-FO 1.
coordinate: ITRS, international terrestrial reference system
            (belong to ECEF, Earth-Centered, Earth-Fixed)  
time line: yyyy mm dd hh mm 0.000
POD data: position(km) and velocity(dm/s), not orbit elements.

POD_readin.py: read in POD data and transformat coordinate
   read_sp3_pv: read in given time position(r) and velocity(v) from the SP3 file 
               input: data path; start_time; end_time; satellite_id
               output(Tuple|List): time_epoch; r_itrs(m); v_itrs(m/s);  
   build_pod_from_sp3: coorinate transformation ITRS -> GCRS 
               input(Tuple|List, read_sp3_pv output): time_epoch; r_itrs(m); v_itrs(m/s); 
               output:t_gcrs, pos_gcrs, vel_gcrs

PODfile_validate.py: validate coordinate transformation

CD_estimate.py: estimation of Cd
   
