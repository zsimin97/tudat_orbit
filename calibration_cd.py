import numpy as np
from datetime import datetime, timezone
from POD_readin import find_sp3_files_for_date
from estimate_cd import estimate_cd_and_state 
from tudatpy.interface import spice
import os

spice.load_standard_kernels()
sp3_dir = "./poddata/"
sat_id = "L65"
initial_cd_guess = 2.2
calibration_dates = [
    (2024, 1, 7),
    (2024, 2, 3),
    (2024, 2, 19),
    (2024, 3, 17),
    (2024, 4, 25)
]

results_table = []
all_estimated_cds = []

for y, m, d in calibration_dates:
    doy = datetime(y, m, d).timetuple().tm_yday
    start_dt = datetime(y, m, d, 0, 0, 0, tzinfo=timezone.utc)
    end_dt   = datetime(y, m, d, 23, 59, 45, tzinfo=timezone.utc)
    
    files = find_sp3_files_for_date(y, m, d, sp3_dir)
    print(f"\n>>> date: {y}-{m:02d}-{d:02d}")
    print(f"    in {len(files)} SP3 files:")
    for f in files:
        print(f"    - {os.path.basename(f)}")

    if not files:
        print(f"{y}-{m:02d}-{d:02d} | {doy:<4} | {'N/A':<10} | {'N/A':<10} | Missing SP3")
        continue
    
    try:
        init_vec, final_vec, final_resi = estimate_cd_and_state(
            files, start_dt, end_dt, initial_cd_guess, sat_id=sat_id
        )
        
        est_cd = final_vec[-1]
        all_estimated_cds.append(est_cd)
        
        print(f"{y}-{m:02d}-{d:02d} | {doy:<4} | {est_cd:<10.4f} | {final_resi:<10.4f} | Success")
        results_table.append([y, m, d, doy, est_cd, final_resi])
        
    except Exception as e:
        print(f"{y}-{m:02d}-{d:02d} | {doy:<4} | {'Error':<10} | {'--':<10} | {str(e)[:20]}...")

if all_estimated_cds:
    print("Summary of Results:")
    print(f"{'Date':<12} | {'DOY':<4} | {'Est. Cd':<10} | {'Final Resi':<10}")
    print("-" * 60)
 
    for res in results_table:
        y, m, d, doy, est_cd, final_resi = res
        print(f"{y}-{m:02d}-{d:02d} | {doy:<4} | {est_cd:<10.4f} | {final_resi:<10.4f}")

    locked_cd = np.mean(all_estimated_cds)
    std_cd = np.std(all_estimated_cds)
    
    print("\n" + "="*50)
    print(f"CALIBRATION FINAL REPORT")
    print(f"Locked Cd Baseline: {locked_cd:.6f}")
    print(f"Standard Deviation: {std_cd:.6f}")
    print(f"Relative Uncertainty: {(std_cd/locked_cd)*100:.2f}%")
    print("="*50)
else:
    print("\nError Estimation")