from datetime import datetime, timezone, timedelta
from estimate_cd import estimate_cd_and_state 
from result_plot import plot_cd_heatmap_plt, plot_combined_heatmaps 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from tudatpy.interface import spice

if __name__ == "__main__":
 
    spice.load_standard_kernels()
    files = [
        "poddata/GFZOP_RSO_L65_G_20240218_220000_20240219_120000_v03.sp3",
        "poddata/GFZOP_RSO_L65_G_20240219_100000_20240220_000000_v03.sp3",
    ]
    t_start = datetime(2024, 2, 19, 0, 0, 0, tzinfo=timezone.utc)
    
    durations_hr = [3, 6, 9, 12, 15, 18, 21, 24]    
    # durations_hr = list(range(1, 25))    
    cd_guesses = np.arange(1.0, 3.1, 0.5)
    results_list = []
    # t_end   = datetime(2024, 2, 19, 6, 0, 0, tzinfo=timezone.utc)
    print(f"{'Duration':>8} | {'Guess':>6} | {'Final Cd':>10} | {'Status'}")
    print("-" * 45)

    
    for hr in durations_hr:
        t_end = t_start + timedelta(hours=hr)
        
        for guess in cd_guesses:      
            
            try:
                init_vec, final_vec, final_resi = estimate_cd_and_state(
                    files, t_start, t_end, cd_guess=guess
                )
                
                final_cd = final_vec[6]
                results_list.append({
                    "Duration": hr,
                    "Initial_Cd": guess,
                    "Estimated_Cd": final_cd,
                    "Residual": final_resi
                })
                print(f"{hr:8d} | {guess:6.1f} | {final_cd:10.4f} | {final_resi:10.4f} | OK")
                
            except Exception as e:
                print(f"{hr:8d} | {guess:6.1f} | {'FAILED':>10} | {str(e)[:15]}")

    #df = pd.DataFrame(results_list)
    #plot_cd_heatmap_plt(results_list)
    plot_combined_heatmaps(results_list)

    #try:
        #init_vec, final_vec, final_resi = estimate_cd_and_state(files, t_start, t_end, cd_guess=2.0)
        
        # print("\n" + "="*40)
        #print("--- before ---")
        #print(f"initial states: {init_vec[:6]}")
        #print(f"initial Cd:  {init_vec[6]:.6f}")
        #print("-" * 20)
        #print(f"initial states: {final_vec[:6]}")
        #print(f"estimated Cd:  {final_vec[6]:.6f}")
        #print(f"Cd corrected: {final_vec[6] - init_vec[6]:.6f}")
        #print("="*40)
        
    #except Exception as e:
        #print(f"\nerror: {e}")