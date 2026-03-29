import numpy as np
from datetime import datetime, timezone
from POD_readin import find_sp3_files_for_date, build_pod_from_sp3
from dynamics_setup import make_bodies
from propagate_orbit import propagate_orbit, propagate_state_and_covariance
from tudatpy.interface import spice
import os
import matplotlib.pyplot as plt

spice.load_standard_kernels()
sp3_dir = "./poddata/"

dates = [(2024,1,7),]
for y, m, d in dates:
    doy = datetime(y, m, d).timetuple().tm_yday
    start_dt = datetime(y, m, d, 0, 0, 0, tzinfo=timezone.utc)
    end_dt   = datetime(y, m, d, 23, 59, 45, tzinfo=timezone.utc)
    files = find_sp3_files_for_date(y, m, d, sp3_dir)
    print(f"\n>>> date: {y}-{m:02d}-{d:02d}")
    print(f"    in {len(files)} SP3 files:")
    for f in files:
        print(f"    - {os.path.basename(f)}")

    t_gcrs, pos_gcrs, vel_gcrs = build_pod_from_sp3(files, start_dt, end_dt, sat_id="L65")
    start_epoch = float(t_gcrs[0])
    end_epoch = float(t_gcrs[-1]) 
    initial_state = np.hstack((pos_gcrs[0], vel_gcrs[0]))

    satellite_name = "grace_fo"
    cd_guess = 2.2 
    mass = 600.0
    reference_area = 2.0
    bodies = make_bodies(
        space_weather_file="sw19571001.txt",
        satellite_name=satellite_name,
        mass=mass,
        reference_area=reference_area,
        cd_guess=cd_guess,
    )

    #state_history = propagate_orbit(
    #    start_epoch=start_epoch,
    #    end_epoch=end_epoch,
    #    bodies=bodies,
    #    initial_state=initial_state,
    #    grav_rank=20,
    #    step_size=10.0,
    #    satellite_name=satellite_name,
    #)

    initial_covariance = np.diag([
        0.1**2,   # x position variance (m²) - 10cm std
        0.1**2,   # y position variance (m²)
        0.1**2,   # z position variance (m²)
        0.001**2, # vx velocity variance (m²/s²) - 1mm/s std
        0.001**2, # vy velocity variance (m²/s²)
        0.001**2  # vz velocity variance (m²/s²)
    ])

    process_noise = 1e-5 * 0.36
    print(f"    Process noise sigma: {process_noise:.2e} m/s²")

    #SNC 
    times_with_snc, states_with_snc, cov_with_snc= propagate_state_and_covariance(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        bodies=bodies,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        process_noise=process_noise,
        satellite_name="grace_fo",
        grav_rank=20,
        step_size=60.0
    )
    
    #no SNC
    times_no_snc, states_no_snc, cov_no_snc= propagate_orbit(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        bodies=bodies,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        satellite_name="grace_fo",
        grav_rank=20,
        step_size=60.0
    )

    time_hours = (times_with_snc - times_with_snc[0]) / 3600
    
    pos_rss_no_snc = np.sqrt([np.trace(cov[:3, :3]) for cov in cov_no_snc])
    vel_rss_no_snc = np.sqrt([np.trace(cov[3:, 3:]) for cov in cov_no_snc])
    
    pos_rss_with_snc = np.sqrt([np.trace(cov[:3, :3]) for cov in cov_with_snc])
    vel_rss_with_snc = np.sqrt([np.trace(cov[3:, 3:]) for cov in cov_with_snc])
    
    # 4. 打印对比
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    duration_hours = time_hours[-1]
    
    print(f"\nDuration: {duration_hours:.2f} hours ({duration_hours*60:.1f} minutes)")
    print(f"Process noise: {process_noise:.2e} m/s²")
    
    print(f"\n{'Metric':<30} {'Without SNC':<20} {'With SNC':<20} {'Ratio':<10}")
    print("-"*80)
    
    # Initial
    print(f"{'Initial Position Std (m)':<30} {pos_rss_no_snc[0]:<20.3f} {pos_rss_with_snc[0]:<20.3f} {pos_rss_with_snc[0]/pos_rss_no_snc[0]:<10.2f}")
    print(f"{'Initial Velocity Std (mm/s)':<30} {vel_rss_no_snc[0]*1000:<20.3f} {vel_rss_with_snc[0]*1000:<20.3f} {vel_rss_with_snc[0]/vel_rss_no_snc[0]:<10.2f}")
    
    # Final
    print(f"{'Final Position Std (m)':<30} {pos_rss_no_snc[-1]:<20.3f} {pos_rss_with_snc[-1]:<20.3f} {pos_rss_with_snc[-1]/pos_rss_no_snc[-1]:<10.2f}")
    print(f"{'Final Velocity Std (mm/s)':<30} {vel_rss_no_snc[-1]*1000:<20.3f} {vel_rss_with_snc[-1]*1000:<20.3f} {vel_rss_with_snc[-1]/vel_rss_no_snc[-1]:<10.2f}")
    
    # Growth
    growth_no_snc = pos_rss_no_snc[-1] - pos_rss_no_snc[0]
    growth_with_snc = pos_rss_with_snc[-1] - pos_rss_with_snc[0]
    
    print(f"{'Position Growth (m)':<30} {growth_no_snc:<20.3f} {growth_with_snc:<20.3f} {growth_with_snc/growth_no_snc:<10.2f}")
    
    growth_rate_no_snc = growth_no_snc / duration_hours if duration_hours > 0 else 0
    growth_rate_with_snc = growth_with_snc / duration_hours if duration_hours > 0 else 0
    
    print(f"{'Growth Rate (m/hour)':<30} {growth_rate_no_snc:<20.3f} {growth_rate_with_snc:<20.3f} {growth_rate_with_snc/growth_rate_no_snc if growth_rate_no_snc > 0 else 0:<10.2f}")
    
    # 5. 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position RSS
    ax1 = axes[0, 0]
    ax1.plot(time_hours, pos_rss_no_snc, 'b-', linewidth=2, label='Without SNC', marker='o', markersize=4)
    ax1.plot(time_hours, pos_rss_with_snc, 'r-', linewidth=2, label='With SNC', marker='s', markersize=4)
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('Position RSS Std (m)', fontsize=11)
    ax1.set_title('Position Uncertainty Growth', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Velocity RSS
    ax2 = axes[0, 1]
    ax2.plot(time_hours, vel_rss_no_snc * 1000, 'b-', linewidth=2, label='Without SNC', marker='o', markersize=4)
    ax2.plot(time_hours, vel_rss_with_snc * 1000, 'r-', linewidth=2, label='With SNC', marker='s', markersize=4)
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_ylabel('Velocity RSS Std (mm/s)', fontsize=11)
    ax2.set_title('Velocity Uncertainty Growth', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Difference (absolute)
    ax3 = axes[1, 0]
    pos_diff = pos_rss_with_snc - pos_rss_no_snc
    ax3.plot(time_hours, pos_diff, 'g-', linewidth=2, marker='d', markersize=4)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (hours)', fontsize=11)
    ax3.set_ylabel('Difference (With SNC - Without SNC) [m]', fontsize=11)
    ax3.set_title('SNC Contribution to Position Uncertainty', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(time_hours, 0, pos_diff, alpha=0.3, color='green')
    
    # Ratio
    ax4 = axes[1, 1]
    ratio = pos_rss_with_snc / pos_rss_no_snc
    ax4.plot(time_hours, ratio, 'm-', linewidth=2, marker='^', markersize=4)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='No difference')
    ax4.set_xlabel('Time (hours)', fontsize=11)
    ax4.set_ylabel('Ratio (With SNC / Without SNC)', fontsize=11)
    ax4.set_title('Relative Effect of SNC', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snc_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved as 'snc_comparison.png'")
    plt.show()
    
    # 6. 详细分析
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    print(f"\n📊 Key Observations:")
    
    if pos_diff[-1] > 0.01:  # 如果差异 > 1cm
        print(f"  • SNC adds {pos_diff[-1]:.3f} m to final position uncertainty")
        print(f"  • This is {(ratio[-1]-1)*100:.1f}% more than without SNC")
        print(f"  • SNC contribution grows over time (from {pos_diff[0]:.3f}m to {pos_diff[-1]:.3f}m)")
    else:
        print(f"  ⚠️  SNC contribution is very small ({pos_diff[-1]*100:.2f} cm)")
        print(f"     This suggests:")
        print(f"     - Time period too short, OR")
        print(f"     - Process noise too small")
    
    # 判断process noise是否合理
    if duration_hours > 1:
        expected_contribution = duration_hours * 0.5  # 粗略估计：0.5m/hour
        if pos_diff[-1] < expected_contribution * 0.3:
            print(f"\n  ⚠️  WARNING: Process noise seems too small!")
            print(f"     Expected SNC contribution: ~{expected_contribution:.2f} m")
            print(f"     Actual: {pos_diff[-1]:.2f} m")
            print(f"     Consider increasing process_noise")
        elif pos_diff[-1] > expected_contribution * 3:
            print(f"\n  ⚠️  WARNING: Process noise seems too large!")
            print(f"     Expected SNC contribution: ~{expected_contribution:.2f} m")
            print(f"     Actual: {pos_diff[-1]:.2f} m")
            print(f"     Consider decreasing process_noise")
        else:
            print(f"\n  ✅ Process noise appears reasonable")
            print(f"     SNC contribution: {pos_diff[-1]:.2f} m over {duration_hours:.1f} hours")
    
   

    # print(f"\n    Propagation Results:")
    # print(f"    - Total steps: {len(times)}")
    # print(f"    - Duration: {(times[-1] - times[0])/3600:.2f} hours")
        
    # # Final uncertainty
    # final_cov = covariances[-1]
    # pos_std = np.sqrt(np.diag(final_cov[:3, :3]))
    # vel_std = np.sqrt(np.diag(final_cov[3:, 3:]))       
    # print(f"    - Final position std: [{pos_std[0]:.2f}, {pos_std[1]:.2f}, {pos_std[2]:.2f}] m")
    # print(f"    - Final velocity std: [{vel_std[0]*1000:.2f}, {vel_std[1]*1000:.2f}, {vel_std[2]*1000:.2f}] mm/s")
        
    # time_hours = (times - times[0]) / 3600
    # pos_rss = np.sqrt([np.trace(cov[:3, :3]) for cov in covariances])
    # vel_rss = np.sqrt([np.trace(cov[3:, 3:]) for cov in covariances])
        
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
    # ax1.plot(time_hours, pos_rss, 'b-', linewidth=2, label='RSS Position Std')
    # ax1.set_ylabel('Position Std (m)', fontsize=12)
    # ax1.set_xlabel('Time (hours)', fontsize=12)
    # ax1.grid(True, alpha=0.3)
    # ax1.legend()
    # ax1.set_title(f'Covariance Propagation with SNC (σ_a = {process_noise:.2e} m/s²)', fontsize=14)
        
    # ax2.plot(time_hours, vel_rss * 1000, 'r-', linewidth=2, label='RSS Velocity Std')
    # ax2.set_ylabel('Velocity Std (mm/s)', fontsize=12)
    # ax2.set_xlabel('Time (hours)', fontsize=12)
    # ax2.grid(True, alpha=0.3)
    # ax2.legend()
        
    # plt.tight_layout()
    # plt.savefig(f'snc_covariance_{y}{m:02d}{d:02d}.png', dpi=150)
    # plt.show()