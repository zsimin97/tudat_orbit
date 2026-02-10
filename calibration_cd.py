import numpy as np
from datetime import datetime, timezone
from POD_readin import find_sp3_files_for_date
from estimate_cd import estimate_cd_and_state 
from tudatpy.interface import spice
import os
from matplotlib import pyplot as plt

spice.load_standard_kernels()
sp3_dir = "./poddata/"
sat_id = "L65"
initial_cd_guess = 2.2
calibration_dates = [
    (2024, 1, 7),
  #  (2024, 2, 3),
  #  (2024, 2, 19),
  #  (2024, 3, 17),
  #  (2024, 4, 25)
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
        init_vec, final_vec, final_resi, covariance_output = estimate_cd_and_state(
            files, start_dt, end_dt, initial_cd_guess, sat_id=sat_id
        )
        
        formal_errors = covariance_output.formal_errors
        covariance = covariance_output.covariance
        original_eigenvalues, original_eigenvectors = np.linalg.eig(covariance)
        print(f'Formal Errors:\n{formal_errors**2}')
        print(f'covariance:\n\n{covariance}\n')
        print(f'Eigenvalues of Covariance Matrix:\n\n {original_eigenvalues}\n')
        
        #sorted_indices = np.argsort(original_eigenvalues)[::-1]
        #eigenvalues = original_eigenvalues[sorted_indices]
        #eigenvectors = original_eigenvectors[:, sorted_indices]
        #print(f"Sorted Eigenvalues (variances along principal axes):\n\n{eigenvalues}\n")   
        #print(f"Sorted Eigenvectors (directions of principal axes):\n\n{eigenvectors}\n")
        #COV_sub = covariance[np.ix_(np.sort(sorted_indices)[:3], np.sort(sorted_indices)[:3])]  
        #x_star_sub = init_vec[sorted_indices[:3]]

        COV_sub = covariance[0:3, 0:3] 
        eigenvalues, eigenvectors = np.linalg.eig(COV_sub)
        print(f"Eigenvalues (Position):\n\n{eigenvalues}\n")   
        print(f"Eigenvectors (Position):\n\n{eigenvectors}\n")
        if np.any(eigenvalues <= 0):
            raise ValueError(f"$Covariance$ submatrix is not positive definite. Eigenvalues must be positive.\n")

        phi = np.linspace(0, np.pi, 50)
        theta = np.linspace(0, 2 * np.pi,50)
        phi, theta = np.meshgrid(phi, theta)
        x_star_sub = final_vec[0:3]
        # Generate points on the unit sphere and multiply each direction by the corresponding eigenvalue
        x_ell= np.sqrt(eigenvalues[0])*  np.sin(phi) * np.cos(theta)
        y_ell = np.sqrt(eigenvalues[1])* np.sin(phi) * np.sin(theta)
        z_ell = np.sqrt(eigenvalues[2])* np.cos(phi)
        ell = np.stack([x_ell, y_ell, z_ell], axis=0)
        #Rotate the Ellipsoid(s). This is done by multiplying ell and diagonal_ell by the corresponding eigenvector matrices
        ellipsoid_boundary_3_sigma = 3 * np.tensordot(eigenvectors, ell, axes=1)
        ellipsoid_boundary_1_sigma = 1 * np.tensordot(eigenvectors, ell, axes=1)
        #ellipsoid_boundary_3_sigma = ellipsoid_boundary_3_sigma + x_star_sub[:, np.newaxis, np.newaxis]
        #ellipsoid_boundary_1_sigma = ellipsoid_boundary_1_sigma + x_star_sub[:, np.newaxis, np.newaxis]
        # Plot the ellipsoid in 3D
        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(ellipsoid_boundary_3_sigma[0], ellipsoid_boundary_3_sigma[1], ellipsoid_boundary_3_sigma[2], color='cyan', alpha=0.4, label = '3-sigma (covariance)')
        ax.plot_surface(ellipsoid_boundary_1_sigma[0], ellipsoid_boundary_1_sigma[1], ellipsoid_boundary_1_sigma[2], color='blue', alpha=0.4, label = '1-sigma (covariance)')
        ax.plot(ellipsoid_boundary_1_sigma[0], ellipsoid_boundary_1_sigma[2], 'r+', alpha=0.1, zdir='y', zs=2*np.max(ellipsoid_boundary_3_sigma[1]))
        ax.plot(ellipsoid_boundary_1_sigma[1], ellipsoid_boundary_1_sigma[2], 'r+',alpha=0.1, zdir='x', zs=-2*np.max(ellipsoid_boundary_3_sigma[0]))
        ax.plot(ellipsoid_boundary_1_sigma[0], ellipsoid_boundary_1_sigma[1], 'r+',alpha=0.1, zdir='z', zs=-2*np.max(ellipsoid_boundary_3_sigma[2]))
        ax.plot(ellipsoid_boundary_3_sigma[0], ellipsoid_boundary_3_sigma[2], 'b+', alpha=0.1, zdir='y', zs=2*np.max(ellipsoid_boundary_3_sigma[1]))
        ax.plot(ellipsoid_boundary_3_sigma[1], ellipsoid_boundary_3_sigma[2], 'b+',alpha=0.1, zdir='x', zs=-2*np.max(ellipsoid_boundary_3_sigma[0]))
        ax.plot(ellipsoid_boundary_3_sigma[0], ellipsoid_boundary_3_sigma[1], 'b+',alpha=0.1, zdir='z', zs=-2*np.max(ellipsoid_boundary_3_sigma[2]))
        ax.set_xlabel('(x-x^*)')
        ax.set_ylabel('(y-y^*)')
        ax.set_zlabel('(z-z^*)')
        ax.set_aspect('equal')
        ax.set_title('3D Position Confidence Ellipsoid and Projections')
        plt.legend()
        plt.tight_layout()
        plt.savefig('covariance_Ellipsoid.png', dpi=300)
        plt.show()

        #plt.figure(figsize=(7, 6))
        #plt.imshow((np.abs(covariance_output.correlations)), aspect='auto', interpolation='none')
        #plt.colorbar()
        #plt.title("Correlation Matrix")
        #plt.xlabel("Index - Estimated Parameter")
        #plt.ylabel("Index - Estimated Parameter")
        #plt.tight_layout()
        #plt.savefig('covariance_output_correlations.png', dpi=300)
        #plt.show()  

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
    print(f"Calibration Final")
    print(f"Locked Cd Baseline: {locked_cd:.6f}")
    print(f"Standard Deviation: {std_cd:.6f}")
    print(f"Relative Uncertainty: {(std_cd/locked_cd)*100:.2f}%")
    print("="*50)
else:
    print("\nError Estimation")