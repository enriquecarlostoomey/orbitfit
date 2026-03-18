import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import orbitfit.rotate as rot
from astropy.coordinates import get_sun
from astropy.time import Time

class PLOTRESULTS:

    def __init__(self, versor_arr_real, versor_arr_real_filt, versor_arr_meas, versor_arr_comp, df_client_real_unfilt, df_client_real, 
                 df_client_init, df_state_final, df_servicer, b_matrix = None, n_loops = 0, rMax = 4e7, save_dir=None):
        """
        Class to plotresults for both residuals and final fit fior the Angle Based Least Square Optimization (AngularBatchEst)

        Inputs: 
            index: List of all dates from propagation (format yyyy-mm-ddThh:mm:ss.ss   es: 2021-03-09T16:08:14.991)
            versor_arr_real: real (reference) versor from unperturbed unfiltered simulated data
            versor_arr_rejected: rejected values from the filters (FOV, sun) 
            versor_arr_meas: measured (simulated, perturbed) versor utilised for the LS optimization
            versor_arr_comp; final optimized versors corresponding to final fitted orbit
            df_client_real: real (unperturbed) simulated r,v dataframe (in m, m/s) (lenght N)
            df_client_init: initial guess r,v dataframe for initialize the LS algorithm (in m, m/s) (lenght N)
            df_state_final: final r,v dataframe of the final fitted orbit (in m, m/s) (lenght N) 
            b_matrix = matrix of temporal evolution of residuals vector (norm over x,y,z dimensions)
            n_loops: Number of iteration loops (optional)  
            rMax = Max radius / semi-major axis (for near circular orbit) for the plot-scale. Default r ~ r_GEO (in m)
        Outputs:
            plots
        """
        self.versor_arr_real_unfilt = versor_arr_real
        self.versor_arr_real = versor_arr_real_filt
        self.versor_arr_meas = versor_arr_meas
        self.versor_arr_comp = versor_arr_comp
        self.df_client_ECI_unfilt = df_client_real_unfilt
        self.df_client_ECI_m = df_client_real
        self.df_client_ECI_fit = df_client_init
        self.df_state_final = df_state_final
        self.n_loops = n_loops
        self.rMax = rMax
        self.df_servicer = df_servicer
        self.df_residuals_matrix = b_matrix
        self.save_dir = save_dir

    def plotResiduals(self):

        # ==============================================================================
        # ---      FINAL RESULTS ON RESIDUALS (RELATIVE DIRECTION)                   ---
        # ==============================================================================

        # 1. Calculate residuals (magnitude of the error relative to the truth)
        res_meas = np.linalg.norm(self.versor_arr_real - self.versor_arr_meas, axis=1)
        res_comp = np.linalg.norm(self.versor_arr_real - self.versor_arr_comp, axis=1) 

        n_meas = len(self.versor_arr_meas)
        time_steps = range(n_meas)

        # First we rotate everything in ECEF frame (LVLH for GEO)
        time_index = self.df_client_ECI_m.index
        time_index2 = self.df_client_ECI_unfilt.index
        cols = [f'randv_mks_{i}' for i in range(6)]         # 0 to 5 for compatibility with rotate_gps

        # initialyze empty array (we have to "fake" velocities)
        zeri = np.zeros_like(self.versor_arr_real)
        zeri2 = np.zeros_like(self.versor_arr_real_unfilt)

        # --- REAL VERSORS ---
        versors_real_6d = np.hstack((self.versor_arr_real_unfilt, zeri2))
        df_versors_real = pd.DataFrame(versors_real_6d, index=time_index2, columns=cols)
        df_versors_real_rot = rot.rotate_gps(df_versors_real, method="I2E")
        versors_real = df_versors_real_rot.iloc[:, 0:3].values             # only position estracted

        # --- MEASURED VERSORS ---
        versors_meas_6d = np.hstack((self.versor_arr_meas, zeri))
        df_versors_meas = pd.DataFrame(versors_meas_6d, index=time_index, columns=cols)
        df_versors_meas_rot = rot.rotate_gps(df_versors_meas, method="I2E")
        versors_meas = df_versors_meas_rot.iloc[:, 0:3].values

        # --- FINAL VERSORS ---
        versors_final_6d = np.hstack((self.versor_arr_comp, zeri))
        df_versors_final = pd.DataFrame(versors_final_6d, index=time_index, columns=cols)
        df_versors_final_rot = rot.rotate_gps(df_versors_final, method="I2E")
        versors_final = df_versors_final_rot.iloc[:, 0:3].values

        # Inityializing the figure
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])
        ax_xy = fig.add_subplot(gs[0])

        # Real, unfiltered Trajectory
        ax_xy.plot(versors_real[:, 0], versors_real[:, 1], label='Real Trajectory (Truth)', color='grey', linewidth=1)
        ax_xy.plot(versors_real[0, 0], versors_real[0, 1], marker='*', color='grey', markersize=10) # Start point

        # Initial Guess / Measured
        ax_xy.plot(versors_meas[:, 0], versors_meas[:, 1], label='Initial Guess (Measured)', color='red', alpha=0.5, linestyle='None', marker='.')
        ax_xy.plot(versors_meas[0, 0], versors_meas[0, 1], marker='*', color='red', markersize=10) # Start point

        # Final Computed
        ax_xy.plot(versors_final[:, 0], versors_final[:, 1], label='Final Fitted Trajectory', color='blue', linestyle='None', linewidth=2, marker='.')
        ax_xy.plot(versors_final[0, 0], versors_final[0, 1], marker='*', color='blue', markersize=10) # Start point

        ax_xy.set_title('Relative Motion X-Y Plane - Versors in ECEF')
        ax_xy.set_xlabel('Versor X (ECEF)')
        ax_xy.set_ylabel('Versor Y (ECEF)')
        ax_xy.set_aspect('equal', adjustable='box') # Mantiene le proporzioni corrette
        ax_xy.set_xlim([-1.1, 1.1])
        ax_xy.set_ylim([-1.1, 1.1])
        ax_xy.grid(True, linestyle='--', alpha=0.6)
        ax_xy.legend()


        # --- ROW 2: Residuals Comparison (Absolute Error) ---
        # Spans across all 3 columns (gs[1, :])
        ax_res = fig.add_subplot(gs[1, :])

        ax_res.plot(time_steps, res_meas, marker='o', linestyle='-', color='red', alpha=0.5, markersize=4, label='|Real - Measured| (Initial Noise)')
        ax_res.plot(time_steps, res_comp, marker='s', linestyle='-', color='blue', alpha=0.8, markersize=4, label='|Real - Computed| (Final Fit Error)')

        ax_res.set_xlabel('Measurement Index')
        ax_res.set_ylabel('Residual Magnitude')
        ax_res.set_title('Versor Residuals Comparison: Initial Noise vs Final Fit')
        ax_res.grid(True, linestyle='--', alpha=0.7)
        ax_res.legend()

        plt.tight_layout()
        if self.save_dir:
            import os
            fig.savefig(os.path.join(self.save_dir, "residualsplot.png"), dpi=200)
        plt.show(block=False) 



    def plotResidualsEvolution(self):
        """
        Plots the evolution of the residual magnitude (3D Norm) across all optimizer iterations
        using a continuous colormap and a reference target line.
        """
        # ==============================================================================
        # --- 1. CALCULATE REFERENCE TARGET (INITIAL NOISE) AND FINAL FIT           ---
        # ==============================================================================
        
        # Calculate the 3D magnitude (Norm) of the error relative to the truth
        # Initial Measured Noise (Target to beat)
        res_meas_magnitude = np.linalg.norm(self.versor_arr_real - self.versor_arr_meas, axis=1)
        # Final Computed Fit Error
        res_comp_magnitude = np.linalg.norm(self.versor_arr_real - self.versor_arr_comp, axis=1) 

        n_measurements = len(self.versor_arr_meas)
        measurement_index = np.arange(n_measurements)

        # ==============================================================================
        # --- 2. INITIALIZE FIGURE                                                   ---
        # ==============================================================================
        fig, ax = plt.subplots(figsize=(14, 8))

        # ==============================================================================
        # --- 3. PLOT EVOLUTION ACROSS ITERATIONS USING COLORMAP                     ---
        # ==============================================================================
        
        # Check if the residuals history matrix was passed to the class
        if hasattr(self, 'df_residuals_matrix') and self.df_residuals_matrix is not None:
            
            n_cols = self.df_residuals_matrix.shape[1]      # Total columns (X, Y, Z per iter)
            n_iter = n_cols // 3                            # Number of iterations performed
            
            # Setup a continuous colormap
            cmap_name = 'plasma_r'
            cmap = plt.get_cmap(cmap_name)
            norm_colors = plt.Normalize(vmin=0, vmax=n_iter - 1) # Normalize iteration number to 0-1

            # Loop through each iteration to plot the error magnitude
            for it in range(n_iter):
                idx_start = it * 3
                
                # Extract X, Y, Z residuals for the current iteration using position (.iloc)
                res_x = self.df_residuals_matrix.iloc[:, idx_start].values
                res_y = self.df_residuals_matrix.iloc[:, idx_start + 1].values
                res_z = self.df_residuals_matrix.iloc[:, idx_start + 2].values
                
                # Calculate the 3D Magnitude (Norm) of the error for each measurement point
                magnitude_res = np.sqrt(res_x**2 + res_y**2 + res_z**2)
                
                # Get the color corresponding to this iteration
                color = cmap(norm_colors(it))
                
                # Plot the curve for this specific iteration
                # Note: We don't add labels here to keep the legend clean
                ax.plot(measurement_index, magnitude_res, linestyle='-', 
                        color=color, alpha=0.8, linewidth=1.5)
            
            # --- Add Colorbar ---
            # Create a scalar mappable for the colorbar to map iteration numbers to colors
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_colors)
            sm.set_array([]) # Required for colorbar to work
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Optimizer Iteration', rotation=270, labelpad=20, fontsize=12)

        else:
            print("[WARNING] df_residuals_matrix not found. Cannot plot iteration evolution.")
            # If matrix is missing, still plot the final fit as fallback
            ax.plot(measurement_index, res_comp_magnitude, color='blue', label='Final Fit Error', linewidth=2)

        # ==============================================================================
        # --- 4. PLOT REFERENCE TARGETS (SINGLE PLOT CALLS)                         ---
        # ==============================================================================

        # Plot the INITIAL MEASURED NOISE once as a thin grey dashed line (The 'Target')
        ax.plot(measurement_index, res_meas_magnitude, color='grey', linestyle='--', 
                linewidth=1, label='Initial Measured Noise (Target to Beat)')

        # ==============================================================================
        # --- 5. FORMATTING AND LEGEND (CUSTOM HANDLES)                              ---
        # ==============================================================================
        ax.set_xlabel('Measurement Index', fontsize=12)
        ax.set_ylabel('Residual Magnitude (Norm 3D)', fontsize=12)
        ax.set_title('Versor Residuals Evolution: Initial Noise vs Convergence History', fontsize=14)
        ax.grid(True, linestyle='--', alpha=1)
        
        # 1. Get the existing handles and labels (the black dashed line)
        handles, labels = ax.get_legend_handles_labels()
        
        # 2. Create proxy artists (fake lines) for the first and last iterations
        if hasattr(self, 'df_residuals_matrix') and self.df_residuals_matrix is not None:
            # Extract the exact colors from the colormap
            color_start = cmap(norm_colors(0))             # Iteration 0 color
            color_end = cmap(norm_colors(n_iter - 1))      # Last iteration color

            # Create Line2D objects for the legend
            line_start = plt.Line2D([0], [0], color=color_start, lw=2, label='Initial Guess (Iter 0)')
            line_end = plt.Line2D([0], [0], color=color_end, lw=2, label=f'Final Fit (Iter {n_iter - 1})')

            # 3. Add them to the existing legend handles
            handles.extend([line_start, line_end])

        # 4. Draw the updated legend
        ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=10)
        
        # Enable Logarithmic Scale for dynamic range
        ax.set_yscale('log')

        plt.tight_layout()
        if self.save_dir:
            import os
            fig.savefig(os.path.join(self.save_dir, "residuals_history.png"), dpi=200)
        plt.show(block=False)

       
    def compute_errors(self, df_a, df_b):
     # find difference in r, v (for final plot)

        pos_cols = ['randv_mks_0', 'randv_mks_1', 'randv_mks_2']
        vel_cols = ['randv_mks_3', 'randv_mks_4', 'randv_mks_5']
        
        pos_error = np.linalg.norm(df_a[pos_cols].values - df_b[pos_cols].values, axis=1)
        vel_error = np.linalg.norm(df_a[vel_cols].values - df_b[vel_cols].values, axis=1)
        
        return pos_error, vel_error

    def plotFinalFit(self):
        
        # ==============================================================================
        # ---       FINAL RESULTS ON THE ORBIT (ABSOLUTE POSITION in ECI)            ---
        # ==============================================================================

        # Scaling factor
        lim_min = -4e7
        lim_max = self.rMax * 1.1
    
        # 3D plot of original client orbit self.df_client_ECI_m and the perturbated initial guess self.df_client_ECI_fit:

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot original unfiltered client orbit
        ax.scatter(self.df_client_ECI_unfilt["randv_mks_0"],
                self.df_client_ECI_unfilt["randv_mks_1"],
                self.df_client_ECI_unfilt["randv_mks_2"],
                label="Unfiltered Orbit", color="grey", s=0.5, linestyle=':')

        # Plot original client orbit
        ax.scatter(self.df_client_ECI_m["randv_mks_0"],
                self.df_client_ECI_m["randv_mks_1"],
                self.df_client_ECI_m["randv_mks_2"],
                label="Original Client Orbit", color="red", s=5)

        # Plot final fitted orbit
        ax.scatter(self.df_client_ECI_fit["randv_mks_0"],
                self.df_client_ECI_fit["randv_mks_1"],
                self.df_client_ECI_fit["randv_mks_2"],
                label="Initial guess Orbit", color="green", s=5)
        
                # Plot sun direction
        t_start = Time(self.df_client_ECI_m.index[0])
        sun_coor = get_sun(t_start)
        sun_pos = sun_coor.cartesian.xyz.to('m').value
        sun_vec = sun_pos / np.linalg.norm(sun_pos)
        light_direction = -sun_vec * lim_max * 0.5
        # Target position to plot incoming light
        target_pos = np.mean(self.df_client_ECI_m.iloc[:, :3].values, axis=0)
        start_point = target_pos + (sun_vec * lim_max * 0.5)

        # Plot Quiver
        ax.quiver(start_point[0], start_point[1], start_point[2], 
                  light_direction[0], light_direction[1], light_direction[2], 
                  color='orange', label='Incoming Sunlight', 
                  linewidth=1.5, arrow_length_ratio=0.3)
        # plot a little sun
        ax.scatter(start_point[0], start_point[1], start_point[2], 
                   color='orange', s=50, marker='*')

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Original vs Initial guess Orbit")
        ax.legend()

        ax.set_xlim([lim_min, lim_max])
        ax.set_ylim([lim_min, lim_max])
        ax.set_zlim([lim_min, lim_max])
        ax.set_box_aspect([1, 1, 1])
        if self.save_dir:
            import os
            fig.savefig(os.path.join(self.save_dir, "Original_VS_Initial.png"), dpi=200)

        # 3D plot of original client orbit self.df_client_ECI_m and the final, fitted self.df_state_final:

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot original unfiltered client orbit
        ax.scatter(self.df_client_ECI_unfilt["randv_mks_0"],
                self.df_client_ECI_unfilt["randv_mks_1"],
                self.df_client_ECI_unfilt["randv_mks_2"],
                label="Unfiltered Orbit", color="grey", s=0.5, linestyle=':')

        # Plot original client orbit
        ax.scatter(self.df_client_ECI_m["randv_mks_0"],
                self.df_client_ECI_m["randv_mks_1"],
                self.df_client_ECI_m["randv_mks_2"],
                label="Original Client Orbit", color="red", s=5)

        # Plot final fitted orbit
        ax.scatter(self.df_state_final["randv_mks_0"],
                self.df_state_final["randv_mks_1"],
                self.df_state_final["randv_mks_2"],
                label="Final Fitted Orbit", color="green", s=5)
        
        # Plot Servicer Orbit
        ax.plot(self.df_servicer["randv_mks_0"],
                self.df_servicer["randv_mks_1"],
                self.df_servicer["randv_mks_2"],
                label="Servicer Orbit (Observer)", 
                color="blue", linestyle='--', linewidth=1, alpha=0.7) 

        # single point for the current/starting position of the servicer
        ax.scatter(self.df_servicer["randv_mks_0"].iloc[0],
                   self.df_servicer["randv_mks_1"].iloc[0],
                   self.df_servicer["randv_mks_2"].iloc[0],
                   color="blue", s=5, marker='o')
        
        # Plot sun direction
        t_start = Time(self.df_client_ECI_m.index[0])
        sun_coor = get_sun(t_start)
        sun_pos = sun_coor.cartesian.xyz.to('m').value
        sun_vec = sun_pos / np.linalg.norm(sun_pos)
        light_direction = -sun_vec * lim_max * 0.5
        # Target position to plot incoming light
        target_pos = np.mean(self.df_client_ECI_m.iloc[:, :3].values, axis=0)
        start_point = target_pos + (sun_vec * lim_max * 0.5)

        # Plot Quiver
        ax.quiver(start_point[0], start_point[1], start_point[2], 
                  light_direction[0], light_direction[1], light_direction[2], 
                  color='orange', label='Incoming Sunlight', 
                  linewidth=1.5, arrow_length_ratio=0.3)
        # plot a little sun
        ax.scatter(start_point[0], start_point[1], start_point[2], 
                   color='orange', s=50, marker='*')


        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Original vs Final Fitted Orbit")
        ax.legend()


        ax.set_xlim([lim_min, lim_max])
        ax.set_ylim([lim_min, lim_max])
        ax.set_zlim([lim_min, lim_max])
        ax.set_box_aspect([1, 1, 1])
        if self.save_dir:
            fig.savefig(os.path.join(self.save_dir, "Original_VS_Final.png"), dpi=200)


        # compute and plot: 1) difference between original and initial guessed orbit\\ 2) difference between original and final fitted orbit\\ 3) difference between initial guess and final fitted orbit

        df_true = self.df_client_ECI_m
        df_initial = self.df_client_ECI_fit
        df_final = self.df_state_final

        # 1) Original vs Initial Guess
        err_pos_orig_init, err_vel_orig_init = self.compute_errors(df_true, df_initial)

        # 2) Original vs Final Fit
        err_pos_orig_final, err_vel_orig_final = self.compute_errors(df_true, df_final)

        # 3) Initial Guess vs Final Fit
        err_pos_init_final, err_vel_init_final = self.compute_errors(df_initial, df_final)


        fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Sottoplot 1: Posizione
        ax[0].plot(df_true.index, err_pos_orig_init, label='Original vs Initial Guess', color='red')
        ax[0].plot(df_true.index, err_pos_init_final, label='Initial Guess vs Final Fit', color='orange', linestyle=':')
        ax[0].plot(df_true.index, err_pos_orig_final, label='Original vs Final Fit (Residual)', color='green', linewidth=2, linestyle='--')

        ax[0].set_ylabel('Position Error [m]')
        ax[0].set_title('Comparison of Orbital Position Differences')
        ax[0].legend()
        ax[0].grid(True, which='both', linestyle='--', alpha=0.5)
        ax[0].set_yscale('log') # Usiamo scala logaritmica per vedere il miglioramento

        # Sottoplot 2: Velocità
        ax[1].plot(df_true.index, err_vel_orig_init, label='Original vs Initial Guess', color='red')
        ax[1].plot(df_true.index, err_vel_init_final, label='Initial Guess vs Final Fit', color='orange', linestyle=':')
        ax[1].plot(df_true.index, err_vel_orig_final, label='Original vs Final Fit (Residual)', color='green', linewidth=2, linestyle='--')

        ax[1].set_ylabel('Velocity Error [m/s]')
        ax[1].set_xlabel('Time')
        ax[1].legend()
        ax[1].grid(True, which='both', linestyle='--', alpha=0.5)
        ax[1].set_yscale('log')

        plt.tight_layout()
        plt.tight_layout()
        if self.save_dir:
            fig.savefig(os.path.join(self.save_dir, "finalresults.png"), dpi=200)

        print()
        print("ratio between final and original error (hopefully <<1):")
        print((np.linalg.norm(err_pos_orig_final)/np.linalg.norm(err_pos_orig_init)))
        print()
        print("norm of error between original and initial guess:")
        print(np.linalg.norm(err_pos_orig_init))
        print()
        print("norm of error between original and final guess:")
        print(np.linalg.norm(err_pos_orig_final))
        print()
        print("Max error between final and real positions:")
        print(np.max(err_pos_orig_final))
        print()
        if self.n_loops!= 0:
            print("Iterations:")
            print({self.n_loops})
        # save the outputs in the txt file:
        note_path = os.path.join(self.save_dir, "note.txt")
        with open(note_path, 'a') as f:
            f.write("\nratio between final and original error (hopefully <<1):\n")
            f.write(str(np.linalg.norm(err_pos_orig_final)/np.linalg.norm(err_pos_orig_init)) + "\n")
            f.write("\nnorm of error between original and initial guess:\n")
            f.write(str(np.linalg.norm(err_pos_orig_init)) + "\n")
            f.write("\nnorm of error between original and final guess:\n")
            f.write(str(np.linalg.norm(err_pos_orig_final)) + "\n")
            f.write("\nMax error between final and real positions:\n")
            f.write(str(np.max(err_pos_orig_final)) + "\n")
        plt.show(block=False)

def plot_detectability(phi, d, m_v, m_v_threshold, save_dir=None):
    """
    Plots the evolution of Phase Angle, Distance, and Magnitude.
    Input:
        parameters from the detectability filter:
        phi: Sun phase angle for each time [deg]
        d: relative distance for each time [m]
        m_v: magnitude of incoming radiation in logaritmic scale
        m_v_threshold: threshold for the accept/reject logic
    Output:
        Plot()
    """
    # Creazione della figura
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    time_axis = range(len(phi)) # Oppure usa self.df_servicer.index se vuoi le date

    # 1. Plot Phase Angle
    ax1.plot(time_axis, np.rad2deg(phi), color='orange', linewidth=2)
    ax1.set_ylabel('Sun Phase Angle [deg]')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title('Detectability Factors Evolution')

    # 2. Plot Distance
    ax2.plot(time_axis, d, color='royalblue', linewidth=2)
    ax2.set_ylabel('Distance [m]')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Plot Magnitude
    ax3.plot(time_axis, m_v, color='crimson', linewidth=2, label='Apparent Mag ($m_v$)')
    # Plot threshold
    ax3.axhline(y=m_v_threshold, color='black', linestyle='--', linewidth=1.5, label='Threshold')
    
    # Invert magnitude axis (under the line: rejected)
    ax3.set_ylim(bottom=max(m_v_threshold + 2, np.max(m_v[m_v < 1e6])), 
                top=min(np.min(m_v) - 1, 0)) 
    ax3.set_ylabel('Visual Magnitude')
    ax3.set_xlabel('Step [M]')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_dir:
        import os
        fig.savefig(os.path.join(save_dir, "sunFilter.png"), dpi=200)
    plt.show(block=False)