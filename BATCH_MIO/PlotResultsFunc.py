import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import orbitfit.rotate as rot

class PLOTRESULTS:

    def __init__(self, versor_arr_real, versor_arr_meas, versor_arr_comp, df_client_real, df_client_init, df_state_final, n_loops = 0, rMax = 4e7):
        """
        Class to plotresults for both residuals and final fit fior the Angle Based Least Square Optimization (AngularBatchEst)

        Inputs: 
            index: List of all dates from propagation (format yyyy-mm-ddThh:mm:ss.ss   es: 2021-03-09T16:08:14.991)
            versor_arr_real: real (reference) versor from unperturbed simulated data 
            versor_arr_meas: measured (simulated, perturbed) versor utilised for the LS optimization
            versor_arr_comp; final optimized versors corresponding to final fitted orbit
            df_client_real: real (unperturbed) simulated r,v dataframe (in m, m/s) (lenght N)
            df_client_init: initial guess r,v dataframe for initialize the LS algorithm (in m, m/s) (lenght N)
            df_state_final: final r,v dataframe of the final fitted orbit (in m, m/s) (lenght N) 
            n_loops: Number of iteration loops (optional)  
            rMax = Max radius / semi-major axis (for near circular orbit) for the plot-scale. Default r ~ r_GEO (in m)
        Outputs:
            plots
        """
        self.versor_arr_real = versor_arr_real
        self.versor_arr_meas = versor_arr_meas
        self.versor_arr_comp = versor_arr_comp
        self.df_client_ECI_m = df_client_real
        self.df_client_ECI_fit = df_client_init
        self.df_state_final = df_state_final
        self.n_loops = n_loops
        self.rMax = rMax

    def plotResiduals(self):

        # ==============================================================================
        # ---      FINAL RESULTS ON RESIDUALS (RELATIVE DIRECTION)                   ---
        # ==============================================================================

        # 1. Calculate residuals (magnitude of the error relative to the truth)
        res_meas = np.linalg.norm(self.versor_arr_real - self.versor_arr_meas, axis=1)
        res_comp = np.linalg.norm(self.versor_arr_real - self.versor_arr_comp, axis=1) 

        n_meas = len(self.versor_arr_meas)
        time_steps = range(n_meas)

        # Create figure with GridSpec for layout management
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1]) # Top row slightly taller

        # --- ROW 1: Evolution of individual components X, Y, Z ---
        # Each subplot compares Real, Measured, and Computed for a single axis

        # First we rotate everything in ECEF frame (LVLH for GEO)
        time_index = self.df_client_ECI_m.index
        cols = [f'randv_mks_{i}' for i in range(6)]  # 0 to 5
        cols_pos = ['randv_mks_0', 'randv_mks_1', 'randv_mks_2'] # position columns

        # initialyze empty array (we have to "fake" velocities)
        zeri = np.zeros_like(self.versor_arr_real)

        # --- REAL VERSORS ---
        versors_real_6d = np.hstack((-self.versor_arr_real, zeri))
        df_versors_real = pd.DataFrame(versors_real_6d, index=time_index, columns=cols)
        df_versors_real_rot = rot.rotate_gps(df_versors_real, method="I2E")
        versors_real = df_versors_real_rot[cols_pos].values             # only position estracted

        # --- MEASURED VERSORS ---
        versors_meas_6d = np.hstack((-self.versor_arr_meas, zeri))
        df_versors_meas = pd.DataFrame(versors_meas_6d, index=time_index, columns=cols)
        df_versors_meas_rot = rot.rotate_gps(df_versors_meas, method="I2E")
        versors_meas = df_versors_meas_rot[cols_pos].values

        # --- FINAL VERSORS ---
        versors_final_6d = np.hstack((-self.versor_arr_comp, zeri))
        df_versors_final = pd.DataFrame(versors_final_6d, index=time_index, columns=cols)
        df_versors_final_rot = rot.rotate_gps(df_versors_final, method="I2E")
        versors_final = df_versors_final_rot[cols_pos].values

        # 1. X-Y Plane per i Versori REALI (Truth)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(versors_real[:, 0], versors_real[:, 1], label='Real Trajectory', color='green', linewidth=2)
        # Segna il punto di partenza
        ax1.plot(versors_real[0, 0], versors_real[0, 1], marker='*', color='black', markersize=10, label='Start') 
        ax1.set_title('Relative Motion X-Y (Real)')
        ax1.set_xlabel('Versor X (ECEF)')
        ax1.set_ylabel('Versor Y (ECEF)')
        ax1.set_aspect('equal', adjustable='datalim') # FONDAMENTALE per non distorcere l'orbita
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # 2. X-Y Plane per i Versori CALCOLATI (Final Fit)
        # Condividiamo X e Y con ax1 per un confronto visivo diretto 1:1
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1) 
        ax2.plot(versors_final[:, 0], versors_final[:, 1], label='Fitted Trajectory', color='blue', linestyle='--', linewidth=2)
        # Segna il punto di partenza
        ax2.plot(versors_final[0, 0], versors_final[0, 1], marker='*', color='black', markersize=10, label='Start')
        ax2.set_title('Relative Motion X-Y (Final Fit)')
        ax2.set_xlabel('Versor X (ECEF)')
        # ax2.set_ylabel('Versor Y (ECEF)') # Nascosto perché condivide l'asse Y con ax1
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        # # 3. Z Component (Index 2)
        # ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
        # ax3.plot(time_steps, versors_real[:, 2], label='Real (Truth)', color='green', linewidth=2)
        # ax3.plot(time_steps, versors_meas[:, 2], label='Measured (Noisy)', color='red', alpha=0.5)
        # ax3.plot(time_steps, versors_final[:, 2], label='Final Computed', color='blue', linestyle='--', linewidth=2)
        # ax3.set_title('Z Component Evolution')
        # ax3.set_xlabel('Measurement Index')
        # ax3.grid(True, linestyle='--', alpha=0.6)
        # ax3.legend()


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
        plt.show() 

       
    def compute_errors(self, df_a, df_b):
     # find difference in r, v (for final plot)

        pos_cols = ['randv_mks_0', 'randv_mks_1', 'randv_mks_2']
        vel_cols = ['randv_mks_3', 'randv_mks_4', 'randv_mks_5']
        
        pos_error = np.linalg.norm(df_a[pos_cols].values - df_b[pos_cols].values, axis=1)
        vel_error = np.linalg.norm(df_a[vel_cols].values - df_b[vel_cols].values, axis=1)
        
        return pos_error, vel_error

    def plotFinalFit(self):
        
        # ==============================================================================
        # ---       FINAL RESULTS ON THE ORBIT (ABSOLUTE POSITION)                   ---
        # ==============================================================================


        # 3D plot of original client orbit self.df_client_ECI_m and the perturbated initial guess self.df_client_ECI_fit:

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

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

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Original vs Initial guess Orbit")
        ax.legend()

        lim_min = -4e7
        lim_max = self.rMax * 1.1
        ax.set_xlim([lim_min, lim_max])
        ax.set_ylim([lim_min, lim_max])
        ax.set_zlim([lim_min, lim_max])
        ax.set_box_aspect([1, 1, 1])

        # 3D plot of original client orbit self.df_client_ECI_m and the final, fitted self.df_state_final:

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

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

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Original vs Final Fitted Orbit")
        ax.legend()


        ax.set_xlim([lim_min, lim_max])
        ax.set_ylim([lim_min, lim_max])
        ax.set_zlim([lim_min, lim_max])
        ax.set_box_aspect([1, 1, 1])


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
        #ax[0].set_yscale('log') # Usiamo scala logaritmica per vedere il miglioramento

        # Sottoplot 2: Velocità
        ax[1].plot(df_true.index, err_vel_orig_init, label='Original vs Initial Guess', color='red')
        ax[1].plot(df_true.index, err_vel_init_final, label='Initial Guess vs Final Fit', color='orange', linestyle=':')
        ax[1].plot(df_true.index, err_vel_orig_final, label='Original vs Final Fit (Residual)', color='green', linewidth=2, linestyle='--')

        ax[1].set_ylabel('Velocity Error [m/s]')
        ax[1].set_xlabel('Time')
        ax[1].legend()
        ax[1].grid(True, which='both', linestyle='--', alpha=0.5)
        #ax[1].set_yscale('log')

        plt.tight_layout()
        plt.show()

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