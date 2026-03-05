import orbitfit.orbitfit as orb
from orbitfit.utils import (rv2oe, oe2rv, oe2ee, ee2oe)
from orbitfit.tle_utils import load_tle
import copy
import pandas as pd
import numpy as np

class Optimizer:
        # Function to initialize and launch the LS Loop optimization
        #
        # input:
        #       df_client,
        #       df_servicer: dataset of size (N,6) contaioning r,v points in [m,m/s], in ECI frame
        #       versor_arr_meas: dataset of size (N,3) containing measured relative directions, ECI frame 
        #       ee_initialguess:
        #       config: propagator configurations
        #       config_0: full configuration from JSON file
        #
        # Output:


    def __init__(self, ee_initialguess, df_client, df_servicer, versor_arr_meas, config, config_0, damping_lambda=0.001, max_loops=30):
        self.df_client = df_client                      # otherwise we can propagate it inside from e_initial
        self.df_servicer = df_servicer
        self.versor_arr_meas = versor_arr_meas
        self.n_measurements = len(versor_arr_meas)
        self.propagation_config = config
        self.prop_config_0 = config_0
        self.ee_initial = ee_initialguess
        self.max_loops = max_loops
        self.damping_lambda = damping_lambda            


    def find_relative (self, df_client_current):  #(df_client, df_servicer)
        # Code to obtain the relative direction of client (optimizing) and servicer (known, fixed) 
        # input:
        #       df_client_current: dataset of size (N,6) contaioning r,v points in [m,m/s], in ECI frame
        # Output:
        #       versor_arr_comp: dataset of size (N,3) containing relative position, normalizied (versor)
        #                        of the position of client wrt servicer, ECI frame

        diff = df_client_current.iloc[:, :3].values - self.df_servicer.iloc[:, :3].values
        ranges = np.linalg.norm(diff, axis=1)[:, np.newaxis] # Shape (N, 1)
        return diff / ranges # Shape (N, 3)


    def a_matrix (self, ee_step, df_state, prop_config_loop):
        # loop start: A matrix computation (from find_a)
        # Input:
        #       df_state: propagated dataframe (r,v) of current state, dimension (N,6)
        #       ee_step: current equinoctial elements (used for df_state),  dimension (1x6)
        #       prop_config_loop: 
        # Output:
        #       a: Jacobian of the current state (6xNx3)

        # we have to feed find_a with the equinoctial state [a, h, k, p, q, lambda] (ee_step) since this is the one used by Orekit  

        # def find_a(self, state, deltaamtchg=1e-6, percentchg=0.001):                                                          # could add these parameters !!!
        a = []                                                                                                                  # Preallocate the space (6xNx3)???
        deltaamtchg=1e-7
        percentchg=1e-6
        for i in range(len(ee_step)):                      # cycling through the 6 elements of the state
                    ee_state_mod = copy.copy(ee_step)        
                    counter = 0
                    percentchg_local = percentchg
                    deltaamt = ee_state_mod[i] * percentchg_local
                    while abs(deltaamt) < deltaamtchg and counter <=5:
                        counter += 1
                        percentchg_local = 1.4 * percentchg_local
                        deltaamt = ee_state_mod[i] * percentchg_local
                    #debug(f"Calculating a for state {i}: {ee_initial_guess[i]} with deltaamt: {deltaamt}")
                    ee_state_mod[i] = ee_state_mod[i] + deltaamt
                    ee_state_mod = orb.fix_state(ee_state_mod)

                    # convert back to OE and then to RV for the propagation 
                    oe_state_mod = ee2oe(*ee_state_mod)
                    rv_state_mod = np.array(oe2rv(*oe_state_mod)).flatten()*1e3
                    
                    # propagate the perturbed orbit  (modified state)
                    prop_config_loop["Propagation"]["InitialState"] = rv_state_mod.tolist()
                    index_mod, data_mod = orb.propagate_orbits_wrapper(prop_config_loop)
                    data_mod = np.array([list(item) for item in data_mod])
                    df_state_mod = pd.DataFrame(data_mod, index=index_mod, columns=['randv_mks_0', 'randv_mks_1', 'randv_mks_2', 'randv_mks_3', 'randv_mks_4', 'randv_mks_5'])

                    # Here comes the difference: we have to convert from "state" to "versor", that serves us as estimate for the precision (calculate residuals)

                    # conversion into versors

                    # Original-unperturbed (each step):
                    versor_arr_comp = self.find_relative(df_state)

                    # Perturbated:
                    versor_arr_computed_mod = self.find_relative(df_state_mod)

                    # compute the difference in versor components and divide by deltaamt to get the partial derivative (A matrix component)
                    df_versor_diff = versor_arr_computed_mod - versor_arr_comp
                    a.append(df_versor_diff/deltaamt)

        return np.asarray(a) 
    

    def LevenbergMarquardt (self, b, abw, awat, ee_step, max_diag, loop):
        # Function to
        # Input:
        #       df_state: propagated dataframe (r,v) of current state, dimension (N,6)
        #       ee_step: current equinoctial elements (used for df_state),  dimension (1x6)
        #       prop_config_loop: 
        # Output:
        #       a: Jacobian of the current state (6xNx3)        

        # 1. Initialization

        # Cost function: control of the goodness of the next step (if too long, reduce the step)
        # (if new<old accept new step)
        cost_old = np.sum(np.linalg.norm(b, axis=1)**2)

        # Damping parameter and scaling


        # # A-priori covariance matrix to penalize and lock the semi-major axis (a)                                                To implement as optional args?
        # # Assuming 'a' is at index 2, 'af' at 0, 'ag' at 1 based on oe2ee conversion
        # P_inv_apriori = np.zeros((6, 6))
        # P_inv_apriori[0, 0] = 1e9  # Lock eccentricity component
        # P_inv_apriori[1, 1] = 1e9  # Lock eccentricity component
        # P_inv_apriori[2, 2] = 1e9  # Lock semi-major axis

        # 3. INNER LOOP (Levenberg-Marquardt Accept/Reject Logic) 
         
        step_accepted = False
        flag = True                                                 

        while not step_accepted:
            
            # Apply damping and A-priori penalty
            matrice_damping = np.eye(6) * (self.damping_lambda * max_diag)
            awat_damped = awat + matrice_damping # + P_inv_apriori                                                        # if "a-priori" is implemented

            # Solve for the state update step (dx)
            inv_awat = np.linalg.inv(awat_damped) 
            d_state = np.dot(inv_awat, abw)

            # Compute TRIAL state (provisional state k+1)
            d_state_filtered = orb.filter_dstate(d_state, ee_step, loop)
            ee_trial = [x + dx for x, dx in zip(ee_step, d_state_filtered)]
            ee_trial = orb.fix_state(ee_trial)

            # Convert and propagate the TRIAL state to verify if the error decreases
            oe_trial = ee2oe(*ee_trial)
            rv_trial = np.array(oe2rv(*oe_trial)).flatten() * 1e3


            self.prop_config_trial["Propagation"]["InitialState"] = rv_trial.tolist()
            index_trial, data_trial = orb.propagate_orbits_wrapper(self.prop_config_trial)
            df_state_trial = pd.DataFrame(data=np.array(list(data_trial)), index=pd.DatetimeIndex(index_trial), columns=["randv_mks_{}".format(j) for j in range(6)])

            # Compute TRIAL versors
            versor_trial = self.find_relative(df_state_trial)

            # Compute TRIAL residuals and cost
            b_trial = self.versor_arr_meas - versor_trial
            cost_trial = np.sum(np.linalg.norm(b_trial, axis=1)**2)


            # STEP GOODNESS EVALUATION              
            
            if cost_trial < cost_old:
                # STEP ACCEPTED: The error decreased. 
                print(f"   -> STEP ACCEPTED: Cost decreased from {cost_old:.6f} to {cost_trial:.6f}")
                
                # 1. Update the actual state and residuals for the next main iteration
                ee_step = ee_trial
                versor_arr_comp = versor_trial
                b = b_trial                     
                
                if flag ==  True:               # the damping parameter is increaed only if also the previous step was accepted
                    # 2. Decrease damping factor to take larger Newton-like steps next time
                    self.damping_lambda = max(1e-5, self.damping_lambda / 10.0) 
                    print(f"Damping paramter reduced to {self.damping_lambda:.6f}.")
                else: 
                    print(f"Damping paramter maintained at {self.damping_lambda:.6f}.")
                
                flag == True
                # 3. Exit the inner while loop to proceed to iteration k+1
                step_accepted = True            
                
            else:
                # STEP REJECTED: The error increased (Overshooting).
                print(f"   -> STEP REJECTED: Cost increased to {cost_trial:.6f}. Increasing damping.")
                
                # 1. Do NOT update ee_step or b. We stay at iteration k.
                # 2. Increase damping factor to force a smaller, safer Gradient Descent step
                self.damping_lambda = min(1e9, self.damping_lambda * 10.0)
                flag == False

                # If damping gets unreasonably high, we are stuck in a local minimum
                if self.damping_lambda >= 1e9:
                    print("   -> WARNING: Damping limit reached. Stopping optimization.")
                    step_accepted = True # Force exit to prevent infinite loop
                else: 
                    print(f"Damping paramter increased to {self.damping_lambda:.6f}.")
            
        return df_state_trial, ee_step, versor_arr_comp, b
         

    def lsqr (self, b_init, w, W, versor_arr_init, epsilon=1e-9):                                                   # HOW TO CHANGE THE CALL FOR EPSILON?
        # Input:
        #       b_init:
        #       versor_arr_init:
        #       W:
        #       
        # Output:
        #       df_state:
        #       oe_final:
        #
        # We are setting the LS algortihm:
        # dx = (A' W A)^(-1) A' b
        # dx is the correction vector, 1x6 (correction to the initial state elements)
        # To avoid huge matrices, we use the einstein notation to compute the products A' A and A' b without explicitly forming the large A matrix.
        # A is the Jacobian matrix of the versor components with respect to the initial state elements, shape (6 x N x 3) (six matrices of 3 perturbed parameters for each measurement)
        # b is the residual vector (computed versor - measured versor), shape (N x 3)
        # Otherwise, we would need to flatten A, b for the three elemment of each measurement (r_x r_y r_z) to go from 3xN to a 1D array of size 3N (more computational expensive but more intuitive)

        versor_arr_comp = versor_arr_init
        b = b_init
        loop = 1
        sigmanew = np.mean(b ** 2 * w)
        sigmaold = 20000.0
        sigmaold2 = 30000.0
        ee_step = self.ee_initial
        df_step = self.df_client
        self.prop_config_loop = copy.deepcopy(orb.STK_CONFIG)
        self.prop_config_loop["Propagation"] = self.propagation_config
        self.prop_config_0 = copy.deepcopy(self.prop_config_loop)
        self.prop_config_trial = copy.deepcopy(self.prop_config_loop)

        # Compute initial error on versors
        versor_cost = np.linalg.norm(versor_arr_comp - self.versor_arr_meas, axis=1)
        print("")
        print(f"Initial absolut error on versors : {np.sum(versor_cost)}")
        print("")
        # LOOP: run till convergency is reached

        while ((#(abs((sigmanew - sigmaold) / sigmaold) >= epsilon) and 
                (loop < self.max_loops) and (sigmanew >= epsilon)) and 
                not ((sigmanew > sigmaold) and (sigmaold > sigmaold2) and (sigmanew > 500000.0))):
            sigmaold2 = sigmaold
            sigmaold = sigmanew

            a = self.a_matrix(ee_step, df_step, self.prop_config_loop)


        
            aw = a * W
            awat = np.einsum("ijk,ljk->il", aw, a)
            abw = np.einsum("ijk,jk->i", aw, b)
            
            # CALL Levenberg-Marquardt logic function
            max_diag = np.max(np.diag(awat))
            
            df_step, ee_step, versor_arr_comp, b = self.LevenbergMarquardt(b, abw, awat, ee_step, max_diag, loop)

            # Update sigmanew for the control in "while..."
            sigmanew = np.mean(b ** 2 * w)

            loop += 1
            # Compute error on versors
            versor_cost = np.linalg.norm(versor_arr_comp - self.versor_arr_meas, axis=1)
            
            print("")
            print(f"Iteration {loop}: Absolut erro on versors : {np.sum(versor_cost)}")
            print("")

        return df_step, ee_step, loop


    def LSLoop (self):  #(df_client, df_servicer, versor_arr_meas)
        # Function to initialize and launch the LS Loop optimization
        #
        # input:
        #       df_client,
        #       df_servicer: dataset of size (N,6) contaioning r,v points in [m,m/s], in ECI frame
        #       versor_arr_meas: dataset of size (N,3) containing measured relative directions, ECI frame 
        # Output:
        #       

        # Find initial relative direction (from initial guess orbit propagation)
        versor_arr_init = self.find_relative(self.df_client)

        # Initialize the residual wrt the measured directions
        b = self.versor_arr_meas - versor_arr_init      # residual vector (computed versor - measured versor) 

        # Weight function (u_x u_y u_z)
        w_i = np.array([1, 1, 1])                       # for the moment a equal weight has been assumed for each component of the versor (r_x r_y r_z)
        # weight matrix (3N x 3N , diag)
        W = np.tile(w_i, (self.n_measurements,1))       # shape (N, 3)
    
        # Call the loop
        df_optimized, ee_final, loop = self.lsqr (b, w_i, W, versor_arr_init)

        return df_optimized, ee_final, loop
