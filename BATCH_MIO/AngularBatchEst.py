import orbitfit.orbitfit as orb
from orbitfit.utils import (rv2oe, oe2rv, oe2ee, ee2oe)
import copy
import pandas as pd
import numpy as np
# for sun direction:
from astropy.time import Time
from astropy.coordinates import get_sun
class Optimizer:
    # Function to initialize and run the LS Loop optimization
    # fov_offset never insert
    # We could add other parameters as input instead of fixed

    def __init__(self, df_client, df_client_real, df_servicer, versor_arr_meas, config, config_0=None, 
                 damping_lambda=0.001, max_loops=30, epsilon=1e-9, w_i=None, deltaamtchg=1e-7, percentchg=1e-6, fov=0, fov_offset = 0, alphamax=0, m_v_threshold=1e6):
        """
        Initializes the Optimizer class with the initial state guess, reference datasets, 
        measurements, propagation configurations, and Levenberg-Marquardt solver parameters.

        Input:
            --df_client (DataFrame): Initial propagated state of the client in ECI frame from unflitered initial guess [m, m/s] (N, 6).
            --df_client_real (DataFrame): 
            --df_servicer (DataFrame): Known, fixed state of the servicer in ECI frame (N, 6).              [m, m/s] 
            --versor_arr_meas (ndarray): Measured relative directions (versors) in ECI frame (N, 3).        [-]
            --config (dict): Orekit propagation configuration (Step, Start, End).                           [-]
            --config_0 (dict): Full STK_CONFIG dictionary for the propagator template.                      [-]
            --damping_lambda (float): Initial damping parameter for the Levenberg-Marquardt algorithm.      [-]
            --max_loops (int): Maximum number of iterations allowed before forcing termination.             [-]
            --epsilon (float, optional): Relative tolerance threshold for the Levenberg-Marquardt convergence check (default 1e-10).        [-]
            --w_i (vector[]): Weight vector for the 3 spatial components. Defaults to [1.0, 1.0, 1.0].                                      [-]
            --deltaamtchg (float, optional): Minimum absolute perturbation step   for the Jacobian finite differences (default 1e-7).       [-]
            --percentchg (float, optional): Relative perturbation percentage for the Jacobian finite differences (default 1e-6).            [%]
            --fov: FOV semi-aperture to filter out-of-sight measurements, in degrees (if == 0 filter not applied - default)                     [deg]
            --fov_offset: Pointing direction. if positive pointing forward, otherwise pointin backward (if <=1e-6 or not specified: zenithal)   [deg]
            --alphamax: Maximum sun phase angle to see the target, in degrees (if == 0 filter not applied - default)                            [deg]
            --m_v_threshold: Maximum magnitude of reflected light to see the target, in log scale (if == 0 filter not applied - default)        [-]
                                                                                (suggested value 13, see the function for more details)      
        Output:
            Results and final plots saved in the folder  " .\tests\test_PropTime_StepLenght_MagnitudeValue_StartingDate "
        """

        self.df_client = df_client                      # otherwise we can propagate it inside from e_initial
        self.df_servicer = df_servicer
        self.df_client_real = df_client_real
        self.versor_arr_meas = versor_arr_meas
        self.propagation_config = config
        self.max_loops = max_loops
        self.damping_lambda = damping_lambda
        self.epsilon = epsilon
        self.deltaamtchg = deltaamtchg
        self.percentchg = percentchg   
        self.b_history = []                             

        # Initialize the mask and filters values
        self.mask = np.ones(len(self.versor_arr_meas), dtype=bool) 
        self.FOV = fov
        self.FOVoffset = fov_offset
        self.alphaMax = alphamax
        self.m_v_threshold = m_v_threshold
        self.phi = 0

        # weighting matrix initialization
        if w_i is None:
            self.w_i = np.array([1.0, 1.0, 1.0])
        else:
            self.w_i = np.array(w_i)       
        # config_0 initialization
        if config_0 is None:
            self.prop_config_0 = copy.deepcopy(orb.STK_CONFIG)
        else:
            self.prop_config_0 = copy.deepcopy(config_0)     


    def find_relative (self, df_client_current): 
        """
        Calculates the normalized relative position vector between the client (current estimated state) 
        and the servicer (fixed state).
        
        Input:
            df_client_current (DataFrame): Current propagated state of the client in ECI frame (N, 6).            
        Output:
            versor_arr_comp (ndarray): Computed normalized relative directions (versors) in ECI frame (N, 3).
        """

        diff = df_client_current.iloc[:, :3].values - self.df_servicer.iloc[:, :3].values
        ranges = np.linalg.norm(diff, axis=1)[:, np.newaxis] # Shape (N, 1)
        return diff / ranges # Shape (N, 3)
        

    #------------------------------------------#
    #       Filters implementation:            #
    #------------------------------------------#


    def FOV_filter(self):
        """
        Find the angle between earth normal-servicer-client to exclude out of FOV situations (zenithal pointing assumed)

        Input:
            FOV: semi aperture of Field of View (in degrees)
        Output:
            Updates self.mask keeping only the values inside the FOV.
        """        

        ###############################
        # ADD an offset if not zenithal pointing (instead of normal_versors)
        ###########

        if  self.FOV == 0:
            return
        
        print (f"--> Entering in FOV filter with an aperture of {self.FOV} and a offset equal to {self.FOVoffset}\n")
        # Normalized position vectors
        normal_versors = self.df_servicer[["randv_mks_0", "randv_mks_1", "randv_mks_2"]].values.astype(float)
        norms = np.linalg.norm(normal_versors, axis=1, keepdims=True)
        normal_versors = normal_versors / norms

        if self.FOVoffset <= 1e-6:       # Zenithal pointing
            pointing_versors = normal_versors
        else:
            # Extract servicer's position and velocity vectors to find the orbital plane
            pos = self.df_servicer.iloc[:, 0:3].values
            vel = self.df_servicer.iloc[:, 3:6].values
            
            # Compute the cross-track direction (perpendicular to the orbital plane)
            # angular momentum vector = r x v
            cross_track = np.cross(pos, vel)
            cross_norms = np.linalg.norm(cross_track, axis=1, keepdims=True)
            cross_versors = cross_track / cross_norms
            
            # Compute the true along-track direction (the exact "forward" direction)
            along_track = np.cross(cross_versors, normal_versors)
            
            # Apply the rotation in the orbital plane
            # A positive FOVoffset rotates the pointing vector towards the along_track vector
            pointing_versors = (normal_versors * np.cos(np.radians(self.FOVoffset)) + 
                                along_track * np.sin(np.radians(self.FOVoffset)))
            
            # Normalize just to be safe against floating-point drifts
            pointing_versors = pointing_versors / np.linalg.norm(pointing_versors, axis=1, keepdims=True)

        # Compare with visibility condition
        self.cos_angles = np.sum(pointing_versors * self.versor_arr_meas, axis=1)
        cos_lim = np.cos(np.radians(self.FOV))
        fov_mask = self.cos_angles >= cos_lim 

        # combine with the existing mask
        self.mask = self.mask & fov_mask

        if np.sum(self.mask) == 0:
            print ("All the points are filtered out. Exiting the simulation.")
            self.max_loops = 0          # Force the exit from lsqr loop
            return None
        else:
            print(f"valid points after FOV filter: {np.sum(self.mask)} over {len(self.mask)}")


    def sunVisibility_filter(self):
        """
        Find the angle between sun-servicer-client to exclude non-visibility situations (backlight)
        The direction between sun and servicer is been assumed to be the same of sun-earth.

        Input:
            alphaMax: Max angle to gvuarantee target visibility (offset 90 deg)
        Output:
            self.mask: Mask of "good" values 
        """

        if  self.alphaMax == 0:
            return
        
        print (f"--> Entering in sun visibility filter with a maximum sun phase angle of {self.alphaMax}\n")
        # time vector
        times = Time(self.df_servicer.index)

        #sun-earth position
        sun_coords = get_sun(times)
        sun_x = sun_coords.cartesian.x.to_value('m')
        sun_y = sun_coords.cartesian.y.to_value('m')
        sun_z = sun_coords.cartesian.z.to_value('m')
        light_arr = np.column_stack((-sun_x, -sun_y, -sun_z))              # invert sign to find incoming light direction
        norms = np.linalg.norm(light_arr, axis=1, keepdims=True)
        light_versors = light_arr / norms

        # compute the angle between sun-servicer-client (dot product for each line)
        cos_angles = np.sum(light_versors * self.versor_arr_meas, axis=1)  # each row correspond to cos(sun-client relative angle)
        cos_lim = np.cos(np.radians(self.alphaMax))
        sun_mask = cos_angles >= cos_lim                                 # saving only the angles>=alphaMax

        # combine with the existing mask
        self.mask = self.mask & sun_mask

        if np.sum(self.mask) == 0:
            print ("All the points are filtered out. Exiting the simulation.")
            self.max_loops = 0          # Force the exit from lsqr loop
            return None
        else:
            print(f"valid points after sun direction filter: {np.sum(self.mask)} over {len(self.mask)}")

        # This filter is applied AFTER the propagation, so Orekit computes all the points and then we cancle the unwanted ones. 
        # Anyway, the computational effort from Orekit would be the same if we filter the data before the call, since it has 
        # to propagate the orbit from the first to the last point --> no need to change it


    def detectability_filter(self, sun_flux=1361.0):
        """
        Find the incoming light reaching the camera from the servicer and compare with threshold detectability value.
        Formulas based on: "Observations and Modeling of GEO Satellites at Large Phase Angles", Rita L. Cognion (2013)

        Input:
            sun_flux: Solar flux constant [W/m^2]
        Output:
            self.mask: Mask of "good" values based on visual magnitude threshold           
        """

        # VALUES FOR MAX MAGNITUDE:
        # from Luis Calvo "Validation of models employed in the Far-Range Image Generator Software"
        # considering an exposure time of 1s, 30|40 deg temperature:
        # -No security factor: ~ 15.4|15.1 mag   -Security factor 10 ~ 13.3|13.0 mag    -Security factor 20 ~ 12.1|12.0 mag
        # considering an exposure time of 0.1s, 30|40 deg temperature:
        # -No security factor: ~ 12.6|12.1 mag   -Security factor 10 ~ 10.6|10.6 mag    -Security factor 20 ~ 10|9.1 mag
        # In our case: we don't need fast imaging, we can assume 1s exposure, security factor 10 --> magMax = 13

        if self.m_v_threshold == 1e6:
            return
        
        print (f"--> Entering in detectability filter with a maximum detectable magnitude of {self.m_v_threshold}\n")
        # Time vector
        times = Time(self.df_servicer.index)

        # Extract the area from the nested dictionary
        target_area = self.prop_config_0['SpaceObject']['Area'] 
        
        # Convert cross-sectional area to equivalent radius r 
        r = np.sqrt(target_area / np.pi)
        
        # Compute relative distance vector and magnitude (d)
        diff = self.df_client_real.iloc[:, :3].values - self.df_servicer.iloc[:, :3].values
        d = np.linalg.norm(diff, axis=1)
        self.d = d

        # Sun-Earth position        
        sun_coords = get_sun(times)
        sun_x = sun_coords.cartesian.x.to_value('m')
        sun_y = sun_coords.cartesian.y.to_value('m')
        sun_z = sun_coords.cartesian.z.to_value('m')
        
        # Invert sign to find incoming light direction
        light_arr = np.column_stack((-sun_x, -sun_y, -sun_z))
        norms = np.linalg.norm(light_arr, axis=1, keepdims=True)
        light_versors = light_arr / norms

        # Compute the angle between sun-servicer-client (dot product for each line)
        dot_prod = np.sum(light_versors * self.versor_arr_meas, axis=1)
        
        # Clip dot product to avoid floating point errors (e.g., cos=1.0000001)
        phi = np.arccos(np.clip(dot_prod, -1.0, 1.0)) 

        # -----------------------------------------------------------------
        # COGNION (2013) EMPIRICAL PHASE FUNCTION CALCULATION
        # -----------------------------------------------------------------
        # The polynomials directly model the brightness variation. 
        # We clip the angle for the polynomial input between 25 and 150 deg 
        # to ensure the empirical model behaves correctly at the boundaries.
        
        phi_calc = np.clip(phi, np.radians(25), np.radians(150))
        a_0 = np.zeros_like(phi)

        # Case 1: phi < 100 deg (First Polynomial)
        mask_poly1 = (phi_calc < np.radians(100))
        p1 = phi_calc[mask_poly1]
        a_0[mask_poly1] = (3.1765 * p1**6 - 22.0968 * p1**5 + 
                           62.182 * p1**4 - 90.0993 * p1**3 + 
                           70.3031 * p1**2 - 27.9227 * p1 + 4.7373)

        # Case 2: 100 <= phi <= 150 deg (Second Polynomial)
        mask_poly2 = (phi_calc >= np.radians(100))
        p2 = phi_calc[mask_poly2]
        a_0[mask_poly2] = (0.510905 * p2**3 - 2.72607 * p2**2 + 4.96646 * p2 - 3.02085)

        # Case 3: phi > 150 deg 
        # Forward scattering region / Eclipse: force albedo term to 0
        a_0[phi > np.radians(150)] = 0.0

        # Reflected flux (Lambertian phase function removed, Cognion polynomial already accounts for it)
        f_diff = (2/3) * a_0 * (r**2 / (np.pi * d**2))

        # Initialize apparent magnitude with a very high value (30.0 = completely dark/invisible)
        self.m_v = np.full_like(phi, 30.0) 
        
        # Calculate valid magnitudes (where flux is strictly positive)
        valid = (f_diff > 0)
        self.m_v[valid] = -26.74 - 2.5 * np.log10(f_diff[valid] / sun_flux)

        self.phi = phi
        
        # Detectability Mask: Keep only measurements brighter than the threshold
        det_mask = self.m_v < self.m_v_threshold

        # Update global mask
        self.mask = self.mask & det_mask
        
        if np.sum(self.mask) == 0:
            print("All the points are filtered out. Exiting the simulation.")
            self.max_loops = 0          # Force the exit from lsqr loop
            return None
        else:
            print(f"Valid points after sun direction filter: {np.sum(self.mask)} over {len(self.mask)}")


    def applyMask(self, df):
        '''
        Apply the filter Mask found in FOV and SunPhaseAngle
        input:
            df: unfiltered dataframe
        output:
            df_filtered: filtered dataframe
        '''
        if np.any(self.mask):       
            df = df[self.mask]
        else:
            print("Empty mask. Not applied.\n")
        return df        
    

    #-----------------------------------------------------------#
    #       LSQR functions and logic implementation:            #
    #-----------------------------------------------------------#


    def a_matrix (self, ee_step, df_state, prop_config_loop):
        """
        Computes the Jacobian matrix (A) using finite differences. It perturbs each of the 
        6 equinoctial elements one by one, propagates the perturbed orbit, and calculates 
        the resulting variation in the computed versors.
        
        Input:
            ee_step: Current estimated equinoctial elements (6,).
            df_state: Propagated state corresponding to ee_step (N, 6).
            prop_config_loop: Orekit configuration template to use for propagation.            
        Output:
            a (ndarray): Jacobian matrix containing partial derivatives of versor components 
                         with respect to the 6 initial state elements. Shape: (6, N, 3).
        """

        # we have to feed find_a with the equinoctial state [a, h, k, p, q, lambda] (ee_step) since this is the one used by Orekit  

        a = []   

        for i in range(len(ee_step)):                      # cycling through the 6 elements of the state
                    ee_state_mod = copy.copy(ee_step)        
                    counter = 0
                    percentchg_local = self.percentchg
                    deltaamt = ee_state_mod[i] * percentchg_local
                    while abs(deltaamt) < self.deltaamtchg and counter <=5:
                        counter += 1
                        percentchg_local = 1.4 * percentchg_local
                        deltaamt = ee_state_mod[i] * percentchg_local
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
                    df_state_mod = self.applyMask(df_state_mod)

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
    

    def LevenbergMarquardt (self, b, abw, awat, ee_step, loop, df_step_old, versor_arr_comp_old, max_diag = 1):
        """
        Executes the inner loop of the Levenberg-Marquardt algorithm. Applies a scaled damping 
        factor to the normal equations and solves for the state correction step (dx). 
        Propagates the "trial" state to evaluate the new cost: 
            - If cost decreases: accepts the step and reduces the damping factor (closer to Gauss-Newton).
            - If cost increases (or integrator crashes): rejects the step and increases the damping 
              factor (closer to Gradient Descent) to take a safer, smaller step.
        
        Input:
            b (ndarray): Current residual vector (Measured Versor - Computed Versor) (N, 3).
            abw (ndarray): The right-hand side of the normal equations (A' * W * b) (1x6).
            awat (ndarray): The left-hand side of the normal equations (A' * W * A) (6x6).
            ee_step (list/array): Current accepted equinoctial elements (1x6).
            max_diag (float): Maximum value on the diagonal of awat, used to scale the damping.
            loop (int): Current iteration index (used for filtering rules).
            df_state_old: Propagated state from previous step, used if the optimization step fails (N, 6).
            versor_arr_comp_old: Computed versors corresponding to df_state_old (same rationale) (N, 3).            
        Output:
            df_state_trial: New accepted propagated state of the client (N, 6).
            ee_step: Updated and accepted equinoctial elements (6,).
            versor_arr_comp: Computed versors corresponding to df_state_trial (N, 3).
            b: Updated residual vector for the next iteration (N, 3).
        """       

        # 1. Initialization

        # Cost function: control of the goodness of the next step (if too long, reduce the step)
        # (if new<old accept new step)

        cost_old = np.sum((np.linalg.norm(b* self.W, axis=1)**2))

        # Damping parameter and scaling

        # # A-priori covariance matrix to penalize and lock the semi-major axis (a)                                       To implement as optional args?
        # # Assuming 'a' is at index 2, 'af' at 0, 'ag' at 1 based on oe2ee conversion
        # P_inv_apriori = np.zeros((6, 6))
        # P_inv_apriori[0, 0] = 1e9  # Lock eccentricity component
        # P_inv_apriori[1, 1] = 1e9  # Lock eccentricity component
        # P_inv_apriori[2, 2] = 1e9  # Lock semi-major axis

        # 3. INNER LOOP (Levenberg-Marquardt Accept/Reject Logic) 
         
        step_accepted = False
        flag = True                                                 

        while not step_accepted:
            
            # extracting the values for each lambda parameter.
            diag_awat = np.diag(awat)
            diag_awat_safe = np.where(diag_awat > 1e-12, diag_awat, 1e-12)  # correct to avoid singularities (no division by 0)

            # Apply damping and A-priori penalty
            matrice_damping = np.diag(self.damping_lambda * diag_awat_safe)

            # Otherwise (unweighted lambda):
            #matrice_damping = np.eye(6) * (self.damping_lambda * max_diag)

            awat_damped = awat + matrice_damping # + P_inv_apriori          # if "a-priori" is implemented

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

            # try/except: Gemini suggestion to avoid a crash due to propagation of an impossible step (correction values too high):
            # If the input configuration is non-physical, the stepsize is reduced to dump the correction:

            try:                                                            
                index_trial, data_trial = orb.propagate_orbits_wrapper(self.prop_config_trial)
                df_state_trial = pd.DataFrame(data=np.array(list(data_trial)), index=pd.DatetimeIndex(index_trial), columns=["randv_mks_{}".format(j) for j in range(6)])
                df_state_trial = self.applyMask(df_state_trial)

                # Compute TRIAL versors
                versor_trial = self.find_relative(df_state_trial)

                # Compute TRIAL residuals and cost
                b_trial = self.versor_arr_meas - versor_trial
                cost_trial = np.sum((np.linalg.norm(b_trial*self.W, axis=1)**2))

            except Exception as e:                 
                # INTEGRATOR CRASHED: The trial state is physically impossible.
                # Treat this exactly as if the cost went to infinity.
                print("   [!] INTEGRATOR CRASH: Trial state is non-physical. Rejecting step.")
                cost_trial = 2*cost_old # Force the "STEP REJECTED" branch below
                df_state_trial = None # Just to have a placeholder


            # STEP GOODNESS EVALUATION              
            
            if cost_trial < cost_old:
                # STEP ACCEPTED: The error decreased. 
                print(f"   -> STEP ACCEPTED: Cost decreased from {cost_old:.6f} to {cost_trial:.6f}")
                
                # 1. Update the actual state and residuals for the next main iteration
                ee_step = ee_trial
                versor_arr_comp = versor_trial
                b = b_trial                     
                
                if flag ==  True:               # the damping parameter is increaed only if the previous step was accepted
                    # 2. Decrease damping factor to take larger Newton-like steps next time
                    self.damping_lambda = max(1e-6, self.damping_lambda / 10.0) 
                    print(f"Damping paramter reduced to {self.damping_lambda:.6f}.")
                else: 
                    print(f"Damping paramter maintained at {self.damping_lambda:.6f}.")
                
                flag = True
                # 3. Exit the inner while loop to proceed to iteration k+1
                step_accepted = True            
                
            else:
                # STEP REJECTED: The error increased (Overshooting).
                print(f"   -> STEP REJECTED: Cost increased to {cost_trial:.6f}. Increasing damping.")
                
                # 1. Do NOT update ee_step or b. We stay at iteration k.
                # 2. Increase damping factor to force a smaller, safer Gradient Descent step
                self.damping_lambda = min(1e9, self.damping_lambda * 10.0)
                flag = False

                # If damping gets unreasonably high, we are stuck in a local minimum
                if self.damping_lambda >= 1e9:
                    print("   !! WARNING: Damping limit reached. Stopping optimization.")
                    print("\nForcing the exit from the minimization loop ( step(t)=step(t-1) ).\n")
                    print("We're stucked!")                    
                    step_accepted = True # Force exit to prevent infinite loop
                    df_state_trial = df_step_old
                    versor_arr_comp = versor_arr_comp_old
                else: 
                    print(f"Damping paramter increased to {self.damping_lambda:.6f}.")
            
        return df_state_trial, ee_step, versor_arr_comp, b
         

    def lsqr (self, b_init, versor_arr_init):                                                   # HOW TO CHANGE THE CALL FOR EPSILON?
        """
        Runs the main iterative Batch Least Squares loop. It checks for convergence 
        based on the relative variation of the cost function (epsilon). In each iteration, 
        it computes the Jacobian matrix, builds the normal equations using Einstein summation 
        (no flattening), and calls the Levenberg-Marquardt logic to update the state.
        
        Input:
            b_init (ndarray): Initial residual vector (N, 3).
            w (ndarray): Weight vector for variance calculation (1x3).
            W (ndarray): Expanded weight matrix for normal equations contraction (N, 3).
            versor_arr_init (ndarray): Initial computed versors from the first guess (N, 3).            
        Output:
            df_step (DataFrame): Final optimized propagated state of the client (N, 6).
            ee_step (array): Final optimized equinoctial elements (6,).
            loop (int): Total number of iterations performed.
        """

        # We are setting the LS algortihm:
        # dx = (A' W A)^(-1) A' b
        # dx is the correction vector, 1x6 (correction to the initial state elements)
        # To avoid huge matrices, we use the einstein notation to compute the products A' A and A' b without explicitly forming the large A matrix.
        # A is the Jacobian matrix of the versor components with respect to the initial state elements, shape (6 x N x 3) (six matrices of 3 perturbed parameters for each measurement)
        # Otherwise, we would need to flatten A, b for the three elemment of each measurement (r_x r_y r_z) to go from 3xN to a 1D array of size 3N 
        # (more computational expensive but more intuitive)

        versor_arr_comp = versor_arr_init
        b = b_init
        loop = 1
        sigmanew = np.mean(b ** 2 * self.w_i)
        sigmaold = 20000.0
        sigmaold2 = 30000.0
        ee_step = self.ee_initial
        df_step = self.df_client
        self.prop_config_0["Propagation"] = self.propagation_config
        self.prop_config_loop = copy.deepcopy(self.prop_config_0)
        self.prop_config_trial = copy.deepcopy(self.prop_config_0)

        # Compute initial error on versors
        versor_cost = np.linalg.norm(versor_arr_comp - self.versor_arr_meas, axis=1)
        print("")
        print(f"Initial absolut error on versors : {np.sum(versor_cost)}")
        print("")
        # LOOP: run till convergency is reached

        while (((abs((sigmanew - sigmaold) / sigmaold) >= self.epsilon) and 
                (loop < self.max_loops) and (sigmanew >= self.epsilon)) and 
                not ((sigmanew > sigmaold) and (sigmaold > sigmaold2) and (sigmanew > 500000.0))):
            sigmaold2 = sigmaold
            sigmaold = sigmanew

            a = self.a_matrix(ee_step, df_step, self.prop_config_loop)


        
            aw = a * self.W
            awat = np.einsum("ijk,ljk->il", aw, a)
            abw = np.einsum("ijk,jk->i", aw, b)
            
            # CALL Levenberg-Marquardt logic function
            max_diag = np.max(np.diag(awat))
            
            df_step, ee_step, versor_arr_comp, b = self.LevenbergMarquardt(b, abw, awat, ee_step, loop, df_step, versor_arr_comp, max_diag)

            # Update sigmanew for the control in "while..."
            sigmanew = np.mean(b ** 2 * self.w_i)
            self.b_history.append(pd.DataFrame(b, columns=[f'Res_X_it{loop}', f'Res_Y_it{loop}', f'Res_Z_it{loop}']))           # converting b in panda format

            loop += 1
            # Compute error on versors
            versor_cost = np.linalg.norm(versor_arr_comp - self.versor_arr_meas, axis=1)
            
            print("")
            print(f"Iteration {loop} - Sum of all the versors residuals: {np.sum(versor_cost)}")
            print("")

            # Check termination reasons
        print("\n--- Optimization Terminated ---")
        if not (abs((sigmanew - sigmaold) / sigmaold) >= self.epsilon):
            print(f"Termination: Convergence reached (Relative change in sigma {abs((sigmanew - sigmaold) / sigmaold):.2e} < epsilon)")
        
        if not (loop < self.max_loops):
            print(f"Termination: Maximum number of iterations reached (loop = {loop})")
        
        if not (sigmanew >= self.epsilon):
            print(f"Termination: Absolute error sigma is below threshold (sigmanew = {sigmanew:.2e})")
            
        if (sigmanew > sigmaold) and (sigmaold > sigmaold2) and (sigmanew > 500000.0):
            print("Termination: Divergence detected (Sigma has increased for two consecutive steps and exceeds safety limit)")

        # exportinmg the covariance matrix P to see the predicted quality of the final fit
        inv_awat = np.linalg.inv(awat)
        sigma_squared = (sigmanew**2)

        # compute adn save covariance and std of the 6 unknowns (sqrt(variance))
        self.covariance_matrix = inv_awat * sigma_squared 
        self.param_errors = np.sqrt(np.diag(self.covariance_matrix))       

        self.versor_arr_comp = versor_arr_comp
        
        return df_step, ee_step, loop


    def LSLoop (self): 
        """
        Entry point for the Angular Batch Estimator. It extracts the initial computed versors 
        from the first guess, initializes the residual vector (b), constructs the weight 
        matrix (W), and triggers the main optimization loop (`lsqr`).
        
        Input:
            None (class must be correctly initiaized).            
        Output:
            df_optimized (DataFrame): Final optimized state of the client (N, 6).
            ee_final (array): Final optimized equinoctial elements (6,).
            loop (int): Total number of iterations it took to converge (or max out).
        """

        ## ADD IF w=111 --> UNWEIGHTED case      

        # filter out for FOV and Sun Phase Angle
        if self.alphaMax != 0:
            self.sunVisibility_filter() 
        if self.FOV != 0:
            self.FOV_filter()
        if self.m_v_threshold != 1e6:
            self.detectability_filter()
        # Check if mask is empty
        if np.sum(self.mask) == 0:
            return 0, 0, 0      
        
        self.df_client = self.applyMask(self.df_client)
        self.df_servicer = self.applyMask(self.df_servicer)
        self.versor_arr_meas = self.applyMask(self.versor_arr_meas)
        self.n_measurements = len(self.versor_arr_meas)

        self.oe_initial = rv2oe(self.df_client.iloc[0, 0:3].values * 1e-3, self.df_client.iloc[0, 3:6].values * 1e-3)
        self.ee_initial = np.array(oe2ee(*self.oe_initial), dtype=float)

        # Find initial relative direction (from initial guess orbit propagation)
        versor_arr_init = self.find_relative(self.df_client)

        # Initialize the residual wrt the measured directions
        b = self.versor_arr_meas - versor_arr_init      # residual vector (measured - computed versor) 
        self.b_history.append(pd.DataFrame(b, columns=[f'Res_X_it{0}', f'Res_Y_it{0}', f'Res_Z_it{0}']))

        # weight matrix (3N x 3N , diag)
        self.W = np.tile(self.w_i, (self.n_measurements,1))       # shape (N, 3)
    
        # Call the loop
        df_optimized, ee_final, loop = self.lsqr (b, versor_arr_init)


        self.b_history = pd.concat(self.b_history, axis=1)

        return df_optimized, ee_final, loop
