import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

import copy
import numpy as np
import pandas as pd
import datetime
import dateutil.parser
import scipy.optimize
import matplotlib.pyplot as plt
from sgp4.earth_gravity import wgs84
from sgp4.propagation import sgp4, sgp4init

from orbdetpy.propagation import propagate_orbits
import orbdetpy
from orbdetpy.conversion import get_J2000_epoch_offset, get_UTC_string
from .utils import interp, rv2oe, oe2rv, ee2oe, oe2ee
from .astronomical_constants import R_mean_earth, mu_earth

import logging

logger = logging.getLogger(__name__)


PRECISE_CONFIG = {'Gravity': {'Degree': 80, 'Order': 80},
                  'OceanTides': {'Degree': 20, 'Order': 20},
                  'Drag': {'Model': 'MSISE2000', 'Coefficient': {'Value': 2.0}},
                  'SolidTides': {'Sun': True, 'Moon': True},
                  'ThirdBodies': {'Sun': True, 'Moon': True},
                  'RadiationPressure': {'Sun': True, 'Creflection': {'Value': 1.5}},
                  'SpaceObject': {'Mass': 8, 'Area': 0.3,
                                  'Attitude': {'Provider': 'NADIR_POINTING', 'SpinVelocity': [0.0, 0.0, 0.0],
                                               'SpinAcceleration': [0.0, 0.0, 0.0]}}}
STK_CONFIG = {'Gravity': {'Degree': 21, 'Order': 21},
              'OceanTides': {'Degree': -1, 'Order': -1},
              'Drag': {'Model': 'MSISE2000', 'Coefficient': {'Value': 1.63}},
              'SolidTides': {'Sun': True, 'Moon': True},
              'ThirdBodies': {'Sun': True, 'Moon': True},
              'RadiationPressure': {'Sun': True, 'Creflection': {'Value': 1.0}},
              'SpaceObject': {'Mass': 8, 'Area': 0.3,
                              'Attitude': {'Provider': 'NADIR_POINTING', 'SpinVelocity': [0.0, 0.0, 0.0],
                                           'SpinAcceleration': [0.0, 0.0, 0.0]}}}


def propagate_orbits_wrapper(cfg):
    config = orbdetpy.configure(prop_start=get_J2000_epoch_offset(cfg["Propagation"]["Start"]),
    prop_initial_state=cfg["Propagation"]["InitialState"],
    prop_end=get_J2000_epoch_offset(cfg["Propagation"]["End"]),
    prop_step=cfg["Propagation"]["Step"],
    gravity_degree=cfg["Gravity"]['Degree'],
    gravity_order=cfg["Gravity"]['Order'],
    ocean_tides_degree=cfg["OceanTides"]["Degree"],
    ocean_tides_order=cfg["OceanTides"]["Order"],
    third_body_sun=cfg["ThirdBodies"]["Sun"],
    third_body_moon=cfg["ThirdBodies"]["Moon"],
    solid_tides_sun=cfg["SolidTides"]["Sun"],
    solid_tides_moon=cfg["SolidTides"]["Moon"],
    drag_model=getattr(orbdetpy.DragModel, cfg["Drag"]["Model"], 2),
    drag_coefficient=orbdetpy.Parameter(value=cfg["Drag"]["Coefficient"]["Value"], min=1.0, max=3.0, estimation=orbdetpy.EstimationType.UNDEFINED),
    rp_sun=cfg["RadiationPressure"]["Sun"],
    rp_coeff_reflection=orbdetpy.Parameter(value=cfg["RadiationPressure"]["Creflection"]["Value"], min=1.0, max=2.0, estimation=orbdetpy.EstimationType.UNDEFINED),
    rso_mass=cfg["SpaceObject"]["Mass"],
    rso_area=cfg["SpaceObject"]["Area"],
    rso_attitude_provider=getattr(orbdetpy.AttitudeType,cfg["SpaceObject"]["Attitude"]["Provider"], 0),
    rso_spin_velocity=cfg["SpaceObject"]["Attitude"]["SpinVelocity"],
    rso_spin_acceleration=cfg["SpaceObject"]["Attitude"]["SpinAcceleration"])

    if "Maneuvers" in cfg:
        for maneuver in cfg["Maneuvers"]:
            orbdetpy.add_maneuver(cfg=config, time=get_J2000_epoch_offset(maneuver["Time"]),
             trigger_event = getattr(orbdetpy.ManeuverTrigger, maneuver["TriggerEvent"], 0),
             trigger_params = maneuver['TriggerParams'],
             maneuver_type=getattr(orbdetpy.ManeuverType, maneuver["ManeuverType"], 0),
             maneuver_params=maneuver["ManeuverParams"])

    output = propagate_orbits([config])
    index = []
    data = []
    for x in output[0].array:
        index.append(get_UTC_string(x.time))
        data.append(x.true_state)
    return index, data


def inclination_change_get_directions(df_gps, propulsion_start, total_propulsion_time_s, inclination=1):
    propulsion_end = propulsion_start + datetime.timedelta(seconds=total_propulsion_time_s)
    df_gps_subset = df_gps[df_gps.index > propulsion_start]
    df_gps_subset["orbital_momentum_direction"] = -1*inclination
    positive_mask = np.dot(df_gps_subset[[f"randv_mks_{i}" for i in [3,4,5]]].values, np.array([0,0,1])) > 0
    df_gps_subset.loc[positive_mask, "orbital_momentum_direction"] = 1*inclination

    manoeuvres = []
    start_date = propulsion_start
    df_gps_orbital_changes = df_gps_subset["orbital_momentum_direction"][df_gps_subset["orbital_momentum_direction"].diff(-1) != 0.0]
    for end_date, value in df_gps_orbital_changes.items():
        if end_date > propulsion_end:
            end_date = propulsion_end
        duration = (end_date - start_date).total_seconds()
        if duration < 1.0:
            continue
        manoeuvres.append({"direction": value*np.array([0,1,0]), "date": start_date, "duration":duration})
        start_date = end_date

    return manoeuvres


def add_maneuver_to_config_dict(config_dict, propulsion_date, propulsion_time, direction, thrust_N, isp=9999):
    """
    Adda a maneuver entry to config_dict for orbdetpy simulation
    :param config_dict: config dict in which the maneuvers are added under the 'Maneuvers' key
    :param propulsion_date: datetime object with propulsion starting time
    :param propulsion_time: propulsion duration (constant thrust assumed)
    :param direction: direction of thrust vector in orbital frame (numpy unit vector array)
    :param thrust_N: Mean thrust value in Newtons
    :param isp: Specific Impulse. By default is taken as 9999, having little impact in the mass loss of the spacecraft.
    :return: orbdetpy config_dict with maneuver added.
    """
    maneuver_config = dict()
    maneuver_config["TriggerEvent"] = "DATE_TIME"
    maneuver_config["Time"] = (propulsion_date).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    maneuver_config["TriggerParams"] = [0]
    maneuver_config["ManeuverType"] = "CONSTANT_THRUST"
    maneuver_config["ManeuverParams"] = direction.tolist()+[propulsion_time, thrust_N, isp]
    if "Maneuvers" in config_dict:
        config_dict["Maneuvers"].append(maneuver_config)
    else:
        config_dict["Maneuvers"] = [maneuver_config]
    return config_dict


# This are the two functions that need to change depending on TLE propagator or OREKIT

class GenericFit:

    def get_state_error(self, state, df_pos_vel, inverse=False):
        df_out = self.propagate_state(state)
        if inverse:
            df_diff = df_pos_vel - df_out
        else:
            df_diff = df_out - df_pos_vel
        posvel_error_km = df_diff[["randv_mks_{}".format(i) for i in range(6)]].values / 1000
        return posvel_error_km

    def propagate_state(self, state):
        """
        This functions depends on the propagator type, and the format of the measurements.
        """
        pass

    def fix_state(self, state, statetype="e"):
        """Checks that the state describes a posible orbit"""
        if statetype == "e":
            # En el codigo de vallado, el "a" esta definido a partir de no, y no de no_kozai
            a = state[2]
            if a < 1.0:
                state[2] = a = 1.01 #Equivalent to 60km alt
                logger.warning("a can be less than 1.0, changes to 1.01")
            no_min = 1e-5 #missing units for this
            tumin = np.sqrt(R_mean_earth**3/mu_earth) / 60.0
            a_no_min = (no_min*tumin)**(-2.0 / 3.0)
            if a > a_no_min:
                state[2] = a = 0.99*a_no_min  # Equivalent to 60km alt
                logger.warning("due to minimum no of 1e-5, a can be less than {}. Changin a to {}".format(a_no_min, a))
            inclo = 2 * np.arctan(np.sqrt(state[4] ** 2 + state[5] ** 2))
            nodeo = np.arctan2(state[4], state[5])
            if nodeo < 0.0:
                nodeo += 2*np.pi
                state[4] = (np.tan(inclo * 0.5) * np.sin(nodeo))
                state[5] = (np.tan(inclo * 0.5) * np.cos(nodeo))
            argpo = np.arctan2(state[1], state[0]) - nodeo
            ecco = np.sqrt(state[0] ** 2 + state[1] ** 2)
            if ecco > 1.0:
                ecco = 0.9
                state[0] = (ecco * np.cos(argpo + nodeo))
                state[1] = (ecco * np.sin(argpo + nodeo))
        return state

    def filter_dstate(self, d_state, state, loop):
        d_state_filtered = copy.copy(d_state)
        for i in range(len(d_state)):
            if loop > -1 and abs(d_state[i]/state[i]) > 1000:
                d_state_filtered[i] = 0.1 * state[i] * np.sign(d_state[i])
            elif loop > 0 and abs(d_state[i]/state[i]) > 200:
                d_state_filtered[i] = 0.3 * state[i] * np.sign(d_state[i])
            elif loop > 0 and abs(d_state[i]/state[i]) > 100:
                d_state_filtered[i] = 0.7 * state[i] * np.sign(d_state[i])
            elif loop > 0 and abs(d_state[i]/state[i]) > 10:
                d_state_filtered[i] = 0.9 * state[i] * np.sign(d_state[i])
        return d_state_filtered

    def find_a(self, state, deltaamtchg=1e-6, percentchg=0.001):
        a = []
        df_state = self.propagate_state(state)
        for i in range(len(state)):
            state_mod = copy.copy(state)
            counter = 0
            percentchg_local = percentchg
            deltaamt = state_mod[i] * percentchg_local
            while abs(deltaamt) < deltaamtchg and counter <=5:
                counter += 1
                percentchg_local = 1.4 * percentchg_local
                deltaamt = state_mod[i] * percentchg_local
            logger.debug(f"Calculating a for state {i}: {state[i]} with deltaamt: {deltaamt}")
            state_mod[i] += deltaamt
            state_mod = self.fix_state(state_mod)
            logger.debug(f"state {i} changed to: {state_mod[i]}")
            posvel_error_km = self.get_state_error(state_mod, df_state)
#            if logger.level == logging.DEBUG:
#                fig, ax = plt.subplots(2, 1)
#                fig.suptitle(f"Error for state {i} with deltaamt:{deltaamt}")
#                ax[0].plot(df_pos_vel.index, error_pos_km)
#                ax[0].set_ylabel("Pos error [km]")
#                ax[1].plot(df_pos_vel.index, error_vel_km_s)
#                ax[1].set_ylabel("Vel error [km/s]")
#                plt.show()
            a.append(posvel_error_km/deltaamt)
        return np.asarray(a)

    def lsqr_loop(self, state, b, w, loop, deltaamtchg, percentchg, svd=False):
        a = self.find_a(state, deltaamtchg, percentchg)
        aw = a * w
        awat = np.einsum("ijk,ljk->il", aw, a)
        abw = np.einsum("ijk,jk->i", aw, b)

        inv_awat = np.linalg.inv(awat)
        d_state = np.dot(inv_awat, abw)
        d_state_filtered = self.filter_dstate(d_state, state, loop)
        new_state = [x + dx for x, dx in zip(state, d_state_filtered)]
        new_state = self.fix_state(new_state)
        return new_state

    def _run_fit(self, max_loops=5, percentchg=0.01, deltaamtchg=1e-6, epsilon=1e-8):
        """
        Fit position propagation to gps inertial data
        :param max_loops: Maximum number of optimization loops.
        :param percentchg:
        :param deltaamtchg:
        :param epsilon:
        :param optimize_maneuver:
        :return:
        """
        # <---
        state = self.initial_state
        posvel_error_km = self.get_state_error(state, self.df_gps, inverse=True)
        b = posvel_error_km
        w = np.array([1, 1, 1, .01, .01, .01])
        sigmanew = np.mean(b ** 2 * w)
        sigmaold = 20000.0
        sigmaold2 = 30000.0
        loop = 0

        logger.info("{}: {}".format(-1, sigmanew))
        logger.info("\tstate = {}".format(state))
        while (((abs((sigmanew - sigmaold) / sigmaold) >= epsilon)
                and (loop < max_loops) and (sigmanew >= epsilon)) and not
               ((sigmanew > sigmaold) and (sigmaold > sigmaold2) and (sigmanew > 500000.0))):
            sigmaold2 = sigmaold
            sigmaold = sigmanew
            try:
                new_state = self.lsqr_loop(state, b, w, loop, deltaamtchg, percentchg)
            except Exception as e:
                logger.exception(e)
                logger.warning(f"Something went wrong! returning state from loop {loop}")
                break
            posvel_error_km = self.get_state_error(new_state, self.df_gps, inverse=True)
            b = posvel_error_km
            w = np.array([1, 1, 1, 1, 1, 1])
            sigmanew = np.mean(b ** 2 * w)
            state = copy.copy(new_state)
            logger.info("{}: {}".format(loop, sigmanew))
            logger.info("\tstate = {}".format(state))
            loop += 1
        return state


class OrekitFit(GenericFit):

    def __init__(self, df_gps_eci, baseconfig=STK_CONFIG, step=10, optimize_area=False, optimize_maneuver=False):
        """
        :param df_gps_eci: Pandas dataframe with "randv_mks_{0-2}" keys for position x,y,z, "randv_mks_{3-5}" for velocity
        x,y,z, and timestamp as index. Position and velocity are expresed in ECI frame.
        :param initial_state: If not None, should be a 6 element list with position<xyz> in [m] and velocity<xyz> iin [m/s],
        corresponding to first GPS time.
        :param baseconfig: Dict with basic configuration for OREKIT simulation. By default it uses STK HPOP equivalent config.
        :param step: Simulation step in sec.
        :param optimize_maneuver:
        """
        skip_initialization = False
        if "Propagation" in baseconfig.keys():
            logger.info("Existing propagation setup in config file, using it")
            try:
                start_time = dateutil.parser.parse(baseconfig["Propagation"]['Start']).replace(tzinfo=df_gps_eci.index.tzinfo)
                assert abs(start_time - df_gps_eci.index[0]) < datetime.timedelta(seconds=step), "Mismatch in Start"
                end_time = dateutil.parser.parse(baseconfig["Propagation"]['End']).replace(tzinfo=df_gps_eci.index.tzinfo)
                assert abs(end_time - df_gps_eci.index[-1]) < datetime.timedelta(seconds=step), "Mismatch in End"
                assert len(baseconfig["Propagation"]["InitialState"]) == 6, "Mismatch in state size"
            except Exception as e:
                logger.info("Fail to use existing propagation values")
                logger.info(e)
            else:
                skip_initialization = True
        if not skip_initialization:
            propagation_config = dict()
            propagation_config["InitialState"] = df_gps_eci[["randv_mks_{}".format(i) for i in range(6)]].iloc[0].values.tolist()
            propagation_config["Step"] = step
            propagation_config['Start'] = df_gps_eci.index[0].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            propagation_config['End'] = df_gps_eci.index[-1].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            baseconfig["Propagation"] = propagation_config
        self.baseconfig = baseconfig
        self.optimize_area = optimize_area
        self.optimize_maneuver = optimize_maneuver
        self.df_gps = df_gps_eci
        self.initial_state = self.config2state(baseconfig)

    def state2config(self, state):
        """Uses baseconfig, bug"""
        config = copy.deepcopy(self.baseconfig)
        oe = ee2oe(*state[:6])
        r, v = oe2rv(*oe)
        config["Propagation"]["InitialState"] = (1000 * r).tolist() + (1000 * v).tolist()
        if self.optimize_area:
            config["SpaceObject"]["Area"] = state[6]
            maneuver_index = 7
        else:
            maneuver_index = 6
        if self.optimize_maneuver and "Maneuvers" in config.keys():
            for thrust, maneuver in zip(state[maneuver_index:], config["Maneuvers"]):
                if thrust < 0:
                    # orbdetpy does not accept negative values of thrust, therefore direction must be make negative
                    direction = np.array(maneuver["ManeuverParams"][:3])
                    maneuver["ManeuverParams"][:3] = (-1 * direction).tolist()
                    maneuver["ManeuverParams"][4] = -1 * thrust
                else:
                    maneuver["ManeuverParams"][4] = thrust
        return config

    def config2state(self, config):
        rv = config["Propagation"]["InitialState"]
        r = np.array(rv[0:3]) / 1000
        v = np.array(rv[3:6]) / 1000
        oe = rv2oe(r, v)
        ee = oe2ee(*oe)
        state = list(ee)
        if self.optimize_area:
            area = config["SpaceObject"]["Area"]
            state.append(area)
        if self.optimize_maneuver and "Maneuvers" in config.keys():
            for maneuver in config["Maneuvers"]:
                thrust = maneuver["ManeuverParams"][4]
                state.append(thrust)
        return state

    def propagate_state(self, state):
        config_dict_out = self.state2config(state)
        index, data = propagate_orbits_wrapper(config_dict_out)
        df_out = pd.DataFrame(data=np.array(data), index=pd.DatetimeIndex(index), columns=["randv_mks_{}".format(i) for i in range(6)])
        # propagated values should always be interpolated to the times of the available data.
        df_out_interpolated = interp(df_out, self.df_gps.index)
        return df_out_interpolated

    def run_fit(self, max_loops=5, percentchg=0.01, deltaamtchg=1e-6, epsilon=1e-8):
        """wrapper over original _run_fit to return config instead of state"""
        state = self._run_fit(max_loops, percentchg, deltaamtchg, epsilon)
        df_out = self.propagate_state(state)
        config_out = self.state2config(state)
        return df_out, config_out


class TLEFit(GenericFit):

    def __init__(self, df_gps_teme, initial_TLE=None, statesize=7):
        """
        :param df_gps_teme: Pandas dataframe with "randv_mks_{0-2}" keys for position x,y,z, "randv_mks_{3-5}" for velocity
        x,y,z, and timestamp as index. Position and velocity are expresed in TEME frame.
        :param intial_TLE: SGP4 TLE object
        :param propulsion_maneuvre:
        """
        if initial_TLE:
            self.initial_tle = initial_TLE
            self.tle_epoch = self.initial_tle.epoch
        else:
            self.initialize_TLE_from_gps_data()
        self.df_gps = df_gps_teme
        self.pos_gps_km = df_gps_teme[["randv_mks_{}".format(i) for i in range(3)]].values / 1000
        self.vel_gps_km = df_gps_teme[["randv_mks_{}".format(i) for i in range(3, 6)]].values / 1000
        self.initial_state = self.tle2state(initial_TLE, statesize=statesize)

    def initialize_TLE_from_gps_data(self):
        initial_state_vector = self.df_gps.iloc[0][[f'randv_mks_{i}' for i in range(6)]]
        epoch = self.df_gps.iloc[0].index
        self.initial_tle = self.generate_tle_from_state_vector(epoch, initial_state_vector)

    def generate_tle_from_state_vector(self, epoch, initial_state_vector):
        tle = object()
        return tle

    def tle2state(self, tle, statetype="e", statesize=7):
        state = [0] * statesize
        if statetype == "t":
            state[0] = tle.a
            state[1] = tle.ecco
            state[2] = tle.inclo
            state[3] = tle.nodeo
            state[4] = tle.argpo
            state[5] = tle.mo
        elif statetype == "e":
            state[0] = tle.ecco * np.cos(tle.argpo + tle.nodeo)  # ke, af
            state[1] = tle.ecco * np.sin(tle.argpo + tle.nodeo)  # he, ag
            state[2] = (tle.no_kozai * wgs84.tumin) ** (-2.0 / 3.0)
            state[3] = np.mod(tle.mo + tle.argpo + tle.nodeo, 2 * np.pi)  # L
            state[4] = np.tan(0.5 * tle.inclo) * np.sin(tle.nodeo)  # pe
            state[5] = np.tan(0.5 * tle.inclo) * np.cos(tle.nodeo)  # qe
        else:
            raise RuntimeError("{} is not a valid statetype".format(statetype))
        if statesize > 6:
            state[6] = tle.bstar
        return state

    def state2tle(self, state, statetype="e"):
        tle_out = copy.copy(self.initial_tle)
        if statetype == "t":
            tle_out.a = state[0]
            # que hacer si a < 1.0
            tle_out.no_kozai = (tle_out.a) ** (-3.0 / 2.0) / wgs84.tumin
            tle_out.ecco = state[1]
            tle_out.inclo = state[2]
            tle_out.nodeo = state[3]
            tle_out.argpo = state[4]
            tle_out.mo = state[5]
        elif statetype == "e":
            # En el codigo de vallado, el "a" esta definido a partir de no, y no de no_kozai
            tle_out.a = state[2]
            tle_out.no_kozai = (tle_out.a) ** (-3.0 / 2.0) / wgs84.tumin
            tle_out.ecco = np.sqrt(state[0] ** 2 + state[1] ** 2)
            tle_out.inclo = 2 * np.arctan(np.sqrt(state[4] ** 2 + state[5] ** 2))
            tle_out.nodeo = np.arctan2(state[4], state[5])
            tle_out.argpo = np.arctan2(state[1], state[0]) - tle_out.nodeo
            tle_out.mo = np.mod(state[3] - tle_out.nodeo - tle_out.argpo, 2 * np.pi)
        if len(state) > 6:
            tle_out.bstar = state[6]
        sgp4init(wgs84, False, tle_out.satnum, tle_out.jdsatepoch - 2433281.5, tle_out.bstar, tle_out.ecco,
                 tle_out.argpo, tle_out.inclo, tle_out.mo, tle_out.no_kozai, tle_out.nodeo, tle_out)
        return tle_out

    def propagate_state(self, state):
        tle_out = self.state2tle(state)
        # propagated values should always be interpolated to the times of the available data.
        dt_min = (self.df_gps.index - self.initial_tle.epoch.replace(tzinfo=self.df_gps.index.tzinfo)).total_seconds() / 60
        pos_tle, vel_tle = sgp4(tle_out, dt_min)
        data = np.hstack((np.array(pos_tle).T*1000, np.array(vel_tle).T*1000))
        return pd.DataFrame(data=data, index=self.df_gps.index, columns=["randv_mks_{}".format(i) for i in range(6)])

    def run_fit(self, max_loops=50, percentchg=0.001, deltaamtchg=1e-10, epsilon=1e-11):
        """wrapper over original _run_fit to return config instead of state"""
        state = self._run_fit(max_loops, percentchg, deltaamtchg, epsilon)
        tle = self.state2tle(state)
        return tle



