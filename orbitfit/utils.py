import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
import copy
from sgp4.propagation import sgp4
import logging
import os
from quaternions import Quaternion
from .orbital_elements import OrbitalElements as oe
from .astronomical_constants import R_mean_earth, mu_earth

logger = logging.getLogger()


def rv2oe(r, v, threshold=1e-7):
    """
    Computes Orbital Elements from cartesian coordinates.
    Input position and velocity are in ECI (inertial reference frame) in [km] and [km/s].
    All output angles are in radians. a in [km]
    returns: a, ecco, inclo, nodeo, argpo, mo
    """
    rnorm = np.linalg.norm(r)
    vnorm = np.linalg.norm(v)
    a = -mu_earth / (2 * (vnorm ** 2 / 2 - mu_earth / rnorm))
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    e_vec = 1 / mu_earth * np.cross(v, h) - r / rnorm
    ecco = np.linalg.norm(e_vec)
    inclo = np.arccos(h[2] / h_norm)
    if abs(inclo % (2 * np.pi)) > threshold:
        line_of_nodes = np.cross(np.array([0, 0, 1]), h)
        nodeo = np.arctan2(line_of_nodes[1], line_of_nodes[0])
        nodeo = nodeo % (2 * np.pi)
        if ecco > threshold:
            cos_w = np.dot(line_of_nodes, e_vec) / (ecco * np.linalg.norm(line_of_nodes))
            if e_vec[2] >= 0:
                argpo = np.arccos(cos_w)
            else:
                argpo = 2 * np.pi - np.arccos(cos_w)
            argpo = argpo % (2 * np.pi)
            sigma0 = np.dot(r, v) / np.sqrt(mu_earth)
            E0 = np.arctan2(sigma0 / np.sqrt(a), 1 - rnorm / a)
            mo = E0 - ecco * np.sin(E0)
        else:
            argpo = 0
            sigma0 = np.dot(r, v) / np.sqrt(mu_earth)
            mo = np.arctan2(sigma0 / np.sqrt(a), 1 - rnorm / a)
    else:
        nodeo = np.NaN
        argpo = np.NaN
        if ecco > threshold:
            sigma0 = np.dot(r, v) / np.sqrt(mu_earth)
            E0 = np.arctan2(sigma0 / np.sqrt(a), 1 - rnorm / a)
            mo = E0 - ecco * np.sin(E0)
    return a, ecco, inclo, nodeo, argpo, mo

def oe2rv(a, ecco, inclo, nodeo, argpo, mo):
    p = a * (1 - ecco ** 2)
    f = lambda E: mo - (E - ecco * np.sin(E))
    E0 = scipy.optimize.newton(f, x0=mo, rtol=1e-9)
    no = 2 * np.arctan(np.sqrt((1 + ecco) / (1 - ecco)) * np.tan(E0 / 2))
    temp = p / (1 + ecco * np.cos(no))
    rpqw = np.array([temp * np.cos(no),
                        temp * np.sin(no),
                        0.0])
    vpwq = np.array([-np.sin(no) * np.sqrt(mu_earth / p),
                        (ecco + np.cos(no)) * np.sqrt(mu_earth / p),
                        0.0])
    q_argpo = Quaternion.from_rotation_vector(np.array([0.0, 0.0, -argpo]))
    q_inclo = Quaternion.from_rotation_vector(np.array([-inclo, 0.0, 0.0]))
    q_nodeo = Quaternion.from_rotation_vector(np.array([0.0, 0.0, -nodeo]))
    q_total = q_nodeo * q_inclo * q_argpo
    r = q_total * rpqw
    v = q_total * vpwq
    return r, v

def oe2ee(a, ecco, inclo, nodeo, argpo, mo):
    af = ecco * np.cos(argpo + nodeo)
    ag = ecco * np.sin(argpo + nodeo)
    a = a / R_mean_earth
    L = np.mod(mo + argpo + nodeo, 2 * np.pi)
    pe = np.tan(0.5 * inclo) * np.sin(nodeo)
    qe = np.tan(0.5 * inclo) * np.cos(nodeo)
    return af, ag, a, L, pe, qe

def ee2oe(af, ag, a, L, pe, qe):
    a = a * R_mean_earth
    ecco = np.sqrt(af ** 2 + ag ** 2)
    # que hacer si ecco > 1.0
    inclo = 2 * np.arctan(np.sqrt(pe ** 2 + qe ** 2))
    nodeo = np.arctan2(pe, qe)
    # que hacer si nodeo < 0.0
    argpo = np.arctan2(ag, af) - nodeo
    mo = np.mod(L - nodeo - argpo, 2 * np.pi)
    return a, ecco, inclo, nodeo, argpo, mo



def angle2dcm(angle1, angle2, angle3, axis_order):
    """
    Angles are expected in rads. axis_order is a string. Returns the transformation matrix composed by the three
    consecutive transformations defined by angle1,2&3 and axis_order.
    :param angle1: first angle of rotation [rad]
    :param angle2: second angle of rotation [rad]
    :param angle3: third angle of rotation [rad]
    :param axis_order: 3 letter string. All angles
    :return: 3x3 numpy.array
    """
    assert all([axis in 'XYZ' for axis in axis_order]) and len(axis_order) >= 3, "Expects a 3 letter combination of" \
                                                                                 " 'X', 'Y' or 'Z'"
    dcm = np.eye(3, 3)
    for angle, axis in zip([angle1, angle2, angle3], axis_order):
        if axis == 'X':
            dcm = np.dot(np.array([[1, 0, 0], [0, np.cos(angle), np.sin(angle)],
                                    [0, -np.sin(angle), np.cos(angle)]]), dcm)
        elif axis == 'Y':
            dcm = np.dot(np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0],
                                    [np.sin(angle), 0, np.cos(angle)]]), dcm)
        elif axis == 'Z':
            dcm = np.dot(np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]]), dcm)
    return dcm


def interp(df, new_index):
    """Return a new DataFrame with all numerical columns values interpolated
    to the new_index values."""
    df = df.drop_duplicates()
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name
    x = np.array([t.timestamp() for t in df.index])
    x_interp = np.array([t.timestamp() for t in df_out.index])

    for colname, col in df.items():
        if np.issubdtype(col.dtype, np.number):
            f = scipy.interpolate.interp1d(x, col.values, kind="cubic", fill_value="extrapolate", assume_sorted=True)
            df_out[colname] = f(x_interp)
        else:
            df_out[colname] = col
    return df_out


def distance_between_position(pos_1, pos_2):
    """
    Expects two nx3 array of satellite coordinates. Express the distance between both arrays in
    AAR coordinate systems, using the first array as base for the coordinate system
    :param pos_1: nx3 array
    :param pos_2: nx3 array
    :return:  nx3 array
    """
    pos_error = pos_2 - pos_1
    teme2arr = find_inertial2aar_dcm(pos_1)
    return np.einsum("ijk,ik->ij", teme2arr, pos_error)


def calculate_gps_tle_error(df_gps, tle_list_with_data):
    """
    Calculates gps - tle difference in along-track, across-track, radial refenrece system (based on GPS position).
    :param df_gps: dataframe with gps position in TEME reference frame. As all position data dataframe, expects keys
    "randv_mks_<0-5>"
    :param tle_list_with_data: list of available TLE for the duration of df_gps. Each element in the list is a dict
     with the following keys:
        tle: tle element
        start: starting time for period in which the tle is applicable
        end: ending time for period in which the tle is applicable
    :return: pandas dataframe with keys: "randv_mks_tle_<0-5>", "alongtrack_error_km", "acrosstrack_error_km",
     "radial_error_km", "error_norm_km" and same index as the df_gps
    """
    df_gps_tle_error = pd.DataFrame(index= df_gps.index,
                                    columns=[f"randv_mks_tle_{i}" for i in range(6)]+
                                            ["alongtrack_error_km", "acrosstrack_error_km", "radial_error_km",
                                             "error_norm_km"],
                                    data=None)
    for data in tle_list_with_data:
        tle = data["tle"]
        start = data["start"]
        end = data["end"]
        df_gps_teme_slice = df_gps[(df_gps.index > start) & (df_gps.index < end)]
        if df_gps_teme_slice.empty:
            continue
        dt_min = np.asarray((df_gps_teme_slice.index
                             - tle.epoch.replace(tzinfo=df_gps_teme_slice.index.tz)).total_seconds()/60)
        r_teme, v_teme = sgp4(tle, dt_min)
        tle_rteme_km = np.vstack(r_teme).T
        tle_vteme_km = np.vstack(v_teme).T

        for i in range(3):
            df_gps_tle_error.loc[df_gps_teme_slice.index, f"randv_mks_tle_{i}"] = tle_rteme_km[:, i] * 1000
            df_gps_tle_error.loc[df_gps_teme_slice.index, f"randv_mks_tle_{i+3}"] = tle_vteme_km[:, i] * 1000

        gps_rteme_km = df_gps_teme_slice[["randv_mks_{}".format(i) for i in range(3)]].values.astype('double') / 1000.0
        errors_aar = distance_between_position(gps_rteme_km, tle_rteme_km)
        error_norm_km = np.linalg.norm(errors_aar, axis=1)
        df_gps_tle_error.loc[df_gps_teme_slice.index, "alongtrack_error_km"] = errors_aar[:, 0]
        df_gps_tle_error.loc[df_gps_teme_slice.index, "acrosstrack_error_km"] = errors_aar[:, 1]
        df_gps_tle_error.loc[df_gps_teme_slice.index, "radial_error_km"] = errors_aar[:, 2]
        df_gps_tle_error.loc[df_gps_teme_slice.index, "error_norm_km"] = error_norm_km
    return df_gps_tle_error


def find_inertial2aar_dcm(coord):
    """
    Given an nx3 array of satellite inertial coordinates, it builds rotation matrix
    to go from inertial to along-track, across-track, radial coordinate systems. This is a
    moving reference systems, so a diferent dcm is calculated for every coordinate.
    :param coord: nx3 numpy array with interial coordinates.
    :return: inertial2arr: nx3x3 numpy array with a of dcm for each coordinate.
    """
    G = coord.sum(axis=0) / coord.shape[0]
    u, s, vh = np.linalg.svd(coord - G)
    r = (coord.T / np.linalg.norm(coord, axis=1)).T
    h_estimate = np.cross(r[0, :], r[1, :])
    h0 = np.sign(np.dot(h_estimate, vh[2, :])) * vh[2, :]# unitary vector normal to the plane that best fits coords
    h = np.repeat(h0[:, np.newaxis].T, coord.shape[0], axis=0)
    v = (np.cross(h0, r).T / np.linalg.norm(np.cross(h0, r), axis=1)).T
    v = v.reshape((v.shape[0], 1, v.shape[1]))
    h = h.reshape((h.shape[0], 1, h.shape[1]))
    r = r.reshape((r.shape[0], 1, r.shape[1]))
    inertial2arr = np.hstack([v, h, r])
    return inertial2arr


def get_mean_oe(df_gps_eci):
    df_mean_oe = pd.DataFrame(index=df_gps_eci.index, data=None, columns=["mean_sma", "mean_ecc", "mean_inc"])
    for idx, row in df_gps_eci.iterrows():
        pos_teme_km = row[[f"randv_mks_{i}" for i in range(3)]].values.astype('double') * 1e-3
        vel_teme_km = row[[f"randv_mks_{i}" for i in range(3, 6)]].values.astype('double') * 1e-3
        orbital = oe.from_cartesian_coordinates(pos_teme_km, vel_teme_km)
        mean_elements = orbital.compute_mean_elements()
        df_mean_oe.loc[idx, 'mean_sma'] = mean_elements.mean_semimajor_axis
        df_mean_oe.loc[idx, 'mean_ecc'] = mean_elements.mean_eccentricity
        df_mean_oe.loc[idx, 'mean_inc'] = mean_elements.mean_inclination
        df_mean_oe.loc[idx, 'mean_RAAN'] = mean_elements.mean_RAAN
        df_mean_oe.loc[idx, 'mean_argpo'] = mean_elements.mean_arg_perigee
        df_mean_oe.loc[idx, 'mean_anomaly'] = mean_elements.mean_mean_anomaly
    return df_mean_oe


def filter_with_mean_oe(df_gps_eci, thresholds=None, method="median", plot=False, output=None, suffix="all"):
    if bool(thresholds) is False:
        thresholds = {"mean_sma": 0.75, "mean_ecc": 0.001, "mean_inc": 0.017}
    else:
        assert all([key in ["mean_sma", "mean_ecc", "mean_inc"] for key in thresholds.keys()]), ...
        f"threshold keys not recognize: {thresholds}"
    logger.info(f"Filter mean oe with thresholds = {thresholds}")
    df_mean_oe = get_mean_oe(df_gps_eci)
    for key, threshold in thresholds.items():
        if method == "median":
            avg = df_mean_oe[key].median(axis=0)
        elif method == "mean":
            avg = df_mean_oe[key].mean(axis=0)
        df_mean_oe[f"filtered_{key}"] = 1  # Everything is filter until its not
        df_mean_oe.loc[abs(avg - df_mean_oe[key]) < threshold, f"filtered_{key}"] = 0
        if plot:
            fig, ax = plt.subplots(2, 1)
            df_mean_oe[key].plot(ax=ax[0], style='*')
            df_mean_oe.loc[df_mean_oe[f"filtered_{key}"] == 0, key].plot(ax=ax[0], style='*')
            ax[0].plot([df_mean_oe.index[0], df_mean_oe.index[-1]], [avg, avg], '--k')
            ax[0].plot([df_mean_oe.index[0], df_mean_oe.index[-1]], [avg+threshold, avg+threshold], '--r')
            ax[0].plot([df_mean_oe.index[0], df_mean_oe.index[-1]], [avg-threshold, avg-threshold], '--r')
            ax[0].set_ylabel(key)

            df_mean_oe[key].plot.hist(ax=ax[1], bins=25)
            ax[1].plot([avg, avg], [0, 1], '--k')
            ax[1].plot([avg + threshold, avg + threshold], [0, 1], '--r')
            ax[1].plot([avg - threshold, avg - threshold], [0, 1], '--r')
            plt.tight_layout()
            if output is not None:
                fig.savefig(os.path.join(output, f"{key}_filter_{suffix}.png"))
            plt.show()
    df_gps_eci_filtered = copy.deepcopy(df_gps_eci[df_mean_oe[[f"filtered_{key}" for key in thresholds.keys()]].sum(axis=1) == 0])
    return df_gps_eci_filtered


def filter_with_tle(df_gps_teme, tle, threshold_km=3, plot=False, output=None):
    # Hack to put the input TLE file, which is an SGP4 object, into a list of dicts. This list of dicts is usefull when
    # the GPS data to filter is long and a single TLE is not enought to apply the filter for all the period.
    tle_list_with_data = [{"tle": tle,
                           "start": df_gps_teme.index[0] - datetime.timedelta(days=0.5),
                           "end": df_gps_teme.index[-1]+datetime.timedelta(days=0.5)}]
    df_gps_tle_error = calculate_gps_tle_error(df_gps_teme, tle_list_with_data)
    # plot filter results
    if plot:
        fig, ax = plt.subplots(1, 1)
        df_gps_tle_error["error_norm_km"].plot(ax=ax, style='*')
        df_gps_tle_error.loc[df_gps_tle_error["error_norm_km"] < threshold_km, "error_norm_km"].plot(ax=ax, style="*")
        ax.plot([df_gps_tle_error.index[0], df_gps_tle_error.index[-1]], [threshold_km, threshold_km], '--r')
        ax.set_ylabel("error norm [km]")
        if output is not None:
            fig.savefig(os.path.join(output, "tle_filter.png"))
        plt.show()
    df_gps_teme_filtered = copy.deepcopy(df_gps_teme.loc[df_gps_tle_error["error_norm_km"] < threshold_km])
    return df_gps_teme_filtered


def plot_error_aar(df_gps_1, df_gps_2, dates=()):
    pos_1 = df_gps_1[[f"randv_mks_{i}" for i in range(3)]].values.astype('double')
    pos_2 = df_gps_2[[f"randv_mks_{i}" for i in range(3)]].values.astype('double')
    error_aar = distance_between_position(pos_1, pos_2)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df_gps_1.index, error_aar)
    for date in dates:
        plt.plot([date, date], [np.min(error_aar), np.max(error_aar)], label=date.strftime("%Y-%m-%d %H:%M:%S"))
    plt.ylabel("diff [m]")
    plt.legend(["along-track", "cross-track", "radial"])
    plt.tight_layout()
    plt.show()
    return fig


def get_gps_error_percentiles(df_gps, df_true, quantiles_of_interest=(0.66, 0.90, 0.99), plot=True, output=None):
    pos1 = df_true[[f'randv_mks_{i}' for i in range(3)]].values
    pos2 = df_gps[[f'randv_mks_{i}' for i in range(3)]].values
    d = distance_between_position(pos1, pos2)
    error_norm = np.linalg.norm(d, 2, 1)
    error_norm_sorted = np.sort(error_norm)
    quantiles_values = [np.quantile(error_norm_sorted, q) for q in quantiles_of_interest]
    if plot:
        fig = plt.figure()
        plt.semilogx(error_norm_sorted, np.arange(len(error_norm_sorted)) / len(error_norm_sorted) * 100)
        for q, v in zip(quantiles_of_interest, quantiles_values):
            plt.semilogx([0, v], [q * 100, q * 100], '--k')
            plt.semilogx([v, v], [0, q * 100], '--k')
        plt.grid()
        plt.ylabel('Quantile [%]')
        plt.xlabel('Error [km]')
        plt.xlim([0, 1000])
        if output is not None:
            fig.savefig(os.path.join(output, "Quantiles.png"))
    # store quantiles of interest in a dataframe
    df = pd.DataFrame(data=zip([str(q * 100) + '%' for q in quantiles_of_interest], quantiles_values)).T
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    return df

