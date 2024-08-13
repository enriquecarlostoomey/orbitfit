import numpy as np
import pandas as pd

from .teme2ecef import ecef2teme_full, teme2ecef_dcm_full, teme2ecef_full
from .I2E import dcmeci2ecef_full, GREENWICH_DOT_RADSEC, get_reference_I2E_matrix


def rotate_gps(df_gps, prefix="randv_mks_", method="E2I"):
    """
    Rotates position and velocity values in df_gps dataframe applying the desired method
    :param df_gps: Dataframe with DatetimeIndex. Must contain at least 6 colums ending in 0-6,, where the first 0-2
    corresponds to xyz cartesion position and 3-6 corresponds to xyz cartesian velocity.
    :param prefix: Prefix of the column name. By defult is randv_mks_, but could be randv_mks_raw_ for gps_raw telemtry.
    :param method: Methods implemented here are: I2E (ECI to ECEF), E2I (ECEF to ECI), E2T (ECEF to TEME), T2E (TEME to ECEF),
    I2E_alt (ECI to ECEF using reference matrix), E2I_alt (ECEF to ECI using reference matrix)
    :return: The rotated Dataframe mantening the column prefix.
    """
    if method == "I2T" or method == "T2I":
        mid_idx = df_gps.index[0] + (df_gps.index[-1] - df_gps.index[0])/2
        i2e = np.asarray(dcmeci2ecef_full(mid_idx))
        t2e = teme2ecef_dcm_full(mid_idx)
        rotmatrix = np.dot(t2e.T, i2e)
        if method == "T2I":
            rotmatrix = rotmatrix.T
        pos_original = df_gps[[f"{prefix}{i}" for i in (0, 1, 2)]].values.astype('double')
        vel_original = df_gps[[f"{prefix}{i}" for i in (3, 4, 5)]].values.astype('double')
        pos_rotated = np.dot(rotmatrix, pos_original.T).T
        vel_rotated = np.dot(rotmatrix, vel_original.T).T
        df_gps_rotated = pd.DataFrame(index=df_gps.index, columns=[f"{prefix}{i}" for i in range(6)],
                                      data = np.hstack((pos_rotated, vel_rotated)))
    else:
        pos_vel_data_rotated = []
        for idx, data in df_gps.iterrows():
            pos_original = data[[f"{prefix}{i}" for i in (0, 1, 2)]].values.astype('double')
            vel_original = data[[f"{prefix}{i}" for i in (3, 4, 5)]].values.astype('double')
            if method == "E2I":
                rotmatrix = np.asarray(dcmeci2ecef_full(idx).T)
                pos_rotated = np.dot(rotmatrix, pos_original)
                vel_rotated = np.dot(rotmatrix, vel_original + np.cross([0, 0, GREENWICH_DOT_RADSEC], pos_original))
            elif method == "I2E":
                rotmatrix = np.asarray(dcmeci2ecef_full(idx))
                pos_rotated = np.dot(rotmatrix, pos_original)
                vel_rotated = np.dot(rotmatrix, vel_original - np.cross([0, 0, GREENWICH_DOT_RADSEC], pos_original))
            elif method == "E2T":
                pos_rotated, vel_rotated = ecef2teme_full(pos_original, vel_original, idx)
            elif method == "T2E":
                pos_rotated, vel_rotated = teme2ecef_full(pos_original, vel_original, idx)
            elif method == "E2I_alt":
                rotmatrix = np.asarray(get_reference_I2E_matrix(idx.timestamp()).T)
                pos_rotated = np.dot(rotmatrix, pos_original)
                vel_rotated = np.dot(rotmatrix, vel_original + np.cross([0, 0, GREENWICH_DOT_RADSEC], pos_original))
            elif method == "I2E_alt":
                rotmatrix = np.asarray(get_reference_I2E_matrix(idx.timestamp()))
                pos_rotated = np.dot(rotmatrix, pos_original)
                vel_rotated = np.dot(rotmatrix, vel_original - np.cross([0, 0, GREENWICH_DOT_RADSEC], pos_original))
            else:
                raise RuntimeError(f"{method} not implements. Available methods are E2I, I2E, E2T, T2E")
            aux_dic = {}
            aux_dic.update({f"{prefix}{ax}": p for p, ax in zip(pos_rotated, "012")})
            aux_dic.update({f"{prefix}{ax}": v for v, ax in zip(vel_rotated, "345")})
            pos_vel_data_rotated.append(aux_dic)
        df_gps_rotated = pd.DataFrame(data=pos_vel_data_rotated)
        df_gps_rotated.index = df_gps.index
    return df_gps_rotated