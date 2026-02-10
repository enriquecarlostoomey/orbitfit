import pandas as pd
import matplotlib.pyplot as plt


def load_gps_data(gps_filename):
    """Load GPS data from pickle file.
    :param gps_filename: Path to pickle file with GPS data
    :return: DataFrame with GPS data
    """
    return pd.read_pickle(gps_filename)


def parse_gps_data(df_gps, save=False, plot=False, filter_status=True):
    """loads pickle with tfrs data
    :param df_gps: dataframe with columns ["utc_date", "navigation_state", "ecef_pos_x", "ecef_pos_y", "ecef_pos_z", "ecef_vel_x", "ecef_vel_y", "ecef_vel_z"
    :param save: If True, the parsed data is plot.
    :param plot: If True, saves the parsed data into pickle file with name filename_orbit_{source}.pkl
    :param filter_status: If True, all gps values with status != 1 are filtered (status=1 -> lock)
    """
    df_gps.index = pd.to_datetime(df_gps['utc_time'], unit='s')
    df_gps.sort_index(inplace=True)

    if filter_status:
        # Filter all points with no lock (data_good=False)
        df_gps = df_gps[df_gps.navigation_state == 3]

    df_gps = df_gps[["ecef_pos_x", "ecef_pos_y", "ecef_pos_z", "ecef_vel_x", "ecef_vel_y", "ecef_vel_z" ]]
    df_gps.rename(columns={f"ecef_pos_{x}":f"randv_mks_{i}" for x,i in zip("xyz", range(3))}, inplace=True)
    df_gps.rename(columns={f"ecef_vel_{x}":f"randv_mks_{i}" for x,i in zip("xyz", range(3,6))}, inplace=True)

    if plot:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        fig.suptitle("Input GPS data")
        df_gps[['randv_mks_0', 'randv_mks_1', 'randv_mks_2']].plot(ax=ax[0], style='--*',
                                                                                             grid=True)
        ax[0].set_ylabel('pos [m]')
        df_gps[['randv_mks_0', 'randv_mks_1', 'randv_mks_2']].plot(ax=ax[1], style='--*',
                                                                                             grid=True)
        ax[1].set_ylabel('vel [m/s]')

#   percentage_of_fixes = len(df_gps[df_gps['status'] != '00']) / len(df_gps['status']) * 100
#   a = (df_gps['status'] != '00').values
#   max_consecutive_fixes = np.argmin(np.append(a[~np.logical_and.accumulate(~a)], False))
#   print("percentage of fixes  : {:.2f}%".format(percentage_of_fixes))
#   print("max consecutive fixes: {:d}".format(max_consecutive_fixes))
    return df_gps
