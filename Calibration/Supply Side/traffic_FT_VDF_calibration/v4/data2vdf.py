# In[0] Import necessary packages 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import datetime

g_number_of_plink = 0
g_plink_id_dict = {}
g_plink_nb_seq_dict = {}
g_parameter_list = []
g_vdf_group_list = []


# In[1] input data
def input_data():
    data_df = pd.read_csv('./test/link_performance.csv',
                          encoding='UTF-8')  # create data frame from the input link_performance.csv
    # data_df=data_df[data_df['FT']==1]
    # data_df=data_df[data_df['AT']==1]
    # data_df = data_df[data_df['assignment_period'] == '2300_0600']
    data_df = data_df.drop(
        data_df[(data_df.volume == 0) | (data_df.speed == 0)].index)  # drop all rows that have 0 volume or speed
    data_df.dropna(axis=0, how='any', inplace=True)  # drop all rows that have any null value
    data_df.reset_index(drop=True, inplace=True)  # reset the index of the dataframe
    # Calculate some derived properties for each link
    data_df['volume_per_lane'] = data_df['volume'] / data_df[
        'lanes']  # add an additional column volume_per_lane in the dataframe
    data_df['hourly_volume_per_lane'] = data_df['volume_per_lane'] * (
            60 / TIME_INTERVAL_IN_MIN)  # add an additional column hourly_volume_per_lane
    # in the link_performance.csv, the field "volume" is link volume within the time interval 
    data_df['density'] = data_df['hourly_volume_per_lane'] / data_df['speed']  # add an additional column density
    return data_df


# In[2] Traffic flow models and volume delay function (BPR function)
def density_speed_function(density, free_flow_speed, critical_density,
                           mm):  # fundamental diagram model (density-speed function):
    # More informantion the density-speed function: https://www.researchgate.net/publication/341104050_An_s-shaped_three-dimensional_S3_traffic_stream_model_with_consistent_car_following_relationship
    k_over_k_critical = density / critical_density
    denominator = np.power(1 + np.power(k_over_k_critical, mm), 2 / mm)
    return free_flow_speed / denominator


def volume_speed_func(x, ffs, alpha, beta, critical_density, mm):  # fundamental diagram  (volume_delay fuction)
    # 1. input: assigned volume 
    # 2. output: converted volume on S3 model 
    speed = bpr_func(x, ffs, alpha, beta)
    kernal = np.power(np.power(ffs / speed, mm), 0.5)
    return speed * critical_density * np.power(kernal - 1, 1 / mm)


def bpr_func(x, ffs, alpha, beta):  # BPR volume delay function input: volume over capacity
    return ffs / (1 + alpha * np.power(x, beta))


# In[3] Calibrate traffic flow model
def calibrate_traffic_flow_model(vdf_training_set, vdf_index):
    # 1. set the lower bound and upper bound of the free flow speed value 
    lower_bound_FFS = vdf_training_set['speed'].quantile(
        0.9)  # Assume that the lower bound of freeflow speed should be larger than the mean value of speed
    upper_bound_FFS = np.maximum(vdf_training_set['speed'].max(), lower_bound_FFS + 0.1)
    # Assume that the upper bound of freeflow speed should at least larger than the lower bound, and less than the maximum value of speed

    # 2. generate the outer layer of density-speed  scatters 
    vdf_training_set_after_sorting = vdf_training_set.sort_values(
        by='speed')  # sort speed value from the smallest to the largest
    vdf_training_set_after_sorting.reset_index(drop=True, inplace=True)  # reset the index
    step_size = np.maximum(1, int((vdf_training_set['speed'].max() - vdf_training_set[
        'speed'].min()) / LOWER_BOUND_OF_OUTER_LAYER_SAMPLES))  # determine the step_size of each segment to generate the outer layer
    X_data = []
    Y_data = []
    for k in range(0, int(np.ceil(vdf_training_set['speed'].max())), step_size):
        segment_df = vdf_training_set_after_sorting[
            (vdf_training_set_after_sorting.speed < k + step_size) & (vdf_training_set_after_sorting.speed >= k)]
        Y_data.append(segment_df.speed.mean())
        threshold = segment_df['density'].quantile(OUTER_LAYER_QUANTILE)
        X_data.append(segment_df[(segment_df['density'] >= threshold)]['density'].mean())
    XY_data = pd.DataFrame({'X_data': X_data, 'Y_data': Y_data})
    XY_data = XY_data[~XY_data.isin([np.nan, np.inf, -np.inf]).any(1)]  # delete all the infinite and null values
    if len(XY_data) == 0:
        print('WARNING: No available data within all speed segments')
        exit()
    density_data = XY_data.X_data.values
    speed_data = XY_data.Y_data.values
    # 3. calibrate traffic flow model using scipy function curve_fit. More information about the function, see https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.curve_fit.html
    popt, pcov = curve_fit(density_speed_function, density_data, speed_data,
                           bounds=[[lower_bound_FFS, 0, 0], [upper_bound_FFS, UPPER_BOUND_CRITICAL_DENSITY, 15]])

    free_flow_speed = popt[0]
    critical_density = popt[1]
    mm = popt[2]
    speed_at_capacity = free_flow_speed / np.power(2, 2 / mm)
    ultimate_capacity = speed_at_capacity * critical_density
    print('--speed_at_capacity=', speed_at_capacity)
    print('--ultimate_capacity=', ultimate_capacity)
    print('--critical_density=', critical_density)
    print('--free_flow_speed=', free_flow_speed)
    print('--mm=', mm)

    xvals = np.linspace(0, UPPER_BOUND_JAM_DENSITY, 100)  # all the data points with density values
    plt.plot(vdf_training_set_after_sorting['density'], vdf_training_set_after_sorting['speed'], '*', c='k',
             label='observations', markersize=1)
    plt.plot(xvals, density_speed_function(xvals, *popt), '--', c='b', markersize=6)
    plt.scatter(density_data, speed_data, edgecolors='r', color='r', label='outer layer', zorder=30)
    plt.legend()
    plt.title('Density-speed fundamental diagram, VDF: ' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])))
    plt.xlabel('Density')
    plt.ylabel('Speed')
    plt.savefig('./1_FD_speed_density_' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + '.png')
    plt.close()

    plt.plot(vdf_training_set_after_sorting['hourly_volume_per_lane'], vdf_training_set_after_sorting['speed'], '*',
             c='k', label='observations', markersize=1)
    plt.plot(xvals * density_speed_function(xvals, *popt), density_speed_function(xvals, *popt), '--', c='b',
             markersize=6)
    plt.scatter(density_data * speed_data, speed_data, edgecolors='r', color='r', label='outer layer', zorder=30)
    plt.legend()
    plt.title('Volume-speed fundamental diagram,VDF: ' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])))
    plt.xlabel('Volume')
    plt.ylabel('Speed')
    plt.savefig('./1_FD_speed_volume_' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + '.png')
    plt.close()

    plt.plot(vdf_training_set_after_sorting['density'], vdf_training_set_after_sorting['hourly_volume_per_lane'], '*',
             c='k', label='original values', markersize=1)
    plt.plot(xvals, xvals * density_speed_function(xvals, *popt), '--', c='b', markersize=6)
    plt.scatter(density_data, density_data * speed_data, edgecolors='r', color='r', label='outer layer', zorder=30)
    plt.legend()
    plt.title('Density-volume fundamental diagram,VDF: ' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])))
    plt.xlabel('Density')
    plt.ylabel('Volume')
    plt.savefig('./1_FD_volume_density_' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + '.png')
    plt.close()

    return speed_at_capacity, ultimate_capacity, critical_density, free_flow_speed, mm


# In[4] VDF calibration
def vdf_calculation(internal_period_vdf_daily_link_df, vdf_index, period_index, speed_at_capacity, ultimate_capacity,
                    critical_density, free_flow_speed, mm):
    p0 = np.array([free_flow_speed, 0.15, 4])
    lowerbound_fitting = [free_flow_speed, 0.15, 1.01]  # upper bound and lower bound of free flow speed, alpha and beta
    upperbound_fitting = [free_flow_speed * 1.1, 10, 10]
    if DOC_RATIO_METHOD == 'VBM':
        print('Volume method calibration...')
        internal_period_vdf_daily_link_df['hourly_demand_over_capacity'] = internal_period_vdf_daily_link_df.apply(
            lambda x: (ultimate_capacity + (
                    ultimate_capacity - x.period_mean_hourly_volume_per_lane)) / ultimate_capacity if x.period_mean_daily_speed < speed_at_capacity else x.period_mean_hourly_volume_per_lane / ultimate_capacity,
            axis=1)

        X_data = []
        Y_data = []
        # load data
        for k in range(0, len(internal_period_vdf_daily_link_df)):
            # stage 1Hourly hourly_demand_over_capacity data 
            for kk in range(WEIGHT_HOURLY_DATA):
                Y_data.append(internal_period_vdf_daily_link_df.loc[k, 'period_mean_daily_speed'])  # x first y second
                X_data.append(internal_period_vdf_daily_link_df.loc[k, 'hourly_demand_over_capacity'])
            # stage 2Period hourly_demand_over_capacity data
            for kk in range(WEIGHT_PERIOD_DATA):
                Y_data.append(internal_period_vdf_daily_link_df['period_mean_daily_speed'].mean())
                X_data.append(internal_period_vdf_daily_link_df['period_demand_over_capacity'].mean())
            # stage 3
            for kk in range(WEIGHT_UPPER_BOUND_DOC_RATIO):
                Y_data.append(0.001)
                X_data.append(
                    upper_bound_doc_ratio_dict[period_index])  # out it outside the loop

        # curve fitting
        x_demand_over_capacity = np.array(X_data)
        y_speed = np.array(Y_data)
        popt, pcov = curve_fit(bpr_func, x_demand_over_capacity, y_speed, p0,
                               bounds=[lowerbound_fitting, upperbound_fitting])
        # RMSE=np.power((np.sum(np.power((bpr_func(x_demand_over_capacity, *popt)-y_speed),2))/len(x_demand_over_capacity)),0.5)
        RSE = np.sum(np.power((bpr_func(x_demand_over_capacity, *popt) - y_speed), 2)) / np.sum(
            np.power((bpr_func(x_demand_over_capacity, *popt) - y_speed.mean()), 2))

    if DOC_RATIO_METHOD == 'DBM':
        print('Density method calibration...')
        internal_period_vdf_daily_link_df['hourly_demand_over_capacity'] = internal_period_vdf_daily_link_df.apply(
            lambda x: x.period_mean_density / critical_density, axis=1)
        X_data = []
        Y_data = []
        for k in range(0, len(internal_period_vdf_daily_link_df)):
            # Hourly hourly_demand_over_capacity data 
            for kk in range(WEIGHT_HOURLY_DATA):
                Y_data.append(internal_period_vdf_daily_link_df.loc[k, 'period_mean_daily_speed'])
                X_data.append(internal_period_vdf_daily_link_df.loc[k, 'hourly_demand_over_capacity'])
            # Period hourly_demand_over_capacity data
            for kk in range(WEIGHT_PERIOD_DATA):
                Y_data.append(internal_period_vdf_daily_link_df['period_mean_daily_speed'].mean())
                X_data.append(internal_period_vdf_daily_link_df['period_demand_over_capacity'].mean())
            for kk in range(WEIGHT_UPPER_BOUND_DOC_RATIO):
                Y_data.append(0.001)
                X_data.append(upper_bound_doc_ratio_dict[period_index])
        x_demand_over_capacity = np.array(X_data)
        y_speed = np.array(Y_data)
        popt, pcov = curve_fit(bpr_func, x_demand_over_capacity, y_speed, p0,
                               bounds=[lowerbound_fitting, upperbound_fitting])
        RMSE = np.power(
            (np.sum(np.power((bpr_func(x_demand_over_capacity, *popt) - y_speed), 2)) / len(x_demand_over_capacity)),
            0.5)
        RSE = np.sum(np.power((bpr_func(x_demand_over_capacity, *popt) - y_speed), 2)) / np.sum(
            np.power((bpr_func(x_demand_over_capacity, *popt) - y_speed.mean()), 2))

    if DOC_RATIO_METHOD == 'QBM':
        print('Queue based method method calibration...')
        internal_period_vdf_daily_link_df['hourly_demand_over_capacity'] = internal_period_vdf_daily_link_df.apply(
            lambda x: x.demand / ultimate_capacity, axis=1)
        X_data = []
        Y_data = []
        for k in range(0, len(internal_period_vdf_daily_link_df)):
            # Hourly hourly_demand_over_capacity data 
            for kk in range(WEIGHT_HOURLY_DATA):
                Y_data.append(internal_period_vdf_daily_link_df.loc[k, 'congestion_period_mean_speed'])
                X_data.append(internal_period_vdf_daily_link_df.loc[k, 'hourly_demand_over_capacity'])
            # Period hourly_demand_over_capacity data
            for kk in range(WEIGHT_PERIOD_DATA):
                Y_data.append(internal_period_vdf_daily_link_df['period_mean_daily_speed'].mean())
                X_data.append(internal_period_vdf_daily_link_df['period_demand_over_capacity'].mean())
            for kk in range(WEIGHT_UPPER_BOUND_DOC_RATIO):
                Y_data.append(0.001)
                X_data.append(upper_bound_doc_ratio_dict[period_index])
        x_demand_over_capacity = np.array(X_data)
        y_speed = np.array(Y_data)
        popt, pcov = curve_fit(bpr_func, x_demand_over_capacity, y_speed,
                               bounds=[lowerbound_fitting, upperbound_fitting])

    RMSE = np.power(
        (np.sum(np.power((bpr_func(x_demand_over_capacity, *popt) - y_speed), 2)) / len(x_demand_over_capacity)), 0.5)
    RSE = np.sum(np.power((bpr_func(x_demand_over_capacity, *popt) - y_speed), 2)) / np.sum(
        np.power((bpr_func(x_demand_over_capacity, *popt) - y_speed.mean()), 2))

    xvals = np.linspace(0, 5, 50)
    popt_fundamental_diagram = np.array([critical_density, mm])
    plt.plot(x_demand_over_capacity, y_speed, '*', c='k', label='original values', markersize=3)
    plt.plot(volume_speed_func(xvals, *popt, *popt_fundamental_diagram) / ultimate_capacity, bpr_func(xvals, *popt),
             c='b')
    plt.plot(xvals, bpr_func(xvals, *popt), '--', c='r', markersize=6)
    plt.title(DOC_RATIO_METHOD + ' ' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + ' ' + str(
        period_index) + ',RSE=' + str(
        round(RSE, 2)) + '%,ffs=' + str(round(popt[0], 2)) + ',alpha=' + str(round(popt[1], 2)) + ',beta=' + str(
        round(popt[2], 2)))
    plt.xlabel('Hourly_demand_over_capacity')
    plt.ylabel('Speed')
    plt.savefig('./3_hourly_VDF_' + DOC_RATIO_METHOD + '_' + str(
        'FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + '_' + str(
        period_index) + '.png')
    plt.close()

    internal_period_vdf_daily_link_df['period_mean_volume'] = internal_period_vdf_daily_link_df['period_volume'].mean()
    internal_period_vdf_daily_link_df['period_mean_speed'] = internal_period_vdf_daily_link_df[
        'period_mean_daily_speed'].mean()

    xvals = np.linspace(0, 20000, 100) / internal_period_vdf_daily_link_df.period_capacity.mean()
    plt.plot(internal_period_vdf_daily_link_df.period_volume, internal_period_vdf_daily_link_df.period_mean_daily_speed,
             '*', c='k', label='original values', markersize=3)
    plt.plot(np.linspace(0, 20000, 100), bpr_func(xvals, *popt), '--', c='r', markersize=6)
    plt.plot(internal_period_vdf_daily_link_df['period_volume'].mean(),
             internal_period_vdf_daily_link_df['period_mean_daily_speed'].mean(), 'o', c='r', markersize=8)
    plt.title(DOC_RATIO_METHOD + ' ' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + ' ' + str(
        period_index) + ',RSE=' + str(
        round(RSE, 2)) + '%,ffs=' + str(round(popt[0], 2)) + ',alpha=' + str(round(popt[1], 2)) + ',beta=' + str(
        round(popt[2], 2)))
    plt.xlabel('Assigned_volume')
    plt.ylabel('Speed')
    plt.savefig('./3_period_VDF_' + DOC_RATIO_METHOD + '_' + str(
        'FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + '_' + str(
        period_index) + '.png')
    plt.close()

    internal_period_vdf_daily_link_df['alpha'] = round(popt[1], 2)
    internal_period_vdf_daily_link_df['beta'] = round(popt[2], 2)

    return internal_period_vdf_daily_link_df


def vdf_calculation_daily(all_calibration_period_vdf_daily_link_results, vdf_index, speed_at_capacity,
                          ultimate_capacity, critical_density, free_flow_speed):
    p0 = np.array([free_flow_speed, 0.15, 4])
    lowerbound_fitting = [free_flow_speed, 0.15, 1.01]
    upperbound_fitting = [free_flow_speed * 1.1, 10, 10]
    X_data = []
    Y_data = []

    for k in range(0, len(all_calibration_period_vdf_daily_link_results)):
        # Hourly hourly_demand_over_capacity data 
        for kk in range(WEIGHT_HOURLY_DATA):
            Y_data.append(all_calibration_period_vdf_daily_link_results.loc[k, 'congestion_period_mean_speed'])
            X_data.append(all_calibration_period_vdf_daily_link_results.loc[k, 'hourly_demand_over_capacity'])
        # Period hourly_demand_over_capacity data
        # Period hourly_demand_over_capacity data
        for kk in range(WEIGHT_PERIOD_DATA):
            Y_data.append(all_calibration_period_vdf_daily_link_results.loc[k, 'period_mean_speed'])
            X_data.append(all_calibration_period_vdf_daily_link_results.loc[k, 'period_mean_volume'] /
                          all_calibration_period_vdf_daily_link_results.loc[k, 'period_capacity'])
        for kk in range(WEIGHT_UPPER_BOUND_DOC_RATIO):
            Y_data.append(0.001)
            X_data.append(
                upper_bound_doc_ratio_dict[all_calibration_period_vdf_daily_link_results.loc[k, 'assignment_period']])
        x_demand_over_capacity = np.array(X_data)
        y_speed = np.array(Y_data)

    popt_daily, pcov = curve_fit(bpr_func, x_demand_over_capacity, y_speed,
                                 bounds=[lowerbound_fitting, upperbound_fitting])
    # daily_RMSE=np.power((np.sum(np.power((bpr_func(x_demand_over_capacity, *popt_daily)-y_speed),2))/len(x_demand_over_capacity)),0.5)
    daily_RSE = np.sum(np.power((bpr_func(x_demand_over_capacity, *popt_daily) - y_speed), 2)) / np.sum(
        np.power((bpr_func(x_demand_over_capacity, *popt_daily) - y_speed.mean()), 2))

    xvals = np.linspace(0, 5, 50)
    plt.plot(x_demand_over_capacity, y_speed, '*', c='k', label='original values', markersize=3)
    plt.plot(xvals, bpr_func(xvals, *popt_daily), '--', c='r', markersize=6)

    plt.title(
        'Daily_' + DOC_RATIO_METHOD + ',' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + ',RSE=' + str(
            round(daily_RSE, 2)) + '%,ffs=' + str(round(popt_daily[0], 2)) + ',alpha=' + str(
            round(popt_daily[1], 2)) + ',beta=' + str(round(popt_daily[2], 2)))
    plt.xlabel('hourly_demand_over_capacity')
    plt.ylabel('speed')
    plt.savefig('./4_hourly_VDF_' + DOC_RATIO_METHOD + '_' + str(
        'FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + '_day.png')
    plt.close()

    plt.plot(all_calibration_period_vdf_daily_link_results.period_volume,
             all_calibration_period_vdf_daily_link_results.period_mean_daily_speed, '*', c='k',
             label='derived training data', markersize=3)
    maximum_value = all_calibration_period_vdf_daily_link_results.period_volume.max() * 2
    capacity_period_dict = dict(zip(all_calibration_period_vdf_daily_link_results.assignment_period,
                                    all_calibration_period_vdf_daily_link_results.period_capacity, ))

    for kk in list(all_calibration_period_vdf_daily_link_results.assignment_period.unique()):
        xvals = np.linspace(0, maximum_value, 100) / capacity_period_dict[kk]
        plt.plot(np.linspace(0, maximum_value, 100), bpr_func(xvals, *popt_daily), '--', label=kk, markersize=6)
        df = all_calibration_period_vdf_daily_link_results[
            all_calibration_period_vdf_daily_link_results.assignment_period == kk]
        plt.plot(df['period_mean_volume'], df['period_mean_speed'], 'o', label=kk, markersize=8)

    plt.legend()
    plt.title(
        'Daily_' + DOC_RATIO_METHOD + ' ' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + ',RSE=' + str(
            round(daily_RSE, 2)) + '%,ffs=' + str(round(popt_daily[0], 2)) + ',alpha=' + str(
            round(popt_daily[1], 2)) + ',beta=' + str(round(popt_daily[2], 2)))
    plt.xlabel('Assigned_volume')
    plt.ylabel('Speed')
    plt.savefig('./4_period_VDF_' + DOC_RATIO_METHOD + '_' + str(
        'FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + '_day.png')
    plt.close()

    all_calibration_period_vdf_daily_link_results['daily_alpha'] = round(popt_daily[1], 2)
    all_calibration_period_vdf_daily_link_results['daily_beta'] = round(popt_daily[2], 2)

    return all_calibration_period_vdf_daily_link_results


# In[5] Calculate demand and congestion period
def calculate_congestion_duration(speed_series, volume_per_lane_series, hourly_volume_per_lane_series,
                                  speed_at_capacity, ultimate_capacity, length, free_flow_speed):
    global PSTW  # preferred service time window
    nb_time_stamp = len(speed_series)
    min_speed = min(speed_series)
    min_index = speed_series.index(min(speed_series))  # The index of speed with minimum value

    # start time and ending time of prefered service time window
    PSTW_start_time = max(min_index - 2, 0)
    PSTW_ending_time = min(min_index + 1, nb_time_stamp)
    if PSTW_ending_time - PSTW_start_time < 3:
        if PSTW_start_time == 0:
            PSTW_ending_time = PSTW_ending_time + (3 - (PSTW_ending_time - PSTW_start_time))
        if PSTW_ending_time == nb_time_stamp:
            PSTW_start_time = PSTW_start_time - (3 - (PSTW_ending_time - PSTW_start_time))
    PSTW = (PSTW_ending_time - PSTW_start_time + 1) * (TIME_INTERVAL_IN_MIN / 60)
    PSTW_volume = np.array(volume_per_lane_series[PSTW_start_time:PSTW_ending_time + 1]).sum()
    PSTW_speed = np.array(speed_series[PSTW_start_time:PSTW_ending_time + 1]).mean()

    # Determine 
    t3 = nb_time_stamp - 1
    t0 = 0
    if min_speed <= speed_at_capacity:
        for i in range(min_index, nb_time_stamp):
            if speed_series[i] > speed_at_capacity:
                t3 = i - 1
                break
        for j in range(min_index, -1, -1):
            # t0=PSTW_start_time
            if speed_series[j] > speed_at_capacity:
                t0 = j + 1
                break
        congestion_duration = (t3 - t0 + 1) * (TIME_INTERVAL_IN_MIN / 60)
        # peak_hour_factor_method='SBM' # if the min_speed of the link within the assignment period is less than the speed_at_capacity, then we use SBM to calculate the peak hour factor

    elif min_speed > speed_at_capacity:
        t0 = 0
        t3 = 0
        congestion_duration = 0
        # peak_hour_factor_method='VBM' # if the min_speed of the link within the assignment period is larger than the speed_at_capacity, then we use VBM to calculate the peak hour factor

    average_discharge_rate = np.mean(hourly_volume_per_lane_series[t0:t3 + 1])

    gamma = 0
    max_queue_length = 0
    # average_waiting_time=0
    if congestion_duration > PSTW:
        demand = np.array(volume_per_lane_series[t0:t3 + 1]).sum()
        congestion_period_mean_speed = np.array(speed_series[t0:t3 + 1]).mean()
        if min_speed <= speed_at_capacity:
            congestion_period_speed_series = np.array(speed_series[t0:t3 + 1])
            congestion_period_travel_time_series = length / congestion_period_speed_series
            free_flow_travel_time = length / speed_at_capacity  # free_flow_speed
            congestion_period_delay_series = congestion_period_travel_time_series - free_flow_travel_time

            time_stamp = np.array([*range(t0, t3 + 1)]) / (60.0 / TIME_INTERVAL_IN_MIN)
            t0_ph = t0 / (60.0 / TIME_INTERVAL_IN_MIN)
            t3_ph = t3 / (60.0 / TIME_INTERVAL_IN_MIN)

            waiting_time_term = (1 / (4 * average_discharge_rate)) * (time_stamp - t0_ph) * (time_stamp - t3_ph) * (
                    time_stamp - t3_ph) * (time_stamp - t0_ph)
            linreg = LinearRegression(fit_intercept=False)  # do not calculate the intercept for this model
            derived_waiting_time_term = waiting_time_term.reshape(waiting_time_term.shape[0], 1)
            linreg.fit(derived_waiting_time_term, congestion_period_delay_series) # , in X Y
            gamma = linreg.coef_[0]
            queue_series = waiting_time_term * average_discharge_rate * gamma
            max_queue_length = np.max(queue_series)

            # average_waiting_time=np.mean(congestion_period_delay_series)
            # average_waiting_time=(gamma/(120*average_discharge_rate))*(np.array(volume_per_lane_series[t0:t3+1]).sum()/average_discharge_rate)
            # print(1)
    elif congestion_duration <= PSTW:
        demand = PSTW_volume
        congestion_period_mean_speed = PSTW_speed

    return t0, t3, congestion_duration, PSTW_start_time, PSTW_ending_time, PSTW, demand, average_discharge_rate, congestion_period_mean_speed, peak_hour_factor_method, gamma, max_queue_length


# In[6] Main
if __name__ == "__main__":
    # Step 0: Parameter setting
    # Step 0.1. Parameters set in external setting.csv
    with open('./test/setting.csv', encoding='UTF-8') as setting_file:
        setting_csv = csv.reader(setting_file)
        for row in setting_csv:
            if row[0] == "DOC_RATIO_METHOD":
                # volume based method, VBM; density based method, DBM, and queue based method QBM, or BPR_X
                DOC_RATIO_METHOD = row[1]

    # Step 0.2. Internal parameters
    LOG_FILE = 1  # Open the log file or not, 1 and 0
    TIME_INTERVAL_IN_MIN = 15  # the time stamp in minutes in the observation records
    UPPER_BOUND_CRITICAL_DENSITY = 50  # we assume that the upper bound of critical density is 50 vehicle/mile
    UPPER_BOUND_JAM_DENSITY = 220  # we assume that the upper bound of jam density is 220 vehicle/mile
    OUTER_LAYER_QUANTILE = 0.9  # The quantile threshold to generate the outer layer to calibrate traffic flow model
    LOWER_BOUND_OF_OUTER_LAYER_SAMPLES = 20  # number_of_outer_layer_samples
    MIN_THRESHOLD_SAMPLING = 1  # if the missing data of a link during a peak period less than the threshold delete the data
    WEIGHT_HOURLY_DATA = 1  # Weight of hourly data during calibratio
    WEIGHT_PERIOD_DATA = 10  # Weight of average period speed and volume during the calibration
    WEIGHT_UPPER_BOUND_DOC_RATIO = 100  # Weight of prompting the VDF curve to 0 when the DOC is close to its maximum values
    peak_hour_factor_method = 'SBM'

    # Step 1: Read input data
    print('Step 1:Read input data...')
    training_set = input_data()  # training_set is a data frame of pandas to store the whole link_performance.csv file
    ASSIGNMENT_PERIOD = training_set.assignment_period.unique().tolist()  # array of assignment periods
    PERIOD_LENGTH = []  # list of the length of assignment periods
    NUMBER_OF_RECORDS = []  # list of the number of records of assignment periods
    UPPER_BOUND_DOC_RATIO = []  # list of the upper bound of demand over capacity ratio
    for period in ASSIGNMENT_PERIOD:  # parsor HHMM time, period length
        time_ss = [int(var[0:2]) for var in period.split('_')]
        if time_ss[0] > time_ss[1]:
            period_length = time_ss[1] + 24 - time_ss[0]
            # e.g. if the assignment period is 1800_0600, then we will calculate that 6-18+24=12
            upper_bound_doc_ratio = np.minimum(5, period_length)
            # assume that the maximum value of D over C ratio is 6 hours and should not be larger than the length of assignment periods
            number_of_records = period_length * (60 / TIME_INTERVAL_IN_MIN)
            # calculate the complete number of records in the time-series data of a link during an assignment period, e.g if  assignment period 0600_0900 should have 3 hours * 4 records (if time stamp is 15 minutes)
        else:
            period_length = time_ss[1] - time_ss[0]
            upper_bound_doc_ratio = np.minimum(5, period_length)
            number_of_records = period_length * (60 / TIME_INTERVAL_IN_MIN)

        PERIOD_LENGTH.append(period_length)
        NUMBER_OF_RECORDS.append(number_of_records)
        UPPER_BOUND_DOC_RATIO.append(upper_bound_doc_ratio)

    # create three hash table to map assignment periods to period lenght/ upper bound of DOC/ complete number of records
    period_length_dict = dict(zip(ASSIGNMENT_PERIOD, PERIOD_LENGTH))
    upper_bound_doc_ratio_dict = dict(zip(ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO))
    number_of_records_dict = dict(zip(ASSIGNMENT_PERIOD, NUMBER_OF_RECORDS))

    # Step 2: For each VDF type, calibrate basic coefficients for fundamental diagrams
    # Step 2.1: Group the data frame by VDF types. Each VDF type is a combination of facility type (FT) and area type (AT)
    vdf_group = training_set.groupby(['FT', 'AT'])
    output_df_daily = pd.DataFrame()  # Create empty dataframe
    output_df = pd.DataFrame()  # Create empty dataframe
    alpha_dict = {}  # Create empty dictionary
    beta_dict = {}  # Create empty dictionary

    for vdf_index, vdf_training_set in vdf_group:  # vdf_index is a pair of facility type and area type e.g. vdf_index = (1,1) implies that FT=1 and AT=1

        vdf_training_set.reset_index(drop=True, inplace=True)  # reset index of the sub dataframe
        print('Step 2: Calibrate key coefficients in traffic stream models in VDF type ' + str(
            'FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + ' ...')
        # For each VDF type, we have a unique index for each VDF type e.g. (1) VDF type FT = 1 and AT = 1:  AT*100+FT=100*1+1=101 (2) VDF type FT = 3 and AT = 2:  AT*200+FT=100*2+3=203
        speed_at_capacity, ultimate_capacity, critical_density, free_flow_speed, mm = calibrate_traffic_flow_model(
            vdf_training_set, vdf_index)  # calibrate parameters of traffic flow model
        # calibrate the four key parameters : speed at capacity, ultimate capacity , critical density and free flow speed. 

        all_calibration_period_vdf_daily_link_results = pd.DataFrame()  # Create empty dataframe

        # Step 3: For each VDF and assignment period, calibrate congestion duration, demand, t0, t3, alpha and beta
        print('Step 3: Calibrate VDF function of links for VDF_type: ' + str(vdf_index) + ' and time period...')
        # group the vdf_train_set according to time periods
        period_vdf_group = vdf_training_set.groupby(['assignment_period'])
        for period_index, period_vdf_training_set in period_vdf_group:  # _period_based_vdf_training_set
            internal_period_vdf_daily_link_df = pd.DataFrame()
            # day_period_link_based_vdf_training_set
            daily_link_group = period_vdf_training_set.groupby(['link_id', 'from_node_id', 'to_node_id', 'date'])
            daily_link_list = []
            period_vdf_training_set_reorder = pd.DataFrame()  # store the dataframe reordered by the timestamp
            for daily_link_index, daily_link_training_set in daily_link_group:
                daily_link_training_set.reset_index(drop=True, inplace=True)  # reset index of the sub dataframe
                link_id = daily_link_index[0]
                from_node_id = daily_link_index[1]
                to_node_id = daily_link_index[2]
                date = daily_link_index[3]
                FT = vdf_index[0]
                AT = vdf_index[1]
                period_index = period_index
                if (len(daily_link_training_set) < number_of_records_dict[period_index]) and ((1 - len(
                        daily_link_training_set) / number_of_records_dict[period_index]) >= MIN_THRESHOLD_SAMPLING):
                    print('WARNING:  link ', link_id, 'does not have enough time series records in assignment period',
                          period_index, 'at', daily_link_index[3], '...')
                    continue
                period_volume = daily_link_training_set[
                    'volume_per_lane'].sum()  # summation of all volume per lane within the period, +++++
                period_mean_hourly_volume_per_lane = daily_link_training_set[
                    'hourly_volume_per_lane'].mean()  # mean hourly value
                period_mean_daily_speed = daily_link_training_set['speed'].mean()
                period_mean_density = daily_link_training_set['density'].mean()

                # We need to resort the dataframe if assignment period is like 2000_0600
                try:
                    daily_link_training_set[['start_time', 'end_time']] = \
                        (daily_link_training_set['assignment_period'].str.split('_', expand=True))
                    daily_link_training_set[['start_time_each', 'end_time_each']] = \
                        (daily_link_training_set['time_period'].str.split('_', expand=True))
                    daily_link_training_set['start_time0'] = \
                        (daily_link_training_set['start_time'].str[0:2]).astype(int)
                    daily_link_training_set['end_time0'] = (daily_link_training_set['end_time'].str[0:2]).astype(int)
                    if list(daily_link_training_set['start_time0'])[0] > list(daily_link_training_set['end_time0'])[0]:
                        start_index = daily_link_training_set[daily_link_training_set['start_time_each'] ==
                                                              daily_link_training_set['start_time']].index[0]
                        daily_link_training_set = pd.concat(
                            [daily_link_training_set[start_index:], daily_link_training_set[:start_index]]).reset_index(
                            drop=True)
                    daily_link_training_set = daily_link_training_set.drop(
                        ['start_time', 'end_time', 'start_time_each', 'end_time_each', 'start_time0', 'end_time0'],
                        axis=1)
                except:
                    daily_link_training_set = daily_link_training_set
                period_vdf_training_set_reorder = period_vdf_training_set_reorder.append(daily_link_training_set)

                volume_per_lane_series = daily_link_training_set.volume_per_lane.to_list()
                speed_series = daily_link_training_set.speed.to_list()
                hourly_volume_per_lane_series = daily_link_training_set.hourly_volume_per_lane.to_list()  # --> hourly_volume_per_lane
                link_length = daily_link_training_set.length.mean()  # obtain the length of the link
                link_free_flow_speed = free_flow_speed

                # Step 3.1 Calculate Demand over capacity and congestion duration
                t0, t3, congestion_duration, PSTW_start_time, PSTW_ending_time, PSTW, demand, average_discharge_rate, congestion_period_mean_speed, peak_hour_factor_method, gamma, max_queue_length \
                    = calculate_congestion_duration(speed_series, volume_per_lane_series, hourly_volume_per_lane_series,
                                                    speed_at_capacity, ultimate_capacity, link_length,
                                                    link_free_flow_speed)

                # Step 3.2 Calculate peak hour factor for each link
                vol_hour_max = np.max(hourly_volume_per_lane_series)
                EPS = ultimate_capacity / 7  # setting a lower bound of demand
                if peak_hour_factor_method == 'SBM':
                    peak_hour_factor = period_volume / max(demand, EPS)
                if peak_hour_factor_method == 'VBM':
                    peak_hour_factor = period_volume / vol_hour_max  # per link peak hour factor

                daily_link = [link_id, from_node_id, to_node_id, date, FT, AT, period_index, period_volume,
                              period_mean_hourly_volume_per_lane, period_mean_daily_speed, period_mean_density, t0, t3,
                              demand, average_discharge_rate, peak_hour_factor, congestion_duration,
                              congestion_period_mean_speed, gamma, max_queue_length]
                daily_link_list.append(daily_link)

            if len(daily_link_list) == 0:
                print('WARNING: all the links of ' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(
                    vdf_index[1])) + ' during assignment period ' + period_index + ' are not qualified...')
                continue

            internal_period_vdf_daily_link_df = pd.DataFrame(daily_link_list)
            internal_period_vdf_daily_link_df. \
                rename(columns={0: 'link_id', 1: 'from_node_id', 2: 'to_node_id', 3: 'date', 4: 'FT',
                                5: 'AT', 6: 'assignment_period', 7: 'period_volume',
                                8: 'period_mean_hourly_volume_per_lane', 9: 'period_mean_daily_speed',
                                10: 'period_mean_density', 11: 't0', 12: 't3', 13: 'demand',
                                14: 'average_discharge_rate', 15: 'peak_hour_factor', 16: 'congestion_duration',
                                17: 'congestion_period_mean_speed', 18: 'gamma', 19: 'max_queue_length'}, inplace=True)
            internal_period_vdf_daily_link_df.to_csv(
                './2_training_set_' + str('FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1])) + '_' + str(
                    period_index) + '.csv',
                index=False)

            # Step 3.3 calculate the peak hour factor for each period and VDF type 
            period_peak_hour_factor = np.mean(internal_period_vdf_daily_link_df.peak_hour_factor)
            internal_period_vdf_daily_link_df['period_peak_hour_factor'] = period_peak_hour_factor
            internal_period_vdf_daily_link_df['ultimate_capacity'] = ultimate_capacity
            internal_period_vdf_daily_link_df['period_capacity'] = ultimate_capacity * period_peak_hour_factor
            internal_period_vdf_daily_link_df['period_demand_over_capacity'] = \
                internal_period_vdf_daily_link_df['period_volume'].mean() / (
                        ultimate_capacity * period_peak_hour_factor)

            # Step 3.4 calculate alpha and beta for each period and VDF type
            calibration_period_vdf_daily_link_results = \
                vdf_calculation(internal_period_vdf_daily_link_df, vdf_index, period_index, speed_at_capacity,
                                ultimate_capacity, critical_density, free_flow_speed, mm)

            # Step 3.5 calculate gamma for each period and VDF type: link-based then average
            allday_link_group = \
                period_vdf_training_set_reorder.groupby(['link_id', 'from_node_id', 'to_node_id'], sort=False)
            all_link_gamma = []
            for link_index, link_training_set in allday_link_group:
                period_vdf_training_set_mean = \
                    link_training_set.groupby(['time_period'], sort=False).mean().reset_index()  # average all dates
                volume_per_lane_series = period_vdf_training_set_mean.volume_per_lane.to_list()
                speed_series = period_vdf_training_set_mean.speed.to_list()
                hourly_volume_per_lane_series = period_vdf_training_set_mean.hourly_volume_per_lane.to_list()  # --> hourly_volume_per_lane
                link_length = period_vdf_training_set_mean.length.mean()  # obtain the length of the link
                link_free_flow_speed = free_flow_speed
                # Step 3.1 Calculate Demand over capacity and congestion duration
                t0, t3, congestion_duration, PSTW_start_time, PSTW_ending_time, PSTW, demand, average_discharge_rate, congestion_period_mean_speed, peak_hour_factor_method, gamma, max_queue_length \
                    = calculate_congestion_duration(speed_series, volume_per_lane_series, hourly_volume_per_lane_series,
                                                    speed_at_capacity, ultimate_capacity, link_length,
                                                    link_free_flow_speed)
                all_link_gamma.append(gamma)

            all_calibration_period_vdf_daily_link_results = pd.concat(
                [all_calibration_period_vdf_daily_link_results, calibration_period_vdf_daily_link_results], sort=False)
            para = [vdf_index, 'FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1]), vdf_index[0], vdf_index[1],
                    period_index,
                    speed_at_capacity, ultimate_capacity, critical_density, free_flow_speed, mm,
                    period_peak_hour_factor, calibration_period_vdf_daily_link_results.alpha.mean(),
                    calibration_period_vdf_daily_link_results.beta.mean(), np.mean(all_link_gamma)]
            g_parameter_list.append(para)

        # Step 4 Calibrate daily VDF function 
        print('Step 4: Calibrate daily VDF function for VDF_type:' + str(vdf_index) + '...\n')

        all_calibration_period_vdf_daily_link_results = all_calibration_period_vdf_daily_link_results.reset_index(
            drop=True)

        # change t0, t3 to time format
        all_calibration_period_vdf_daily_link_results[['start_time', 'end_time']] = \
            (all_calibration_period_vdf_daily_link_results['assignment_period'].str.split('_', expand=True))
        all_calibration_period_vdf_daily_link_results['start_time'] = \
            pd.to_datetime('2020-01-01 ' + all_calibration_period_vdf_daily_link_results['start_time'] + '00')
        all_calibration_period_vdf_daily_link_results['timedelta_t0'] = pd.to_timedelta(
            all_calibration_period_vdf_daily_link_results['t0'] * TIME_INTERVAL_IN_MIN, 'm')
        all_calibration_period_vdf_daily_link_results['timedelta_t3'] = pd.to_timedelta(
            all_calibration_period_vdf_daily_link_results['t3'] * TIME_INTERVAL_IN_MIN + TIME_INTERVAL_IN_MIN, 'm')
        all_calibration_period_vdf_daily_link_results['new_t0'] = \
            all_calibration_period_vdf_daily_link_results['timedelta_t0'] + \
            all_calibration_period_vdf_daily_link_results['start_time']
        all_calibration_period_vdf_daily_link_results['new_t3'] = \
            all_calibration_period_vdf_daily_link_results['timedelta_t3'] + \
            all_calibration_period_vdf_daily_link_results['start_time']
        all_calibration_period_vdf_daily_link_results['new_t0'] = all_calibration_period_vdf_daily_link_results[
            'new_t0'].dt.strftime('%H:%M')
        all_calibration_period_vdf_daily_link_results['new_t3'] = all_calibration_period_vdf_daily_link_results[
            'new_t3'].dt.strftime('%H:%M')
        all_calibration_period_vdf_daily_link_results.loc[
            all_calibration_period_vdf_daily_link_results['t3'] == 0, ['new_t0', 'new_t3']] = 0
        all_calibration_period_vdf_daily_link_results = all_calibration_period_vdf_daily_link_results.drop(
            ['start_time', 'end_time', 'timedelta_t0', 'timedelta_t3'], axis=1)

        all_calibration_period_vdf_daily_link_results[
            'VDF_TYPE'] = 'FT_' + (all_calibration_period_vdf_daily_link_results.FT).astype(str) + '_AT_' + (
            all_calibration_period_vdf_daily_link_results.AT).astype(str)
        all_calibration_period_vdf_daily_link_results = vdf_calculation_daily(
            all_calibration_period_vdf_daily_link_results, vdf_index, speed_at_capacity, ultimate_capacity,
            critical_density, free_flow_speed)
        output_df_daily = pd.concat([output_df_daily, all_calibration_period_vdf_daily_link_results], sort=False)

        para = ['FT_' + str(vdf_index[0]) + '_AT_' + str(vdf_index[1]), vdf_index[0], vdf_index[1], 'daily', '--', '--',
                '--', '--', '--', '--', '--', all_calibration_period_vdf_daily_link_results.daily_alpha.mean(),
                all_calibration_period_vdf_daily_link_results.daily_beta.mean(), '--']
        g_parameter_list.append(para)

    if len(g_parameter_list) == 0:
        print('WARNING: No available data')
        exit()

    # Step 6 Output results
    print('Step 5: Output...\n')
    para_df = pd.DataFrame(g_parameter_list)
    para_df.rename(
        columns={0: 'VDF', 1: 'VDF_TYPE', 2: 'FT', 3: 'AT', 4: 'period', 5: 'speed_at_capacity', 6: 'ultimate_capacity',
                 7: 'critical_density', 8: 'free_flow_speed', 9: 'mm', 10: 'peak_hour_factor', 11: 'alpha',
                 12: 'beta', 13: 'gamma'}, inplace=True)
    para_df.to_csv('./5_VDF_summary.csv', index=False)  # vdf calibratio summary
    output_df_daily.to_csv('./5_calibration_daily_output.csv', index=False)  # day by day
    print('END...')
