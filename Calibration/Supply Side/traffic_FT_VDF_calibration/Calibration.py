# In[0] Import necessary packages 
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import heapq
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import random

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# os.chdir(r'C:\Users\huson\PycharmProjects\DTA_P3\traffic_FT_VDF_calibration')

# In[1] Set parameters
TIM_STAMP = 15

WEIGHT_CAP_DROP = 1
WEIGHT_REASON_JAM = 100
REASON_JAM = 220  # The upper bound of density
SPD_CAP = 1  # speed based period capacity
METHOD = "QBM"  # volume based method, VBM; density based method, DBM, and queue based method QBM, or BPR_X
OUTPUT = "DAY"  # DAY, or PERIOD
INCOMP_SAMPLE = 0.5  # if the incomplete of the records larger than the threshold, we will give up the link's data during a time-period
FILE = 1

weight_hourly_data = 1
weight_period_data = 5
weight_max_cong_period = 100

g_number_of_plink = 0
g_plink_id_dict = {}
g_plink_nb_seq_dict = {}
g_parameter_list = []
g_vdf_group_list = []


# In[2] Upper bound and lower bound setting
def max_cong_period(period, vdf_name):
    if period == "1400_1800":
        if vdf_name[0] == 1:
            return 4
        else:
            return 4
    if period == "0600_0900":
        if vdf_name[0] == 1:
            return 3
        else:
            return 3
    if period == "0900_1400":
        if vdf_name[0] == 1:
            return 4
        else:
            return 5
    if period == "1800_0600":
        if vdf_name[0] == 1:
            return 4
        else:
            return 12
    if period == "Day":
        if vdf_name[0] == 1:
            return 4
        else:
            return 12


# In[3] input data
def input_data():
    data_df = pd.read_csv('./0_input/link_performance.csv', encoding='UTF-8')
    data_df = data_df.drop(data_df[(data_df.volume == 0) | (data_df.speed == 0)].index)  # delete the records speed=0
    data_df['volume_pl'] = data_df['volume'] / data_df['lanes']

    # Filtering 
    data_df = data_df[(data_df['FT'] == 1) | (data_df['FT'] == 6)]
    data_df = data_df[(data_df['AT'] == 1) | (data_df['AT'] == 2) | (data_df['AT'] == 3)]
    # data_df=data_df[(data_df['assignment_period']=='1400_1800') ]
    data_df.reset_index(drop=True, inplace=True)

    # Calculate hourly vol and density
    data_df['volume_hourly'] = data_df['volume_pl'] * 4
    data_df['density'] = data_df['volume_hourly'] / data_df['speed']

    # data_df.to_csv('./1_calibration_output/0_training_set.csv',index=False)
    # data_df=pd.read_csv('./1_calibration_output/0_training_set.csv',encoding='UTF-8')
    return data_df


# In[4] Traffic flow models and volume delay function (BPR function)
def dens_spd_func(x, ffs, k_critical, mm):  # fundamental diagram model (density-speed function)
    x_over_k = x / k_critical
    dominator = 1 + np.power(x_over_k, mm)
    order = 2 / mm
    return ffs / np.power(dominator, order)


def volume_speed_func(x, ffs, alpha, beta, K_CRI, mm):  # fundamental diagram  (volume_delay fuction)
    speed = bpr_func(x, ffs, alpha, beta)
    temp_1 = np.power(ffs / speed, mm)
    temp_2 = np.power(temp_1, 0.5)
    return speed * K_CRI * np.power(temp_2 - 1, 1 / mm)


def bpr_func(x, ffs, alpha, beta):  # BPR volume delay function
    return ffs / (1 + alpha * np.power(x, beta))


# In[5] Calibrate traffic flow model 
def calibrate_traffic_flow(training_set, vdf_name):
    training_set_1 = training_set.sort_values(by='speed')
    training_set_1.reset_index(drop=True, inplace=True)

    lower_bound_FFS = training_set_1['speed'].mean()  # The lower bound of freeflow speed (mean value of speed)
    upper_bound_FFS = np.maximum(training_set_1['speed_limit'].mean(),
                                 lower_bound_FFS + 0.1)  # The upper bound of freeflow speed (mean value of speed limit)

    # fitting speed density fundamental diagram
    plt.plot(training_set_1['density'], training_set_1['speed'], '*', c='k', label='original values', markersize=2)
    X_data = []
    Y_data = []
    for k in range(0, len(training_set_1), 10):
        Y_data.append(training_set_1.loc[k:k + 10, 'speed'].mean())
        threshold = training_set_1.loc[k:k + 10, 'density'].quantile(0.9)  # setting threshold for density
        intern_training_set_1 = training_set_1[k:k + 10]
        X_data.append(intern_training_set_1[(intern_training_set_1['density'] >= threshold)]['density'].mean())
    x = np.array(X_data)
    y = np.array(Y_data)

    popt, pcov = curve_fit(dens_spd_func, x, y, bounds=[[lower_bound_FFS, 0, 0], [upper_bound_FFS, REASON_JAM, 10]])

    xvals = np.sort(x)
    plt.plot(training_set_1['density'], training_set_1['speed'], '*', c='k', label='original values', markersize=1)
    plt.plot(xvals, dens_spd_func(xvals, *popt), '--', c='r', markersize=6)
    plt.title('Traffic flow function fitting,VDF: ' + str(vdf_name))
    plt.xlabel('density (vpmpl)')
    plt.ylabel('speed (mph)')
    plt.savefig('./1_calibration_output/1_FD_speed_density_' + str(vdf_name) + '.png')
    plt.close()

    plt.plot(training_set_1['volume_hourly'], training_set_1['speed'], '*', c='k', label='original values',
             markersize=1)
    plt.plot(xvals * dens_spd_func(xvals, *popt), dens_spd_func(xvals, *popt), '--', c='r', markersize=6)
    plt.title('Traffic flow function fitting,VDF: ' + str(vdf_name))
    plt.xlabel('volume (vphpl)')
    plt.ylabel('speed (mph)')
    plt.savefig('./1_calibration_output/1_FD_speed_volume_' + str(vdf_name) + '.png')
    plt.close()

    plt.plot(training_set_1['density'], training_set_1['volume_hourly'], '*', c='k', label='original values',
             markersize=1)
    plt.plot(xvals, xvals * dens_spd_func(xvals, *popt), '--', c='r', markersize=6)
    plt.title('Traffic flow function fitting,VDF: ' + str(vdf_name))
    plt.xlabel('density (vpmpl)')
    plt.ylabel('volume (vphpl)')
    plt.savefig('./1_calibration_output/1_FD_volume_density_' + str(vdf_name) + '.png')
    plt.close()

    FFS = popt[0]
    K_CRI = popt[1]
    mm = popt[2]
    CUT_OFF_SPD = FFS / np.power(2, 2 / mm)
    ULT_CAP = CUT_OFF_SPD * K_CRI
    print('--CUT_OFF_SPD=', CUT_OFF_SPD)
    print('--ULT_CAP=', ULT_CAP)
    print('--K_CRI=', K_CRI)
    print('--FFS=', FFS)
    print('--mm=', mm)
    return CUT_OFF_SPD, ULT_CAP, K_CRI, FFS, mm


# In[8] VDF calibration
def vdf_calculation(internal_vdf_dlink_df, vdf_name, period_name, CUT_OFF_SPD, ULT_CAP, K_CRI, FFS, mm,
                    peak_factor_avg):
    internal_vdf_dlink_df['VOC_period'] = internal_vdf_dlink_df['vol_period'].mean() / (ULT_CAP * peak_factor_avg)
    p0 = np.array([FFS, 0.15, 4])
    lowerbound_fitting = [FFS, 0.15, 1.01]  # upper bound and lower bound of free flow speed, alpha and beta
    upperbound_fitting = [FFS * 1.1, 10, 10]

    popt_1 = np.array([K_CRI, mm])
    if METHOD == 'VBM':
        print('Volume method calibration...')
        internal_vdf_dlink_df['VOC'] = internal_vdf_dlink_df.apply(lambda x: (ULT_CAP + (
                ULT_CAP - x.vol_period_hourly)) / ULT_CAP if x.speed_period < CUT_OFF_SPD else x.vol_period_hourly / ULT_CAP,
                                                                   axis=1)

        X_data = []
        Y_data = []

        for k in range(0, len(internal_vdf_dlink_df)):
            # Hourly VOC data 
            for kk in range(weight_hourly_data):
                Y_data.append(internal_vdf_dlink_df.loc[k, 'speed_period'])
                X_data.append(internal_vdf_dlink_df.loc[k, 'VOC'])
            # Period VOC data
            for kk in range(weight_period_data):
                Y_data.append(internal_vdf_dlink_df['speed_period'].mean())
                X_data.append(internal_vdf_dlink_df['VOC_period'].mean())
            for kk in range(weight_max_cong_period):
                Y_data.append(0.001)
                X_data.append(max_cong_period(period_name, vdf_name))

        x = np.array(X_data)
        y = np.array(Y_data)
        popt_VBM, pcov = curve_fit(bpr_func, x, y, p0, bounds=[lowerbound_fitting, upperbound_fitting])
        VBM_RMSE = np.power((np.sum(np.power((bpr_func(x, *popt_VBM) - y), 2)) / len(x)), 0.5)
        VBM_RSE = np.sum(np.power((bpr_func(x, *popt_VBM) - y), 2)) / np.sum(
            np.power((bpr_func(x, *popt_VBM) - y.mean()), 2))

        xvals = np.linspace(0, 5, 50)

        plt.plot(x, y, '*', c='k', label='original values', markersize=3)

        plt.plot(xvals, bpr_func(xvals, *popt_VBM), '--', c='r', markersize=6)
        # plt.plot(volume_speed_func(xvals,*popt_VBM,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_VBM),c='b')
        VBM_PE = np.mean(np.abs((bpr_func(x, *popt_VBM) - y) / y))
        # print('VBM,'+str(vdf_name)+' '+str(period_name)+',RMSE='+str(round(VBM_RMSE,2))+'RSE='+str(round(VBM_RSE,2)))
        plt.title('VBM,' + str(vdf_name) + ' ' + str(period_name) + ',RSE=' + str(round(VBM_RSE, 3)) + '% ,ffs=' + str(
            round(popt_VBM[0], 2)) + ',alpha=' + str(round(popt_VBM[1], 2)) + ',beta=' + str(round(popt_VBM[2], 2)))
        plt.xlabel('VOC')
        plt.ylabel('speed (mph)')
        plt.savefig('./1_calibration_output/2_VDF_VBM_' + str(vdf_name) + '_' + str(period_name) + '.png')
        plt.close()
        internal_vdf_dlink_df['alpha'] = round(popt_VBM[1], 2)
        internal_vdf_dlink_df['beta'] = round(popt_VBM[2], 2)

    # -------------------------------------
    if METHOD == 'DBM':
        print('Density method calibration...')
        internal_vdf_dlink_df['VOC'] = internal_vdf_dlink_df.apply(lambda x: x.density_period / K_CRI, axis=1)
        X_data = []
        Y_data = []
        for k in range(0, len(internal_vdf_dlink_df)):
            # Hourly VOC data 
            for kk in range(weight_hourly_data):
                Y_data.append(internal_vdf_dlink_df.loc[k, 'speed_period'])
                X_data.append(internal_vdf_dlink_df.loc[k, 'VOC'])
            # Period VOC data
            for kk in range(weight_period_data):
                Y_data.append(internal_vdf_dlink_df['speed_period'].mean())
                X_data.append(internal_vdf_dlink_df['VOC_period'].mean())
            for kk in range(weight_max_cong_period):
                Y_data.append(0.001)
                X_data.append(max_cong_period(period_name, vdf_name))

        x = np.array(X_data)
        y = np.array(Y_data)
        popt_DBM, pcov = curve_fit(bpr_func, x, y, p0, bounds=[lowerbound_fitting, upperbound_fitting])
        DBM_RMSE = np.power((np.sum(np.power((bpr_func(x, *popt_DBM) - y), 2)) / len(x)), 0.5)
        DBM_RSE = np.sum(np.power((bpr_func(x, *popt_DBM) - y), 2)) / np.sum(
            np.power((bpr_func(x, *popt_DBM) - y.mean()), 2))

        xvals = np.linspace(0, 5, 50)
        plt.plot(x, y, '*', c='k', label='original values', markersize=3)

        plt.plot(xvals, bpr_func(xvals, *popt_DBM), '--', c='r', markersize=6)
        # plt.plot(volume_speed_func(xvals,*popt_DBM,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_DBM),c='b')
        DBM_PE = np.mean(np.abs((bpr_func(x, *popt_DBM) - y) / y))
        # print('DBM,'+str(vdf_name)+' '+str(period_name)+',RMSE='+str(round(DBM_RMSE,2))+'RSE='+str(round(DBM_RSE,2)))
        plt.title('DBM,' + str(vdf_name) + ' ' + str(period_name) + ',RSE=' + str(round(DBM_RSE, 2)) + '% ,ffs=' + str(
            round(popt_DBM[0], 2)) + ',alpha=' + str(round(popt_DBM[1], 2)) + ',beta=' + str(round(popt_DBM[2], 2)))
        plt.xlabel('VOC')
        plt.ylabel('speed (mph)')
        plt.savefig('./1_calibration_output/2_VDF_DBM_' + str(vdf_name) + '_' + str(period_name) + '.png')
        plt.close()
        internal_vdf_dlink_df['alpha'] = round(popt_DBM[1], 2)
        internal_vdf_dlink_df['beta'] = round(popt_DBM[2], 2)

    if METHOD == 'QBM':
        print('Queue based method method calibration...')
        internal_vdf_dlink_df['VOC'] = internal_vdf_dlink_df.apply(lambda x: x.Demand / ULT_CAP, axis=1)

        X_data = []
        Y_data = []
        for k in range(0, len(internal_vdf_dlink_df)):
            # Hourly VOC data 
            for kk in range(weight_hourly_data):
                Y_data.append(internal_vdf_dlink_df.loc[k, 'speed_period'])
                X_data.append(internal_vdf_dlink_df.loc[k, 'VOC'])
            # Period VOC data
            for kk in range(weight_period_data):
                Y_data.append(internal_vdf_dlink_df['speed_period'].mean())
                X_data.append(internal_vdf_dlink_df['VOC_period'].mean())
            for kk in range(weight_max_cong_period):
                Y_data.append(0.001)
                X_data.append(max_cong_period(period_name, vdf_name))

        x = np.array(X_data)
        y = np.array(Y_data)
        popt_QBM, pcov = curve_fit(bpr_func, x, y, bounds=[lowerbound_fitting, upperbound_fitting])
        QBM_RMSE = np.power((np.sum(np.power((bpr_func(x, *popt_QBM) - y), 2)) / len(x)), 0.5)
        QBM_RSE = np.sum(np.power((bpr_func(x, *popt_QBM) - y), 2)) / np.sum(
            np.power((bpr_func(x, *popt_QBM) - y.mean()), 2))

        xvals = np.linspace(0, 5, 50)
        plt.plot(x, y, '*', c='k', label='original values', markersize=3)
        plt.plot(xvals, bpr_func(xvals, *popt_QBM), '--', c='r', markersize=6)
        # plt.plot(volume_speed_func(xvals,*popt_QBM,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_QBM),c='b')

        QBM_PE = np.mean(np.abs((bpr_func(x, *popt_QBM) - y) / y))
        # print('QBM,'+str(vdf_name)+' '+str(period_name)+',RMSE='+str(round(QBM_RMSE,2))+'RSE='+str(round(QBM_RSE,2)))
        plt.title('QBM,' + str(vdf_name) + ' ' + str(period_name) + ',RSE=' + str(round(QBM_RSE, 2)) + '%,ffs=' + str(
            round(popt_QBM[0], 2)) + ',alpha=' + str(round(popt_QBM[1], 2)) + ',beta=' + str(round(popt_QBM[2], 2)))
        plt.xlabel('VOC')
        plt.ylabel('speed (mph)')
        plt.savefig('./1_calibration_output/2_VDF_QBM_' + str(vdf_name) + '_' + str(period_name) + '.png')
        plt.close()
        internal_vdf_dlink_df['alpha'] = round(popt_QBM[1], 2)
        internal_vdf_dlink_df['beta'] = round(popt_QBM[2], 2)

    if METHOD == 'BPR_X':
        print('BPR_X method calibration...')
        internal_vdf_dlink_df['VOC'] = internal_vdf_dlink_df.apply(
            lambda x: x.Demand / x.avg_discharge_rate if x.congestion_period >= PSTW else x.Demand / ULT_CAP, axis=1)

        X_data = []
        Y_data = []
        for k in range(0, len(internal_vdf_dlink_df)):
            # Hourly VOC data 
            for kk in range(weight_hourly_data):
                Y_data.append(internal_vdf_dlink_df.loc[k, 'speed_period'])
                X_data.append(internal_vdf_dlink_df.loc[k, 'VOC'])
            # Period VOC data
            for kk in range(weight_period_data):
                Y_data.append(internal_vdf_dlink_df['speed_period'].mean())
                X_data.append(internal_vdf_dlink_df['VOC_period'].mean())
            for kk in range(weight_max_cong_period):
                Y_data.append(0.001)
                X_data.append(max_cong_period(period_name, vdf_name))

        x = np.array(X_data)
        y = np.array(Y_data)
        popt_BPR_X, pcov = curve_fit(bpr_func, x, y, bounds=[lowerbound_fitting, upperbound_fitting])
        BPR_X_RMSE = np.power((np.sum(np.power((bpr_func(x, *popt_BPR_X) - y), 2)) / len(x)), 0.5)
        BPR_X_RSE = np.sum(np.power((bpr_func(x, *popt_BPR_X) - y), 2)) / np.sum(
            np.power((bpr_func(x, *popt_BPR_X) - y.mean()), 2))

        xvals = np.linspace(0, 5, 50)
        plt.plot(x, y, '*', c='k', label='original values', markersize=3)
        plt.plot(xvals, bpr_func(xvals, *popt_BPR_X), '--', c='r', markersize=6)
        # plt.plot(volume_speed_func(xvals,*popt_BPR_X,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_BPR_X),c='b')

        BPR_X_PE = np.mean(np.abs((bpr_func(x, *popt_BPR_X) - y) / y))
        # print('BPR_X,'+str(vdf_name)+' '+str(period_name)+',RMSE='+str(round(BPR_X_RMSE,2))+'RSE='+str(round(BPR_X_RSE,2)))
        plt.title(
            'BPR_X,' + str(vdf_name) + ' ' + str(period_name) + ',RSE=' + str(round(BPR_X_RSE, 2)) + '%,ffs=' + str(
                round(popt_BPR_X[0], 2)) + ',alpha=' + str(round(popt_BPR_X[1], 2)) + ',beta=' + str(
                round(popt_BPR_X[2], 2)))
        plt.xlabel('VOC')
        plt.ylabel('speed (mph)')
        plt.savefig('./1_calibration_output/2_VDF_BPR_X ' + str(vdf_name) + '_' + str(period_name) + '.png')
        plt.close()
        internal_vdf_dlink_df['alpha'] = round(popt_BPR_X[1], 2)
        internal_vdf_dlink_df['beta'] = round(popt_BPR_X[2], 2)

    return internal_vdf_dlink_df


def vdf_calculation_daily(temp_daily_df, vdf_name, CUT_OFF_SPD, ULT_CAP, K_CRI, FFS, mm):
    p0 = np.array([FFS, 0.15, 4])
    lowerbound_fitting = [FFS, 0.15, 1.01]
    upperbound_fitting = [FFS * 1.1, 10, 10]
    popt_1 = np.array([K_CRI, mm])

    X_data = []
    Y_data = []

    # output_df_daily['d_over_c_bprx']=output_df_daily.apply(lambda x: x.density_period/K_CRI,axis=1)

    for k in range(0, len(temp_daily_df)):
        # Hourly VOC data 
        for kk in range(weight_hourly_data):
            Y_data.append(temp_daily_df.loc[k, 'speed_period'])
            X_data.append(temp_daily_df.loc[k, 'VOC'])
        # Period VOC data
        # Period VOC data
        for kk in range(weight_period_data):
            Y_data.append(temp_daily_df['speed_period'].mean())
            X_data.append(temp_daily_df['VOC_period'].mean())
        for kk in range(weight_max_cong_period):
            Y_data.append(0.001)
            X_data.append(max_cong_period("Day", vdf_name))
        x = np.array(X_data)
        y = np.array(Y_data)

    popt_daily, pcov = curve_fit(bpr_func, x, y, bounds=[lowerbound_fitting, upperbound_fitting])
    daily_RMSE = np.power((np.sum(np.power((bpr_func(x, *popt_daily) - y), 2)) / len(x)), 0.5)
    daily_RSE = np.sum(np.power((bpr_func(x, *popt_daily) - y), 2)) / np.sum(
        np.power((bpr_func(x, *popt_daily) - y.mean()), 2))

    xvals = np.linspace(0, 5, 50)
    plt.plot(x, y, '*', c='k', label='original values', markersize=3)
    plt.plot(xvals, bpr_func(xvals, *popt_daily), '--', c='r', markersize=6)
    # plt.plot(volume_speed_func(xvals,*popt_daily,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_daily),c='b')

    daily_PE = np.mean(np.abs((bpr_func(x, *popt_daily) - y) / y))
    # print('Daily_'+METHOD+','+str(vdf_name)+',RMSE='+str(round(daily_RMSE,2))+'RSE='+str(round(daily_RSE,2)))
    plt.title('Daily_' + METHOD + ',' + str(vdf_name) + ',RSE=' + str(round(daily_RSE, 2)) + '%,ffs=' + str(
        round(popt_daily[0], 2)) + ',alpha=' + str(round(popt_daily[1], 2)) + ',beta=' + str(round(popt_daily[2], 2)))
    plt.xlabel('VOC')
    plt.ylabel('speed (mph)')
    plt.savefig('./1_calibration_output/2_VDF_' + METHOD + '_' + str(vdf_name) + '_day.png')
    plt.close()
    alpha_dict[temp_daily_df.VDF_TYPE.unique()[0]] = round(popt_daily[1], 2)
    beta_dict[temp_daily_df.VDF_TYPE.unique()[0]] = round(popt_daily[2], 2)
    temp_daily_df['alpha_day'] = round(popt_daily[1], 2)
    temp_daily_df['beta_day'] = round(popt_daily[2], 2)

    return alpha_dict, beta_dict, temp_daily_df


# In[9] Calculate demand and congestion period
def calculate_congestion_period(speed_15min, volume_15min, waiting_time_15min, CUT_OFF_SPD, ULT_CAP):
    global PSTW
    nb_time_stamp = len(speed_15min)
    min_speed = min(speed_15min)
    min_index = speed_15min.index(min(speed_15min))  # The index of speed with minimum value
    # start time and ending time of prefered service time window
    PSTW_st = max(min_index - 2, 0)
    PSTW_ed = min(min_index + 1, nb_time_stamp)
    if PSTW_ed - PSTW_st < 3:
        if PSTW_st == 0:
            PSTW_ed = PSTW_ed + (3 - (PSTW_ed - PSTW_st))
        if PSTW_ed == nb_time_stamp:
            PSTW_st = PSTW_st - (3 - (PSTW_ed - PSTW_st))
    PSTW = (PSTW_ed - PSTW_st + 1) * (TIM_STAMP / 60)
    PSTW_volume = np.array(volume_15min[PSTW_st:PSTW_ed + 1]).sum()
    PSTW_speed = np.array(speed_15min[PSTW_st:PSTW_ed + 1]).mean()

    # Determine 
    t3 = nb_time_stamp - 1
    t0 = 0
    if min_speed <= CUT_OFF_SPD:
        for i in range(min_index, nb_time_stamp):
            if speed_15min[i] > CUT_OFF_SPD:
                t3 = i - 1
                break
        for j in range(min_index, 0, -1):
            if speed_15min[j] > CUT_OFF_SPD:
                t0 = j + 1
                break
    elif min_speed > CUT_OFF_SPD:
        t0 = 0
        t3 = 0
    congestion_period = (t3 - t0 + 1) * (TIM_STAMP / 60)
    Mu = np.mean(volume_hour[t0:t3 + 1])
    # gamma=(plink.waiting_time.mean()*120*plink.Mu)/np.power(plink.congestion_period,4)
    rho = np.mean(np.array(waiting_time_15min) * 36 * Mu) / (congestion_period ** 3)
    if congestion_period > PSTW:
        Demand = np.array(volume_15min[t0:t3 + 1]).sum()
        speed_period = np.array(speed_15min[t0:t3 + 1]).mean()
    elif congestion_period <= PSTW:
        Demand = PSTW_volume
        speed_period = PSTW_speed
    return t0, t3, congestion_period, Demand, Mu, speed_period, rho


# # In[9] Calculate demand and congestion period
# def calculate_congestion_period_new(speed_15min, volume_15min, waiting_time_15min, CUT_OFF_SPD, ULT_CAP):
#     # global PSTW
#     nb_time_stamp = len(speed_15min)
#     min_speed = min(speed_15min)
#     min_index = speed_15min.index(min(speed_15min))  # The index of speed with minimum value
#     # Determine
#     t3 = nb_time_stamp - 1
#     t0 = 0
#     if min_speed <= CUT_OFF_SPD:
#         for i in range(min_index, nb_time_stamp):
#             if speed_15min[i] > CUT_OFF_SPD:
#                 t3 = i - 1
#                 break
#         for j in range(min_index, 0, -1):
#             if speed_15min[j] > CUT_OFF_SPD:
#                 t0 = j + 1
#                 break
#     elif min_speed > CUT_OFF_SPD:
#         t0 = 0
#         t3 = 0
#     congestion_period = (t3 - t0 + 1) * (TIM_STAMP / 60)
#     Mu = np.mean(volume_hour[t0:t3 + 1])
#     # gamma=(plink.waiting_time.mean()*120*plink.Mu)/np.power(plink.congestion_period,4)
#     rho = np.mean(np.array(waiting_time_15min) * 36 * Mu) / (congestion_period ** 3)
#     # if congestion_period > PSTW:
#     Demand = np.array(volume_15min[t0:t3 + 1]).sum()
#     speed_period = np.array(speed_15min[t0:t3 + 1]).mean()
#     # elif congestion_period <= PSTW:
#     #     Demand = PSTW_volume
#     #     speed_period = PSTW_speed
#     return t0, t3, congestion_period, Demand, Mu, speed_period, rho


# In[10] Validations
def validation(ffs, alpha, beta, K_CRI, mm, volume, capacity):
    u_assign = ffs / (1 + alpha * np.power(volume / capacity, beta))
    A = np.power(np.power(ffs / u_assign, mm), 0.5)
    flow_assign = u_assign * K_CRI * np.power(A - 1, 1 / mm)

    return flow_assign


# In[11] Check whether the samples are complete
def nb_sample_checking(period):
    if period == "1400_1800":
        return 4 * (60 / TIM_STAMP)
    if period == "0600_0900":
        return 3 * (60 / TIM_STAMP)
    if period == "0900_1400":
        return 5 * (60 / TIM_STAMP)
    if period == "1800_0600":
        return 12 * (60 / TIM_STAMP)


# In[12] Main function
if __name__ == "__main__":
    # Step 1: Input data...
    if FILE == 1:
        log_file = open("./1_calibration_output/log.txt", "w")
        log_file.truncate()
        log_file.write('Step 1:Input data...\n')

    print('Step 1:Input data...')
    start_time = time.time()

    training_set = input_data()

    end_time = time.time()
    print('CPU time:', end_time - start_time, 's\n')

    if FILE == 1:
        log_file.write('CPU time:' + str(end_time - start_time) + 's\n\n')

    # Group based on VDF types...
    FT_set = training_set['FT'].unique()
    AT_set = training_set['AT'].unique()
    vdf_group = training_set.groupby(['FT', 'AT'])  # Group by VDF types
    output_df_daily = pd.DataFrame()  # build up empty dataframe
    output_df = pd.DataFrame()  # build up empty dataframe
    alpha_dict = {}
    beta_dict = {}

    iter = 0
    for vdf_name, vdf_trainingset in vdf_group:
        '''
        vdf_name = (1, 1)
        vdf_trainingset = training_set[(training_set['FT'] == 1) & (training_set['AT'] == 1)]
        '''
        temp_daily_df = pd.DataFrame()  # build up empty dataframe
        # Step 2: For each VDF, calibrate basic coefficients for fundamental diagrams
        print('Step 2: Calibrate' + str(vdf_name) + ' key coeeficients...')
        if FILE == 1:
            log_file.write('Step 2: Calibrate' + str(vdf_name) + ' key coeeficients...\n')

        start_time = time.time()
        vdf_trainingset.reset_index(drop=True, inplace=True)
        CUT_OFF_SPD, ULT_CAP, K_CRI, FFS, mm = calibrate_traffic_flow(vdf_trainingset, vdf_name)

        end_time = time.time()

        print('CPU time:', end_time - start_time, 's\n')

        if FILE == 1:
            log_file.write('CPU time:' + str(end_time - start_time) + 's\n\n')

        # Step 3: For each VDF and period, calibrate alpha and beta
        print('Step 3: Calibrate VDF function of links for VDF_type: ' + str(vdf_name) + ' and time period...')
        if FILE == 1:
            log_file.write(
                'Step 3: Calibrate VDF function of links for VDF_type: ' + str(vdf_name) + ' and time period...\n')

        start_time = time.time()
        # 
        pvdf_group = vdf_trainingset.groupby(['assignment_period'])
        for period_name, pvdf_trainset in pvdf_group:
            '''
            period_name = '1400_1800'
            pvdf_trainset = vdf_trainingset[vdf_trainingset['assignment_period'] == period_name]
            '''
            internal_vdf_dlink_df = pd.DataFrame()

            dlink_group = pvdf_trainset.groupby(['link_id', 'from_node_id', 'to_node_id', 'Date'])
            vdf_dlink_list = []

            # Step 3.1 Calculate the VOC (congestion period)
            print('Step 3.1: Calculate the VOCs of links: ' + str(vdf_name) + ' and time period ' + str(period_name))
            if FILE == 1:
                log_file.write('Step 3.1: Calculate the VOCs of links: ' + str(vdf_name) + ' and time period ' + str(
                    period_name) + '\n')

            for dlink_name, dlink_training_set in dlink_group:
                '''
                dlink_name=('10363AB', 12596, 12614, '1/1/2018')
                dlink_training_set = pvdf_trainset[
                    (pvdf_trainset['link_id'] == '10360AB') & (pvdf_trainset['from_node_id'] == 12596) & (
                                pvdf_trainset['to_node_id'] == 12614) & (pvdf_trainset['Date'] == '1/1/2018')]
                '''
                dlink_id = dlink_name[0]
                from_node_id = dlink_name[1]
                to_node_id = dlink_name[2]
                date = dlink_name[3]
                FT = vdf_name[0]
                AT = vdf_name[1]
                period = period_name
                vol_period = dlink_training_set['volume_pl'].sum()  # summation of all volume within the period
                vol_period_hourly = dlink_training_set['volume_hourly'].mean()
                speed_period = dlink_training_set['speed'].mean()
                density_period = dlink_training_set['density'].mean()
                if len(dlink_training_set) < nb_sample_checking(period_name):
                    print('WARNING:  link ', dlink_id, 'in period', period_name,
                          'does not have all 15 minutes records...')
                    print((1 - len(dlink_training_set) / nb_sample_checking(period_name)) * 100,
                          '% of records of the link of the time period are missing...\n')
                    if (1 - len(dlink_training_set) / nb_sample_checking(period_name)) >= INCOMP_SAMPLE:
                        continue

                volume_15min = dlink_training_set.volume_pl.to_list()
                speed_15min = dlink_training_set.speed.to_list()
                volume_hour = dlink_training_set.volume_hourly.to_list()
                # Calculate waiting time
                waiting_time_15min = (1 / dlink_training_set.speed - 1 / dlink_training_set.speed_limit).to_list()
                waiting_time_15min = [0 if ii < 0 else ii for ii in waiting_time_15min]
                # Step 4.1 Calculate VOC
                t0, t3, congestion_period, Demand, avg_discharge_rate, speed_period_1, rho = calculate_congestion_period(
                    speed_15min, volume_15min, waiting_time_15min, CUT_OFF_SPD, ULT_CAP)
                # d_over_c_bprx is the VOC for queue-based method
                # Calculate peak factor for each link
                vol_hour_max = np.max(volume_hour)
                if SPD_CAP == 1:
                    peak_factor = vol_period / max(Demand, ULT_CAP / 7)
                    # if peak_factor == 1:
                    #     print('WARNING: peak factor is 1,delete the link')
                    #     continue
                else:
                    # vol_hour_max=np.max(volume_hour)
                    peak_factor = vol_period / vol_hour_max

                dlink = [dlink_id, from_node_id, to_node_id, date, FT, AT, period, vol_period, vol_period_hourly, \
                         speed_period, density_period, t0, t3, Demand, avg_discharge_rate, peak_factor,
                         congestion_period, rho, np.mean(waiting_time_15min)]
                vdf_dlink_list.append(dlink)

            internal_vdf_dlink_df = pd.DataFrame(vdf_dlink_list)
            internal_vdf_dlink_df.rename(columns={0: 'link_id',
                                                  1: 'from_node_id',
                                                  2: 'to_node_id',
                                                  3: 'Date',
                                                  4: 'FT',
                                                  5: 'AT',
                                                  6: 'period',
                                                  7: 'vol_period',
                                                  8: 'vol_period_hourly',
                                                  9: 'speed_period',
                                                  10: 'density_period',
                                                  11: 't0',
                                                  12: 't3',
                                                  13: 'Demand',
                                                  14: 'avg_discharge_rate',
                                                  15: 'peak_factor',
                                                  16: 'congestion_period',
                                                  17: 'rho',
                                                  18: 'waiting_time'}, inplace=True)
            internal_vdf_dlink_df.to_csv(
                './1_calibration_output/1_' + str(vdf_name) + ',' + str(period_name) + 'training_set.csv', index=False)
            peak_factor_avg = np.mean(internal_vdf_dlink_df.peak_factor)

            # Step 4.2 VDF calibration
            print('Step 3.2 :VDF calibration: ' + str(vdf_name) + ' and time period ' + str(period_name))
            if FILE == 1:
                log_file.write(
                    'Step 3.2 :VDF calibration: ' + str(vdf_name) + ' and time period ' + str(period_name) + '\n')

            calibration_vdf_dlink_results = vdf_calculation(internal_vdf_dlink_df, vdf_name, period_name, CUT_OFF_SPD,
                                                            ULT_CAP, K_CRI, FFS, mm, peak_factor_avg)
            # calibration_vdf_dlink_results.to_csv('./1_calibration_output/1_'+str(vdf_name)+','+str(period_name)+'training_output.csv',index=False)

            # grouyby_results=calibration_vdf_dlink_results.groupby(['link_id','from_node_id','to_node_id','FT','AT','period'])
            grouyby_results = calibration_vdf_dlink_results.groupby(
                ['link_id', 'from_node_id', 'to_node_id', 'FT', 'AT', 'period'])
            vdf_link_list = []
            for link_name, calibration_outputs in grouyby_results:
                link_id = link_name[0]
                from_node_id = link_name[1]
                to_node_id = link_name[2]
                FT = link_name[3]
                AT = link_name[4]
                period = link_name[5]
                vol_period = calibration_outputs.vol_period.mean()
                vol_period_hourly = calibration_outputs.vol_period_hourly.mean()
                speed_period = calibration_outputs.speed_period.mean()
                density_period = calibration_outputs.density_period.mean()
                t0 = calibration_outputs.t0.mean()
                t3 = calibration_outputs.t3.mean()
                Demand = calibration_outputs.Demand.mean()
                VOC = calibration_outputs.VOC.mean()
                VOC_period = calibration_outputs.VOC_period.mean()
                alpha = calibration_outputs.alpha.mean()
                beta = calibration_outputs.beta.mean()
                peak_factor = calibration_outputs.peak_factor.mean()
                period_capacity = peak_factor * ULT_CAP
                vol_valid = validation(FFS, alpha, beta, K_CRI, mm, vol_period, period_capacity)
                demand_est = VOC * period_capacity
                vdf_link = [link_id, from_node_id, to_node_id, FT, AT, period, vol_period, vol_period_hourly,
                            speed_period, density_period, t0, t3, Demand, VOC, \
                            VOC_period, alpha, beta, peak_factor, period_capacity, vol_valid, demand_est]
                vdf_link_list.append(vdf_link)

            internal_vdf_link_df = pd.DataFrame(vdf_link_list)
            internal_vdf_link_df.rename(columns={0: 'link_id',
                                                 1: 'from_node_id',
                                                 2: 'to_node_id',
                                                 3: 'FT',
                                                 4: 'AT',
                                                 5: 'period',
                                                 6: 'vol_period',
                                                 7: 'vol_period_hourly',
                                                 8: 'speed_period',
                                                 9: 'density_period',
                                                 10: 't0',
                                                 11: 't3',
                                                 12: 'Demand',
                                                 13: 'VOC',
                                                 14: 'VOC_period',
                                                 15: 'alpha',
                                                 16: 'beta',
                                                 17: 'peak_factor',
                                                 18: 'period_cap',
                                                 19: 'vol_valid',
                                                 20: 'demand_est'}, inplace=True)

            temp_daily_df = pd.concat([temp_daily_df, calibration_vdf_dlink_results], sort=False)
            output_df = pd.concat([output_df, internal_vdf_link_df], sort=False)
            per_error = np.mean(np.abs(internal_vdf_link_df['vol_period_hourly'] - internal_vdf_link_df['vol_valid']) /
                                internal_vdf_link_df['vol_valid'])
            per_error_demand = np.mean(
                np.abs(internal_vdf_link_df['vol_period'] - internal_vdf_link_df['demand_est']) / internal_vdf_link_df[
                    'demand_est'])
            alpha_1 = np.mean(internal_vdf_link_df.alpha)
            beta_1 = np.mean(internal_vdf_link_df.beta)
            peak_factor_1 = np.mean(internal_vdf_link_df.peak_factor)
            para = [vdf_name, 100 * vdf_name[1] + vdf_name[0], vdf_name[0], vdf_name[1], period_name, CUT_OFF_SPD,
                    ULT_CAP, K_CRI, FFS, mm, peak_factor_1, alpha_1, beta_1, \
                    per_error, per_error_demand]
            g_parameter_list.append(para)
            # para=[vdf_name,vdf_name[0], vdf_name[1],period_name, CUT_OFF_SPD,ULT_CAP,K_CRI,FFS,mm]
            # g_parameter_list.append(para)
            iter = iter + 1
            end_time = time.time()
            print('CPU time:', end_time - start_time, 's\n')
            if FILE == 1:
                log_file.write('CPU time:' + str(end_time - start_time) + 's\n\n')

        # Step 4 Calibrate daily VDF function 
        print('Step 4: Calibrate daily VDF function for VDF_type:' + str(vdf_name) + '...\n')

        if FILE == 1:
            log_file.write('Step 4: Calibrate daily VDF function for VDF_type:' + str(vdf_name) + '...\n')

        start_time = time.time()

        temp_daily_df = temp_daily_df.reset_index(drop=True)
        temp_daily_df['VDF_TYPE'] = 100 * temp_daily_df.AT + temp_daily_df.FT
        alpha_dict, beta_dict, temp_daily_df = vdf_calculation_daily(temp_daily_df, vdf_name, CUT_OFF_SPD, ULT_CAP,
                                                                     K_CRI, FFS, mm)
        output_df_daily = pd.concat([output_df_daily, temp_daily_df], sort=False)

        end_time = time.time()
        print('CPU time:', end_time - start_time, 's\n')

        if FILE == 1:
            log_file.write('CPU time:' + str(end_time - start_time) + 's\n\n')

    # Step 6 Output results
    print('Step 5: Output...\n')
    if FILE == 1:
        log_file.write('Step 5: Output...\n')
    para_df = pd.DataFrame(g_parameter_list)
    para_df.rename(columns={0: 'VDF',
                            1: 'VDF_TYPE',
                            2: 'FT',
                            3: 'AT',
                            4: 'period',
                            5: 'CUT_OFF_SPD',
                            6: 'ULT_CAP',
                            7: 'K_CRI',
                            8: 'FFS',
                            9: 'mm',
                            10: 'peak_factor',
                            11: 'alpha',
                            12: 'beta',
                            13: 'per_error',
                            14: 'per_error_demand'}, inplace=True)
    para_df.to_csv('./1_calibration_output/3_summary.csv', index=False)
    output_df_daily.to_csv('./1_calibration_output/3_calibration_daily_output.csv', index=False)
    output_df.to_csv('./1_calibration_output/3_calibration_output.csv', index=False)

    pk_spd_cap = pd.read_csv('./0_input/cap_speed_vdf.csv', encoding='utf-8')
    pk_spd_cap_1 = pk_spd_cap.copy()

    # Update capacity+FFS
    time_period = 'AM'
    cap_period = time_period + '_CAP'
    para_df_am = para_df[para_df.period == '0600_0900']
    am_cap_dict = dict(zip(para_df_am['VDF_TYPE'], para_df_am['ULT_CAP'] * para_df_am['peak_factor']))
    pk_spd_cap[cap_period] = pk_spd_cap.apply(
        lambda x: am_cap_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in am_cap_dict.keys() else x[cap_period],
        axis=1)

    time_period = 'MD'
    cap_period = time_period + '_CAP'
    para_df_md = para_df[para_df.period == '0900_1400']
    md_cap_dict = dict(zip(para_df_md['VDF_TYPE'], para_df_md['ULT_CAP'] * para_df_md['peak_factor']))
    pk_spd_cap[cap_period] = pk_spd_cap.apply(
        lambda x: md_cap_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in md_cap_dict.keys() else x[cap_period],
        axis=1)

    time_period = 'PM'
    cap_period = time_period + '_CAP'
    para_df_pm = para_df[para_df.period == '1400_1800']
    pm_cap_dict = dict(zip(para_df_pm['VDF_TYPE'], para_df_pm['ULT_CAP'] * para_df_pm['peak_factor']))
    pk_spd_cap[cap_period] = pk_spd_cap.apply(
        lambda x: pm_cap_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in pm_cap_dict.keys() else x[cap_period],
        axis=1)

    time_period = 'NT'
    cap_period = time_period + '_CAP'
    para_df_nt = para_df[para_df.period == '1800_0600']
    nt_cap_dict = dict(zip(para_df_nt['VDF_TYPE'], para_df_nt['ULT_CAP'] * para_df_nt['peak_factor']))
    pk_spd_cap[cap_period] = pk_spd_cap.apply(
        lambda x: nt_cap_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in nt_cap_dict.keys() else x[cap_period],
        axis=1)

    speed_dict = dict(zip(para_df['VDF_TYPE'], para_df['FFS']))
    pk_spd_cap['SPEED'] = pk_spd_cap.apply(
        lambda x: speed_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in am_cap_dict.keys() else x['SPEED'], axis=1)

    # Update alpha and beta

    if OUTPUT == "PERIOD":
        # am 
        am_alpha_dict = dict(zip(para_df_am['VDF_TYPE'], para_df_am['alpha']))
        am_beta_dict = dict(zip(para_df_am['VDF_TYPE'], para_df_am['beta']))
        pk_spd_cap['ALPHA'] = pk_spd_cap.apply(
            lambda x: am_alpha_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in am_cap_dict.keys() else x['ALPHA'],
            axis=1)
        pk_spd_cap['BETA'] = pk_spd_cap.apply(
            lambda x: am_beta_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in am_cap_dict.keys() else x['BETA'],
            axis=1)
        pk_spd_cap.to_csv('./1_calibration_output/4_pk_spd_cap_am.csv', index=False)

        # md 
        md_alpha_dict = dict(zip(para_df_md['VDF_TYPE'], para_df_md['alpha']))
        md_beta_dict = dict(zip(para_df_md['VDF_TYPE'], para_df_md['beta']))
        pk_spd_cap['ALPHA'] = pk_spd_cap.apply(
            lambda x: md_alpha_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in md_cap_dict.keys() else x['ALPHA'],
            axis=1)
        pk_spd_cap['BETA'] = pk_spd_cap.apply(
            lambda x: md_beta_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in md_cap_dict.keys() else x['BETA'],
            axis=1)
        pk_spd_cap.to_csv('./1_calibration_output/4_pk_spd_cap_md.csv', index=False)

        # pm 
        pm_alpha_dict = dict(zip(para_df_pm['VDF_TYPE'], para_df_pm['alpha']))
        pm_beta_dict = dict(zip(para_df_pm['VDF_TYPE'], para_df_pm['beta']))
        pk_spd_cap['ALPHA'] = pk_spd_cap.apply(
            lambda x: pm_alpha_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in pm_cap_dict.keys() else x['ALPHA'],
            axis=1)
        pk_spd_cap['BETA'] = pk_spd_cap.apply(
            lambda x: pm_beta_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in pm_cap_dict.keys() else x['BETA'],
            axis=1)
        pk_spd_cap.to_csv('./1_calibration_output/4_pk_spd_cap_pm.csv', index=False)

        # nt 
        nt_alpha_dict = dict(zip(para_df_nt['VDF_TYPE'], para_df_nt['alpha']))
        nt_beta_dict = dict(zip(para_df_nt['VDF_TYPE'], para_df_nt['beta']))
        pk_spd_cap['ALPHA'] = pk_spd_cap.apply(
            lambda x: nt_alpha_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in nt_cap_dict.keys() else x['ALPHA'],
            axis=1)
        pk_spd_cap['BETA'] = pk_spd_cap.apply(
            lambda x: nt_beta_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in nt_cap_dict.keys() else x['BETA'],
            axis=1)
        pk_spd_cap.to_csv('./1_calibration_output/4_pk_spd_cap_nt.csv', index=False)

    if OUTPUT == "DAY":
        # alpha_dict = dict(zip(output_df_daily['VDF_TYPE'], output_df_daily['alpha_day']))
        # beta_dict = dict(zip(output_df_daily['VDF_TYPE'], output_df_daily['beta_day']))
        pk_spd_cap['ALPHA'] = pk_spd_cap.apply(
            lambda x: alpha_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in am_cap_dict.keys() else x['ALPHA'],
            axis=1)
        pk_spd_cap['BETA'] = pk_spd_cap.apply(
            lambda x: beta_dict[int(x['VDF_TYPE'])] if int(x['VDF_TYPE']) in am_cap_dict.keys() else x['BETA'], axis=1)
        pk_spd_cap.to_csv('./1_calibration_output/4_pk_spd_cap_day.csv', index=False)

    print('END...')
    if FILE == 1:
        log_file.write('END...\n')
        log_file.close()
