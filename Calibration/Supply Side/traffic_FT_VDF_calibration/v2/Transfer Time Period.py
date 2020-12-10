import pandas as pd

Raw_Data = pd.read_csv(
    r'C:\Users\huson\PycharmProjects\DTA_P3\traffic_FT_VDF_calibration\0_input\link_performance-initial.csv')

Raw_Data['assignment_period'] = '1900_0700'
Raw_Data.loc[(Raw_Data['time_period'].str[0:2].astype(int) <= 9) & (
        Raw_Data['time_period'].str[0:2].astype(int) >= 7), 'assignment_period'] = '0700_0900'
Raw_Data.loc[(Raw_Data['time_period'].str[0:2].astype(int) <= 17) & (
        Raw_Data['time_period'].str[0:2].astype(int) >= 9), 'assignment_period'] = '0900_1700'
Raw_Data.loc[(Raw_Data['time_period'].str[0:2].astype(int) <= 19) & (
        Raw_Data['time_period'].str[0:2].astype(int) >= 17), 'assignment_period'] = '1700_1900'

Raw_Data.to_csv(
    r'C:\Users\huson\PycharmProjects\DTA_P3\traffic_FT_VDF_calibration\0_input\link_performance.csv', index=0)
