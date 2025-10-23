#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import glob

import random
import math
import time
import datetime
import warnings

warnings.filterwarnings("ignore")


# In[7]:


AIS_20_01_01_filepath = (
    r"D:\\Maritime_Vessel_monitoring\\csv_extracted_data\AIS_2020_01_01",
)
AIS_20_01_02_filepath = (
    r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_02",
)
AIS_20_01_03_filepath = (
    r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_03",
)
AIS_20_01_04_filepath = (
    r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_04",
)
AIS_20_01_05_filepath = (
    r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_05",
)
AIS_20_01_06_fiilepath = (
    r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_06",
)
AIS_20_01_07_fiilepath = (
    r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_07",
)
AIS_20_01_08_fiilepath = (
    r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_08",
)
AIS_20_01_09_fiilepath = (
    r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_09",
)
AIS_20_01_10_fiilepath = (
    r"D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_10"
)


# In[ ]:


AIS_2020_01_03 = pd.read_csv(
    "D:\Maritime_Vessel_monitoring\csv_extracted_data\AIS_2020_01_03\AIS_2020_01_03.csv"
)

AIS_2020_01_03.head(10)


# In[9]:


AIS_2020_01_03["BaseDateTime"] = pd.to_datetime(
    AIS_2020_01_03["BaseDateTime"],
    format="%Y-%m-%dT%H:%M:%S",  # our data’s format: YYYY-MM-DDTHH:MM:SS
    errors="coerce",
)

AIS_2020_01_03.head(10)


# In[10]:


AIS_2020_01_03.info()


# In[ ]:





# # Predictive Analytics & Anomaly Detection
# 
# 
# •	Develop and validate vessel trajectory prediction models using statistical and machine learning methods (Kalman Filters, ARIMA, LSTM, etc.).
# 
# •	Implement trajectory consistency checks to verify whether a vessel’s latest reported position aligns with its historical movement pattern.
# 
# •	Create anomaly detection algorithms for identifying sudden deviations, spoofing, or inconsistent data patterns.
# 

# 

# In[ ]:





# In[ ]:




