import numpy as np
import pandas as pd 
import os

def enron_finclean():
    df1 = pd.read_csv('enronfinance.csv')
    df2 = pd.read_csv('enron_crimpros.csv')
    df2 = df2.set_index('insider')
    df1 = df1.set_index('insider')
    df1 = df1.fillna(0)
    df1 = df1.replace('-',0)
    df1 = df1.astype(float)
    df_f = df1.merge(df2,how='left',left_index=True, right_index=True)
    df_f['POI'] = df_f['POI'].fillna('No')
    df_f['bonus_salary_ratio'] = df_f['bonus']/df_f['salary']
    df_f['bonus_salary_ratio'] = df_f['bonus_salary_ratio'].fillna(0)
    return df_f

df = enron_finclean()

path = os.getcwd() + '/enroninsider.csv'
df.to_csv (path, index = True, header=True)
