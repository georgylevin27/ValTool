import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, max_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

from ValTool.ml_logic.data import clean_data, classify_market_condition, market_condition, preprocess
from ValTool.ml_logic.model import train_model, predict
from ValTool.ml_logic.api import *

def model():
    df = pd.read_csv("/home/larryxmiller/code/larryxmiller/ValTool/data/Data.csv")
    clean_data(df)
    market_condition(df)
    preprocess(df)
    train_model(df)
 
def get_pred():
    #Get X_new from API
    load_model()
    predict(X_new)
    
if __name__ == '__main__':
    model()
    #get_pred()