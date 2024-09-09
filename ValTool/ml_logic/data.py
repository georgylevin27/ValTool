import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, max_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

def clean_data(df):
    #Change objects into floats and ints
    df['Target_Revenues']=df['Target_Revenues'].str.replace(',','')
    df['Target_Revenues'] = df.Target_Revenues.astype(float)

    df['Target_EBITDA']=df['Target_EBITDA'].str.replace(',','')
    df['Target_EBITDA']=df.Target_EBITDA.astype(float)

    df['Target_EV']=df['Target_EV'].str.replace(',','')
    df['Target_EV']=df.Target_EV.astype(float)

    df['Year']=df['Year'].str.replace(',','')
    df['Year']=df.Year.astype(float)
    df['Year']=df.Year.astype(int)

    #Remove "Government and Agencies" from target industry
    df = df[df['Target_Industry_Macro'] != 'Government and Agencies']

    #Remove nulls
    df = df.dropna()

    #Remove outliers
    df = df[df['EV_Rev'] < 100]
    df = df[df['EV_EBITDA'] > -100]  #was -100
    df = df[df['EV_EBITDA'] < 100]  #was 500
    df = df[df['Target_Revenues'] > df['Target_EBITDA']]

    return df
  
def market_condition(df):
    #Engineer "Market_Condition" feature from date
    # Step 1: Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    # Step 2: Create 'YearMonth' from the 'Date' column for grouping and calculations
    df['YearMonth'] = df['Date'].dt.to_period('M')
    # Step 3: Calculate the monthly median EV/EBITDA for each YearMonth
    monthly_median = df.groupby('YearMonth')['EV_EBITDA'].median().reset_index()
    # Step 4: Rename the median column for clarity
    monthly_median.rename(columns={'EV_EBITDA': 'EV_EBITDA_MonthlyMedian'}, inplace=True)
    # Step 5: Merge the monthly medians back into the original DataFrame
    df = pd.merge(df, monthly_median, on='YearMonth', how='left')
    # Step 6: Calculate quantiles for the monthly medians
    quantiles = df['EV_EBITDA_MonthlyMedian'].quantile([0.1, 0.9])
    q1 = quantiles[0.1]  # First quantile
    q3 = quantiles[0.9]  # Third quantile
    # Step 7: Create 'Market_Condition' column based on quantile cuts
    df['Market_Condition'] = pd.cut(df['EV_EBITDA_MonthlyMedian'],
                                    bins=[-float('inf'), q1, q3, float('inf')],
                                    labels=['LM', 'NM', 'HM'],
                                    right=False
                                    )                                     
    # Step 8: One-hot encode
    df = pd.get_dummies(df, columns=['Market_Condition'], prefix='Market')
    return df

def preprocess(df):
    #Dictionary of countries to regions
    country_to_region = {
        'United States': 'North America',
    'Canada': 'North America',
    'Greenland': 'North America',
    'France': 'Western Europe',
    'Germany': 'Western Europe',
    'Netherlands': 'Western Europe',
    'Belgium': 'Western Europe',
    'Luxembourg': 'Western Europe',
    'Switzerland': 'Western Europe',
    'Austria': 'Western Europe',
    'Liechtenstein': 'Western Europe',
    'Monaco': 'Western Europe',
    'United Kingdom': 'UK',
    'Jersey': 'UK',
    'Guernsey': 'UK',
    'Isle of Man': 'UK',
    'Ireland': 'UK',
    'Sweden': 'Nordics',
    'Norway': 'Nordics',
    'Denmark': 'Nordics',
    'Finland': 'Nordics',
    'Iceland': 'Nordics',
    'Poland': 'Eastern Europe',
    'Bosnia and Herzegovina': 'Eastern Europe',
    'Lithuania': 'Eastern Europe',
    'Bulgaria': 'Eastern Europe',
    'Russia': 'Eastern Europe',
    'Estonia': 'Eastern Europe',
    'Latvia': 'Eastern Europe',
    'Hungary': 'Eastern Europe',
    'Romania': 'Eastern Europe',
    'Ukraine': 'Eastern Europe',
    'Moldova': 'Eastern Europe',
    'Serbia': 'Eastern Europe',
    'Slovenia': 'Eastern Europe',
    'North Macedonia': 'Eastern Europe',
    'Montenegro': 'Eastern Europe',
    'Slovakia': 'Eastern Europe',
    'Czech Republic': 'Eastern Europe',
    'Portugal': 'Southern Europe',
    'Italy': 'Southern Europe',
    'Spain': 'Southern Europe',
    'Greece': 'Southern Europe',
    'Croatia': 'Southern Europe',
    'Cyprus': 'Southern Europe',
    'Malta': 'Southern Europe',
    'Gibraltar': 'Southern Europe',
    'Turkey': 'Southern Europe'
    }
    #Apply mapping to create "Target_Region" column
    df['Target_Region'] = df['Target_Nation'].map(country_to_region)
    
    #One hot encoding of categorical features
    df = pd.get_dummies(df, columns=['Target_Region'])
    df = pd.get_dummies(df, columns=['Target_Industry_Macro'])
    df = pd.get_dummies(df, columns=['Target_Status'])
    df = df.drop(columns=['Target_Status_Private'])
    
    #Log transformations
    df['log_Target_EV'] = df['Target_EV'].apply(lambda x: np.log(x))
    df['log_Target_Revenues'] = df['Target_Revenues'].apply(lambda x: np.log(x))
    df['log_Target_EBITDA'] = df['Target_EBITDA'].apply(lambda x: -np.log(abs(x)) if x < 0 else np.log(x))
    #Remove -inf value
    df = df[df['log_Target_EBITDA'] != -np.inf]
    
    return df


mydf=pd.read_csv('./data/Data.csv')

df=clean_data(mydf)
df=market_condition(df)


preprocess(df)
train_model(df)