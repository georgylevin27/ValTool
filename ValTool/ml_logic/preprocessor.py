import pandas as pd
import numpy as np

#DON'T NEED THIS SCRIPT##

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

    
    
    
    
    
    