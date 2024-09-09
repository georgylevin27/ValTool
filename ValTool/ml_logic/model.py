def train_model(df):
    # Step 1: Prepare the Data
    X = df[['log_Target_Revenues', 'log_Target_EBITDA', 'Market_HM', 'Market_NM', 'Market_LM', \
        'Target_Status_Public', 'Target_Region_Eastern Europe', 'Target_Region_Nordics', \
        'Target_Region_North America', 'Target_Region_Southern Europe', 'Target_Region_UK', \
        'Target_Region_Western Europe', 'Target_Industry_Macro_Consumer Products and Services', \
        'Target_Industry_Macro_Consumer Staples','Target_Industry_Macro_Energy and Power', \
        'Target_Industry_Macro_Financials','Target_Industry_Macro_Healthcare', \
        'Target_Industry_Macro_High Technology','Target_Industry_Macro_Industrials', \
        'Target_Industry_Macro_Materials','Target_Industry_Macro_Media and Entertainment', \
        'Target_Industry_Macro_Real Estate','Target_Industry_Macro_Retail', \
        'Target_Industry_Macro_Telecommunications']]
    y = df['log_Target_EV']
    
    # Step 2: Best parameters
    best_params = {
        'n_estimators': 346,       
        'max_depth': 37,           
        'min_samples_split': 6,  
        'min_samples_leaf': 1,
        'max_features': 0.6192134961637232
    }
    
    # Step 3: initialize model with best parameters
    model = RandomForestRegressor(
        **best_params
    )
    
    # Step 4: Fit the model
    model.fit(X, y)
    print(f"✅ Model fitting complete.")
    return model
    
def save_model(model):
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models")
    model.save(model_path)
    return None

def load_model():
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    #local_model_paths = glob.glob(f"{local_model_directory}/*")
    #latest_model = models.load_model(most_recent_model_path_on_disk)

def predict(model, X):
    y_pred = model.predict(X)
    return y_pred
    
    
    