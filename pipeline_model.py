import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

class LaptopDataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer to clean laptop data and engineer features."""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # 1. Drop Unnamed: 0
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            
        # 2. Clean Ram and Weight
        if 'Ram' in df.columns:
            df['Ram'] = df['Ram'].astype(str).str.replace('GB', '').astype('int32')
        if 'Weight' in df.columns:
            df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').astype('float32')
            
        # 3. Screen Features
        if 'ScreenResolution' in df.columns:
            df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
            df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
            new = df['ScreenResolution'].str.split('x', n=1, expand=True)
            df['X_res'] = new[0].str.extract(r'(\d+)$').astype(float)
            df['Y_res'] = new[1].astype(float)
            df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype('float32')
            df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)
            
        # 4. CPU Brand
        if 'Cpu' in df.columns:
            df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
            def fetch_processor(text):
                if text in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
                    return text
                elif text.split()[0] == 'Intel':
                    return 'Other Intel Processor'
                else:
                    return 'AMD Processor'
            df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
            df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)
            
        # 5. Memory
        if 'Memory' in df.columns:
            df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
            df["Memory"] = df["Memory"].str.replace('GB', '').str.replace('TB', '000')
            new = df["Memory"].str.split("+", n=1, expand=True)
            df["first"] = new[0].str.strip()
            df["second"] = new[1].fillna("0")

            for col, name in [("first", "Layer1"), ("second", "Layer2")]:
                df[f"{name}HDD"] = df[col].apply(lambda x: 1 if "HDD" in x else 0)
                df[f"{name}SSD"] = df[col].apply(lambda x: 1 if "SSD" in x else 0)
                df[f"{name}Hybrid"] = df[col].apply(lambda x: 1 if "Hybrid" in x else 0)
                df[f"{name}Flash"] = df[col].apply(lambda x: 1 if "Flash Storage" in x else 0)
                df[col] = df[col].str.extract(r'(\d+)').fillna(0).astype(int)

            df["HDD"] = (df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"])
            df["SSD"] = (df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"])
            df["Hybrid"] = (df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"])
            df["Flash_Storage"] = (df["first"] * df["Layer1Flash"] + df["second"] * df["Layer2Flash"])

            df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                             'Layer1Flash', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                             'Layer2Flash', 'Memory'], inplace=True)
                             
        # 6. GPU Brand
        if 'Gpu' in df.columns:
            df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
            df.drop(columns=['Gpu'], inplace=True)
            
        # 7. OS
        if 'OpSys' in df.columns:
            def cat_os(inp):
                if inp in ['Windows 10', 'Windows 7', 'Windows 10 S']:
                    return 'Windows'
                elif inp in ['macOS', 'Mac OS X']:
                    return 'Mac'
                else:
                    return 'Others/No OS/Linux'
            df['os'] = df['OpSys'].apply(cat_os)
            df.drop(columns=['OpSys'], inplace=True)
            
        return df

if __name__ == "__main__":
    print("Loading raw data...")
    df = pd.read_csv('laptop_data.csv')
    
    # Define Target and Features
    X = df.drop(columns=['Price'])
    y = np.log1p(df['Price']) # Log-transform target for better prediction
    
    # Filter out rare ARM GPUs before splitting
    mask = X['Gpu'].apply(lambda x: x.split()[0]) != 'ARM'
    X = X[mask]
    y = y[mask]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
    
    print("Building scikit-learn Pipeline...")
    
    # 1. Custom cleaner for feature engineering
    cleaner = LaptopDataCleaner()
    
    # 2. Column Transformer for Encoding and Scaling
    categorical_cols = ['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'os']
    numerical_cols = ['Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD', 'Hybrid', 'Flash_Storage']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ],
        remainder='passthrough'
    )
    
    # 3. Select Best Model (Random Forest selected based on previous comparison)
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    
    # Complete Pipeline
    pipeline = Pipeline(steps=[
        ('cleaner', cleaner),
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    print("Training Pipeline Model... (This runs data cleaning -> encoding -> training)")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = pipeline.predict(X_test)
    
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    # Convert predictions back from log scale for MAE
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
    print(f"Mean Absolute Error: Rs {mae:,.0f}")
    
    # Export Pipeline
    joblib.dump(pipeline, 'laptop_pipeline_model.pkl')
    print("Pipeline model saved to 'laptop_pipeline_model.pkl' successfully!")
    print("You can now load this model and pass raw input data directly to it.")
