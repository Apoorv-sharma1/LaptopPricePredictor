import numpy as np
import pandas as pd
import re

def process_data():
    # Load data
    df = pd.read_csv('laptop_data.csv')
    
    # 1. Drop initial index column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # 2. Clean Ram and Weight
    df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
    df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')
    
    # 3. Extract Screen Features
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
    
    # Extract resolution
    new = df['ScreenResolution'].str.split('x', n=1, expand=True)
    df['X_res'] = new[0].str.extract(r'(\d+)$').astype(int)
    df['Y_res'] = new[1].astype(int)
    
    # PPI calculation
    df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype('float32')
    
    # Drop intermediate resolution columns and Inches/ScreenResolution
    df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)
    
    # 4. Extract CPU brand
    def fetch_processor(text):
        if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
            return text
        else:
            if text.split()[0] == 'Intel':
                return 'Other Intel Processor'
            else:
                return 'AMD Processor'

    df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
    df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
    df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)
    
    # 5. Process Memory
    df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
    df["Memory"] = df["Memory"].str.replace('GB', '')
    df["Memory"] = df["Memory"].str.replace('TB', '000')
    new = df["Memory"].str.split("+", n=1, expand=True)

    df["first"] = new[0].str.strip()
    df["second"] = new[1]

    df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer1Flash"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

    df['first'] = df['first'].str.extract(r'(\d+)').astype(int)

    df["second"].fillna("0", inplace=True)

    df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer2Flash"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

    df['second'] = df['second'].str.extract(r'(\d+)').astype(int)

    df["HDD"] = (df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"])
    df["SSD"] = (df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"])
    df["Hybrid"] = (df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"])
    df["Flash_Storage"] = (df["first"] * df["Layer1Flash"] + df["second"] * df["Layer2Flash"])

    df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                     'Layer1Flash', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                     'Layer2Flash', 'Memory'], inplace=True)
    
    # 6. Process GPU Brand
    df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
    # Filter out rarely occurring GPUs (like ARM)
    df = df[df['Gpu brand'] != 'ARM']
    df.drop(columns=['Gpu'], inplace=True)
    
    # 7. Group OS
    def cat_os(inp):
        if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
            return 'Windows'
        elif inp == 'macOS' or inp == 'Mac OS X':
            return 'Mac'
        else:
            return 'Others/No OS/Linux'

    df['os'] = df['OpSys'].apply(cat_os)
    df.drop(columns=['OpSys'], inplace=True)
    
    # 8. Log transformation of Price
    df['Price'] = np.log1p(df['Price'])
    
    # Final cleanup
    print("Columns in refined dataset:", df.columns.tolist())
    df.to_csv('refined_laptop_data.csv', index=False)
    print("Refined data saved to refined_laptop_data.csv")

if __name__ == "__main__":
    process_data()
