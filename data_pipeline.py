import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Read and preprocess the data
    df = pd.read_csv(file_path)
    df.fillna(method="ffill", inplace=True)
    return df

def select_columns(df):
    print("Please select the column number(s) from the list below that you want to process:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    selected_column_indices = input("Enter column number(s) separated by spaces: ")
    selected_column_indices = [int(index) - 1 for index in selected_column_indices.split()]

    selected_columns = [df.columns[index] for index in selected_column_indices]
    return selected_columns

def preprocess_data(df, selected_columns):
    selected_df = df[selected_columns].copy()  # Make a copy to avoid modifying the original dataframe

    categorical_columns = selected_df.select_dtypes(include=['object']).columns
    numerical_columns = selected_df.select_dtypes(include=['number']).columns

    numerical_imputer = SimpleImputer(strategy="mean")
    selected_df[numerical_columns] = numerical_imputer.fit_transform(selected_df[numerical_columns])

    selected_df_encoded = pd.get_dummies(selected_df, columns=categorical_columns)

    scaler = StandardScaler()
    selected_df_normalized = scaler.fit_transform(selected_df_encoded)

    return selected_df_normalized, numerical_columns