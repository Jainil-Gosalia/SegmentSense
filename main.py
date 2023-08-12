import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import argparse

from utils import generate_report, create_pdf
from data_pipeline import load_and_preprocess_data, select_columns, preprocess_data
from gmm import perform_clustering, generate_insights_and_recommendations

def process_pipeline(input_data_path, output_filename = "cluster_analysis_report.pdf"):
    file_path = "data/Customers.csv"
    file_path = input_data_path
    df = load_and_preprocess_data(file_path)
    
    selected_columns = select_columns(df)
    selected_df_normalized, numerical_columns = preprocess_data(df, selected_columns)
    
    n_components_range = range(1, 11)
    cluster_labels = perform_clustering(selected_df_normalized, n_components_range)
    
    insights, recommendations = generate_insights_and_recommendations(df, selected_columns, cluster_labels)
    
    report = generate_report(insights, recommendations, [], {})  # Provide empty lists/dicts for missing arguments    
    create_pdf(report, output_filename)

def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding argument
    parser.add_argument("-i", "--input", help = "Link to csv file containing the data")
    parser.add_argument("-o", "--output", help = "Link for the cluster analysis report")
    
    # Read arguments from command line
    args = parser.parse_args()
 
    #Report Generation
    process_pipeline(args.input, args.output)

if __name__ == "__main__":
    main()