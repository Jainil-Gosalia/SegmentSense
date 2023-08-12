import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import analyze_clusters

def perform_clustering(selected_df_normalized, n_components_range):
    aic_scores = []
    bic_scores = []

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(selected_df_normalized)
        aic_scores.append(gmm.aic(selected_df_normalized))
        bic_scores.append(gmm.bic(selected_df_normalized))

    optimal_n_components_aic = np.argmin(aic_scores) + 1
    optimal_n_components_bic = np.argmin(bic_scores) + 1

    # Choose the optimal number of components based on your preference
    optimal_n_components = optimal_n_components_aic  # Change to optimal_n_components_bic if preferred

    gmm = GaussianMixture(n_components=optimal_n_components, random_state=0)
    gmm.fit(selected_df_normalized)
    cluster_labels = gmm.predict(selected_df_normalized)

    return cluster_labels

def generate_insights_and_recommendations(df, selected_columns, cluster_labels):
    df["Cluster"] = cluster_labels
    #cluster_features = df.groupby("Cluster")[selected_columns].mean()
    cluster_features = df.groupby("Cluster")[selected_columns].agg("mean")

    insights, recommendations = analyze_clusters(cluster_features)
    return insights, recommendations