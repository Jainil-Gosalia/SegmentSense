from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def analyze_clusters(cluster_features):
    insights = {}
    recommendations = []

    # Example: Compare averages of selected features across clusters
    for column in cluster_features.columns:
        max_avg_cluster = cluster_features[column].idxmax()
        min_avg_cluster = cluster_features[column].idxmin()

        insight = f"Clusters with higher {column} tend to be {max_avg_cluster}."
        insights[column] = insight

        recommendation = f"Consider targeting {max_avg_cluster} clusters with products related to {column}."
        recommendations.append(recommendation)

    return insights, recommendations

def generate_report(insights, recommendations, excluded_variables, variable_prominence):
    report = "Cluster Analysis Report\n\n"
    
    report += "Excluded Variables:\n"
    for variable in excluded_variables:
        report += f"- {variable}\n"
    
    report += "\nInsights:\n"
    for column, insight in insights.items():
        report += f"- {insight}\n"
    
    report += "\nVariable Prominence:\n"
    for column, prominence in variable_prominence.items():
        report += f"- {column}: {prominence:.2f}\n"
    
    report += "\nRecommendations:\n"
    for recommendation in recommendations:
        report += f"- {recommendation}\n"
    
    return report

def generate_cluster_report(cluster_properties, insights, variable_prominence, recommendations, numerical_columns, selected_df):
    report = "Cluster Analysis Report\n\n"
    
    num_clusters = cluster_properties["num_clusters"]
    confidence_level = cluster_properties["confidence_level"]
    report += f"Number of Clusters: {num_clusters}\n"
    report += f"Confidence in Number of Clusters: {confidence_level:.2f}%\n\n"
    
    for cluster_num, insight in insights.items():
        report += f"Cluster {cluster_num}:\n"
        report += f"Features/Insights: {insight}\n"
        
        # Calculate outlier variables for this cluster
        outlier_vars = calculate_outliers(selected_df[selected_df['Cluster'] == cluster_num], numerical_columns)
        
        if outlier_vars:
            report += f"Outlier Variables: {', '.join(outlier_vars)}\n"
        
        report += f"Recommendations:\n- {recommendations[cluster_num]}\n\n"
    
    return report

def calculate_outliers(cluster_df, numerical_columns):
    outlier_threshold = 2  # Adjust the outlier threshold as needed
    outlier_vars = []
    
    for column in cluster_df.columns:
        if column in numerical_columns:
            z_scores = (cluster_df[column] - cluster_df[column].mean()) / cluster_df[column].std()
            outliers = cluster_df[abs(z_scores) > outlier_threshold][column]
            if not outliers.empty:
                outlier_vars.append(column)
    
    return outlier_vars

def create_pdf(data, output_filename):
    c = canvas.Canvas(output_filename, pagesize=letter)

    # Set font and font size
    c.setFont("Helvetica", 12)

    # Position for drawing
    x = 50
    y = 750

    # Split the data into lines
    lines = data.split("\n")

    # Draw the content
    c.setFont("Helvetica", 12)
    for line in lines:
        c.drawString(x, y, line)
        y -= 15  # Move up

    c.save()

