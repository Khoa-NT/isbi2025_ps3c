"""Script for creating feature importance masking table from multiple models

This script processes feature importance results from different models (Gradient Boosting,
Random Forest, and Logistic Regression) and creates a unified masking table. The table
indicates which features are considered important by each model based on their respective
thresholds.
"""

import pandas as pd

###------ Input Paths and Model Names -------###
# CSV file paths for feature importance results
csv_files = [
    '~/PS3C/importance_based_feature_selection/sorted_featrue_importances/gradient_boosting_feature_importance_sorted.csv',
    '~/PS3C/importance_based_feature_selection/sorted_featrue_importances/random_forest_feature_importance_sorted.csv',
    '~/PS3C/importance_based_feature_selection/sorted_featrue_importances/logistic_regression_feature_importance_sorted.csv'
]

# Model names corresponding to each CSV file
model_names = ["Gradient Boosting", "Random Forest", "Logistic Regression"]

###------ Initialize Tables -------###
# List to store processed feature importance tables for each model
rank_tables = []

###------ Process Each Model's Results -------###
for model_name, file in zip(model_names, csv_files):
    # Read CSV file
    df = pd.read_csv(file)
    
    # Extract feature numbers and handle different column formats
    if "Feature" in df.columns:
        # Extract numeric feature ID from feature name (e.g., 'Feature 10' -> '10')
        df['Feature'] = df['Feature'].str.extract(r'(\d+)')
        df = df.dropna(subset=['Feature'])  # Remove rows where feature number couldn't be extracted
        df['Feature'] = df['Feature'].astype(int)
    elif "Mean" in df.columns:
        # For files without feature numbers, create sequential numbering
        df['Feature'] = range(1, len(df) + 1)
    
    # Apply importance thresholds specific to each model
    if model_name == "Random Forest":
        df[model_name] = (df['Importance'] > 3 * 1e-6).astype(int)
    elif model_name == "Logistic Regression":
        df[model_name] = (df['Mean'] > 1e-16).astype(int)
    else:  # Gradient Boosting
        df[model_name] = (df['Importance'] > 0).astype(int)
    
    # Keep only Feature ID and binary importance indicator
    rank_tables.append(df[['Feature', model_name]])

###------ Create Unified Feature Table -------###
# Get complete set of unique features across all models
all_features = set()
for table in rank_tables:
    all_features.update(table['Feature'])
all_features = sorted(all_features)

# Initialize final table with model names and feature columns
final_table = pd.DataFrame(columns=["Model"] + [f'Feature {i}' for i in all_features])
final_table["Model"] = model_names

# Merge importance indicators from all models
for table, model_name in zip(rank_tables, model_names):
    for _, row in table.iterrows():
        feature = f"Feature {row['Feature']}"
        final_table.loc[final_table["Model"] == model_name, feature] = row[model_name]

###------ Handle Missing Values and Save -------###
# Fill missing values with 0 and convert to integer type
for column in final_table.columns:
    if column != "Model":
        final_table[column] = final_table[column].fillna(0).astype(int)

# Save final masking table
output_file = '~/PS3C/importance_based_feature_selection/masking.csv'
final_table.to_csv(output_file, index=False)

print(f"Final masking table has been saved to: {output_file}")
