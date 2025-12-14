# Deployment/powerbi_data_prep.py
import pandas as pd
import os
import numpy as np

# Quick check script

df = pd.read_csv('Deployment/movie_dashboard_data.csv')
print(f"✓ Data loaded: {df.shape} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
print(f"Success categories: {df['success_category'].unique()}")

print("="*60)
print("PREPARING DATA FOR POWERBI DASHBOARD")
print("="*60)

# Load original data
try:
    # Adjust path based on your actual structure
    data_path = '../data/movie_metadata.csv'
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {df.shape}")
except:
    # Try alternative path
    data_path = 'data/movie_metadata.csv'
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded from alt path: {df.shape}")

# Create success categories
df['success_category'] = pd.cut(df['imdb_score'], 
                                bins=[0, 3, 6, 10],
                                labels=['Flop', 'Average', 'Hit'])

# Select and clean key columns for dashboard
dashboard_cols = [
    'movie_title', 'duration', 'budget', 'gross', 
    'imdb_score', 'success_category', 'num_voted_users',
    'movie_facebook_likes', 'director_facebook_likes',
    'title_year', 'genres', 'country', 'language',
    'content_rating', 'aspect_ratio', 'color'
]

# Create dashboard dataframe
dashboard_df = df[dashboard_cols].copy()

# Clean data
dashboard_df['budget'] = dashboard_df['budget'].fillna(0)
dashboard_df['gross'] = dashboard_df['gross'].fillna(0)
dashboard_df['duration'] = dashboard_df['duration'].fillna(dashboard_df['duration'].median())

# Add calculated columns
dashboard_df['profit'] = dashboard_df['gross'] - dashboard_df['budget']
dashboard_df['roi'] = (dashboard_df['profit'] / dashboard_df['budget'] * 100).replace([np.inf, -np.inf], 0)
dashboard_df['budget_millions'] = dashboard_df['budget'] / 1e6
dashboard_df['gross_millions'] = dashboard_df['gross'] / 1e6

# Save for PowerBI
output_path = 'movie_dashboard_data.csv'
dashboard_df.to_csv(output_path, index=False)

print(f"\n✓ Dashboard data prepared!")
print(f"  Rows: {dashboard_df.shape[0]}, Columns: {dashboard_df.shape[1]}")
print(f"  Saved to: {output_path}")
print(f"\nColumns available:")
for col in dashboard_df.columns:
    print(f"  - {col}")

print("\n" + "="*60)
print("READY FOR POWERBI IMPORT!")
print("="*60)
print("\nNext steps:")
print("1. Open PowerBI Desktop")
print("2. Click 'Get Data' → 'Text/CSV'")
print("3. Select 'movie_dashboard_data.csv'")
print("4. Create your interactive dashboard")