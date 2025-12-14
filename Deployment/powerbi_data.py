# create_powerbi_data.py
import pandas as pd
import os

# Load and clean data
df = pd.read_csv('../data/movie_metadata.csv')
df['success'] = pd.cut(df['imdb_score'], bins=[0, 3, 6, 10],
                       labels=['Flop', 'Average', 'Hit'])

# Select key columns for dashboard
dashboard_df = df[[
    'movie_title', 'duration', 'budget', 'gross', 
    'imdb_score', 'success', 'num_voted_users',
    'movie_facebook_likes', 'director_facebook_likes',
    'title_year', 'genres', 'country'
]].copy()

# Save for PowerBI
dashboard_df.to_csv('Deployment/movie_dashboard_data.csv', index=False)
print("✅ Data saved for PowerBI: Deployment/movie_dashboard_data.csv")