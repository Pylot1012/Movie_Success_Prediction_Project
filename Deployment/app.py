# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables (if any)
from dotenv import load_dotenv
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="wide"
)

# Title
st.title("🎬 Movie Success Prediction Dashboard")
st.markdown("Predict whether a movie will be **Hit, Average, or Flop** based on key features")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Predict Success", "Data Analysis", "Model Insights"])

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/movie_metadata.csv')
    df['success'] = pd.cut(df['imdb_score'], 
                          bins=[0, 3, 6, 10],
                          labels=['Flop', 'Average', 'Hit'])
    return df

df = load_data()

if page == "Home":
    st.header("Welcome to Movie Success Predictor")
    st.markdown("""
    ### Project Overview
    This interactive dashboard helps film studios predict movie success categories (Hit/Average/Flop) 
    using machine learning models trained on historical IMDB data.
    
    ### Key Features:
    - **Success Prediction**: Input movie features to get success probability
    - **Data Analysis**: Explore relationships between features and success
    - **Model Insights**: Understand what drives movie success
    - **Visualizations**: Interactive charts and graphs
    
    ### Dataset Summary:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Hit Movies", df['success'].value_counts().get('Hit', 0))
    with col4:
        st.metric("Model Accuracy", "76.81%")
    
    st.image('graph2_success_categories.png', caption='Movie Success Distribution')

elif page == "Predict Success":
    st.header("🎯 Predict Movie Success")
    st.markdown("Enter movie details to predict success category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Duration (minutes)", 60, 240, 120)
        budget = st.number_input("Budget ($ millions)", 1, 500, 50)
        gross = st.number_input("Expected Gross ($ millions)", 1, 1000, 100)
    
    with col2:
        num_voted_users = st.number_input("Expected Voted Users", 1000, 1000000, 50000)
        movie_fb_likes = st.number_input("Movie Facebook Likes", 0, 200000, 5000)
        director_fb_likes = st.number_input("Director Facebook Likes", 0, 10000, 1000)
    
    if st.button("Predict Success", type="primary"):
        # Prepare input
        input_data = pd.DataFrame({
            'duration': [duration],
            'budget': [budget * 1e6],  # Convert to dollars
            'gross': [gross * 1e6],
            'num_voted_users': [num_voted_users],
            'movie_facebook_likes': [movie_fb_likes],
            'director_facebook_likes': [director_fb_likes]
        })
        
        # Train model on the fly (or load pre-trained)
        features = ['duration', 'budget', 'gross', 'num_voted_users', 
                   'movie_facebook_likes', 'director_facebook_likes']
        X = df[features].fillna(df[features].median())
        y = df['success']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            if prediction == 'Hit':
                st.success(f"🎉 **Prediction: {prediction}**")
            elif prediction == 'Average':
                st.warning(f"📊 **Prediction: {prediction}**")
            else:
                st.error(f"⚠️ **Prediction: {prediction}**")
        
        with result_col2:
            st.metric("Confidence", f"{max(probabilities)*100:.1f}%")
        
        # Probability chart
        with result_col3:
            fig, ax = plt.subplots(figsize=(4, 3))
            categories = ['Flop', 'Average', 'Hit']
            ax.bar(categories, probabilities, color=['red', 'orange', 'green'])
            ax.set_ylabel('Probability')
            ax.set_title('Success Probability Distribution')
            st.pyplot(fig)
        
        # Feature importance
        st.subheader("Top Success Factors")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(importance_df, hide_index=True)

elif page == "Data Analysis":
    st.header("📊 Data Analysis")
    
    analysis_type = st.selectbox("Select Analysis", 
                                ["Success Distribution", "Budget vs Gross", 
                                 "Duration Analysis", "Facebook Impact"])
    
    if analysis_type == "Success Distribution":
        st.image('graph2_success_categories.png')
        st.markdown("""
        **Insights:**
        - Majority of movies (68.5%) are Hits (IMDB 6-10)
        - Very few movies (0.9%) are complete Flops
        - Studios should aim for at least Average category
        """)
    
    elif analysis_type == "Budget vs Gross":
        st.image('graph3_budget_vs_gross.png')
        st.markdown("""
        **Insights:**
        - Positive correlation between budget and gross earnings
        - Higher budget doesn't guarantee success, but increases probability
        - Optimal budget range: $50-150 million
        """)
    
    elif analysis_type == "Duration Analysis":
        st.image('graph4_duration_distribution.png')
        st.markdown("""
        **Insights:**
        - Most successful movies: 90-150 minutes
        - Too short (<90 min) may lack substance
        - Too long (>150 min) may lose audience engagement
        """)
    
    else:  # Facebook Impact
        st.image('graph5_facebook_vs_score.png')
        st.markdown("""
        **Insights:**
        - Social media presence correlates with IMDB scores
        - Facebook likes serve as pre-release interest indicator
        - Marketing campaigns should boost social media engagement
        """)

elif page == "Model Insights":
    st.header("🤖 Model Insights")
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Performance Metrics"])
    
    with tab1:
        st.image('graph7_feature_importance.png')
        st.markdown("""
        **Top 3 Success Predictors:**
        1. **Number of Voted Users** (24.3%) - Audience engagement is key
        2. **Movie Duration** (19.1%) - Optimal runtime matters
        3. **Gross Earnings** (16.0%) - Financial success indicator
        
        **Recommendation:** Focus on generating audience participation and optimizing movie length.
        """)
    
    with tab2:
        st.image('graph8_confusion_matrix.png')
        st.markdown("""
        **Model Performance:**
        - **Hit movies**: 85% correct prediction rate
        - **Average movies**: 70% correct prediction rate  
        - **Flop movies**: Limited data, lower accuracy
        
        **Strength:** Excellent at identifying Hits (most important for studios)
        """)
    
    with tab3:
        st.markdown("""
        ### Model Performance Summary
        
        | Metric | Value |
        |--------|-------|
        | Accuracy | 76.81% |
        | Precision | 77.2% |
        | Recall | 76.8% |
        | F1-Score | 76.9% |
        | ROC-AUC | 0.82 |
        
        ### Business Impact:
        - **76.81% accuracy** means 3 out of 4 predictions are correct
        - Can save millions by avoiding potential flops
        - Helps optimize marketing budget allocation
        """)
        
        # Add download button for model
        st.download_button(
            label="Download Sample Dataset",
            data=df.head(100).to_csv(index=False),
            file_name="sample_movie_data.csv",
            mime="text/csv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Movie Success Predictor**  
Boston Institute of Analytics  
Data Science Project  
Developed by: Harmain Aziz
""")