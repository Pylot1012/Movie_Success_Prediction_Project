# 🎬 Movie Success Prediction Project

## 📊 Project Overview
A comprehensive data science project predicting movie success categories (Hit/Average/Flop) with **76.81% accuracy**. Developed for Boston Institute of Analytics' Data Science & Artificial Intelligence course.

**Student:** Harmain Aziz  
**Institute:** Boston Institute of Analytics  
**Submission Date:** December 2025

---

## 🏆 Key Achievements
- **Model Accuracy:** 76.81% (Random Forest Classifier)
- **Top Predictor:** Number of Voted Users (24.3% importance)
- **Business Impact:** Actionable insights for film studio decision-making
- **Complete Pipeline:** End-to-end data science project from EDA to deployment

---

## 📁 Project Structure
Predicting Movie Success Project/
├── 📊 Dashboard/ # Presentation & Visualizations
│ ├── Movie_Success_Presentation.pptx # Canva Presentation (6 Slides)
│ └── Tableau/
│ ├── movie_success_dashboard.twbx # Interactive Tableau Dashboard
│ ├── dashboard_screenshot.png # Dashboard Preview
│ └── tableau_public_url.txt # Live Dashboard URL
│
├── ⚙️ Deployment/ # Production-Ready Code
│ ├── app.py # Streamlit Web Application
│ ├── model.pkl # Trained ML Model
│ ├── scaler.pkl # Feature Scaler
│ ├── powerbi_data_prep.py # Data Preparation Pipeline
│ ├── movie_dashboard_data.csv # Clean Dataset
│ └── requirements.txt # Dependencies
│
├── 📈 Report/ # Documentation
│ └── project_report.docx # Comprehensive 7-Page Report
│
├── 📊 data/ # Data Sources
│ └── movie_metadata.csv # Original Dataset (5,043 movies)
│
├── 🐍 scripts/ # Analysis & Modeling
│ ├── complete_analysis.py # Full EDA & Modeling
│ ├── movie_success_analysis.py # Simplified Analysis
│ ├── test_data.py # Data Validation
│ └── test_simple.py # Environment Test
│
├── 📋 README.md # This File
├── .gitignore # Git Configuration
├── requirements.txt # Project Dependencies
└── minimal_requirements.txt # Streamlit-Specific Dependencies

---

## 🚀 Features & Deliverables

### 1. **Machine Learning Model** (76.81% Accuracy)
- **Algorithm:** Random Forest Classifier
- **Target:** Predict Success Category (Hit/Average/Flop)
- **Accuracy:** 76.81% | Precision: 77.2% | Recall: 76.8%
- **Top Features:** 
  1. Number of Voted Users (24.3%)
  2. Movie Duration (19.1%)
  3. Gross Earnings (16.0%)
  4. Budget (15.9%)
  5. Director Facebook Likes (13.4%)

### 2. **Interactive Tableau Dashboard**
- **4 Visualizations:**
  1. Success Distribution (Pie Chart)
  2. Budget vs Gross Earnings (Scatter Plot)
  3. Top 10 Genres (Bar Chart)
  4. Movies Per Year Trend (Line Chart)
- **Interactive Filters:** Year Slider, Country Dropdown, Genres Checklist
- **Live URL:** [Tableau Public Link]

### 3. **Streamlit Web Application**
- **Real-time Predictions:** Input movie features for instant success prediction
- **Interactive Features:** 
  - Adjustable parameters (budget, duration, social metrics)
  - Probability visualization
  - Feature importance display
- **Deployment:** Ready for Streamlit Cloud deployment

### 4. **Canva Presentation (6 Slides)**
1. **Introduction:** Project overview & dataset
2. **Methodology:** Data preprocessing & model selection
3. **Results:** Model performance & accuracy
4. **Feature Importance:** Key success drivers
5. **Business Insights:** Strategic recommendations
6. **Conclusion & Future Work:** Deployment plan

### 5. **Comprehensive Data Analysis**
- **8+ Visualizations:** IMDB distribution, success categories, correlations, etc.
- **EDA:** Exploratory Data Analysis with pandas & seaborn
- **Data Cleaning:** Missing value imputation, outlier handling, encoding

### 6. **Technical Documentation**
- **Project Report:** 7-page detailed documentation
- **Code Comments:** Well-documented Python scripts
- **Deployment Guide:** Step-by-step setup instructions

---

## 📈 Business Impact & Insights

### 🎬 **Production Strategy**
- **Optimal Duration:** Target 90-150 minute movies (19.1% importance)
- **Budget Allocation:** Use model confidence scores for investment decisions
- **Director Selection:** Prioritize directors with strong social media presence

### 📢 **Marketing Strategy**
- **Audience Engagement:** Focus on generating votes/reviews (#1 predictor)
- **Social Media:** Build pre-release buzz on Facebook/Twitter
- **Targeted Campaigns:** Allocate budget based on predicted success category

### 💰 **Risk Management**
- **Greenlight Decisions:** Assess project viability before approval
- **Portfolio Diversification:** Balance across predicted success categories
- **ROI Optimization:** Data-driven investment decisions

---

## 🔧 Setup Instructions

### Important: Large Files
The following large files are excluded via `.gitignore`:
- `data/movie_metadata.csv` - Download from [Kaggle](https://www.kaggle.com/)
- Trained models (.pkl files) - Will be generated when running scripts
- Generated visualizations (.png files) - Will be created during analysis

To get the dataset:
1. Download `movie_metadata.csv` from Kaggle
2. Place it in the `data/` folder
3. Run `python Deployment/powerbi_data_prep.py` to generate clean data

---

### 🔗 **Links**
- **GitHub Repository:** https://github.com/Pylot1012/Movie_Success_Prediction_Project.git

- **Tableau Public:** https://public.tableau.com/app/profile/harmain.aziz/viz/Movie_Success_Dashboard/MovieSuccessDashboard?publish=yes


- **Streamlit App:** [Your App URL]


**"Transforming film industry decisions with data-driven insights"**
