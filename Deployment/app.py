import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "movie_dashboard_data.csv"

FEATURES = [
    "duration", "budget", "gross", "num_voted_users",
    "movie_facebook_likes", "director_facebook_likes",
]
FEATURE_LABELS = {
    "duration":                "Duration (min)",
    "budget":                  "Budget ($)",
    "gross":                   "Gross ($)",
    "num_voted_users":         "Voted Users",
    "movie_facebook_likes":    "Movie FB Likes",
    "director_facebook_likes": "Director FB Likes",
}
TARGET = "success_category"
COLORS = {"Hit": "#C084FC", "Average": "#818CF8", "Flop": "#38BDF8"}
BG_DARK = "#0D0D0D"

# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            f"Data file not found at `{DATA_PATH}`. "
            "Please ensure `movie_dashboard_data.csv` is committed to the "
            "repository in the same folder as `app.py`."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET])
    return df

# ── Train model ────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df[TARGET]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    cm = confusion_matrix(y_test, model.predict(X_test), labels=["Flop", "Average", "Hit"])
    return model, scaler, acc, report, cm

# ── Load everything ────────────────────────────────────────────────────
df = load_data()
model, scaler, accuracy, report, cm = train_model(df)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Movie Predictor")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🎯 Predict Success", "📊 Data Analysis", "🤖 Model Insights"],
    )
    st.markdown("---")
    st.markdown(
        """
        **Boston Institute of Analytics**  
        Data Science & AI Course  
        *Student: Harmain Aziz*  
        *December 2025*
        """
    )
    st.markdown("---")
    st.markdown("**🔗 Links**")
    st.markdown("[🚀 Live Streamlit App](https://moviesuccesspredictionproject-appyx3a5em6efjwga7ymrvn.streamlit.app/)")
    st.markdown("[📊 Tableau Dashboard](https://public.tableau.com/app/profile/harmain.aziz/viz/Movie_Success_Dashboard/MovieSuccessDashboard)")
    st.markdown("[💻 GitHub Repo](https://github.com/Pylot1012/Movie_Success_Prediction_Project)")

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🎬 Movie Success Prediction Dashboard")
    st.markdown(
        "Predict whether a movie will be a **Hit**, **Average**, or **Flop** "
        "using machine learning trained on 5,043 IMDB movies."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", f"{len(df):,}")
    c2.metric("Features Used", len(FEATURES))
    c3.metric("Model Accuracy", f"{accuracy:.1%}")
    c4.metric("Top Predictor", "Voted Users")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Success Distribution")
        counts = df[TARGET].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.pie(
            counts, names="Category", values="Count",
            color="Category",
            color_discrete_map=COLORS,
            hole=0.45,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8F8F8",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("IMDB Score Distribution")
        fig2 = px.histogram(
            df, x="imdb_score", nbins=40,
            color_discrete_sequence=["#C084FC"],
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8F8F8",
            xaxis_title="IMDB Score",
            yaxis_title="Number of Movies",
            showlegend=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Classification Rules")
    r1, r2, r3 = st.columns(3)
    r1.error("**Flop** — IMDB Score 1 – 3")
    r2.warning("**Average** — IMDB Score 3 – 6")
    r3.success("**Hit** — IMDB Score 6 – 10")

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Success":
    st.title("🎯 Predict Movie Success")
    st.markdown("Enter the movie's details below and click **Predict** to get an instant result.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            duration   = st.slider("Duration (minutes)", 60, 240, 120)
            budget_m   = st.number_input("Budget ($ millions)", 1.0, 500.0, 50.0, step=5.0)
        with c2:
            gross_m    = st.number_input("Gross Earnings ($ millions)", 0.0, 2000.0, 100.0, step=10.0)
            voted      = st.number_input("Expected Voted Users", 1_000, 2_000_000, 50_000, step=5_000)
        with c3:
            movie_fb   = st.number_input("Movie Facebook Likes", 0, 350_000, 5_000, step=1_000)
            dir_fb     = st.number_input("Director Facebook Likes", 0, 25_000, 1_000, step=500)

        submitted = st.form_submit_button("🎬 Predict Success", use_container_width=True)

    if submitted:
        input_raw = np.array([[
            duration,
            budget_m * 1_000_000,
            gross_m  * 1_000_000,
            voted,
            movie_fb,
            dir_fb,
        ]])
        input_scaled = scaler.transform(input_raw)
        prediction   = model.predict(input_scaled)[0]
        probs        = model.predict_proba(input_scaled)[0]
        classes      = model.classes_

        st.markdown("---")
        res_col, prob_col = st.columns([1, 2])

        with res_col:
            if prediction == "Hit":
                st.success(f"## 🎉 {prediction}")
                st.metric("Confidence", f"{max(probs):.1%}")
            elif prediction == "Average":
                st.warning(f"## 📊 {prediction}")
                st.metric("Confidence", f"{max(probs):.1%}")
            else:
                st.error(f"## ⚠️ {prediction}")
                st.metric("Confidence", f"{max(probs):.1%}")

        with prob_col:
            prob_df = pd.DataFrame({
                "Category":    list(classes),
                "Probability": list(probs),
            })
            fig = px.bar(
                prob_df, x="Category", y="Probability",
                color="Category",
                color_discrete_map=COLORS,
                text=prob_df["Probability"].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8F8F8",
                showlegend=False,
                yaxis_range=[0, 1],
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature Contribution")
        imp_df = pd.DataFrame({
            "Feature":    [FEATURE_LABELS[f] for f in FEATURES],
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig2 = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#38BDF8", "#C084FC", "#FBBF24"],
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8F8F8",
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Budget vs Gross", "Duration", "Facebook Impact", "Correlation",
    ])

    with tab1:
        st.subheader("Budget vs Gross Earnings")
        plot_df = df[df["budget_millions"] < 500].copy()
        fig = px.scatter(
            plot_df, x="budget_millions", y="gross_millions",
            color=TARGET, color_discrete_map=COLORS,
            opacity=0.6,
            labels={"budget_millions": "Budget ($ Millions)", "gross_millions": "Gross ($ Millions)"},
            hover_data=["movie_title"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8F8F8",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Higher budgets generally correlate with higher gross earnings, but hits can emerge across all budget levels.")

    with tab2:
        st.subheader("Movie Duration Distribution")
        fig = px.histogram(
            df[df["duration"] < 300], x="duration",
            color=TARGET, barmode="overlay",
            color_discrete_map=COLORS,
            nbins=50, opacity=0.75,
            labels={"duration": "Duration (minutes)"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8F8F8",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Most successful movies fall in the 90–150 minute range.")

    with tab3:
        st.subheader("Facebook Likes vs IMDB Score")
        fig = px.scatter(
            df[df["movie_facebook_likes"] < 200_000],
            x="movie_facebook_likes", y="imdb_score",
            color=TARGET, color_discrete_map=COLORS,
            opacity=0.5,
            labels={
                "movie_facebook_likes": "Movie Facebook Likes",
                "imdb_score": "IMDB Score",
            },
            hover_data=["movie_title"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8F8F8",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Social media presence shows a moderate positive correlation with audience ratings.")

    with tab4:
        st.subheader("Correlation Matrix — Key Features")
        num_cols = FEATURES + ["imdb_score"]
        corr = df[num_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            labels=dict(color="Correlation"),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F8F8F8",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Insights":
    st.title("🤖 Model Insights")

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Performance Metrics"])

    with tab1:
        st.subheader("Feature Importance — Random Forest")
        imp_df = pd.DataFrame({
            "Feature":    [FEATURE_LABELS[f] for f in FEATURES],
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#38BDF8", "#818CF8", "#C084FC", "#FBBF24"],
            text=imp_df["Importance"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8F8F8",
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success("**#1 Predictor:** Number of Voted Users (audience engagement) — more predictive than budget alone.")

    with tab2:
        st.subheader("Confusion Matrix — Random Forest")
        labels = ["Flop", "Average", "Hit"]
        fig = px.imshow(
            cm,
            x=labels, y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F8F8F8",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "The model performs strongest on **Hit** movies — the most important category for studio decisions. "
            "The Flop class is underrepresented (only 46 of 5,043 movies), which limits its recall."
        )

    with tab3:
        st.subheader("Performance Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{accuracy:.1%}")
        c2.metric("Precision", f"{report['weighted avg']['precision']:.1%}")
        c3.metric("Recall",    f"{report['weighted avg']['recall']:.1%}")
        c4.metric("F1-Score",  f"{report['weighted avg']['f1-score']:.1%}")

        st.markdown("---")
        st.subheader("Per-Class Breakdown")
        class_rows = []
        for cls in ["Flop", "Average", "Hit"]:
            if cls in report:
                r = report[cls]
                class_rows.append({
                    "Class":     cls,
                    "Precision": f"{r['precision']:.1%}",
                    "Recall":    f"{r['recall']:.1%}",
                    "F1-Score":  f"{r['f1-score']:.1%}",
                    "Support":   int(r["support"]),
                })
        st.dataframe(pd.DataFrame(class_rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Business Impact")
        st.markdown("""
        | Decision | Insight |
        |---|---|
        | **Greenlight** | Use predicted Hit probability as a go/no-go signal |
        | **Marketing budget** | Allocate more to movies with high Hit confidence |
        | **Risk management** | Flag predicted Flops early in pre-production |
        | **Portfolio balance** | Diversify across predicted success categories |
        """)
