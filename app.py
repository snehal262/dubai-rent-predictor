# ─────────────────────────────────────────────────────────────
# app.py — Dubai Apartment Rent Predictor
# Run: streamlit run app.py
# ─────────────────────────────────────────────────────────────

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dubai Rent Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d2e;
        border-right: 1px solid #2d2f3e;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2235, #252840);
        border: 1px solid #3a3d55;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-label {
        color: #8b8fa8;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        color: #6c6f84;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, #1a2744, #1e3358);
        border: 1px solid #2e4a80;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(30, 80, 180, 0.2);
    }
    .prediction-label {
        color: #7b9fd4;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .prediction-value {
        color: #4da6ff;
        font-size: 42px;
        font-weight: 800;
        line-height: 1;
    }
    .prediction-monthly {
        color: #8ab4d8;
        font-size: 18px;
        margin-top: 10px;
    }
    .prediction-range {
        color: #6b8ab0;
        font-size: 13px;
        margin-top: 8px;
    }

    /* Section headers */
    .section-header {
        color: #e0e4f0;
        font-size: 20px;
        font-weight: 700;
        margin: 8px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #2d3150;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b8fa8;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d3150 !important;
        color: #4da6ff !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    with open(os.path.join(BASE, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(BASE, "shap_explainer.pkl"), "rb") as f:
        explainer = pickle.load(f)
    with open(os.path.join(BASE, "feature_cols.pkl"), "rb") as f:
        meta = pickle.load(f)
    df = pd.read_csv(os.path.join(BASE, "dubai_clean.csv"))
    return model, encoders, explainer, meta, df

model, encoders, explainer, meta, df = load_assets()
FEATURE_COLS = meta["feature_cols"]
CAT_FEATURES = meta["cat_features"]

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ Dubai Rent Predictor")
    st.markdown("---")
    st.markdown("#### Property Details")

    location = st.selectbox(
        "📍 Location",
        options=sorted(encoders["Location"].classes_),
        index=list(sorted(encoders["Location"].classes_)).index("Dubai Marina")
        if "Dubai Marina" in encoders["Location"].classes_ else 0,
    )

    col1, col2 = st.columns(2)
    with col1:
        beds = st.number_input("🛏️ Beds", min_value=0, max_value=10, value=2, step=1)
    with col2:
        baths = st.number_input("🚿 Baths", min_value=1, max_value=10, value=2, step=1)

    area = st.slider(
        "📐 Area (sqft)",
        min_value=int(df["Area_in_sqft"].min()),
        max_value=int(df["Area_in_sqft"].quantile(0.99)),
        value=1000,
        step=50,
    )

    furnishing = st.selectbox(
        "🛋️ Furnishing",
        options=sorted(encoders["Furnishing"].classes_),
        index=0,
    )

    prop_type = st.selectbox(
        "🏢 Property Type",
        options=sorted(encoders["Type"].classes_),
        index=0,
    )

    frequency = st.selectbox(
        "💳 Payment Frequency",
        options=sorted(encoders["Frequency"].classes_),
        index=0,
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Rent", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────
def make_prediction(location, beds, baths, area, furnishing, prop_type, frequency):
    input_dict = {
        "Area_in_sqft"   : area,
        "Beds"           : beds,
        "Baths"          : baths,
        "Beds_x_Baths"   : beds * baths,
        "Location_enc"   : encoders["Location"].transform([location])[0],
        "Furnishing_enc" : encoders["Furnishing"].transform([furnishing])[0],
        "Type_enc"       : encoders["Type"].transform([prop_type])[0],
        "Frequency_enc"  : encoders["Frequency"].transform([frequency])[0],
    }
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    log_pred = model.predict(input_df)[0]
    pred_aed = np.exp(log_pred)
    # ±1 std of log prediction as confidence range
    low  = np.exp(log_pred - 0.25)
    high = np.exp(log_pred + 0.25)
    return pred_aed, low, high, input_df

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("# 🏙️ Dubai Apartment Rent Predictor")
st.markdown("*Predict annual rent for Dubai apartments using machine learning*")
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Rent Predictor", "🗺️ Map View", "🧠 Model Explainer"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — RENT PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        pred_aed, low, high, input_df = make_prediction(
            location, beds, baths, area, furnishing, prop_type, frequency
        )

        # Prediction card
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimated Annual Rent</div>
            <div class="prediction-value">AED {pred_aed:,.0f}</div>
            <div class="prediction-monthly">≈ AED {pred_aed/12:,.0f} / month</div>
            <div class="prediction-range">Likely range: AED {low:,.0f} — AED {high:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric row
        c1, c2, c3, c4 = st.columns(4)
        similar = df[
            (df["Location"] == location) &
            (df["Beds"] == beds)
        ]["Rent"]

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Your Prediction</div>
                <div class="metric-value">AED {pred_aed:,.0f}</div>
                <div class="metric-sub">Annual</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            if len(similar) > 0:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Area Median</div>
                    <div class="metric-value">AED {similar.median():,.0f}</div>
                    <div class="metric-sub">{len(similar)} similar listings</div>
                </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Per Month</div>
                <div class="metric-value">AED {pred_aed/12:,.0f}</div>
                <div class="metric-sub">Monthly equivalent</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            rent_per_sqft = pred_aed / area
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Per Sqft</div>
                <div class="metric-value">AED {rent_per_sqft:,.1f}</div>
                <div class="metric-sub">Annual rent / sqft</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Distribution chart — where does this prediction sit?
        st.markdown('<div class="section-header">How does this compare to similar listings?</div>',
                    unsafe_allow_html=True)

        compare_df = df[df["Location"] == location][["Rent"]].copy()
        if len(compare_df) > 10:
            fig = px.histogram(
                compare_df, x="Rent", nbins=40,
                color_discrete_sequence=["#2d3a6b"],
                title=f"Rent distribution in {location}",
                labels={"Rent": "Annual Rent (AED)"},
            )
            fig.add_vline(
                x=pred_aed, line_color="#4da6ff", line_width=2.5,
                annotation_text=f"Your prediction: AED {pred_aed:,.0f}",
                annotation_font_color="#4da6ff",
                annotation_position="top right",
            )
            fig.update_layout(
                plot_bgcolor="#1a1d2e",
                paper_bgcolor="#1a1d2e",
                font_color="#c0c4d8",
                title_font_color="#e0e4f0",
                xaxis=dict(gridcolor="#2d3150", tickformat=",.0f"),
                yaxis=dict(gridcolor="#2d3150"),
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #6b6f88;">
            <div style="font-size: 48px;">🏙️</div>
            <div style="font-size: 20px; font-weight: 600; margin-top: 16px; color: #9095b0;">
                Configure your property in the sidebar
            </div>
            <div style="font-size: 14px; margin-top: 8px;">
                Select location, bedrooms, area and click Predict Rent
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — MAP VIEW
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Average Rent by Location</div>',
                unsafe_allow_html=True)

    # Filters
    fc1, fc2 = st.columns([1, 1])
    with fc1:
        map_beds = st.selectbox("Filter by Beds", ["All"] + sorted(df["Beds"].dropna().unique().astype(int).tolist()))
    with fc2:
        min_listings = st.slider("Min listings per area", 5, 100, 20)

    map_df = df.copy()
    if map_beds != "All":
        map_df = map_df[map_df["Beds"] == int(map_beds)]

    # Aggregate by location
    map_agg = (
        map_df.groupby("Location")
        .agg(
            median_rent=("Rent", "median"),
            count=("Rent", "count"),
            lat=("Latitude", "mean"),
            lon=("Longitude", "mean"),
        )
        .query(f"count >= {min_listings}")
        .dropna(subset=["lat", "lon"])
        .reset_index()
    )

    if len(map_agg) > 0:
        fig_map = px.scatter_mapbox(
            map_agg,
            lat="lat", lon="lon",
            color="median_rent",
            size="count",
            hover_name="Location",
            hover_data={
                "median_rent": ":,.0f",
                "count": True,
                "lat": False,
                "lon": False,
            },
            color_continuous_scale="Bluered",
            size_max=30,
            zoom=10,
            mapbox_style="carto-darkmatter",
            title="Median Annual Rent by Location",
            labels={"median_rent": "Median Rent (AED)", "count": "Listings"},
        )
        fig_map.update_layout(
            paper_bgcolor="#0f1117",
            font_color="#c0c4d8",
            title_font_color="#e0e4f0",
            coloraxis_colorbar=dict(
                title="Median Rent",
                tickformat=",.0f",
                tickfont=dict(color="#c0c4d8"),
            ),
            margin=dict(t=50, b=0, l=0, r=0),
            height=550,
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Top locations table
        st.markdown('<div class="section-header">Top 15 Most Expensive Locations</div>',
                    unsafe_allow_html=True)
        top_locs = (
            map_agg.sort_values("median_rent", ascending=False)
            .head(15)[["Location", "median_rent", "count"]]
            .rename(columns={"median_rent": "Median Rent (AED)", "count": "Listings"})
            .reset_index(drop=True)
        )
        top_locs.index += 1
        top_locs["Median Rent (AED)"] = top_locs["Median Rent (AED)"].apply(lambda x: f"AED {x:,.0f}")
        st.dataframe(top_locs, use_container_width=True)
    else:
        st.warning("Not enough data with current filters. Try reducing min listings.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL EXPLAINER
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">How does the model make predictions?</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Global Feature Importance")
        st.markdown("Which features have the biggest impact on rent predictions overall?")

        # Compute SHAP on a sample for speed
        # Encode categoricals on the fly for SHAP sample
        sample = df[meta["numeric_features"] + meta["cat_features"]].dropna().sample(min(500, len(df)), random_state=42)
        for col in meta["cat_features"]:
            sample[col + "_enc"] = encoders[col].transform(sample[col].astype(str))
        sample = sample[FEATURE_COLS]
        shap_vals = explainer.shap_values(sample)
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": mean_abs_shap,
        }).sort_values("Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation="h",
            marker=dict(
                color=importance_df["Importance"],
                colorscale="Blues",
                showscale=False,
            ),
        ))
        fig_imp.update_layout(
            plot_bgcolor="#1a1d2e",
            paper_bgcolor="#1a1d2e",
            font_color="#c0c4d8",
            xaxis=dict(gridcolor="#2d3150", title="Mean |SHAP value|"),
            yaxis=dict(gridcolor="#2d3150"),
            margin=dict(t=20, b=40, l=10, r=10),
            height=350,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_b:
        st.markdown("#### Prediction Explainer")
        st.markdown("Configure a property in the sidebar and click Predict to see why the model chose that price.")

        if predict_btn:
            pred_aed, low, high, input_df = make_prediction(
                location, beds, baths, area, furnishing, prop_type, frequency
            )
            sv = explainer.shap_values(input_df)[0]
            base = explainer.expected_value

            # Waterfall chart using plotly
            features   = FEATURE_COLS
            shap_vals_single = sv
            sorted_idx = np.argsort(np.abs(shap_vals_single))[::-1][:6]

            labels = [features[i] for i in sorted_idx]
            values = [shap_vals_single[i] for i in sorted_idx]
            colors = ["#4da6ff" if v > 0 else "#ff6b6b" for v in values]

            fig_wf = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
            ))
            fig_wf.update_layout(
                title=f"Why AED {pred_aed:,.0f}?",
                title_font_color="#e0e4f0",
                plot_bgcolor="#1a1d2e",
                paper_bgcolor="#1a1d2e",
                font_color="#c0c4d8",
                xaxis=dict(gridcolor="#2d3150", title="SHAP value (impact on log rent)"),
                yaxis=dict(gridcolor="#2d3150"),
                margin=dict(t=50, b=40, l=10, r=10),
                height=350,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            # What each feature contributed in plain English
            st.markdown("**Top factors for this prediction:**")
            for i in sorted_idx[:4]:
                direction = "⬆️ pushed rent up" if sv[i] > 0 else "⬇️ pushed rent down"
                st.markdown(f"- **{features[i]}** = `{input_df[features[i]].values[0]}` → {direction}")
        else:
            st.info("Click **Predict Rent** in the sidebar to see the explanation for your property.")

    # Model performance metrics at the bottom
    st.markdown("---")
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">0.906</div>
            <div class="metric-sub">Explains 90.6% of variance</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">MAPE</div>
            <div class="metric-value">18.2%</div>
            <div class="metric-sub">Avg prediction error</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Training Rows</div>
            <div class="metric-value">58,457</div>
            <div class="metric-sub">Dubai apartment listings</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Algorithm</div>
            <div class="metric-value">XGBoost</div>
            <div class="metric-sub">500 trees, depth 6</div>
        </div>""", unsafe_allow_html=True)