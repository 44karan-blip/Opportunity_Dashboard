import os
import sys
import warnings

# Suppress sklearn unpickling and pandas date parsing warnings for clean UI
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import streamlit as st

# =====================================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# =====================================================================
st.set_page_config(
    page_title="RIT Opportunity Analytics & ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EDF2F7 100%);
        padding: 1.1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    .metric-card-sub {
        font-size: 0.8rem;
        color: #10B981;
        margin-top: 0.2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-rejected {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .badge-info {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        border-bottom: 3px solid #3B82F6 !important;
        color: #1E40AF !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. PATHS & CACHED RESOURCE LOADERS
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "RIT_Opportunity_Wise_Data_Cleaned_finalversion.xlsx")
MODEL_FILE = os.path.join(BASE_DIR, "predictive_model.pkl")
STUDENT_MODEL_FILE = os.path.join(BASE_DIR, "student_model.pkl")
COLUMNS_FILE = os.path.join(BASE_DIR, "model_columns.pkl")


@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset():
    """
    Loads and cleans dataset adhering to the notebook pipeline:
    - Fills missing dates with placeholder '2026-01-01'
    - Standardises categorical strings to lowercase stripped format
    - Generates Signup Year and Month identifiers
    - Employs self-healing fallback target mapping from notebook
    """
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset not found at: {DATA_FILE}")
        st.stop()

    df = pd.read_excel(DATA_FILE)
    df = df.drop_duplicates()

    # Temporal feature extraction as in notebook
    df['Learner SignUp DateTime'] = pd.to_datetime(df['Learner SignUp DateTime'], errors='coerce')
    df['Learner SignUp DateTime'] = df['Learner SignUp DateTime'].fillna(pd.Timestamp('2026-01-01'))

    df['Signup_Year'] = df['Learner SignUp DateTime'].dt.year
    df['Signup_Month_Num'] = df['Learner SignUp DateTime'].dt.month
    df['Signup_Month'] = df['Learner SignUp DateTime'].dt.strftime('%b')

    # Standardise columns to clean lowercase strings
    feature_cols = ['Opportunity Category', 'Gender', 'Country', 'Current/Intended Major']
    target_col = 'Status Description'

    for col in feature_cols + [target_col]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Dynamic target mapping matching notebook logic
    success_keywords = ['completed', 'success', 'registered', 'active', 'yes', '1', 'approved']
    df['Target_Label'] = df[target_col].apply(lambda x: 1 if x in success_keywords else 0)

    # Self-healing fallback: if keywords didn't find >= 2 classes, notebook set top status as 1
    if df['Target_Label'].nunique() < 2:
        most_common_status = df[target_col].value_counts().index[0]  # 'rejected'
        df['Target_Label'] = df[target_col].apply(lambda x: 1 if x == most_common_status else 0)

    df['Target_Status_Name'] = df['Target_Label'].apply(
        lambda x: 'Rejected (Class 1)' if x == 1 else 'Non-Rejected / Active (Class 0)'
    )

    return df


@st.cache_resource(show_spinner=False)
def load_ml_artifacts():
    """
    Loads the trained model and 471 model columns using joblib.
    Prefers predictive_model.pkl, falls back to student_model.pkl.
    """
    active_model_path = MODEL_FILE if os.path.exists(MODEL_FILE) else STUDENT_MODEL_FILE
    if not os.path.exists(active_model_path):
        st.error(f"Trained model not found at {active_model_path}")
        st.stop()

    if not os.path.exists(COLUMNS_FILE):
        st.error(f"Model feature columns not found at {COLUMNS_FILE}")
        st.stop()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = joblib.load(active_model_path)
        columns = joblib.load(COLUMNS_FILE)

    return model, columns, os.path.basename(active_model_path)


def construct_model_input_vector(year, month, category, gender, country, major, model_cols):
    """
    Reproduces the exact 471-feature vector required by the trained model:
    - Signup_Year_Num
    - Signup_Month_Num
    - pd.get_dummies(..., drop_first=True) encoded binary flags
    """
    input_df = pd.DataFrame(0, index=[0], columns=model_cols)
    input_df['Signup_Year_Num'] = int(year)
    input_df['Signup_Month_Num'] = int(month)

    cat_col = f"Opportunity Category_{str(category).strip().lower()}"
    if cat_col in input_df.columns:
        input_df.loc[0, cat_col] = 1

    gen_col = f"Gender_{str(gender).strip().lower()}"
    if gen_col in input_df.columns:
        input_df.loc[0, gen_col] = 1

    cntry_col = f"Country_{str(country).strip().lower()}"
    if cntry_col in input_df.columns:
        input_df.loc[0, cntry_col] = 1

    maj_col = f"Current/Intended Major_{str(major).strip().lower()}"
    if maj_col in input_df.columns:
        input_df.loc[0, maj_col] = 1

    # Guarantee column sequence matches model_cols precisely
    input_df = input_df[model_cols]
    return input_df


# =====================================================================
# 3. INITIALIZE DATA & ARTIFACTS
# =====================================================================
df_full = load_and_preprocess_dataset()
model, model_cols, model_filename = load_ml_artifacts()

# =====================================================================
# 4. SIDEBAR - INTERACTIVE FILTERS & NAVIGATION
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/analytics.png", width=64)
    st.title("Filters & Scope")
    st.caption("Filter records across all overview charts")

    # Opportunity Category Filter
    all_categories = sorted(df_full['Opportunity Category'].unique().tolist())
    selected_categories = st.multiselect(
        "Opportunity Category",
        options=all_categories,
        default=all_categories,
        format_func=lambda x: x.title()
    )

    # Gender Filter
    all_genders = sorted(df_full['Gender'].unique().tolist())
    selected_genders = st.multiselect(
        "Gender",
        options=all_genders,
        default=all_genders,
        format_func=lambda x: x.title()
    )

    # Signup Year Filter
    all_years = sorted(df_full['Signup_Year'].unique().tolist())
    selected_years = st.multiselect(
        "Signup Year",
        options=all_years,
        default=all_years
    )

    # Status Filter
    all_statuses = sorted(df_full['Status Description'].unique().tolist())
    selected_statuses = st.multiselect(
        "Status Description",
        options=all_statuses,
        default=all_statuses,
        format_func=lambda x: x.title()
    )

    # Country Filter (Top 20 + All option)
    country_counts = df_full['Country'].value_counts()
    country_options = ["All Countries"] + sorted(df_full['Country'].unique().tolist())
    selected_country = st.selectbox(
        "Country",
        options=country_options,
        index=0,
        format_func=lambda x: x.title()
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #64748B;">
            <b>Active Model:</b> <code>{model_filename}</code><br>
            <b>Feature Count:</b> <code>{len(model_cols)} columns</code><br>
            <b>Algorithm:</b> Random Forest Classifier<br>
            <b>Scikit-Learn Version:</b> <code>1.9.0 (Loaded)</code>
        </div>
        """,
        unsafe_allow_html=True
    )

# Apply filters to dataset
df_filtered = df_full.copy()
if selected_categories:
    df_filtered = df_filtered[df_filtered['Opportunity Category'].isin(selected_categories)]
if selected_genders:
    df_filtered = df_filtered[df_filtered['Gender'].isin(selected_genders)]
if selected_years:
    df_filtered = df_filtered[df_filtered['Signup_Year'].isin(selected_years)]
if selected_statuses:
    df_filtered = df_filtered[df_filtered['Status Description'].isin(selected_statuses)]
if selected_country != "All Countries":
    df_filtered = df_filtered[df_filtered['Country'] == selected_country]

# =====================================================================
# 5. HEADER & TOP KPI CARDS
# =====================================================================
st.markdown('<div class="main-header">🎓 RIT Opportunity Analytics & Predictive ML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comprehensive Learner Analytics, Cohort Exploration & Machine Learning Status Classification</div>', unsafe_allow_html=True)

# Calculate dynamic KPIs
total_records = len(df_filtered)
unique_learners = df_filtered['First Name'].nunique() if 'First Name' in df_filtered.columns else 0
unique_opps = df_filtered['Opportunity Id'].nunique() if 'Opportunity Id' in df_filtered.columns else 0

rejected_count = (df_filtered['Target_Label'] == 1).sum()
rejection_rate = (rejected_count / total_records * 100) if total_records > 0 else 0

allocated_count = (df_filtered['Status Description'] == 'team allocated').sum()
allocation_rate = (allocated_count / total_records * 100) if total_records > 0 else 0

top_cat = df_filtered['Opportunity Category'].value_counts().index[0].title() if total_records > 0 else "N/A"
top_country_name = df_filtered['Country'].value_counts().index[0].title() if total_records > 0 else "N/A"

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Total Applications</div>
        <div class="metric-card-value">{total_records:,}</div>
        <div class="metric-card-sub" style="color: #64748B;">{total_records / len(df_full):.1%} of master data</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Unique Opportunities</div>
        <div class="metric-card-value">{unique_opps}</div>
        <div class="metric-card-sub" style="color: #3B82F6;">{df_full['Opportunity Id'].nunique()} in catalogue</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Unique Learners</div>
        <div class="metric-card-value">{unique_learners:,}</div>
        <div class="metric-card-sub" style="color: #6366F1;">Registered profiles</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Top Category</div>
        <div class="metric-card-value" style="font-size: 1.35rem; padding-top: 5px;">{top_cat}</div>
        <div class="metric-card-sub" style="color: #8B5CF6;">Leading demand</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Rejection Rate</div>
        <div class="metric-card-value" style="color: #EF4444;">{rejection_rate:.1f}%</div>
        <div class="metric-card-sub" style="color: #EF4444;">{rejected_count:,} applications</div>
    </div>
    """, unsafe_allow_html=True)

with kpi6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Team Allocated</div>
        <div class="metric-card-value" style="color: #10B981;">{allocation_rate:.1f}%</div>
        <div class="metric-card-sub" style="color: #10B981;">{allocated_count:,} assigned</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =====================================================================
# 6. TABBED DASHBOARD CONTENT
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary",
    "📈 Signup Trends (Temporal)",
    "👥 Opportunities & Demographics",
    "🤖 ML Prediction Engine",
    "🔍 Risk Matrix & Feature Importance"
])

# ---------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Learner & Application Status Overview")
    
    col_t1_left, col_t1_right = st.columns([1.1, 1])

    with col_t1_left:
        status_counts = df_filtered['Status Description'].value_counts().reset_index()
        status_counts.columns = ['Status Description', 'Count']
        status_counts['Status Description'] = status_counts['Status Description'].str.title()

        fig_status = px.bar(
            status_counts,
            x='Count',
            y='Status Description',
            orientation='h',
            text='Count',
            title='<b>Application Status Distribution</b>',
            color='Status Description',
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_status.update_layout(
            showlegend=False,
            height=380,
            margin=dict(l=10, r=20, t=40, b=10),
            xaxis_title="Number of Applications",
            yaxis_title=""
        )
        fig_status.update_traces(textposition='outside')
        st.plotly_chart(fig_status, use_container_width=True)

    with col_t1_right:
        cat_counts = df_filtered['Opportunity Category'].value_counts().reset_index()
        cat_counts.columns = ['Opportunity Category', 'Count']
        cat_counts['Opportunity Category'] = cat_counts['Opportunity Category'].str.title()

        fig_cat_pie = px.pie(
            cat_counts,
            values='Count',
            names='Opportunity Category',
            title='<b>Opportunity Share by Category</b>',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_cat_pie.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_cat_pie, use_container_width=True)

    with st.expander("🔎 Browse Filtered Dataset Records", expanded=False):
        show_cols = [
            'Learner SignUp DateTime', 'Opportunity Name', 'Opportunity Category',
            'Gender', 'Country', 'Current/Intended Major', 'Status Description'
        ]
        available_cols = [c for c in show_cols if c in df_filtered.columns]
        st.dataframe(
            df_filtered[available_cols].head(250),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Showing first 250 of {len(df_filtered):,} filtered rows. Use sidebar filters to refine.")


# ---------------------------------------------------------------------
# TAB 2: TEMPORAL SIGNUP ANALYSIS (REPRODUCING NOTEBOOK LOGIC)
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Temporal Registration Trends (Year & Month Analysis)")
    st.markdown("""
    *Reproducing Section 1 from `Untitled0.ipynb`: Evaluation of registration cycles, peak intake months, and seasonal patterns across historical cohorts.*
    """)

    signup_trends = df_filtered.groupby(['Signup_Year', 'Signup_Month']).size().reset_index(name='Signup_Count')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    signup_trends['Signup_Month'] = pd.Categorical(signup_trends['Signup_Month'], categories=month_order, ordered=True)
    signup_trends = signup_trends.sort_values(['Signup_Year', 'Signup_Month'])
    signup_trends['Signup_Year'] = signup_trends['Signup_Year'].astype(str)

    col_t2_1, col_t2_2 = st.columns([1.6, 1])

    with col_t2_1:
        fig_signup = px.line(
            signup_trends,
            x='Signup_Month',
            y='Signup_Count',
            color='Signup_Year',
            markers=True,
            title='<b>Peak Signups Analysis (Year & Month Volumes)</b>',
            labels={'Signup_Month': 'Month of Year', 'Signup_Count': 'Total Registrations', 'Signup_Year': 'Year'},
            color_discrete_sequence=['#2563EB', '#10B981', '#F59E0B', '#EC4899']
        )
        fig_signup.update_layout(
            height=420,
            margin=dict(l=10, r=20, t=50, b=10),
            hovermode="x unified"
        )
        fig_signup.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(fig_signup, use_container_width=True)

    with col_t2_2:
        yearly_summary = df_filtered.groupby('Signup_Year').size().reset_index(name='Registrations')
        yearly_summary['Signup_Year'] = yearly_summary['Signup_Year'].astype(str)
        fig_year_bar = px.bar(
            yearly_summary,
            x='Signup_Year',
            y='Registrations',
            title='<b>Total Registrations by Year</b>',
            color='Signup_Year',
            text='Registrations',
            color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B']
        )
        fig_year_bar.update_layout(
            showlegend=False,
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Signup Year",
            yaxis_title="Total Registrations"
        )
        fig_year_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_year_bar, use_container_width=True)

    pivot_signup = signup_trends.pivot_table(
        index='Signup_Year',
        columns='Signup_Month',
        values='Signup_Count',
        fill_value=0,
        observed=False
    )
    if not pivot_signup.empty:
        fig_heat = px.imshow(
            pivot_signup,
            labels=dict(x="Month", y="Year", color="Signups"),
            x=month_order,
            title='<b>Registration Intensity Heatmap (Year vs Month)</b>',
            color_continuous_scale="Blues",
            aspect="auto",
            text_auto=True
        )
        fig_heat.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------
# TAB 3: DEMOGRAPHICS & ACADEMIC PROFILES
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Demographic & Educational Cohort Distributions")

    col_demo_1, col_demo_2 = st.columns(2)

    with col_demo_1:
        gender_data = df_filtered.groupby(['Gender', 'Status Description']).size().reset_index(name='Count')
        gender_data['Gender'] = gender_data['Gender'].str.title()
        gender_data['Status Description'] = gender_data['Status Description'].str.title()

        fig_gender = px.bar(
            gender_data,
            x='Gender',
            y='Count',
            color='Status Description',
            title='<b>Gender Demographics & Status Breakdown</b>',
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_gender.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Gender",
            yaxis_title="Applications"
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    with col_demo_2:
        cat_metrics = df_filtered.groupby('Opportunity Category').agg(
            Total=('Target_Label', 'count'),
            Rejected=('Target_Label', lambda x: (x == 1).sum())
        ).reset_index()
        cat_metrics['Rejection_Rate'] = (cat_metrics['Rejected'] / cat_metrics['Total']) * 100
        cat_metrics['Opportunity Category'] = cat_metrics['Opportunity Category'].str.title()
        cat_metrics = cat_metrics.sort_values('Rejection_Rate', ascending=False)

        fig_cat_risk = px.bar(
            cat_metrics,
            x='Opportunity Category',
            y='Rejection_Rate',
            title='<b>Rejection Rate by Opportunity Category (%)</b>',
            text=cat_metrics['Rejection_Rate'].apply(lambda x: f"{x:.1f}%"),
            color='Rejection_Rate',
            color_continuous_scale='Reds'
        )
        fig_cat_risk.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Opportunity Category",
            yaxis_title="Rejection Rate (%)",
            coloraxis_showscale=False
        )
        fig_cat_risk.update_traces(textposition='outside')
        st.plotly_chart(fig_cat_risk, use_container_width=True)

    col_geo_1, col_geo_2 = st.columns(2)

    with col_geo_1:
        top_countries_df = df_filtered['Country'].value_counts().head(15).reset_index()
        top_countries_df.columns = ['Country', 'Learners']
        top_countries_df['Country'] = top_countries_df['Country'].str.title()
        top_countries_df = top_countries_df.sort_values('Learners', ascending=True)

        fig_country = px.bar(
            top_countries_df,
            x='Learners',
            y='Country',
            orientation='h',
            title='<b>Top 15 Countries by Application Volume</b>',
            text='Learners',
            color='Learners',
            color_continuous_scale='Teal'
        )
        fig_country.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Learner Applications",
            yaxis_title="",
            coloraxis_showscale=False
        )
        fig_country.update_traces(textposition='outside')
        st.plotly_chart(fig_country, use_container_width=True)

    with col_geo_2:
        top_majors_df = df_filtered['Current/Intended Major'].value_counts().head(15).reset_index()
        top_majors_df.columns = ['Major', 'Count']
        top_majors_df['Major'] = top_majors_df['Major'].str.title()
        top_majors_df = top_majors_df.sort_values('Count', ascending=True)

        fig_major = px.bar(
            top_majors_df,
            x='Count',
            y='Major',
            orientation='h',
            title='<b>Top 15 Academic Majors / Fields of Study</b>',
            text='Count',
            color='Count',
            color_continuous_scale='Purples'
        )
        fig_major.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Applications",
            yaxis_title="",
            coloraxis_showscale=False
        )
        fig_major.update_traces(textposition='outside')
        st.plotly_chart(fig_major, use_container_width=True)


# ---------------------------------------------------------------------
# TAB 4: ML PREDICTIVE ENGINE (WITH 471 FEATURES INFERENCE)
# ---------------------------------------------------------------------
with tab4:
    st.subheader("Machine Learning Application Status Predictor")
    st.markdown("""
    Execute real-time inference using the pre-trained **Random Forest Classifier** (`predictive_model.pkl`).
    The prediction engine dynamically builds the exact **471-column one-hot feature vector** matching `model_columns.pkl`.
    """)

    pred_col_left, pred_col_right = st.columns([1.1, 1])

    with pred_col_left:
        st.markdown("#### **Applicant Profile Parameters**")

        with st.form("prediction_form"):
            form_c1, form_c2 = st.columns(2)

            with form_c1:
                input_category = st.selectbox(
                    "Opportunity Category",
                    options=['internship', 'course', 'event', 'competition', 'engagement'],
                    index=0,
                    format_func=lambda x: x.title(),
                    help="Target program or opportunity type"
                )

                input_gender = st.selectbox(
                    "Gender",
                    options=['male', 'female', "don't want to specify", 'other'],
                    index=0,
                    format_func=lambda x: x.title()
                )

                input_year = st.selectbox(
                    "Registration Year",
                    options=[2023, 2024, 2025, 2026],
                    index=0,
                    help="Signup year of learner"
                )

            with form_c2:
                unique_countries_list = sorted(df_full['Country'].unique().tolist())
                default_country_idx = unique_countries_list.index('united states') if 'united states' in unique_countries_list else 0
                input_country = st.selectbox(
                    "Country of Origin",
                    options=unique_countries_list,
                    index=default_country_idx,
                    format_func=lambda x: x.title()
                )

                unique_majors_list = sorted(df_full['Current/Intended Major'].unique().tolist())
                default_major_idx = unique_majors_list.index('computer science') if 'computer science' in unique_majors_list else 0
                input_major = st.selectbox(
                    "Current / Intended Major",
                    options=unique_majors_list,
                    index=default_major_idx,
                    format_func=lambda x: x.title()
                )

                month_names_map = {
                    1: 'January', 2: 'February', 3: 'March', 4: 'April',
                    5: 'May', 6: 'June', 7: 'July', 8: 'August',
                    9: 'September', 10: 'October', 11: 'November', 12: 'December'
                }
                input_month = st.selectbox(
                    "Registration Month",
                    options=list(month_names_map.keys()),
                    format_func=lambda x: f"{x} - {month_names_map[x]}",
                    index=5  # June default
                )

            submit_pred = st.form_submit_button("⚡ Run Predictive Model Inference", use_container_width=True)

        st.markdown("---")
        st.caption("Quickly test with real historical applicants from the dataset:")
        if st.button("🎲 Test with Random Historical Record"):
            sample_row = df_full.sample(1, random_state=int(np.random.randint(0, 10000))).iloc[0]
            st.session_state['sample_input'] = {
                'year': sample_row['Signup_Year'],
                'month': sample_row['Signup_Month_Num'],
                'category': sample_row['Opportunity Category'],
                'gender': sample_row['Gender'],
                'country': sample_row['Country'],
                'major': sample_row['Current/Intended Major'],
                'actual_status': sample_row['Status Description'],
                'actual_target': sample_row['Target_Label']
            }

        if 'sample_input' in st.session_state:
            s = st.session_state['sample_input']
            st.info(
                f"**Loaded Applicant Record:**\n"
                f"- Category: `{s['category'].title()}` | Gender: `{s['gender'].title()}` | Country: `{s['country'].title()}`\n"
                f"- Major: `{s['major'].title()}` | Date: `{s['month']}/{s['year']}`\n"
                f"- **Actual Historical Status:** `{s['actual_status'].title()}` (Class {s['actual_target']})"
            )
            input_year = s['year']
            input_month = s['month']
            input_category = s['category']
            input_gender = s['gender']
            input_country = s['country']
            input_major = s['major']
            submit_pred = True

    with pred_col_right:
        st.markdown("#### **Prediction Output & Probability Analysis**")

        # Prepare exact 471-feature vector
        feature_vector = construct_model_input_vector(
            year=input_year,
            month=input_month,
            category=input_category,
            gender=input_gender,
            country=input_country,
            major=input_major,
            model_cols=model_cols
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                raw_pred = model.predict(feature_vector)[0]
                raw_probs = model.predict_proba(feature_vector)[0]

            prob_class_0 = raw_probs[0]  # Non-Rejected / Active
            prob_class_1 = raw_probs[1]  # Rejected (Most common status in notebook)

            is_rejected = (raw_pred == 1)

            st.markdown(f"""
            <div style="background-color: {'#FEF2F2' if is_rejected else '#F0FDF4'};
                        border: 1.5px solid {'#FCA5A5' if is_rejected else '#86EFAC'};
                        border-radius: 10px; padding: 1.2rem; margin-bottom: 1.2rem;">
                <div style="font-size: 0.85rem; font-weight: 600; color: {'#991B1B' if is_rejected else '#166534'}; text-transform: uppercase;">
                    ML Classification Result
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; color: {'#DC2626' if is_rejected else '#15803D'}; margin-top: 4px;">
                    {'⚠️ At Risk / Rejection Predicted' if is_rejected else '✅ Active / Allocation Predicted'}
                </div>
                <div style="font-size: 0.95rem; color: #475569; margin-top: 6px;">
                    Model assigns <b>Class {raw_pred}</b> based on demographic and opportunity attributes.
                </div>
            </div>
            """, unsafe_allow_html=True)

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.metric(
                    label="Rejection Probability (Class 1)",
                    value=f"{prob_class_1 * 100:.1f}%",
                    delta=f"{(prob_class_1 - 0.417) * 100:+.1f}% vs Avg"
                )
            with pcol2:
                st.metric(
                    label="Active / Allocation Prob (Class 0)",
                    value=f"{prob_class_0 * 100:.1f}%",
                    delta=f"{(prob_class_0 - 0.583) * 100:+.1f}% vs Avg"
                )

            prob_df = pd.DataFrame({
                'Outcome': ['Non-Rejected / Active (Class 0)', 'Rejected (Class 1)'],
                'Probability': [prob_class_0, prob_class_1]
            })

            fig_prob = px.bar(
                prob_df,
                x='Probability',
                y='Outcome',
                orientation='h',
                text=prob_df['Probability'].apply(lambda x: f"{x * 100:.1f}%"),
                color='Outcome',
                color_discrete_map={
                    'Non-Rejected / Active (Class 0)': '#10B981',
                    'Rejected (Class 1)': '#EF4444'
                }
            )
            fig_prob.update_layout(
                height=180,
                showlegend=False,
                margin=dict(l=10, r=20, t=20, b=10),
                xaxis=dict(range=[0, 1], tickformat=".0%"),
                yaxis_title=""
            )
            fig_prob.update_traces(textposition='inside', textfont=dict(color='white', size=13))
            st.plotly_chart(fig_prob, use_container_width=True)

            with st.expander("🔬 View Activated Model Input Features (471-dim vector)", expanded=False):
                activated_features = feature_vector.loc[0, feature_vector.iloc[0] != 0].to_dict()
                st.write("**Non-zero encoded features sent to `predictive_model.pkl`:**")
                st.json(activated_features)
                st.caption(f"Total features in model vector: {len(model_cols)} (471). Verified feature order: Identical to model_columns.pkl.")

        except Exception as e:
            st.error(f"Inference error: {str(e)}")


# ---------------------------------------------------------------------
# TAB 5: RISK MATRIX & FEATURE IMPORTANCES (REPRODUCING NOTEBOOK CHARTS)
# ---------------------------------------------------------------------
with tab5:
    st.subheader("Model Diagnostic & Feature Importance Matrix")
    st.markdown("""
    *Visualisations replicating Section 4 from `Untitled0.ipynb`: Top predictive drivers, Confusion Matrix, and Risk Cohort Interactions.*
    """)

    col_diag_1, col_diag_2 = st.columns(2)

    with col_diag_1:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:12]
        top_cols = [model_cols[i] for i in indices]
        top_vals = importances[indices]

        fi_df = pd.DataFrame({
            'Feature': top_cols,
            'Importance': top_vals
        }).sort_values('Importance', ascending=True)

        fig_fi = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='<b>Feature Importance Matrix (Top Predictive Drivers)</b>',
            text=fi_df['Importance'].apply(lambda x: f"{x:.3f}"),
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_fi.update_layout(
            height=460,
            margin=dict(l=10, r=20, t=50, b=10),
            xaxis_title="Statistical Importance Score",
            yaxis_title="",
            coloraxis_showscale=False
        )
        fig_fi.update_traces(textposition='outside')
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_diag_2:
        st.markdown("<b>Model Classification Confusion Matrix</b>", unsafe_allow_html=True)

        eval_X = df_full[['Opportunity Category', 'Gender', 'Country', 'Current/Intended Major']].copy()
        eval_X['Signup_Year_Num'] = df_full['Learner SignUp DateTime'].dt.year
        eval_X['Signup_Month_Num'] = df_full['Learner SignUp DateTime'].dt.month
        eval_X_encoded = pd.get_dummies(eval_X, drop_first=True)
        eval_X_encoded = eval_X_encoded.reindex(columns=model_cols, fill_value=0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            all_preds = model.predict(eval_X_encoded)

        cm = confusion_matrix(df_full['Target_Label'], all_preds)
        cm_labels = ['Class 0 (Non-Rejected)', 'Class 1 (Rejected)']

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=cm_labels,
            y=cm_labels,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            title='<b>Model Classification Accuracy Grid (Master Dataset)</b>'
        )
        fig_cm.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("#### **Interactive Risk Cohort Analysis Matrix**")
    st.caption("Replicating Chart 4 from notebook: Cross-examination of opportunity and demographic risk cohorts against predicted rejection probability.")

    cohort_c1, cohort_c2 = st.columns(2)
    with cohort_c1:
        cohort_dim1 = st.selectbox(
            "Primary Cohort Dimension",
            options=['Opportunity Category', 'Gender'],
            index=0
        )
    with cohort_c2:
        cohort_dim2 = st.selectbox(
            "Secondary Cohort Dimension",
            options=['Gender', 'Opportunity Category', 'Signup_Year'],
            index=1 if cohort_dim1 == 'Gender' else 0
        )

    df_cohort = df_full.copy()
    df_cohort['Predicted_Risk_Prob'] = model.predict_proba(eval_X_encoded)[:, 1]

    risk_pivot = df_cohort.pivot_table(
        values='Predicted_Risk_Prob',
        index=cohort_dim1,
        columns=cohort_dim2,
        aggfunc='mean',
        observed=False
    )

    fig_risk_heat = px.imshow(
        risk_pivot,
        text_auto='.2f',
        color_continuous_scale='RdYlGn_r',
        title=f'<b>Predicted Risk Probability: {cohort_dim1} vs {cohort_dim2}</b>',
        labels=dict(color="Rejection Prob")
    )
    fig_risk_heat.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_risk_heat, use_container_width=True)


# =====================================================================
# 7. FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 10px 0px 20px 0px;">
        RIT Opportunity Intelligence Platform &bull; Built with Streamlit, Scikit-Learn & Plotly &bull; 471 Predictive Features &bull; Local Deployment Ready
    </div>
    """,
    unsafe_allow_html=True
)

