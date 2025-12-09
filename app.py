import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

# Konfiguracja strony
st.set_page_config(
    page_title="Znajdź Swoją Grupę",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Niestandardowe style CSS
st.markdown("""
    <style>
    /* Główne tło */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1000 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a00 0%, #1a1000 100%);
        border-right: 2px solid #D4AF37;
    }
    
    /* Nagłówki */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
    }
    
    h1 {
        background: linear-gradient(90deg, #FFFFFF 0%, #FFD700 50%, #D4AF37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Tekst */
    p, label, .stMarkdown {
        color: #FFF8DC !important;
    }
    
    /* Selectbox i radio */
    .stSelectbox label, .stRadio label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* Input fields */
    .stSelectbox > div > div {
        background-color: #1a1000 !important;
        border: 2px solid #D4AF37 !important;
        color: #FFFFFF !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: rgba(212, 175, 55, 0.1);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #D4AF37;
    }
    
    /* Metryki */
    [data-testid="stMetricValue"] {
        color: #FFD700 !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* Karty/kontenery */
    .element-container {
        background: rgba(26, 16, 0, 0.3);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    /* Wykresy */
    .js-plotly-plot {
        border-radius: 15px;
        border: 2px solid #D4AF37;
        box-shadow: 0 0 30px rgba(212, 175, 55, 0.3);
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h2 {
        color: #FFD700 !important;
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid #D4AF37;
        margin-bottom: 20px;
    }
    
    /* Buttony (jeśli są) */
    .stButton > button {
        background: linear-gradient(90deg, #D4AF37 0%, #FFD700 100%);
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.5);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #FFD700 0%, #D4AF37 100%);
        box-shadow: 0 0 30px rgba(212, 175, 55, 0.8);
    }
    </style>
""", unsafe_allow_html=True)

# Definicja kolorów
COLORS = {
    'primary': '#D4AF37',      # Złoto główne
    'secondary': '#FFD700',    # Jasne złoto
    'accent': '#FFF8DC',       # Bardzo jasne złoto/cornsilk
    'background': '#000000',   # Czerń
    'text': '#FFFFFF',         # Biel
    'gradient_start': '#D4AF37',
    'gradient_end': '#FFD700'
}

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    all_df.fillna('Brak danych', inplace=True)
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

def create_styled_histogram(data, x_column, title, xaxis_title):
    """Tworzy stylizowany histogram w kolorach biel/czerń/złoto"""
    fig = px.histogram(
        data, 
        x=x_column,
        color_discrete_sequence=[COLORS['primary']]
    )
    
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 20, 'color': COLORS['text'], 'family': 'Arial Black'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=xaxis_title,
        yaxis_title="Liczba osób",
        plot_bgcolor='rgba(0, 0, 0, 0.5)',
        paper_bgcolor='rgba(26, 16, 0, 0.3)',
        font={'color': COLORS['text'], 'size': 12},
        xaxis={
            'gridcolor': 'rgba(212, 175, 55, 0.2)',
            'title_font': {'size': 14, 'color': COLORS['secondary']},
            'tickfont': {'color': COLORS['accent']}
        },
        yaxis={
            'gridcolor': 'rgba(212, 175, 55, 0.2)',
            'title_font': {'size': 14, 'color': COLORS['secondary']},
            'tickfont': {'color': COLORS['accent']}
        },
        margin=dict(t=60, b=40, l=40, r=40),
        hoverlabel=dict(
            bgcolor=COLORS['primary'],
            font_size=14,
            font_family="Arial"
        )
    )
    
    # Dodanie gradientu do słupków
    fig.update_traces(
        marker=dict(
            color=COLORS['primary'],
            line=dict(color=COLORS['secondary'], width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Liczba: %{y}<extra></extra>'
    )
    
    return fig

# Nagłówek główny
st.markdown("""
    <h1 style='text-align: center; font-size: 3rem; margin-bottom: 10px;'>
        👑 Znajdź Swoją Grupę
    </h1>
    <p style='text-align: center; font-size: 1.2rem; color: #FFD700; margin-bottom: 30px;'>
        Odkryj osoby o podobnych zainteresowaniach
    </p>
""", unsafe_allow_html=True)

# Ładowanie modelu i danych
model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()


# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='margin: 0;'>✨ Twój Profil</h2>
            <p style='color: #FFD700; margin-top: 10px;'>Powiedz nam coś o sobie</p>
        </div>
    """, unsafe_allow_html=True)
    
    age = st.selectbox("🎂 Wiek", all_df['age'].unique())#['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("🎓 Wykształcenie", all_df['edu_level'].unique())#['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("🐾 Ulubione zwierzęta", all_df['fav_animals'].unique())#['Brak danych','Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("🌍 Ulubione miejsce", all_df['fav_place'].unique())#['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("👤 Płeć", all_df['gender'].unique())#['Mężczyzna', 'Kobieta', 'Brak danych'])
    
    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])


# Predykcja
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# Wyniki
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(255, 215, 0, 0.1) 100%);
                    padding: 30px; border-radius: 20px; border: 2px solid #D4AF37;
                    box-shadow: 0 0 40px rgba(212, 175, 55, 0.3); margin-bottom: 30px;'>
            <h2 style='color: #FFD700; margin-top: 0;'>
                🎯 Twoja Grupa: {predicted_cluster_data['name']}
            </h2>
            <p style='font-size: 1.1rem; line-height: 1.8; color: #FFF8DC;'>
                {predicted_cluster_data['description']}
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
    st.markdown("""
        <div style='background: linear-gradient(135deg, #D4AF37 0%, #FFD700 100%);
                    padding: 30px; border-radius: 20px; text-align: center;
                    box-shadow: 0 0 40px rgba(212, 175, 55, 0.5);'>
    """, unsafe_allow_html=True)
    st.metric("👥 Twoi Znajomi", f"{len(same_cluster_df)}")
    st.markdown("</div>", unsafe_allow_html=True)

# Sekcja wykresów
st.markdown("""
    <h2 style='text-align: center; margin: 50px 0 30px 0; font-size: 2rem;'>
        📊 Analiza Grupy
    </h2>
""", unsafe_allow_html=True)

# Wykresy w dwóch kolumnach
col1, col2 = st.columns(2)

with col1:
    fig = create_styled_histogram(
        same_cluster_df.sort_values("age"),
        "age",
        "🎂 Rozkład Wieku",
        "Wiek"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    fig = create_styled_histogram(
        same_cluster_df,
        "fav_animals",
        "🐾 Ulubione Zwierzęta",
        "Zwierzęta"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    fig = create_styled_histogram(
        same_cluster_df,
        "gender",
        "👤 Rozkład Płci",
        "Płeć"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = create_styled_histogram(
        same_cluster_df,
        "edu_level",
        "🎓 Wykształcenie",
        "Poziom wykształcenia"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    fig = create_styled_histogram(
        same_cluster_df,
        "fav_place",
        "🌍 Ulubione Miejsca",
        "Miejsca"
    )
    st.plotly_chart(fig, use_container_width=True)

# Stopka
st.markdown("""
    <div style='text-align: center; margin-top: 60px; padding: 30px; 
                border-top: 2px solid #D4AF37; color: #FFD700;'>
        <p style='font-size: 0.9rem;'>
            ✨ Powered by Machine Learning & AI ✨
        </p>
    </div>
""", unsafe_allow_html=True)