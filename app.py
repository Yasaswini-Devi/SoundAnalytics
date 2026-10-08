import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# 1. PAGE & SPOTIFY DARK THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SoundPulse: Music Analytics & Recommendation Platform",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* Main Canvas Background & Base Text */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    
    /* Top Title Styling */
    .main-title {
        color: #1DB954 !important; /* Spotify Green */
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    .sub-title {
        color: #B3B3B3;
        font-size: 1rem;
        margin-bottom: 15px;
    }
    
    /* Top KPI Metric Cards Styling */
    [data-testid="stMetric"] {
        background-color: #181818 !important;
        border: 1px solid #282828 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
    }
    
    /* High Contrast Text on Metric Cards */
    [data-testid="stMetricLabel"] {
        color: #B3B3B3 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #1DB954 !important; /* Spotify Green */
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #181818;
        border-radius: 8px;
        color: #B3B3B3;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1DB954 !important;
        color: #FFFFFF !important;
        font-weight: bold;
    }
    
    /* Gauge Header Titles */
    .gauge-title {
        text-align: center;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: -10px;
    }
    
    /* Custom Spotify Green Buttons */
    .stButton>button {
        background-color: #1DB954 !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
        border-radius: 20px !important;
        border: none !important;
        padding: 10px 24px !important;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1ED760 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    history_df = pd.read_csv('data/spotify_history.csv')
    
    # Standardize history columns dynamically
    history_df['master_metadata_album_artist_name'] = history_df['artist_name']
    history_df['master_metadata_track_name'] = history_df['track_name']
    history_df['master_metadata_album_name'] = history_df['album_name']
    
    # Timestamps & metrics
    history_df['ts'] = pd.to_datetime(history_df['ts'], utc=True)
    history_df['hour'] = history_df['ts'].dt.hour
    history_df['day_name'] = history_df['ts'].dt.day_name()
    history_df['minutes_played'] = history_df['ms_played'] / 60000.0
    
    # Skip logic
    if 'reason_end' in history_df.columns:
        history_df['is_skip'] = (history_df['ms_played'] < 30000) | (history_df['reason_end'].astype(str).str.lower() == 'fwdbtn')
    else:
        history_df['is_skip'] = history_df['ms_played'] < 30000

    tracks_df = pd.read_csv('data/spotify_tracks.csv')
    
    if 'track_name' not in tracks_df.columns and 'name' in tracks_df.columns:
        tracks_df['track_name'] = tracks_df['name']
    if 'artists' not in tracks_df.columns and 'artist_name' in tracks_df.columns:
        tracks_df['artists'] = tracks_df['artist_name']
        
    tracks_df = tracks_df.dropna(subset=['track_name', 'artists'])
    tracks_df = tracks_df.drop_duplicates(subset=['track_name', 'artists'])
    
    # Intersection of tracks present in BOTH files
    history_tracks_lower = set(history_df['master_metadata_track_name'].dropna().astype(str).str.lower())
    common_tracks_mask = tracks_df['track_name'].astype(str).str.lower().isin(history_tracks_lower)
    common_tracks_df = tracks_df[common_tracks_mask].copy()
    
    return history_df, tracks_df, common_tracks_df

try:
    history_df, tracks_df, common_tracks_df = load_and_prep_data()
except Exception as e:
    st.error(f"Error loading datasets from 'data/' folder: {e}")
    st.stop()

AUDIO_FEATURES = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

@st.cache_data
def get_scaled_features(df, features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(df[features])

scaled_feature_matrix = get_scaled_features(tracks_df, AUDIO_FEATURES)

# -----------------------------------------------------------------------------
# 3. TOP HEADER & INLINE CONTROLS
# -----------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>🎵 SoundPulse: Music Analytics & Recommendation Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Interactive Listening Habits Dashboard & Similarity-Based Recommendation Engine</p>", unsafe_allow_html=True)

with st.expander("🎛️ Analytics Controls & Global Filters", expanded=True):
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        platform_list = ["All"] + list(history_df['platform'].dropna().unique())
        selected_platform = st.selectbox("Filter by Device Platform", platform_list)
        
    with col_filter2:
        day_list = ["All", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day = st.selectbox("Filter by Day of Week", day_list)

filtered_history = history_df.copy()
if selected_platform != "All":
    filtered_history = filtered_history[filtered_history['platform'] == selected_platform]
if selected_day != "All":
    filtered_history = filtered_history[filtered_history['day_name'] == selected_day]

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. KPI SUMMARY CARDS
# -----------------------------------------------------------------------------
valid_plays = filtered_history[filtered_history['is_skip'] == False]
total_hours = valid_plays['minutes_played'].sum() / 60.0
total_streams = len(filtered_history)
skip_rate = (filtered_history['is_skip'].sum() / total_streams * 100) if total_streams > 0 else 0
unique_artists = valid_plays['master_metadata_album_artist_name'].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Hours Listened", f"{total_hours:,.1f} hrs")
col2.metric("Total Play Events", f"{total_streams:,}")
col3.metric("Skip Rate", f"{skip_rate:.1f}%")
col4.metric("Unique Artists Played", f"{unique_artists:,}")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. DASHBOARD CHARTS
# -----------------------------------------------------------------------------
tab_time, tab_artists, tab_radar, tab_single_track = st.tabs([
    "⏰ Listening History Analytics", 
    "🎤 Top Artists & Tracks", 
    "🕸️ Audio Profile & Mood Matrix",
    "🎧 Individual Track Analytics"
])

with tab_time:
    col_a, col_b = st.columns(2)
    with col_a:
        hourly_counts = filtered_history.groupby('hour').size().reset_index(name='plays')
        fig_hour = px.bar(
            hourly_counts, x='hour', y='plays',
            title="Listening Activity by Hour of Day (00:00 - 23:00)",
            labels={'hour': 'Hour of Day', 'plays': 'Streams'},
            color='plays', color_continuous_scale=['#121212', '#1DB954']
        )
        fig_hour.update_layout(template="plotly_dark", paper_bgcolor='#121212', plot_bgcolor='#121212')
        st.plotly_chart(fig_hour, use_container_width=True)
        
    with col_b:
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = filtered_history.groupby('day_name').size().reindex(days_order).fillna(0).reset_index(name='plays')
        fig_day = px.line(
            day_counts, x='day_name', y='plays', markers=True,
            title="Weekly Streaming Trend",
            labels={'day_name': 'Day', 'plays': 'Streams'}
        )
        fig_day.update_traces(line_color='#1DB954', line_width=3)
        fig_day.update_layout(template="plotly_dark", paper_bgcolor='#121212', plot_bgcolor='#121212')
        st.plotly_chart(fig_day, use_container_width=True)

with tab_artists:
    col_c, col_d = st.columns(2)
    with col_c:
        top_artists = valid_plays['master_metadata_album_artist_name'].value_counts().head(10).reset_index()
        top_artists.columns = ['artist', 'plays']
        fig_art = px.bar(
            top_artists, x='plays', y='artist', orientation='h',
            title="Top 10 Most Played Artists",
            color='plays', color_continuous_scale=['#1DB954', '#1ed760']
        )
        fig_art.update_layout(template="plotly_dark", paper_bgcolor='#121212', plot_bgcolor='#121212', yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_art, use_container_width=True)
        
    with col_d:
        top_tracks = valid_plays['master_metadata_track_name'].value_counts().head(10).reset_index()
        top_tracks.columns = ['track', 'plays']
        fig_trk = px.bar(
            top_tracks, x='plays', y='track', orientation='h',
            title="Top 10 Most Played Tracks",
            color='plays', color_continuous_scale=['#1DB954', '#1ed760']
        )
        fig_trk.update_layout(template="plotly_dark", paper_bgcolor='#121212', plot_bgcolor='#121212', yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_trk, use_container_width=True)

with tab_radar:
    col_e, col_f = st.columns(2)
    with col_e:
        mean_features = tracks_df[AUDIO_FEATURES].mean().reset_index()
        mean_features.columns = ['Feature', 'Value']
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=mean_features['Value'],
            theta=mean_features['Feature'],
            fill='toself',
            fillcolor='rgba(29, 185, 84, 0.4)',
            line=dict(color='#1DB954', width=2),
            name='Catalog Average'
        ))
        fig_radar.update_layout(
            template="plotly_dark", paper_bgcolor='#121212', plot_bgcolor='#121212',
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Average Audio Feature Profile"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col_f:
        sample_df = tracks_df.sample(min(1200, len(tracks_df)))
        fig_scatter = px.scatter(
            sample_df,
            x='valence', y='energy', color='popularity',
            size='danceability',
            color_continuous_scale='Viridis',
            hover_data=['track_name', 'artists', 'track_genre'],
            title="Mood Quadrant: Positivity (Valence) vs. Intensity (Energy)",
            labels={'valence': 'Valence (Happiness)', 'energy': 'Energy'}
        )
        fig_scatter.update_layout(template="plotly_dark", paper_bgcolor='#121212', plot_bgcolor='#121212')
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab_single_track:
    st.markdown("### 🎧 Single Track Deep-Dive Analyzer")
    
    common_tracks_sorted = common_tracks_df['track_name'].drop_duplicates().sort_values().tolist()
    
    if not common_tracks_sorted:
        st.warning("No matching track names found between your history and track catalog files.")
    else:
        selected_analytics_track = st.selectbox(
            "Select a Track to Analyze (Tracks present in your streaming history):", 
            common_tracks_sorted,
            key="single_track_selector"
        )
        
        track_meta = common_tracks_df[common_tracks_df['track_name'] == selected_analytics_track].iloc[0]
        track_history = history_df[history_df['master_metadata_track_name'].astype(str).str.lower() == selected_analytics_track.lower()]
        track_play_count = len(track_history)
        track_total_mins = track_history['minutes_played'].sum() if track_play_count > 0 else 0
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
        t_col1.metric("Artist", str(track_meta['artists']))
        t_col2.metric("Genre", str(track_meta.get('track_genre', 'N/A')))
        t_col3.metric("Popularity", f"{track_meta.get('popularity', 0)} / 100")
        t_col4.metric("Tempo (BPM)", f"{track_meta.get('tempo', 0):.1f}")
        t_col5.metric("Your Total Streams", f"{track_play_count} plays ({track_total_mins:.1f} mins)")
        
        st.markdown("---")
        
        col_t_left, col_t_right = st.columns(2)
        
        with col_t_left:
            track_features = [track_meta[f] for f in AUDIO_FEATURES]
            catalog_mean_features = tracks_df[AUDIO_FEATURES].mean().tolist()
            
            fig_track_radar = go.Figure()
            fig_track_radar.add_trace(go.Scatterpolar(
                r=catalog_mean_features,
                theta=AUDIO_FEATURES,
                fill='toself',
                fillcolor='rgba(179, 179, 179, 0.2)',
                line=dict(color='#B3B3B3', dash='dash'),
                name='Catalog Average'
            ))
            fig_track_radar.add_trace(go.Scatterpolar(
                r=track_features,
                theta=AUDIO_FEATURES,
                fill='toself',
                fillcolor='rgba(29, 185, 84, 0.5)',
                line=dict(color='#1DB954', width=3),
                name=f"Track: {selected_analytics_track}"
            ))
            
            fig_track_radar.update_layout(
                template="plotly_dark", paper_bgcolor='#121212', plot_bgcolor='#121212',
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"Audio Vector Profile: '{selected_analytics_track}' vs Average"
            )
            st.plotly_chart(fig_track_radar, use_container_width=True)
            
        with col_t_right:
            track_hourly = track_history.groupby('hour').size().reset_index(name='plays')
            all_hours = pd.DataFrame({'hour': range(24)})
            track_hourly = pd.merge(all_hours, track_hourly, on='hour', how='left').fillna(0)
            
            fig_track_time = px.bar(
                track_hourly, x='hour', y='plays',
                title=f"When Do You Listen to '{selected_analytics_track}'? (Hour of Day)",
                labels={'hour': 'Hour of Day (00:00 - 23:00)', 'plays': 'Streams'},
                color='plays', color_continuous_scale=['#181818', '#1DB954']
            )
            fig_track_time.update_layout(template="plotly_dark", paper_bgcolor='#121212', plot_bgcolor='#121212')
            st.plotly_chart(fig_track_time, use_container_width=True)
                
        # ---------------------------------------------------------------------
        # VISIBLE AUDIO FEATURE BREAKDOWN GAUGES
        # ---------------------------------------------------------------------
        st.markdown("#### Audio Feature Breakdown")
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        
        def create_gauge(val):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val * 100,
                number={'suffix': "%", 'font': {'color': "#1DB954", 'size': 28}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#FFFFFF"},
                    'bar': {'color': "#1DB954"},
                    'bgcolor': "#181818",
                    'bordercolor': "#282828"
                }
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor='#121212', height=140, 
                margin=dict(l=15, r=15, t=10, b=10)
            )
            return fig

        with g_col1:
            st.markdown("<p class='gauge-title'>Danceability</p>", unsafe_allow_html=True)
            st.plotly_chart(create_gauge(track_meta.get('danceability', 0)), use_container_width=True)
            
        with g_col2:
            st.markdown("<p class='gauge-title'>Energy</p>", unsafe_allow_html=True)
            st.plotly_chart(create_gauge(track_meta.get('energy', 0)), use_container_width=True)
            
        with g_col3:
            st.markdown("<p class='gauge-title'>Valence (Positivity)</p>", unsafe_allow_html=True)
            st.plotly_chart(create_gauge(track_meta.get('valence', 0)), use_container_width=True)
            
        with g_col4:
            st.markdown("<p class='gauge-title'>Acousticness</p>", unsafe_allow_html=True)
            st.plotly_chart(create_gauge(track_meta.get('acousticness', 0)), use_container_width=True)

# -----------------------------------------------------------------------------
# 6. RECOMMENDATION ENGINE SECTION
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("🎯 Playlist Recommendation Engine")

col_rec1, col_rec2, col_rec3 = st.columns([2, 1, 1])

with col_rec1:
    song_options = tracks_df['track_name'].drop_duplicates().sort_values().tolist()
    selected_song = st.selectbox("Search or Select Seed Track:", song_options)

with col_rec2:
    top_n = st.slider("Number of Songs:", min_value=3, max_value=15, value=5)

with col_rec3:
    filter_explicit = st.checkbox("Exclude Explicit Content", value=False)
    same_genre_only = st.checkbox("Filter by Same Genre Only", value=False)

if st.button("Generate Playlist Recommendations"):
    matched_indices = tracks_df[tracks_df['track_name'] == selected_song].index
    
    if len(matched_indices) == 0:
        st.error("Selected track not found in catalog.")
    else:
        target_idx = matched_indices[0]
        target_genre = tracks_df.iloc[target_idx]['track_genre'] if 'track_genre' in tracks_df.columns else None
        
        target_vector = scaled_feature_matrix[target_idx].reshape(1, -1)
        scores = cosine_similarity(target_vector, scaled_feature_matrix)[0]
        sorted_indices = np.argsort(scores)[::-1]
        
        filtered_indices = []
        for idx in sorted_indices:
            if idx == target_idx:
                continue
                
            candidate_row = tracks_df.iloc[idx]
            
            if filter_explicit and candidate_row.get('explicit', False):
                continue
                
            if same_genre_only and target_genre and candidate_row.get('track_genre') != target_genre:
                continue
                
            filtered_indices.append(idx)
            if len(filtered_indices) >= top_n:
                break
                
        rec_df = tracks_df.iloc[filtered_indices][
            ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'popularity', 'tempo']
        ].copy()
        
        rec_df['Similarity Score'] = [f"{scores[idx] * 100:.1f}%" for idx in filtered_indices]
        
        # Build direct Spotify URL column
        if 'track_id' in rec_df.columns:
            clean_ids = rec_df['track_id'].astype(str).str.replace('spotify:track:', '')
            rec_df['Listen on Spotify'] = "https://open.spotify.com/track/" + clean_ids
        else:
            rec_df['Listen on Spotify'] = "https://open.spotify.com/search/" + rec_df['track_name'].astype(str)

        display_df = rec_df[['track_name', 'artists', 'album_name', 'track_genre', 'Similarity Score', 'Listen on Spotify']].reset_index(drop=True)
        
        st.subheader(f"Recommended Songs Similar to '{selected_song}':")
        
        # Render table with separate readable text column and clickable LinkColumn
        st.dataframe(
            display_df,
            column_config={
                "Listen on Spotify": st.column_config.LinkColumn(
                    "Spotify Link",
                    help="Click to open track directly on Spotify",
                    display_text="▶ Play Track"
                )
            },
            use_container_width=True
        )