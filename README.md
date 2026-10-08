# 🎵 SoundPulse: Music Analytics & Recommendation Platform

**SoundPulse** is an interactive, local music streaming analytics and recommendation web application built with **Streamlit**, **Pandas**, and **Scikit-Learn**. It allows users to explore their personal Spotify listening habits, analyze audio feature metrics, and generate instant, memory-efficient similarity-based playlist recommendations.

---

## ✨ Key Features

* **Interactive Global Control Bar:** Easily filter your entire streaming dashboard by device platform and day of the week.
* **High-Contrast Spotify Aesthetic:** Designed with a sleek dark mode theme (`#121212`) paired with vibrant Spotify Green (`#1DB954`) visual accents and metrics.
* **Comprehensive Analytics Dashboard:**
  * **Listening History:** Hourly activity distribution and weekly streaming trends.
  * **Top Artists & Tracks:** Visual breakdown of your most streamed music.
  * **Audio Profile & Mood Matrix:** Valence (positivity) vs. Energy (intensity) scatter plots and catalog feature distribution radars.
  * **Individual Track Deep-Dive:** Single-track analyzer featuring audio gauge meters and individual history timelines for tracks present in your logs.
* **Similarity Recommendation Engine:** Computes cosine distance vectors on normalized audio features (`danceability`, `energy`, `valence`, `tempo`, etc.) on the fly to recommend tailor-made playlists with direct clickable Spotify links.
* **Memory-Optimized Architecture:** Bypasses large matrix allocation errors by scaling features once and computing similarities dynamically on demand.

---

## 📂 Project Structure

```text
music-analytics/
│
├── data/
│   ├── spotify_history.csv       # Streaming history logs (timestamps, plays, platforms)
│   └── spotify_tracks.csv        # Audio feature catalog (danceability, energy, tempo, etc.)
│
├── app.py                        # Main Streamlit application script
└── README.md                     # Project documentation
```

## 🛠️ Installation & Setup
1. Prerequisites

Ensure you have Python 3.8+ installed on your system.
2. Clone or Setup the Project Directory

Navigate to your project directory:
```bash

cd music-analytics
```

3. Install Required Dependencies

Install the required data science and web framework libraries via pip:
```bash

pip install -m requirements.txt
```
4. Add Your Datasets

Ensure your raw data files are placed inside a data/ folder in the root directory:

    data/spotify_history.csv

    data/spotify_tracks.csv

## 🚀 Running the Application

Launch the Streamlit dashboard locally by running:

```code
streamlit run app.py
```

Streamlit will automatically open the interactive platform in your default web browser (typically at http://localhost:8501).