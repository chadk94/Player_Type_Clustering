import time

import nba_api.stats.library.data
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas
import requests
from bs4 import BeautifulSoup
from nba_api.stats.library.parameters import SeasonTypeAllStar, PlayerOrTeamAbbreviation
from sklearn.cluster import KMeans
from nba_api.stats.endpoints import LeagueGameLog, SynergyPlayTypes, PlayerDashPtShots, shotchartdetail, ShotChartDetail
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static.players import get_players
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from sklearn.preprocessing import StandardScaler
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd



def get_shot_chart_data(player_id, season='2024-25'):
    """Get detailed shot chart data for a player."""
    try:
        time.sleep(1)
        player_id = str(int(float(player_id)))
        print ("getting shot chart",player_id)
        shot_chart = ShotChartDetail(
            player_id=player_id,
            team_id=0,
            season_nullable='2024-25',
            season_type_all_star='Regular Season',
            context_measure_simple='FGA'  # Specify the context measure
        ).get_data_frames()[0]
        # Calculate zone-based metrics
        shot_profile = {
            'PLAYER_ID': player_id,
            'SEASON_ID': season
        }

        # Paint touches
        paint_shots = shot_chart[shot_chart['SHOT_ZONE_BASIC'].isin(['Restricted Area', 'In The Paint (Non-RA)'])]
        shot_profile['paint_shots_per_game'] = len(paint_shots) / 82  # normalized to per game
        shot_profile['paint_fg_pct'] = paint_shots['SHOT_MADE_FLAG'].mean() if len(paint_shots) > 0 else 0

        # Corner 3s
        corner_3s = shot_chart[shot_chart['SHOT_ZONE_BASIC'] == 'Corner 3']
        shot_profile['corner_3_per_game'] = len(corner_3s) / 82
        shot_profile['corner_3_pct'] = corner_3s['SHOT_MADE_FLAG'].mean() if len(corner_3s) > 0 else 0

        # Mid-range
        midrange = shot_chart[shot_chart['SHOT_ZONE_BASIC'] == 'Mid-Range']
        shot_profile['midrange_per_game'] = len(midrange) / 82
        shot_profile['midrange_pct'] = midrange['SHOT_MADE_FLAG'].mean() if len(midrange) > 0 else 0

        # Above break 3s
        above_break_3s = shot_chart[shot_chart['SHOT_ZONE_BASIC'] == 'Above the Break 3']
        shot_profile['above_break_3_per_game'] = len(above_break_3s) / 82
        shot_profile['above_break_3_pct'] = above_break_3s['SHOT_MADE_FLAG'].mean() if len(above_break_3s) > 0 else 0

        # Distance metrics
        shot_profile['avg_shot_distance'] = shot_chart['SHOT_DISTANCE'].mean()
        shot_profile['max_shot_distance'] = shot_chart['SHOT_DISTANCE'].max()

        # Shot clock analysis
        if 'SHOT_CLOCK' in shot_chart.columns:
            shot_profile['avg_shot_clock'] = shot_chart['SHOT_CLOCK'].mean()
            shot_profile['early_shot_clock_freq'] = len(shot_chart[shot_chart['SHOT_CLOCK'] >= 15]) / len(shot_chart)

        # Pressure shots
        if 'PERIOD' in shot_chart.columns:
            clutch_shots = shot_chart[
                (shot_chart['PERIOD'] >= 4) &
                (shot_chart['MINUTES_REMAINING'] <= 5)
                ]
            shot_profile['clutch_fg_pct'] = clutch_shots['SHOT_MADE_FLAG'].mean() if len(clutch_shots) > 0 else 0
        return pd.DataFrame([shot_profile])
    except:
        return None


def enhance_player_data(player_averages, min_games=10):
    """Add shot chart data to player averages."""
    # Ensure PLAYER_ID is formatted correctly

    all_shot_data = pd.DataFrame()
    qualified_players = player_averages

    total_players = len(qualified_players)
    print(f"Processing {total_players} qualified players...")

    for idx, player in qualified_players.iterrows():
        print(f"Processing player {idx + 1}/{total_players}: {player['PLAYER_NAME']} (ID: {player['PLAYER_ID']})")
        if idx==0:
            all_shot_data = pd.DataFrame(get_shot_chart_data(player['PLAYER_ID'], player['SEASON_ID']))
        else:
            shot_data = get_shot_chart_data(player['PLAYER_ID'], player['SEASON_ID'])
        if idx!=0:
            if shot_data is not None:
                shot_data = pd.DataFrame(shot_data, columns=all_shot_data.columns)
                all_shot_data = pd.concat([all_shot_data, shot_data], ignore_index=True)
        time.sleep(1)
    qualified_players=qualified_players.fillna(0)
    qualified_players[['PLAYER_ID','SEASON_ID']]=qualified_players[['PLAYER_ID','SEASON_ID']].astype(int)
    all_shot_data[['PLAYER_ID','SEASON_ID']]=all_shot_data[['PLAYER_ID','SEASON_ID']].astype(int)

    enhanced_data = qualified_players.merge(
        all_shot_data,
        on=['PLAYER_ID', 'SEASON_ID'],
        how='left'
    )
    return enhanced_data


def cluster_players_off(data, n_clusters):
    """Perform k-means clustering on players based on their stats and shooting profiles."""
    # Select features for clustering
    data=data[data['MIN']>10].copy()
    features = [
        'PTS_per36', 'AST_per36', 'OREB_per36', 'DREB_per36', 'TOV_per36',
        'FG%', 'FG3%', 'FT%',
        'paint_shots_per_game_per36', 'paint_fg_pct',
        'corner_3_per_game_per36', 'corner_3_pct',
        'midrange_per_game_per36', 'midrange_pct',
        'above_break_3_per_game_per36', 'above_break_3_pct',
        'avg_shot_distance', 'WEIGHT', 'Height_IN',
    ]

    data['Height_IN']=data['HEIGHT'].str.split('-').apply(lambda x: int(x[0]) * 12 + int(x[1]))

    minutes_factor = 36 / data['MIN'].clip(lower=1)  # Avoid division by zero
    per36_cols = ['PTS', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'TOV','paint_shots_per_game','corner_3_per_game','midrange_per_game','above_break_3_per_game']
    for col in per36_cols:
        if col in data.columns:
            data[f'{col}_per36'] = data[col] * minutes_factor

    # Remove rows with missing values
    clean_data = data.dropna(subset=features)
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clean_data[features])
    if n_clusters is None:
        from sklearn.metrics import silhouette_score

        # Initialize plots for both methods
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        min_clusters=10
        max_clusters=50
        # Lists to store results
        inertia_values = []
        silhouette_scores = []

        print("Determining optimal number of clusters...")

        # Try different numbers of clusters
        for k in range(min_clusters, max_clusters + 1):
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)

            # Get inertia (for Elbow Method)
            inertia_values.append(kmeans.inertia_)

            # Get silhouette score
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            print(f"  Clusters: {k}, Inertia: {kmeans.inertia_:.0f}, Silhouette Score: {silhouette_avg:.3f}")

        # Plot Elbow Method
        ax1.plot(range(min_clusters, max_clusters + 1), inertia_values, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')

        # Calculate the "elbow" point using second derivatives
        deltas = np.diff(inertia_values)
        second_deltas = np.diff(deltas)
        elbow_index = np.argmax(second_deltas) + 1  # +1 due to double differentiation
        elbow_value = min_clusters + elbow_index

        ax1.axvline(x=elbow_value, color='r', linestyle='--', alpha=0.7)
        ax1.text(elbow_value + 0.1, ax1.get_ylim()[0] + 0.7 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                 f'Elbow point: {elbow_value}', color='r')

        # Plot Silhouette Analysis
        ax2.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'go-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')

        # Find optimal clusters from silhouette
        optimal_clusters_sil = min_clusters + np.argmax(silhouette_scores)
        ax2.axvline(x=optimal_clusters_sil, color='r', linestyle='--', alpha=0.7)
        ax2.text(optimal_clusters_sil + 0.1, ax2.get_ylim()[0] + 0.7 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]),
                 f'Optimal: {optimal_clusters_sil}', color='r')

        plt.tight_layout()
        plt.show()

        # Choose the optimal number of clusters
        # If silhouette and elbow methods disagree, prefer silhouette
        if abs(optimal_clusters_sil - elbow_value) <= 1:
            n_clusters = optimal_clusters_sil
        else:
            # If they disagree by more than 1, take silhouette with a note
            n_clusters = optimal_clusters_sil
            print(f"Note: Elbow method suggests {elbow_value} clusters, "
                  f"while Silhouette suggests {optimal_clusters_sil}. "
                  f"Choosing {n_clusters} based on Silhouette score.")

        print(f"Optimal number of clusters: {n_clusters}")

    # Perform k-means clustering with the determined number of clusters

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Add cluster labels to the dataset
    result_data = clean_data.copy()
    result_data['Cluster'] = clusters
    result_data=result_data[result_data['SEASON_ID'] == 22024]
    # Create cluster visualization
    plt.figure(figsize=(12, 8))

    # Create a scatter plot using the first two principal components
    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='viridis')
    plt.title('Player Clusters Based on Performance and Shooting Profile')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Add player names for some key points
    for idx, player in enumerate(result_data['PLAYER_NAME']):
        if result_data.iloc[idx]['PTS'] > 20:  # Label high scorers
            plt.annotate(player, (scaled_features[idx, 0], scaled_features[idx, 1]))

    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    plt.figure(figsize=(12, 8))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
    plt.title('Player Clusters Using PCA (Performance and Shooting Profile)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

    # Add player names for key players
    for idx, player in enumerate(result_data['PLAYER_NAME']):
        if result_data.iloc[idx]['PTS'] > 20 or result_data.iloc[idx]['AST'] > 7:
            plt.annotate(player, (principal_components[idx, 0], principal_components[idx, 1]))

    plt.tight_layout()
    plt.show()

    # Print cluster summaries with more detailed information
    print("\nCluster Summaries:")
    for i in range(n_clusters):
        cluster_players = result_data[result_data['Cluster'] == i]

        print(f"\nCluster {i} ({len(cluster_players)} players):")

        # Top players by different metrics
        print("Top Scorers:", ", ".join(cluster_players.nlargest(3, 'PTS')['PLAYER_NAME'].tolist()))
        if 'AST' in cluster_players.columns:
            print("Top Playmakers:", ", ".join(cluster_players.nlargest(3, 'AST')['PLAYER_NAME'].tolist()))

        # Shooting profile
        print(f"Avg Points: {cluster_players['PTS'].mean():.1f}")
        print(f"Avg 3P%: {cluster_players['FG3%'].mean():.3f}")

        # Shot distribution
        if 'paint_shots_per_game' in cluster_players.columns and 'above_break_3_per_game' in cluster_players.columns:
            paint_pct = cluster_players['paint_shots_per_game'].mean() / (
                        cluster_players['paint_shots_per_game'].mean() +
                        cluster_players['midrange_per_game'].mean() +
                        cluster_players['above_break_3_per_game'].mean() +
                        cluster_players['corner_3_per_game'].mean()) * 100
            print(f"Paint Shot %: {paint_pct:.1f}%")

    return result_data

def cluster_players_def(data, n_clusters):
    """Perform k-means clustering on players based on their stats and shooting profiles."""
    # Select features for clustering
    data=data[data['MIN']>10].copy()
    features = [
        'DREB_per36', 'WEIGHT', 'Height_IN','STL_per36','BLK_per36'
    ]

    data['Height_IN']=data['HEIGHT'].str.split('-').apply(lambda x: int(x[0]) * 12 + int(x[1]))

    minutes_factor = 36 / data['MIN'].clip(lower=1)  # Avoid division by zero
    per36_cols = ['PTS', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'TOV','paint_shots_per_game','corner_3_per_game','midrange_per_game','above_break_3_per_game']
    for col in per36_cols:
        if col in data.columns:
            data[f'{col}_per36'] = data[col] * minutes_factor

    # Remove rows with missing values
    clean_data = data.dropna(subset=features)
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clean_data[features])
    if n_clusters is None:
        from sklearn.metrics import silhouette_score

        # Initialize plots for both methods
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        min_clusters=10
        max_clusters=50
        # Lists to store results
        inertia_values = []
        silhouette_scores = []

        print("Determining optimal number of clusters...")

        # Try different numbers of clusters
        for k in range(min_clusters, max_clusters + 1):
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)

            # Get inertia (for Elbow Method)
            inertia_values.append(kmeans.inertia_)

            # Get silhouette score
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            print(f"  Clusters: {k}, Inertia: {kmeans.inertia_:.0f}, Silhouette Score: {silhouette_avg:.3f}")

        # Plot Elbow Method
        ax1.plot(range(min_clusters, max_clusters + 1), inertia_values, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')

        # Calculate the "elbow" point using second derivatives
        deltas = np.diff(inertia_values)
        second_deltas = np.diff(deltas)
        elbow_index = np.argmax(second_deltas) + 1  # +1 due to double differentiation
        elbow_value = min_clusters + elbow_index

        ax1.axvline(x=elbow_value, color='r', linestyle='--', alpha=0.7)
        ax1.text(elbow_value + 0.1, ax1.get_ylim()[0] + 0.7 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                 f'Elbow point: {elbow_value}', color='r')

        # Plot Silhouette Analysis
        ax2.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'go-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')

        # Find optimal clusters from silhouette
        optimal_clusters_sil = min_clusters + np.argmax(silhouette_scores)
        ax2.axvline(x=optimal_clusters_sil, color='r', linestyle='--', alpha=0.7)
        ax2.text(optimal_clusters_sil + 0.1, ax2.get_ylim()[0] + 0.7 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]),
                 f'Optimal: {optimal_clusters_sil}', color='r')

        plt.tight_layout()
        plt.show()

        # Choose the optimal number of clusters
        # If silhouette and elbow methods disagree, prefer silhouette
        if abs(optimal_clusters_sil - elbow_value) <= 1:
            n_clusters = optimal_clusters_sil
        else:
            # If they disagree by more than 1, take silhouette with a note
            n_clusters = optimal_clusters_sil
            print(f"Note: Elbow method suggests {elbow_value} clusters, "
                  f"while Silhouette suggests {optimal_clusters_sil}. "
                  f"Choosing {n_clusters} based on Silhouette score.")

        print(f"Optimal number of clusters: {n_clusters}")

    # Perform k-means clustering with the determined number of clusters

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Add cluster labels to the dataset
    result_data = clean_data.copy()
    result_data['Cluster'] = clusters
    result_data=result_data[result_data['SEASON_ID'] == 22024]
    # Create cluster visualization
    plt.figure(figsize=(12, 8))

    # Create a scatter plot using the first two principal components
    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='viridis')
    plt.title('Player Clusters Based on Performance and Shooting Profile')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Add player names for some key points
    for idx, player in enumerate(result_data['PLAYER_NAME']):
        if result_data.iloc[idx]['PTS'] > 20:  # Label high scorers
            plt.annotate(player, (scaled_features[idx, 0], scaled_features[idx, 1]))

    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    plt.figure(figsize=(12, 8))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
    plt.title('Player Clusters Using PCA (Performance and Shooting Profile)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

    # Add player names for key players
    for idx, player in enumerate(result_data['PLAYER_NAME']):
        if result_data.iloc[idx]['PTS'] > 20 or result_data.iloc[idx]['AST'] > 7:
            plt.annotate(player, (principal_components[idx, 0], principal_components[idx, 1]))

    plt.tight_layout()
    plt.show()

    # Print cluster summaries with more detailed information
    print("\nCluster Summaries:")
    for i in range(n_clusters):
        cluster_players = result_data[result_data['Cluster'] == i]

        print(f"\nCluster {i} ({len(cluster_players)} players):")

        # Top players by different metrics
        print("Top Steals:", ", ".join(cluster_players.nlargest(3, 'STL')['PLAYER_NAME'].tolist()))
        if 'AST' in cluster_players.columns:
            print("Top Blocks:", ", ".join(cluster_players.nlargest(3, 'BLK')['PLAYER_NAME'].tolist()))

        # Shooting profile
        print(f"Avg Blocks: {cluster_players['BLK'].mean():.1f}")
        print(f"Avg Steals: {cluster_players['STL'].mean():.3f}")

        # Shot distribution
        if 'paint_shots_per_game' in cluster_players.columns and 'above_break_3_per_game' in cluster_players.columns:
            paint_pct = cluster_players['paint_shots_per_game'].mean() / (
                        cluster_players['paint_shots_per_game'].mean() +
                        cluster_players['midrange_per_game'].mean() +
                        cluster_players['above_break_3_per_game'].mean() +
                        cluster_players['corner_3_per_game'].mean()) * 100
            print(f"Paint Shot %: {paint_pct:.1f}%")

    return result_data
def get_player_box(seasontype="Regular Season"):
    # returns all player box scores for the season
    time.sleep(1)
    playerbox = LeagueGameLog(player_or_team_abbreviation='P', season_type_all_star=seasontype,
                              season=['2022-23']).get_data_frames()[0]
    time.sleep(1)
    playerbox = pandas.concat(
        [playerbox, LeagueGameLog(player_or_team_abbreviation='P', season_type_all_star=seasontype,
                                  season=['2023-24']).get_data_frames()[0]])
    time.sleep(1)
    playerbox = pandas.concat([playerbox,
         LeagueGameLog(player_or_team_abbreviation='P', season_type_all_star=seasontype,
                       season=['2024-25']).get_data_frames()[0]])

    print(playerbox.columns)
    # 10 wnba 00 nba 20 g league
    return playerbox


def box_to_avg(data):
    player_averages = data[['PLAYER_NAME', 'PLAYER_ID', 'SEASON_ID', 'MIN', 'FGM',
                            'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                            'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].groupby(
        ['PLAYER_NAME', 'PLAYER_ID', 'SEASON_ID']).mean().reset_index()
    player_averages['FG%'] = player_averages['FGM'] / player_averages['FGA']
    player_averages['FT%'] = player_averages['FTM'] / player_averages['FTA']
    player_averages['FG3%'] = player_averages['FG3M'] / player_averages['FG3A']
    return player_averages


def add_height_weight_pos(data):
    index = nba_api.stats.endpoints.playerindex.PlayerIndex().get_data_frames()[0]
    index = index[['PERSON_ID', 'POSITION', 'HEIGHT', 'WEIGHT']]
    data = index.merge(data, left_on='PERSON_ID', right_on='PLAYER_ID',how='left')
    return data


def add_shot_loc_eff(data):
    print(SynergyPlayTypes(
    player_or_team_abbreviation='P',
    season='2023-24',
    season_type_all_star='Regular Season'
).get_data_frames()[0])
    return


def add_advanced(data):  # ftr usage to ratio
    return


def basic_clustering(data):
    # Step 1: Generate or Load Data
    # Replace this with your actual dataset
    np.random.seed(42)
    data = np.random.rand(100, 2)  # 100 samples with 2 features

    # Step 2: Define Number of Clusters (k)
    num_clusters = 10

    # Step 3: Initialize and Fit the KMeans Model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)

    # Step 4: Access the Results
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Step 5: Visualize the Clusters (Optional for 2D data)
    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x', s=100, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    # Step 6: (Optional) Evaluate or Save Results
    # Print the cluster centers
    print("Cluster Centers:")
    print(cluster_centers)

    # Save the labels or other results as needed
def create_clusters():
    enhanced_data=pd.read_csv("player_data.csv")
    clustered_data = cluster_players_off(enhanced_data, None)
    clustered_data.to_csv('player_clusters_detailed.csv', index=False)
    defcluster=cluster_players_def(enhanced_data, None)
    defcluster.to_csv('def_player_clusters_detailed.csv')
    print(clustered_data)

    return
st.set_page_config(page_title="NBA Player Archetype Dashboard", layout="wide")

@st.cache_data
def load_data():
    playerbox = LeagueGameLog(
        player_or_team_abbreviation='P',
        season_type_all_star='Regular Season',
        season='2024-25'
    ).get_data_frames()[0]

    offcluster = pd.read_csv('player_clusters_detailed.csv')[['PLAYER_ID', 'Cluster']].rename(columns={'Cluster': 'OffCluster'})
    defcluster = pd.read_csv('def_player_clusters_detailed.csv')[['PLAYER_ID', 'Cluster']].rename(columns={'Cluster': 'DefCluster'})

    merged = (
        playerbox
        .merge(offcluster, on='PLAYER_ID', how='left')
        .merge(defcluster, on='PLAYER_ID', how='left')
    )
    merged['GAME_DATE'] = pd.to_datetime(merged['GAME_DATE'])
    return merged


def main():
    merged = load_data()
    st.title("🏀 NBA Player Archetype Dashboard (2024–25 Season)")

    st.sidebar.header("Filters")

    teams = sorted(merged['TEAM_ABBREVIATION'].unique())
    selected_team = st.sidebar.selectbox("Team", ["All"] + teams)

    Opponents = sorted(merged['TEAM_ABBREVIATION'].unique())
    selected_opp = st.sidebar.selectbox("Opponent", ["All"] + teams)


    players = sorted(merged['PLAYER_NAME'].unique())
    selected_player = st.sidebar.selectbox("Player", ["All"] + players)

    off_clusters = sorted(merged['OffCluster'].dropna().unique())
    selected_off_cluster = st.sidebar.selectbox("Offensive Cluster", ["All"] + off_clusters)

    def_clusters = sorted(merged['DefCluster'].dropna().unique())
    selected_def_cluster = st.sidebar.selectbox("Defensive Cluster", ["All"] + def_clusters)

    min_date, max_date = merged['GAME_DATE'].min(), merged['GAME_DATE'].max()
    selected_date = st.sidebar.date_input("Game Date Range", [min_date, max_date])

    # --- Apply Filters once ---
    df_filtered = merged.copy()

    if selected_team != "All":
        df_filtered = df_filtered[df_filtered['TEAM_ABBREVIATION'] == selected_team]
    if selected_opp != "All":
        df_filtered = df_filtered[df_filtered['MATCHUP'].apply(lambda x: x.split()[-1]) == selected_opp]
    if selected_player != "All":
        df_filtered = df_filtered[df_filtered['PLAYER_NAME'] == selected_player]
    if selected_off_cluster != "All":
        df_filtered = df_filtered[df_filtered['OffCluster'] == selected_off_cluster]
    if selected_def_cluster != "All":
        df_filtered = df_filtered[df_filtered['DefCluster'] == selected_def_cluster]
    if len(selected_date) == 2:
        df_filtered = df_filtered[
            (df_filtered['GAME_DATE'] >= pd.to_datetime(selected_date[0])) &
            (df_filtered['GAME_DATE'] <= pd.to_datetime(selected_date[1]))
            ]

    # =====================================================
    # 🔹 Tabs that share this filtered dataset
    # =====================================================
    tab1, tab2 = st.tabs(["📊 Player Stats", "🧠 Clustering Overview"])

    # =====================================================
    # 🟦 TAB 1 — Player Stats
    # =====================================================
    with tab1:
        st.markdown("### Player Stats + Archetypes")
        if df_filtered.empty:
            st.info("No games match your filters.")
        else:
            st.dataframe(
                df_filtered[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_DATE',
                             'MATCHUP', 'PTS', 'REB', 'AST','FG3A','FTA', 'OffCluster', 'DefCluster']]
                .sort_values('GAME_DATE', ascending=False)
                .reset_index(drop=True)
            )

            # --- Summary metrics ---
            st.markdown("### Summary Stats")
            avg_pts = df_filtered['PTS'].mean()
            avg_reb = df_filtered['REB'].mean()
            avg_ast = df_filtered['AST'].mean()
            avg_3pa=  df_filtered['FG3A'].mean()
            avg_FTA=  df_filtered['FTA'].mean()

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Average Points", f"{avg_pts:.1f}")
            col2.metric("Average Rebounds", f"{avg_reb:.1f}")
            col3.metric("Average Assists", f"{avg_ast:.1f}")
            col4.metric("Average 3PA", f"{avg_3pa:.1f}")
            col5.metric("Average FTA", f"{avg_FTA:.1f}")


    # =====================================================
    # 🟩 TAB 2 — Cluster Overview (with % Diff vs Team)
    # =====================================================
    with tab2:
        st.subheader("Cluster Performance Overview")
        cluster_type = st.radio("Select Cluster Type:", ["Offensive", "Defensive"], horizontal=True)

        if cluster_type == "Offensive":
            cluster_options = sorted(df_filtered['OffCluster'].dropna().unique())
            selected_cluster = st.selectbox("Select Offensive Cluster", cluster_options)
            df_cluster = merged[(merged['OffCluster'] == selected_cluster) &
                            (merged['GAME_DATE'] >= min_date) &
                            (merged['GAME_DATE'] <= max_date)]

        else:
            cluster_options = sorted(df_filtered['DefCluster'].dropna().unique())
            selected_cluster = st.selectbox("Select Defensive Cluster", cluster_options)
            df_cluster = merged[(merged['DefCluster'] == selected_cluster) &
                            (merged['GAME_DATE'] >= min_date) &
                            (merged['GAME_DATE'] <= max_date)]
        players_in_cluster=df_cluster['PLAYER_NAME'].unique()
        if df_cluster.empty:
            st.warning("No games match this cluster within your selected filters.")
        else:
            st.markdown(f"### 📈 Average Stats — Cluster {selected_cluster} 5 min minimum")
            st.markdown("**Example Players:**")
            st.markdown("- " + "\n- ".join(players_in_cluster[:5]))
            # Overall averages for this cluster
            avg_all = df_cluster[
                ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']].mean()

            # Vs selected team (based on opponent)
            df_cluster['OPP_TEAM'] = df_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
            if selected_opp != "All":
                df_vs_team = df_cluster[df_cluster['OPP_TEAM'] == selected_opp]
                if df_vs_team.empty:
                    st.info(f"No games for cluster {selected_cluster} vs {selected_opp}.")
                    df_vs_team = pd.DataFrame(columns=avg_all.index)
            else:
                df_vs_team = pd.DataFrame(columns=avg_all.index)
                comparison_df = pd.DataFrame({
                    "Overall Avg": avg_all
                }).T

            if not df_vs_team.empty:
                # Define which stats to normalize by minutes
                counting_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']

                # Convert to per-36 for season average
                df_cluster_per36 = df_cluster.copy()
                for stat in counting_stats:
                    df_cluster_per36[stat] = (df_cluster[stat] / df_cluster['MIN']) * 36

                season_avg_per36 = df_cluster_per36[counting_stats].mean()

                # Convert opponent games to per-36
                df_vs_team_per36 = df_vs_team.copy()
                for stat in counting_stats:
                    df_vs_team_per36[stat] = (df_vs_team[stat] / df_vs_team['MIN']) * 36

                # Calculate % diff for each game (now in per-36)
                if not df_vs_team.empty:
                    counting_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM',
                                      'FTA']
                    MIN_THRESHOLD = 5

                    # Filter valid games
                    df_cluster_valid = df_cluster[df_cluster['MIN'] >= MIN_THRESHOLD]
                    df_vs_team_valid = df_vs_team[df_vs_team['MIN'] >= MIN_THRESHOLD]

                    if df_vs_team_valid.empty:
                        st.info(f"No games with {MIN_THRESHOLD}+ minutes vs {selected_opp}")
                        comparison_df = pd.DataFrame({
                            "Overall Avg": avg_all
                        }).T
                    else:
                        # Vectorized per-36
                        df_cluster_per36 = df_cluster_valid[counting_stats].div(df_cluster_valid['MIN'], axis=0) * 36
                        df_vs_team_per36 = df_vs_team_valid[counting_stats].div(df_vs_team_valid['MIN'], axis=0) * 36

                        season_avg_per36 = df_cluster_per36.mean()
                        avg_vs_team_per36 = df_vs_team_per36.mean()

                        # Calculate % diff for each game, then mean across games
                        pct_diffs = ((df_vs_team_per36 - season_avg_per36) / season_avg_per36 * 100)
                        avg_pct_diff = pct_diffs.mean()  # This averages across games (axis=0 by default)

                        comparison_df = pd.DataFrame({
                            "Season Avg (per 36)": season_avg_per36,
                            f"vs {selected_opp} per 36": avg_vs_team_per36,
                            "Avg % Diff from Expected": avg_pct_diff
                        }).T
                # Option B: Convert back to per-game averages for display
                # avg_mins = df_cluster['MIN'].mean()
                # season_avg_pergame = (season_avg_per36 / 36) * avg_mins
                # avg_vs_team_pergame = (avg_vs_team_per36 / 36) * df_vs_team['MIN'].mean()
                #
                # comparison_df = pd.DataFrame({
                #     "Season Avg": season_avg_pergame,
                #     f"vs {selected_opp} Avg": avg_vs_team_pergame,
                #     "Avg % Diff from Expected": avg_pct_diff
                # }).T

            def color_cells(val, row_name):
                if row_name == '% Diff':
                    try:
                        if val > 0:
                            return 'background-color: #28A745; color: white'  # Bootstrap success green
                        elif val < 0:
                            return 'background-color: #DC3545; color: white'  # Bootstrap danger red
                    except:
                        pass
                return ''

            if '% Diff' in comparison_df.index:
                styled_df = comparison_df.style.format("{:.2f}").apply(
                    lambda row: [color_cells(val, row.name) for val in row], axis=1
                )
            else:
                styled_df = comparison_df.style.format("{:.2f}")

            st.dataframe(styled_df)

        # Summary metrics for quick glance

        st.caption(f"Games in sample: {len(df_cluster):,}")






if __name__ == '__main__':
    main()
    '''  data = get_player_box()
    print ("got data")
    player_averages = box_to_avg(data)
    print ("converetd to avg")
    print (player_averages)
    player_averages = add_height_weight_pos(player_averages)
    print ("added height weight")
    Add shooting data
    enhanced_data = enhance_player_data(player_averages)
    enhanced_data.to_csv('player_data.csv', index=False)

    print ("got more shooting")
    print (enhanced_data)
    Perform clustering'''
    # create_clusters()

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/