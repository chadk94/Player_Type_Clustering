import time

import nba_api.stats.library.data
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas
from nba_api.stats.library.parameters import SeasonTypeAllStar, PlayerOrTeamAbbreviation
from sklearn.cluster import KMeans
from nba_api.stats.endpoints import LeagueGameLog, SynergyPlayTypes, PlayerDashPtShots, shotchartdetail, ShotChartDetail
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static.players import get_players
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster
from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_scoreboard(): ##Returns todays games + opponents. Helper function for build_player_list
    board = scoreboardv2.ScoreboardV2()
    games = board.game_header.get_data_frame()
    matchups=[]
    for index,row in games.iterrows():
        awayTeam=row['VISITOR_TEAM_ID']
        homeTeam=row['HOME_TEAM_ID']
        matchups.append([awayTeam,homeTeam])
    return matchups
def build_player_list(): ##builds a list of players in todays games as well as their opponents  and whether they are home or away.
    matchups=get_scoreboard()
    nba_teams=teams.get_teams()
    playeroutput=pd.DataFrame()
    for matchup in matchups:
        awayabb = [team['abbreviation'] for team in nba_teams if team["id"] == matchup[0]]
        homeabb=[team['abbreviation'] for team in nba_teams if team["id"] == matchup[1]]
        awayid= matchup[0]
        homeid =matchup[1]
        time.sleep(1)
        awayroster=commonteamroster.CommonTeamRoster(team_id=awayid,season='2025')
        time.sleep(1)
        homeroster = commonteamroster.CommonTeamRoster(team_id=homeid,season='2025')
        awayroster=pd.DataFrame(awayroster.get_data_frames()[0].PLAYER_ID)
        awayroster['Home']=False
        awayroster['OPP']=str(homeabb)
        homeroster=pd.DataFrame(homeroster.get_data_frames()[0].PLAYER_ID)
        homeroster['Home']=True
        homeroster['OPP']=str(awayabb)
        playeroutput=pd.concat([playeroutput,awayroster,homeroster])
    return playeroutput

def get_shot_chart_data(player_id, season='2025-26'):
    """Get detailed shot chart data for a player."""
    try:
        time.sleep(1)
        player_id = str(int(float(player_id)))
        print("getting shot chart", player_id)
        shot_chart = ShotChartDetail(
            player_id=player_id,
            team_id=0,
            season_nullable='2025-26',
            season_type_all_star='Regular Season',
            context_measure_simple='FGA'  # Specify the context measure
        ).get_data_frames()[0]
        # Calculate zone-based metrics
        shot_profile = {
            'PLAYER_ID': player_id,
            'SEASON_ID': season
        }

        # Paint touches
        try:
            # Get actual games played
            games_played = shot_chart['GAME_ID'].nunique() if len(shot_chart) > 0 else 0

            if games_played == 0:
                return None

            # Paint shots
            paint_shots = shot_chart[shot_chart['SHOT_ZONE_BASIC'].isin(['Restricted Area', 'In The Paint (Non-RA)'])]
            shot_profile['paint_shots_per_game'] = len(paint_shots) / games_played
            shot_profile['paint_fg_pct'] = paint_shots['SHOT_MADE_FLAG'].mean() if len(paint_shots) > 0 else 0

            # Corner 3s - catches all variations
            corner_3s = shot_chart[shot_chart['SHOT_ZONE_BASIC'].isin(['Corner 3', 'Left Corner 3', 'Right Corner 3'])]
            shot_profile['corner_3_per_game'] = len(corner_3s) / games_played
            shot_profile['corner_3_pct'] = corner_3s['SHOT_MADE_FLAG'].mean() if len(corner_3s) > 0 else 0

            # Mid-range
            midrange = shot_chart[shot_chart['SHOT_ZONE_BASIC'] == 'Mid-Range']
            shot_profile['midrange_per_game'] = len(midrange) / games_played
            shot_profile['midrange_pct'] = midrange['SHOT_MADE_FLAG'].mean() if len(midrange) > 0 else 0

            # Above break 3s
            above_break_3s = shot_chart[shot_chart['SHOT_ZONE_BASIC'] == 'Above the Break 3']
            shot_profile['above_break_3_per_game'] = len(above_break_3s) / games_played
            shot_profile['above_break_3_pct'] = above_break_3s['SHOT_MADE_FLAG'].mean() if len(
                above_break_3s) > 0 else 0

            # Distance metrics
            shot_profile['avg_shot_distance'] = shot_chart['SHOT_DISTANCE'].mean()
            shot_profile['max_shot_distance'] = shot_chart['SHOT_DISTANCE'].max()

            # Shot clock analysis
            if 'SHOT_CLOCK' in shot_chart.columns:
                valid_shot_clock = shot_chart['SHOT_CLOCK'].notna()
                shot_profile['avg_shot_clock'] = shot_chart.loc[valid_shot_clock, 'SHOT_CLOCK'].mean()
                shot_profile['early_shot_clock_freq'] = (
                    len(shot_chart[shot_chart['SHOT_CLOCK'] >= 15]) / len(shot_chart)
                    if len(shot_chart) > 0 else 0
                )

            # Pressure shots
            if 'PERIOD' in shot_chart.columns and 'MINUTES_REMAINING' in shot_chart.columns:
                clutch_shots = shot_chart[
                    (shot_chart['PERIOD'] >= 4) &
                    (shot_chart['MINUTES_REMAINING'] <= 5)
                    ]
                shot_profile['clutch_fg_pct'] = clutch_shots['SHOT_MADE_FLAG'].mean() if len(clutch_shots) > 0 else 0

            # Add metadata for tracking
            shot_profile['games_played'] = games_played
            shot_profile['total_shots'] = len(shot_chart)

            return pd.DataFrame([shot_profile])
        except Exception as e:
            print(f"Error processing shot chart: {e}")
            return None
    except Exception as e:
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
        if idx == 0:
            all_shot_data = pd.DataFrame(get_shot_chart_data(player['PLAYER_ID'], player['SEASON_ID']))
        else:
            shot_data = get_shot_chart_data(player['PLAYER_ID'], player['SEASON_ID'])
        if idx != 0:
            if shot_data is not None:
                shot_data = pd.DataFrame(shot_data, columns=all_shot_data.columns)
                all_shot_data = pd.concat([all_shot_data, shot_data], ignore_index=True)
        time.sleep(1)
    qualified_players = qualified_players.fillna(0)
    qualified_players[['PLAYER_ID', 'SEASON_ID']] = qualified_players[['PLAYER_ID', 'SEASON_ID']].astype(int)
    all_shot_data[['PLAYER_ID', 'SEASON_ID']] = all_shot_data[['PLAYER_ID', 'SEASON_ID']].astype(int)

    enhanced_data = qualified_players.merge(
        all_shot_data,
        on=['PLAYER_ID', 'SEASON_ID'],
        how='left'
    )
    return enhanced_data


def cluster_players_off(data, n_clusters):
    """Perform k-means clustering on players based on their stats and shooting profiles."""
    # Select features for clustering
    data = data[data['MIN'] > 10].copy()
    features = [
        'PTS_per36', 'AST_per36', 'OREB_per36', 'DREB_per36', 'TOV_per36',
        'FG%', 'FG3%', 'FT%',
        'paint_shots_per_game_per36', 'paint_fg_pct',
        'corner_3_per_game_per36', 'corner_3_pct',
        'midrange_per_game_per36', 'midrange_pct',
        'above_break_3_per_game_per36', 'above_break_3_pct',
        'avg_shot_distance', 'WEIGHT', 'Height_IN',
    ]

    data['Height_IN'] = data['HEIGHT'].str.split('-').apply(lambda x: int(x[0]) * 12 + int(x[1]))

    minutes_factor = 36 / data['MIN'].clip(lower=1)  # Avoid division by zero
    per36_cols = ['PTS', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'paint_shots_per_game', 'corner_3_per_game',
                  'midrange_per_game', 'above_break_3_per_game']
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
        min_clusters = 10
        max_clusters = 50
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
        min_silhouette_improvement = 0.01  # Only add clusters if silhouette improves by at least 0.02

        # Find optimal with improvement threshold
        optimal_clusters_sil = min_clusters
        peak_score = silhouette_scores[0]

        for i, score in enumerate(silhouette_scores[1:], start=1):
            if score > peak_score + min_silhouette_improvement:
                optimal_clusters_sil = min_clusters + i
                peak_score = score

        # Update visualization to show threshold-adjusted optimal
        ax2.axvline(x=optimal_clusters_sil, color='r', linestyle='--', alpha=0.7)
        ax2.text(optimal_clusters_sil + 0.1, ax2.get_ylim()[0] + 0.7 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]),
                 f'Optimal: {optimal_clusters_sil}', color='r')

        plt.tight_layout()
        plt.show()

        # For interpretability in basketball, prefer fewer clusters when close
        if abs(optimal_clusters_sil - elbow_value) <= 1:
            n_clusters = min(optimal_clusters_sil, elbow_value)
            print(f"Elbow and Silhouette methods agree (within 1): choosing {n_clusters} clusters")
        else:
            n_clusters = optimal_clusters_sil
            print(f"Note: Elbow method suggests {elbow_value} clusters, "
                  f"while Silhouette suggests {optimal_clusters_sil}. "
                  f"Choosing {n_clusters} based on Silhouette score with improvement threshold.")

        print(f"\nFinal selection: {n_clusters} clusters")
        print(f"Optimal number of clusters: {n_clusters}")

    # Perform k-means clustering with the determined number of clusters

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Add cluster labels to the dataset
    result_data = clean_data.copy()
    result_data['Cluster'] = clusters
    result_data = result_data[result_data['SEASON_ID'] == 22025]
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
    data = data[data['MIN'] > 10].copy()
    features = [
        'DREB_per36', 'WEIGHT', 'Height_IN', 'STL_per36', 'BLK_per36'
    ]

    data['Height_IN'] = data['HEIGHT'].str.split('-').apply(lambda x: int(x[0]) * 12 + int(x[1]))

    minutes_factor = 36 / data['MIN'].clip(lower=1)  # Avoid division by zero
    per36_cols = ['PTS', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'paint_shots_per_game', 'corner_3_per_game',
                  'midrange_per_game', 'above_break_3_per_game']
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
        min_clusters = 10
        max_clusters = 50
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
    result_data = result_data[result_data['SEASON_ID'] == 22025]
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
                              season=['2023-24']).get_data_frames()[0]
    time.sleep(1)
    playerbox = pandas.concat(
        [playerbox, LeagueGameLog(player_or_team_abbreviation='P', season_type_all_star=seasontype,
                                  season=['2024-25']).get_data_frames()[0]])
    time.sleep(1)
    playerbox = pandas.concat([playerbox,
                               LeagueGameLog(player_or_team_abbreviation='P', season_type_all_star=seasontype,
                                             season=['2025-26']).get_data_frames()[0]])

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
    data = index.merge(data, left_on='PERSON_ID', right_on='PLAYER_ID', how='left')
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
    data = get_player_box()
    print ("got data")
    player_averages = box_to_avg(data)
    print ("converetd to avg")
    print (player_averages)
    player_averages = add_height_weight_pos(player_averages)
    print ("added height weight")
    enhanced_data = enhance_player_data(player_averages)
    enhanced_data.to_csv('player_data.csv', index=False)
    enhanced_data = pd.read_csv("player_data.csv")
    clustered_data = cluster_players_off(enhanced_data, None)
    clustered_data.to_csv('player_clusters_detailed.csv', index=False)
    defcluster = cluster_players_def(enhanced_data, None)
    defcluster.to_csv('def_player_clusters_detailed.csv')
    print(clustered_data)

    return


st.set_page_config(page_title="NBA Player Archetype Dashboard", layout="wide")


@st.cache_data
def load_data():
    playerbox = LeagueGameLog(
        player_or_team_abbreviation='P',
        season_type_all_star='Regular Season',
        season='2025-26'
    ).get_data_frames()[0]

    offcluster = pd.read_csv('player_clusters_detailed.csv')[['PLAYER_ID', 'Cluster']].rename(
        columns={'Cluster': 'OffCluster'})
    defcluster = pd.read_csv('def_player_clusters_detailed.csv')[['PLAYER_ID', 'Cluster']].rename(
        columns={'Cluster': 'DefCluster'})

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

    # Initialize session state for filters if not exists
    if 'team_filter' not in st.session_state:
        st.session_state.team_filter = "All"
    if 'opp_filter' not in st.session_state:
        st.session_state.opp_filter = "All"
    if 'player_filter' not in st.session_state:
        st.session_state.player_filter = "All"
    if 'off_cluster_filter' not in st.session_state:
        st.session_state.off_cluster_filter = "All"
    if 'def_cluster_filter' not in st.session_state:
        st.session_state.def_cluster_filter = "All"

    # Function to clear all filters (used as callback)
    def clear_all_filters():
        st.session_state.team_filter = "All"
        st.session_state.opp_filter = "All"
        st.session_state.player_filter = "All"
        st.session_state.off_cluster_filter = "All"
        st.session_state.def_cluster_filter = "All"

    # Handle individual clear button clicks BEFORE creating widgets
    if 'clear_team' in st.session_state and st.session_state.clear_team:
        st.session_state.team_filter = "All"
    if 'clear_opp' in st.session_state and st.session_state.clear_opp:
        st.session_state.opp_filter = "All"
    if 'clear_player' in st.session_state and st.session_state.clear_player:
        st.session_state.player_filter = "All"
    if 'clear_off' in st.session_state and st.session_state.clear_off:
        st.session_state.off_cluster_filter = "All"
    if 'clear_def' in st.session_state and st.session_state.clear_def:
        st.session_state.def_cluster_filter = "All"

    # Clear all filters button (BEFORE widgets, using on_click callback)
    st.sidebar.button("🔄 Clear All Filters", on_click=clear_all_filters)

    # Team filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        teams = sorted(merged['TEAM_ABBREVIATION'].unique())
        selected_team = st.selectbox("Team", ["All"] + teams, key='team_filter')
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_team")

    # Opponent filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        opponents = sorted(merged['TEAM_ABBREVIATION'].unique())
        selected_opp = st.selectbox("Opponent", ["All"] + opponents, key='opp_filter')
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_opp")

    # Player filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        players = sorted(merged['PLAYER_NAME'].unique())
        selected_player = st.selectbox("Player", ["All"] + players, key='player_filter')
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_player")

    # Offensive Cluster filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        off_clusters = sorted(merged['OffCluster'].dropna().unique())
        selected_off_cluster = st.selectbox("Offensive Cluster", ["All"] + off_clusters, key='off_cluster_filter')
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_off")

    # Defensive Cluster filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        def_clusters = sorted(merged['DefCluster'].dropna().unique())
        selected_def_cluster = st.selectbox("Defensive Cluster", ["All"] + def_clusters, key='def_cluster_filter')
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_def")

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
    tab1, tab2, tab3,tab4 = st.tabs(["📊 Player Stats", "🧠 Clustering Overview","Team Best/Worst Cluster Performance","🔥 Today's Best Matchup Opportunities"])

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
                             'MATCHUP', 'MIN','PTS', 'REB', 'AST', 'FG3A', 'FTA', 'OffCluster', 'DefCluster']]
                .sort_values('GAME_DATE', ascending=False)
                .reset_index(drop=True)
            )

            # --- Summary metrics ---
            st.markdown("### Summary Stats")
            avg_pts = df_filtered['PTS'].mean()
            avg_reb = df_filtered['REB'].mean()
            avg_ast = df_filtered['AST'].mean()
            avg_3pa = df_filtered['FG3A'].mean()
            avg_FTA = df_filtered['FTA'].mean()

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
        default_off_cluster = None
        default_def_cluster = None

        # Auto-select clusters based on selected player
        if selected_player != "All":
            player_data = merged[merged['PLAYER_NAME'] == selected_player]
            if not player_data.empty:
                player_data = player_data.iloc[0]
                default_off_cluster = player_data['OffCluster'] if pd.notna(player_data['OffCluster']) else None
                default_def_cluster = player_data['DefCluster'] if pd.notna(player_data['DefCluster']) else None
            else:
                default_off_cluster = None
                default_def_cluster = None

        # Allow user to select both offensive and defensive clusters
        col1, col2 = st.columns(2)

        with col1:
            off_cluster_options = sorted(merged['OffCluster'].dropna().unique())
            if default_off_cluster is not None and default_off_cluster in off_cluster_options:
                default_off_index = off_cluster_options.index(default_off_cluster)
            else:
                default_off_index = 0
            selected_off_analysis = st.selectbox(
                "Offensive Cluster for Analysis",
                off_cluster_options,
                index=default_off_index,
                key=f"off_analysis_{selected_player}"  # Add key with player name
            )

        with col2:
            def_cluster_options = sorted(merged['DefCluster'].dropna().unique())
            if default_def_cluster is not None and default_def_cluster in def_cluster_options:
                default_def_index = def_cluster_options.index(default_def_cluster)
            else:
                default_def_index = 0
            selected_def_analysis = st.selectbox(
                "Defensive Cluster for Analysis",
                def_cluster_options,
                index=default_def_index,
                key=f"def_analysis_{selected_player}"  # Add key with player name
            )

        # Get data for both clusters
        df_off_cluster = merged[(merged['OffCluster'] == selected_off_analysis) &
                                (merged['GAME_DATE'] >= min_date) &
                                (merged['GAME_DATE'] <= max_date)]

        df_def_cluster = merged[(merged['DefCluster'] == selected_def_analysis) &
                                (merged['GAME_DATE'] >= min_date) &
                                (merged['GAME_DATE'] <= max_date)]

        if df_off_cluster.empty or df_def_cluster.empty:
            st.warning("No games match the selected clusters within your filters.")
        else:
            st.markdown(f"### 📈 Cluster Analysis")
            st.markdown(
                f"**Offensive Cluster {selected_off_analysis}** | **Defensive Cluster {selected_def_analysis}**")

            # Show example players from both clusters
            off_players = df_off_cluster['PLAYER_NAME'].unique()[:5]
            def_players = df_def_cluster['PLAYER_NAME'].unique()[:5]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Example Players (Offensive):**")
                st.markdown("- " + "\n- ".join(off_players))
            with col2:
                st.markdown("**Example Players (Defensive):**")
                st.markdown("- " + "\n- ".join(def_players))

            # Define stat categories
            offensive_stats = ['PTS', 'AST','OREB',  'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA','TOV']
            defensive_stats = ['DREB', 'REB', 'STL', 'BLK',  'PF']
            all_counting_stats = offensive_stats + defensive_stats

            MIN_THRESHOLD = 5

            # Calculate per-36 averages for OFFENSIVE cluster (all players in this cluster)
            df_off_valid = df_off_cluster[df_off_cluster['MIN'] >= MIN_THRESHOLD]
            df_off_per36 = df_off_valid[offensive_stats].div(df_off_valid['MIN'], axis=0) * 36
            season_avg_per36_off = df_off_per36.mean()

            # Calculate per-36 averages for DEFENSIVE cluster (all players in this cluster)
            df_def_valid = df_def_cluster[df_def_cluster['MIN'] >= MIN_THRESHOLD]
            df_def_per36 = df_def_valid[defensive_stats].div(df_def_valid['MIN'], axis=0) * 36
            season_avg_per36_def = df_def_per36.mean()

            # If opponent is selected, calculate % differences
            if selected_opp != "All":
                # Add opponent team column
                df_off_cluster.loc[:, 'OPP_TEAM'] = df_off_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                df_def_cluster.loc[:, 'OPP_TEAM'] = df_def_cluster['MATCHUP'].apply(lambda x: x.split()[-1])

                # Filter by opponent
                df_off_vs_team = df_off_cluster[df_off_cluster['OPP_TEAM'] == selected_opp]
                df_def_vs_team = df_def_cluster[df_def_cluster['OPP_TEAM'] == selected_opp]

                # Filter by minutes
                df_off_vs_team_valid = df_off_vs_team[df_off_vs_team['MIN'] >= MIN_THRESHOLD]
                df_def_vs_team_valid = df_def_vs_team[df_def_vs_team['MIN'] >= MIN_THRESHOLD]

                # Initialize variables - calculate offensive and defensive independently
                pct_diff_off = pd.Series(0, index=offensive_stats)
                pct_diff_def = pd.Series(0, index=defensive_stats)
                has_off_history = False
                has_def_history = False

                # Calculate offensive cluster % diff if data exists
                prior_mean = 0.0  # Prior belief: no difference from season average (0%)
                prior_weight = 3.0  # Weight of the prior

                # Calculate offensive cluster % diff with game-by-game Bayesian approach
                if not df_off_vs_team_valid.empty:
                    has_off_history = True

                    # Sort by game date to process chronologically
                    df_off_vs_team_valid_sorted = df_off_vs_team_valid.sort_values('GAME_DATE')

                    # Calculate per-36 stats for each game
                    df_off_vs_team_per36 = df_off_vs_team_valid_sorted[offensive_stats].div(
                        df_off_vs_team_valid_sorted['MIN'], axis=0) * 36

                    # Calculate percentage difference for each game
                    game_pct_diffs_off = (
                            (df_off_vs_team_per36 - season_avg_per36_off[offensive_stats]) /
                            season_avg_per36_off[offensive_stats] * 100
                    )

                    # Bayesian sequential update for EACH STAT
                    pct_diff_off = pd.Series(index=offensive_stats, dtype=float)

                    for stat in offensive_stats:
                        bayesian_estimate = prior_mean
                        total_weight = prior_weight

                        for game_pct_diff in game_pct_diffs_off[stat].values:
                            bayesian_estimate = (total_weight * bayesian_estimate + game_pct_diff) / (total_weight + 1)
                            total_weight += 1

                        pct_diff_off[stat] = bayesian_estimate

                # Calculate defensive cluster % diff with game-by-game Bayesian approach
                if not df_def_vs_team_valid.empty:
                    has_def_history = True

                    # Sort by game date to process chronologically
                    df_def_vs_team_valid_sorted = df_def_vs_team_valid.sort_values('GAME_DATE')

                    # Calculate per-36 stats for each game
                    df_def_vs_team_per36 = df_def_vs_team_valid_sorted[defensive_stats].div(
                        df_def_vs_team_valid_sorted['MIN'], axis=0) * 36

                    # Calculate percentage difference for each game
                    game_pct_diffs_def = (
                            (df_def_vs_team_per36 - season_avg_per36_def[defensive_stats]) /
                            season_avg_per36_def[defensive_stats] * 100
                    )

                    # Bayesian sequential update for EACH STAT
                    pct_diff_def = pd.Series(index=defensive_stats, dtype=float)

                    for stat in defensive_stats:
                        bayesian_estimate = prior_mean
                        total_weight = prior_weight

                        for game_pct_diff in game_pct_diffs_def[stat].values:
                            bayesian_estimate = (total_weight * bayesian_estimate + game_pct_diff) / (total_weight + 1)
                            total_weight += 1

                        pct_diff_def[stat] = bayesian_estimate

                # Combine the % diffs
                avg_pct_diff_combined = pd.Series(index=all_counting_stats, dtype=float)
                for stat in offensive_stats:
                    if stat in pct_diff_off.index:
                        avg_pct_diff_combined[stat] = pct_diff_off[stat]
                for stat in defensive_stats:
                    if stat in pct_diff_def.index:
                        avg_pct_diff_combined[stat] = pct_diff_def[stat]

                # Show comparison table if we have any historical data
                if has_off_history or has_def_history:
                    st.markdown(f"### 📊 Stats Comparison vs {selected_opp}")

                    # Build comparison table with available data
                    comparison_data = {}

                    if has_off_history and has_def_history:
                        # Calculate season averages and projections for offensive stats
                        season_avg_off = season_avg_per36_off[offensive_stats].copy()
                        proj_vs_opp_off = season_avg_per36_off[offensive_stats].copy()

                        # Calculate season averages and projections for defensive stats
                        season_avg_def = season_avg_per36_def[defensive_stats].copy()
                        proj_vs_opp_def = season_avg_per36_def[defensive_stats].copy()

                        # Handle offensive stats
                        for stat in offensive_stats:
                            if stat == 'OREB':
                                # Just normal projection, but we'll use it for REB later
                                pct_change = pct_diff_off[stat] if stat in pct_diff_off.index else 0
                                proj_vs_opp_off.loc[stat] = season_avg_per36_off[stat] * (1 + pct_change / 100)
                            else:
                                pct_change = pct_diff_off[stat] if stat in pct_diff_off.index else 0
                                proj_vs_opp_off.loc[stat] = season_avg_per36_off[stat] * (1 + pct_change / 100)

                        # Handle defensive stats
                        for stat in defensive_stats:
                            if stat == 'REB':
                                # Season avg: OREB (from offensive) + DREB (from defensive)
                                season_avg_def.loc[stat] = season_avg_per36_off['OREB'] + season_avg_per36_def['DREB']

                                # Projection: use adjusted OREB from offensive + adjusted DREB from defensive
                                dreb_pct = pct_diff_def['DREB'] if 'DREB' in pct_diff_def.index else 0
                                oreb_pct = pct_diff_off['OREB'] if 'OREB' in pct_diff_off.index else 0

                                proj_vs_opp_def.loc[stat] = (
                                        season_avg_per36_def['DREB'] * (1 + dreb_pct / 100) +
                                        season_avg_per36_off['OREB'] * (1 + oreb_pct / 100)
                                )
                                avg_pct_diff_combined['REB'] = ((proj_vs_opp_def.loc[stat] / season_avg_def.loc[
                                    stat]) - 1) * 100

                            elif stat == 'DREB':
                                # Normal projection for DREB
                                pct_change = pct_diff_def[stat] if stat in pct_diff_def.index else 0
                                proj_vs_opp_def.loc[stat] = season_avg_per36_def[stat] * (1 + pct_change / 100)
                            else:
                                # Normal projection for other defensive stats
                                pct_change = pct_diff_def[stat] if stat in pct_diff_def.index else 0
                                proj_vs_opp_def.loc[stat] = season_avg_per36_def[stat] * (1 + pct_change / 100)

                        comparison_data = {
                            "Season Avg (per 36)": pd.concat([season_avg_off, season_avg_def]),
                            f"vs {selected_opp} (per 36)": pd.concat([proj_vs_opp_off, proj_vs_opp_def]),
                            "Avg % Diff": avg_pct_diff_combined
                        }

                    elif has_off_history:
                        st.info("⚠️ No defensive cluster history vs this opponent. Showing offensive stats only.")
                        comparison_data = {
                            "Season Avg (per 36)": season_avg_per36_off[offensive_stats],
                            f"vs {selected_opp} (per 36)": season_avg_per36_off[offensive_stats] * (1 + pct_diff_off / 100),
                            "Avg % Diff": pct_diff_off
                        }
                    elif has_def_history:
                        st.info("⚠️ No offensive cluster history vs this opponent. Showing defensive stats only.")
                        comparison_data = {
                            "Season Avg (per 36)": season_avg_per36_def[defensive_stats],
                            f"vs {selected_opp} (per 36)": season_avg_per36_def[defensive_stats] * (1 + pct_diff_def / 100)
,
                            "Avg % Diff": pct_diff_def
                        }

                    comparison_df = pd.DataFrame(comparison_data).T

                    def color_cells(val, row_name):
                        if row_name == 'Avg % Diff':
                            try:
                                if val > 0:
                                    return 'background-color: #28A745; color: white'
                                elif val < 0:
                                    return 'background-color: #DC3545; color: white'
                            except:
                                pass
                        return ''

                    styled_df = comparison_df.style.format("{:.2f}").apply(
                        lambda row: [color_cells(val, row.name) for val in row], axis=1
                    )

                    st.dataframe(styled_df)
                else:
                    st.info(
                        f"⚠️ No historical data for either cluster vs {selected_opp} with {MIN_THRESHOLD}+ minutes. Showing season averages without adjustments.")

                # =====================================================
                # 🟨 Projected Stats Section - ALWAYS SHOW
                # =====================================================
                st.markdown("---")
                st.markdown(f"### 🎯 Projected Stats vs {selected_opp}")

                if has_off_history and has_def_history:
                    st.markdown("*Season averages adjusted by cluster's historical % difference vs this opponent*")
                elif has_off_history:
                    st.markdown(
                        "*Offensive stats adjusted by cluster history. Defensive stats show season averages (no cluster history).*")
                elif has_def_history:
                    st.markdown(
                        "*Defensive stats adjusted by cluster history. Offensive stats show season averages (no cluster history).*")
                else:
                    st.markdown("*Showing season averages (no historical cluster data vs this opponent available)*")

                st.caption(
                    f"**Note:** Offensive stats use Offensive Cluster {selected_off_analysis}, Defensive stats use Defensive Cluster {selected_def_analysis}")

                # Get ALL unique players from the season who have the required clusters
                # We need to aggregate from merged dataset to get season averages per player

                # Group by player to get season averages for offensive stats
                # Group by player to get season averages for offensive stats
                off_players_season = merged[
                    (merged['OffCluster'] == selected_off_analysis) &
                    (merged['GAME_DATE'] >= min_date) &
                    (merged['GAME_DATE'] <= max_date) &
                    (merged['MIN'] >= MIN_THRESHOLD)
                    ].groupby('PLAYER_NAME').agg({
                    'MIN': 'mean',
                    **{stat: 'mean' for stat in offensive_stats}
                }).reset_index()

                # Group by player to get season averages for defensive stats
                def_players_season = merged[
                    (merged['DefCluster'] == selected_def_analysis) &
                    (merged['GAME_DATE'] >= min_date) &
                    (merged['GAME_DATE'] <= max_date) &
                    (merged['MIN'] >= MIN_THRESHOLD)
                    ].groupby('PLAYER_NAME').agg({
                    'MIN': 'mean',
                    **{stat: 'mean' for stat in defensive_stats}
                }).reset_index()

                # NEW: Get player season averages WITHOUT minutes filter for projections
                off_players_no_min_filter = merged[
                    (merged['OffCluster'] == selected_off_analysis) &
                    (merged['GAME_DATE'] >= min_date) &
                    (merged['GAME_DATE'] <= max_date)
                    # NO MIN_THRESHOLD here
                    ].groupby('PLAYER_NAME').agg({
                    'MIN': 'mean',
                    **{stat: 'mean' for stat in offensive_stats}
                }).reset_index()

                def_players_no_min_filter = merged[
                    (merged['DefCluster'] == selected_def_analysis) &
                    (merged['GAME_DATE'] >= min_date) &
                    (merged['GAME_DATE'] <= max_date)
                    # NO MIN_THRESHOLD here
                    ].groupby('PLAYER_NAME').agg({
                    'MIN': 'mean',
                    **{stat: 'mean' for stat in defensive_stats}
                }).reset_index()

                # Merge to get players who have EITHER cluster (outer join)
                # This way each player brings their own cluster's stats
                all_players = off_players_season.merge(
                    def_players_season,
                    on='PLAYER_NAME',
                    how='outer',
                    suffixes=('_off', '_def')
                )

                # NEW: Merge the no-filter versions for projections
                all_players_for_projection = off_players_no_min_filter.merge(
                    def_players_no_min_filter,
                    on='PLAYER_NAME',
                    how='outer',
                    suffixes=('_off', '_def')
                )

                # Calculate median minutes from last 10 games for the slider
                if selected_player != "All":
                    # Get the specific player's last 10 games
                    player_games = merged[merged['PLAYER_NAME'] == selected_player].copy()
                    player_games = player_games.sort_values('GAME_DATE', ascending=False).head(10)

                    if not player_games.empty:
                        median_min = player_games['MIN'].median()
                        default_min = round(median_min, 1)
                    else:
                        default_min = 30.0  # fallback default

                    # Add minutes slider
                    st.subheader("Adjust Minutes")
                    selected_minutes = st.slider(
                        "Minutes per game:",
                        min_value=0.0,
                        max_value=48.0,
                        value=default_min,
                        step=0.5,
                        help=f"Default is median of last 10 games ({default_min:.1f} min)"
                    )

                    # Calculate the minutes multiplier
                    minutes_multiplier = selected_minutes / default_min if default_min > 0 else 1.0
                else:
                    # For "All" players, use their season average (no adjustment)
                    minutes_multiplier = 1.0
                    selected_minutes = None

                # CHANGED: Use all_players_for_projection instead of all_players
                if all_players_for_projection.empty:
                    st.warning("No players found in the selected clusters.")
                else:
                    # Calculate projections for each player

                    players_to_project = []

                    # CHANGED: Iterate over all_players_for_projection
                    for idx, player_row in all_players_for_projection.iterrows():
                        player_name = player_row['PLAYER_NAME']

                        projected_row = {
                            'PLAYER_NAME': player_name,
                            'OffCluster': selected_off_analysis if not pd.isna(player_row.get('MIN_off')) else None,
                            'DefCluster': selected_def_analysis if not pd.isna(player_row.get('MIN_def')) else None
                        }

                        # Use whichever MIN is available (prefer off, then def)
                        if not pd.isna(player_row.get('MIN_off')):
                            base_minutes = player_row['MIN_off']
                        elif not pd.isna(player_row.get('MIN_def')):
                            base_minutes = player_row['MIN_def']
                        else:
                            continue
                        if selected_player != "All" and player_name == selected_player:
                            projected_row['AVG_MIN'] = selected_minutes
                            actual_multiplier = minutes_multiplier
                        else:
                            projected_row['AVG_MIN'] = base_minutes
                            actual_multiplier = 1.0

                        # Process offensive stats if player has offensive cluster
                        if not pd.isna(player_row.get('MIN_off')):
                            for stat in offensive_stats:
                                season_avg = player_row[stat]

                                # Apply the % change from the offensive cluster's performance vs this opponent
                                pct_change = avg_pct_diff_combined[stat] if stat in avg_pct_diff_combined.index else 0
                                projected_value = season_avg * (1 + pct_change / 100) * actual_multiplier

                                projected_row[f'{stat}_Season'] = season_avg
                                projected_row[f'{stat}_Projected'] = projected_value
                                projected_row[f'{stat}_Diff'] = projected_value - season_avg
                        else:
                            # Player doesn't have offensive cluster, set to NaN or 0
                            for stat in offensive_stats:
                                projected_row[f'{stat}_Season'] = 0
                                projected_row[f'{stat}_Projected'] = 0
                                projected_row[f'{stat}_Diff'] = 0

                        # Process defensive stats if player has defensive cluster
                        if not pd.isna(player_row.get('MIN_def')):
                            for stat in defensive_stats:
                                if stat == 'REB':
                                    # Handle rebounds specially using OREB and DREB components
                                    season_avg = player_row['DREB'] + player_row['OREB']

                                    # Apply matchup adjustments to each component separately
                                    dreb_pct = avg_pct_diff_combined[
                                        'DREB'] if 'DREB' in avg_pct_diff_combined.index else 0
                                    oreb_pct = avg_pct_diff_combined[
                                        'OREB'] if 'OREB' in avg_pct_diff_combined.index else 0

                                    projected_value = (
                                                              player_row['DREB'] * (1 + dreb_pct / 100) +
                                                              player_row['OREB'] * (1 + oreb_pct / 100)
                                                      ) * actual_multiplier
                                else:
                                    # Handle all other stats normally
                                    season_avg = player_row[stat]
                                    pct_change = avg_pct_diff_combined[
                                        stat] if stat in avg_pct_diff_combined.index else 0
                                    projected_value = season_avg * (1 + pct_change / 100) * actual_multiplier

                                # Store results (same for all stats)
                                projected_row[f'{stat}_Season'] = season_avg
                                projected_row[f'{stat}_Projected'] = projected_value
                                projected_row[f'{stat}_Diff'] = projected_value - season_avg
                        else:
                            # Player doesn't have defensive cluster, set to NaN or 0
                            for stat in defensive_stats:
                                projected_row[f'{stat}_Season'] = 0
                                projected_row[f'{stat}_Projected'] = 0
                                projected_row[f'{stat}_Diff'] = 0

                        players_to_project.append(projected_row)

                    if not players_to_project:
                        st.warning("No valid players found for projections.")
                    else:
                        projected_df = pd.DataFrame(players_to_project)

                        # Display selector for stats
                        display_stats = st.multiselect(
                            "Select stats to display:",
                            all_counting_stats,
                            default=['PTS', 'REB', 'AST', 'FG3A', 'FTA', 'STL', 'BLK']
                        )

                        if display_stats:
                            # Create display dataframe
                            display_columns = ['PLAYER_NAME', 'AVG_MIN', 'OffCluster', 'DefCluster']
                            for stat in display_stats:
                                display_columns.extend([f'{stat}_Season', f'{stat}_Projected', f'{stat}_Diff'])

                            display_df = projected_df[display_columns].copy()

                            # Filter by selected player if not "All"
                            if selected_player != "All":
                                display_df = display_df[display_df['PLAYER_NAME'] == selected_player]

                            if display_df.empty:
                                st.warning(f"Player {selected_player} not found in the selected clusters.")
                            else:
                                # Rename columns
                                rename_dict = {
                                    'PLAYER_NAME': 'Player',
                                    'AVG_MIN': 'MPG',
                                    'OffCluster': 'Off',
                                    'DefCluster': 'Def'
                                }
                                for stat in display_stats:
                                    rename_dict[f'{stat}_Season'] = f'{stat} (Season)'
                                    rename_dict[f'{stat}_Projected'] = f'{stat} (Proj.)'
                                    rename_dict[f'{stat}_Diff'] = f'{stat} (±)'

                                display_df = display_df.rename(columns=rename_dict)

                                # Sort by projected points if available
                                if 'PTS (Proj.)' in display_df.columns:
                                    display_df = display_df.sort_values('PTS (Proj.)', ascending=False)

                                # Style the dataframe
                                def highlight_diff(val, col_name):
                                    if '(±)' in col_name:
                                        try:
                                            if val > 0:
                                                return 'background-color: #D4EDDA; color: #155724'
                                            elif val < 0:
                                                return 'background-color: #F8D7DA; color: #721C24'
                                        except:
                                            pass
                                    return ''

                                styled_proj = display_df.style.format({
                                    col: "{:.1f}" for col in display_df.columns if col not in ['Player', 'Off', 'Def']
                                }).apply(
                                    lambda row: [highlight_diff(val, col) for col, val in zip(display_df.columns, row)],
                                    axis=1
                                )

                                st.dataframe(styled_proj, use_container_width=True)

                                st.caption(
                                    f"💡 Offensive stats ({', '.join(offensive_stats)}) use ALL players in Off Cluster {selected_off_analysis}")
                                st.caption(
                                    f"💡 Defensive stats ({', '.join(defensive_stats)}) use ALL players in Def Cluster {selected_def_analysis}")
                        else:
                            st.info("Select at least one stat to display projections.")

            else:
                st.info("Select an opponent to see cluster performance analysis and projections.")
            games_off = len(df_off_cluster) if selected_opp == "All" else len(
                df_off_cluster[df_off_cluster['OPP_TEAM'] == selected_opp])
            st.caption(f"Games in offensive cluster sample: {games_off:,}")

            # For defensive cluster
            games_def = len(df_def_cluster) if selected_opp == "All" else len(
                df_def_cluster[df_def_cluster['OPP_TEAM'] == selected_opp])
            st.caption(f"Games in defensive cluster sample: {games_def:,}")


# =====================================================
# 🟦 TAB 3 — Team Cluster Matchup Analysis
# =====================================================
    with tab3:
        st.subheader("Team Cluster Matchup Analysis")

        if selected_opp == "All":
            st.warning("⚠️ Please select a specific opponent team to view cluster matchup analysis.")
        else:
            st.markdown(f"### 📊 Best & Worst Clusters vs {selected_opp}")
            st.markdown(f"*Showing which clusters perform best/worst against {selected_opp}*")

            # Define stat categories (same as Tab 2)
            offensive_stats = ['PTS', 'AST', 'OREB', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV']
            defensive_stats = ['DREB', 'REB', 'STL', 'BLK', 'PF']

            MIN_THRESHOLD = 5
            prior_mean = 0.0
            prior_weight = 3.0

            # Get all unique clusters
            all_off_clusters = sorted(merged['OffCluster'].dropna().unique())
            all_def_clusters = sorted(merged['DefCluster'].dropna().unique())

            # =====================================================
            # Calculate % diff for each offensive cluster vs selected opponent
            # =====================================================
            off_cluster_results = {}

            for cluster in all_off_clusters:
                df_cluster = merged[(merged['OffCluster'] == cluster) &
                                    (merged['GAME_DATE'] >= min_date) &
                                    (merged['GAME_DATE'] <= max_date)]

                if df_cluster.empty:
                    continue

                # Calculate season averages
                df_valid = df_cluster[df_cluster['MIN'] >= MIN_THRESHOLD]
                df_per36 = df_valid[offensive_stats].div(df_valid['MIN'], axis=0) * 36
                season_avg_per36 = df_per36.mean()

                # Filter vs opponent
                df_cluster.loc[:, 'OPP_TEAM'] = df_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                df_vs_team = df_cluster[df_cluster['OPP_TEAM'] == selected_opp]
                df_vs_team_valid = df_vs_team[df_vs_team['MIN'] >= MIN_THRESHOLD]

                if not df_vs_team_valid.empty:
                    # Sort by game date
                    df_vs_team_valid_sorted = df_vs_team_valid.sort_values('GAME_DATE')

                    # Calculate per-36 for each game
                    df_vs_team_per36 = df_vs_team_valid_sorted[offensive_stats].div(
                        df_vs_team_valid_sorted['MIN'], axis=0) * 36

                    # Calculate percentage difference for each game
                    game_pct_diffs = (
                            (df_vs_team_per36 - season_avg_per36[offensive_stats]) /
                            season_avg_per36[offensive_stats] * 100
                    )

                    # Bayesian sequential update for each stat
                    pct_diffs = {}
                    for stat in offensive_stats:
                        bayesian_estimate = prior_mean
                        total_weight = prior_weight

                        for game_pct_diff in game_pct_diffs[stat].values:
                            bayesian_estimate = (total_weight * bayesian_estimate + game_pct_diff) / (total_weight + 1)
                            total_weight += 1

                        pct_diffs[stat] = bayesian_estimate

                    # Get player examples
                    cluster_players = df_cluster['PLAYER_NAME'].unique()
                    player_examples = list(np.random.choice(cluster_players, min(3, len(cluster_players)), replace=False))

                    off_cluster_results[cluster] = {
                        'pct_diffs': pct_diffs,
                        'players': player_examples
                    }

            # =====================================================
            # Calculate % diff for each defensive cluster vs selected opponent
            # =====================================================
            def_cluster_results = {}

            for cluster in all_def_clusters:
                df_cluster = merged[(merged['DefCluster'] == cluster) &
                                    (merged['GAME_DATE'] >= min_date) &
                                    (merged['GAME_DATE'] <= max_date)]

                if df_cluster.empty:
                    continue

                # Calculate season averages
                df_valid = df_cluster[df_cluster['MIN'] >= MIN_THRESHOLD]
                df_per36 = df_valid[defensive_stats].div(df_valid['MIN'], axis=0) * 36
                season_avg_per36 = df_per36.mean()

                # Filter vs opponent
                df_cluster.loc[:, 'OPP_TEAM'] = df_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                df_vs_team = df_cluster[df_cluster['OPP_TEAM'] == selected_opp]
                df_vs_team_valid = df_vs_team[df_vs_team['MIN'] >= MIN_THRESHOLD]

                if not df_vs_team_valid.empty:
                    # Sort by game date
                    df_vs_team_valid_sorted = df_vs_team_valid.sort_values('GAME_DATE')

                    # Calculate per-36 for each game
                    df_vs_team_per36 = df_vs_team_valid_sorted[defensive_stats].div(
                        df_vs_team_valid_sorted['MIN'], axis=0) * 36

                    # Calculate percentage difference for each game
                    game_pct_diffs = (
                            (df_vs_team_per36 - season_avg_per36[defensive_stats]) /
                            season_avg_per36[defensive_stats] * 100
                    )

                    # Bayesian sequential update for each stat
                    pct_diffs = {}
                    for stat in defensive_stats:
                        bayesian_estimate = prior_mean
                        total_weight = prior_weight

                        for game_pct_diff in game_pct_diffs[stat].values:
                            bayesian_estimate = (total_weight * bayesian_estimate + game_pct_diff) / (total_weight + 1)
                            total_weight += 1

                        pct_diffs[stat] = bayesian_estimate

                    # Get player examples
                    cluster_players = df_cluster['PLAYER_NAME'].unique()
                    player_examples = list(np.random.choice(cluster_players, min(3, len(cluster_players)), replace=False))

                    def_cluster_results[cluster] = {
                        'pct_diffs': pct_diffs,
                        'players': player_examples
                    }


            # =====================================================
            # Helper function for heatmap coloring
            # =====================================================
            def get_heatmap_color(pct_value):
                """
                Returns RGB color string based on percentage value
                Scale from -25% (red) to +25% (green)
                """
                # Clamp value between -25 and 25
                clamped = max(-25, min(25, pct_value))

                # Normalize to 0-1 scale
                normalized = (clamped + 25) / 50

                # Red to Green gradient
                if normalized < 0.5:
                    # Red to Yellow (increase green)
                    r = 255
                    g = int(255 * (normalized * 2))
                    b = 0
                else:
                    # Yellow to Green (decrease red)
                    r = int(255 * (2 - normalized * 2))
                    g = 255
                    b = 0

                return f'background-color: rgb({r}, {g}, {b}); color: black; font-weight: bold;'


            # =====================================================
            # Build the results table
            # =====================================================
            if not off_cluster_results and not def_cluster_results:
                st.warning(f"No cluster data available vs {selected_opp} with {MIN_THRESHOLD}+ minutes.")
            else:
                # Store cluster-to-players mapping for display below table
                cluster_players_map = {}

                # Create table data
                table_rows = []
                pct_columns = []  # Track which columns contain percentages for styling

                # Process offensive stats
                for stat in offensive_stats:
                    if not off_cluster_results:
                        continue

                    # Get all clusters with this stat
                    cluster_pcts = [(cluster, data['pct_diffs'][stat], data['players'])
                                    for cluster, data in off_cluster_results.items()
                                    if stat in data['pct_diffs']]

                    if not cluster_pcts:
                        continue

                    # Sort by % diff
                    cluster_pcts_sorted = sorted(cluster_pcts, key=lambda x: x[1], reverse=True)

                    # Get top 3 best and top 3 worst
                    best_3 = cluster_pcts_sorted[:3]
                    worst_3 = cluster_pcts_sorted[-3:]
                    worst_3.reverse()  # Show worst first in order

                    # Build row
                    row = {'Stat': stat}

                    # Add best clusters
                    for i, (cluster, pct, players) in enumerate(best_3, 1):
                        cluster = int(cluster)
                        row[f'Best #{i} Cluster'] = cluster
                        row[f'Best #{i} %'] = pct
                        cluster_players_map[f"Off-{cluster}"] = players
                        if f'Best #{i} %' not in pct_columns:
                            pct_columns.append(f'Best #{i} %')

                    # Add worst clusters
                    for i, (cluster, pct, players) in enumerate(worst_3, 1):
                        cluster = int(cluster)
                        row[f'Worst #{i} Cluster'] = cluster
                        row[f'Worst #{i} %'] = pct
                        cluster_players_map[f"Off-{cluster}"] = players
                        if f'Worst #{i} %' not in pct_columns:
                            pct_columns.append(f'Worst #{i} %')

                    table_rows.append(row)

                # Display offensive stats table
                if table_rows:
                    st.markdown("### Offensive Stats")
                    off_stats_df = pd.DataFrame(table_rows)

                    # Apply styling to percentage columns
                    styled_off = off_stats_df.style.format({col: '{:+.1f}%' for col in pct_columns})
                    cluster_cols = [col for col in off_stats_df.columns if 'Cluster' in col]
                    off_stats_df[cluster_cols] = off_stats_df[cluster_cols].astype(int)

                    # Apply heatmap to percentage columns
                    for col in pct_columns:
                        styled_off = styled_off.applymap(get_heatmap_color, subset=[col])

                    st.dataframe(styled_off, use_container_width=True, height=600)

                    # Display cluster examples in expandable sections
                    st.markdown("#### 📋 Offensive Cluster Examples")
                    off_clusters_used = set()
                    for i in range(1, 4):
                        off_clusters_used.update(off_stats_df[f'Best #{i} Cluster'].values)
                        off_clusters_used.update(off_stats_df[f'Worst #{i} Cluster'].values)
                    off_clusters_used = sorted(off_clusters_used)

                    cols = st.columns(3)
                    for idx, cluster in enumerate(off_clusters_used):
                        with cols[idx % 3]:
                            with st.expander(f"Offensive Cluster {cluster}"):
                                players = cluster_players_map.get(f"Off-{cluster}", [])
                                st.write(", ".join(players))

                # Process defensive stats
                table_rows = []
                pct_columns = []

                for stat in defensive_stats:
                    if not def_cluster_results:
                        continue

                    # Get all clusters with this stat
                    cluster_pcts = [(cluster, data['pct_diffs'][stat], data['players'])
                                    for cluster, data in def_cluster_results.items()
                                    if stat in data['pct_diffs']]

                    if not cluster_pcts:
                        continue

                    # Sort by % diff
                    cluster_pcts_sorted = sorted(cluster_pcts, key=lambda x: x[1], reverse=True)

                    # Get top 3 best and top 3 worst
                    best_3 = cluster_pcts_sorted[:3]
                    worst_3 = cluster_pcts_sorted[-3:]
                    worst_3.reverse()  # Show worst first in order

                    # Build row
                    row = {'Stat': stat}

                    # Add best clusters
                    for i, (cluster, pct, players) in enumerate(best_3, 1):
                        cluster = int(cluster)
                        row[f'Best #{i} Cluster'] = cluster
                        row[f'Best #{i} %'] = pct
                        cluster_players_map[f"Def-{cluster}"] = players
                        if f'Best #{i} %' not in pct_columns:
                            pct_columns.append(f'Best #{i} %')

                    # Add worst clusters
                    for i, (cluster, pct, players) in enumerate(worst_3, 1):
                        cluster = int(cluster)
                        row[f'Worst #{i} Cluster'] = cluster
                        row[f'Worst #{i} %'] = pct
                        cluster_players_map[f"Def-{cluster}"] = players
                        if f'Worst #{i} %' not in pct_columns:
                            pct_columns.append(f'Worst #{i} %')

                    table_rows.append(row)

                # Display defensive stats table
                if table_rows:
                    st.markdown("---")
                    st.markdown("### Defensive Stats")
                    def_stats_df = pd.DataFrame(table_rows)

                    # Convert cluster columns to int
                    cluster_cols = [col for col in def_stats_df.columns if 'Cluster' in col]
                    def_stats_df[cluster_cols] = def_stats_df[cluster_cols].astype(int)

                    # Round percentage columns
                    def_stats_df[pct_columns] = def_stats_df[pct_columns].round(1)

                    # Apply styling to percentage columns
                    styled_def = def_stats_df.style.format({col: '{:+.1f}%' for col in pct_columns})

                    # Apply heatmap to percentage columns
                    for col in pct_columns:
                        styled_def = styled_def.applymap(get_heatmap_color, subset=[col])

                    st.dataframe(styled_def, use_container_width=True, height=400)

                    # Display cluster examples in expandable sections
                    st.markdown("#### 📋 Defensive Cluster Examples")
                    def_clusters_used = set()
                    for i in range(1, 4):
                        def_clusters_used.update(def_stats_df[f'Best #{i} Cluster'].values)
                        def_clusters_used.update(def_stats_df[f'Worst #{i} Cluster'].values)
                    def_clusters_used = sorted(def_clusters_used)

                    cols = st.columns(3)
                    for idx, cluster in enumerate(def_clusters_used):
                        with cols[idx % 3]:
                            with st.expander(f"Defensive Cluster {cluster}"):
                                players = cluster_players_map.get(f"Def-{cluster}", [])
                                st.write(", ".join(players))
                st.caption(
                    f"*Analysis based on games with {MIN_THRESHOLD}+ minutes. % differences calculated using Bayesian approach with game-by-game updates.*")
                st.caption(
                    f"*Heatmap scale: -25% (red) to +25% (green). Click cluster expanders below tables to see player examples.*")
    # =====================================================
    # 🟩 TAB 4 — Today's Best Matchups
    # =====================================================
    # =====================================================
    # 🟩 TAB 4 — Today's Best Matchups
    # =====================================================
    with tab4:
        st.subheader("🔥 Today's Best Matchup Opportunities")

        try:
            # Get today's games and players
            with st.spinner("Loading today's games..."):
                todays_players = build_player_list()

            if todays_players.empty:
                st.info("No games scheduled for today.")
            else:
                st.success(f"Found {len(todays_players)} players in today's games")

                # Clean up the OPP column (remove brackets and quotes)
                todays_players['OPP'] = todays_players['OPP'].str.strip("[]'\"")

                # Merge with our cluster data to get player clusters and stats
                todays_players_merged = todays_players.merge(
                    merged[['PLAYER_ID', 'PLAYER_NAME', 'OffCluster', 'DefCluster']].drop_duplicates('PLAYER_ID'),
                    on='PLAYER_ID',
                    how='inner'
                )

                if todays_players_merged.empty:
                    st.warning("No players from today's games found in cluster data.")
                else:
                    st.markdown(f"**{len(todays_players_merged)} players with cluster data playing today**")

                    # Calculate matchup scores for each player
                    matchup_scores = []

                    MIN_THRESHOLD = 5
                    offensive_stats = ['PTS', 'AST', 'OREB', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV']
                    defensive_stats = ['DREB', 'REB', 'STL', 'BLK', 'PF']

                    # OPTIMIZATION: Pre-filter merged data by date range once
                    merged_filtered = merged[
                        (merged['GAME_DATE'] >= min_date) &
                        (merged['GAME_DATE'] <= max_date)
                        ].copy()

                    # OPTIMIZATION: Pre-calculate cluster averages for all clusters
                    cluster_stats_cache = {}

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for player_idx, (idx, player_row) in enumerate(todays_players_merged.iterrows()):
                        player_id = player_row['PLAYER_ID']
                        player_name = player_row['PLAYER_NAME']
                        opponent = player_row['OPP']
                        off_cluster = player_row['OffCluster']
                        def_cluster = player_row['DefCluster']
                        is_home = player_row['Home']

                        if player_idx % 10 == 0:  # Update status less frequently
                            status_text.text(f"Analyzing matchups... {player_idx + 1}/{len(todays_players_merged)}")

                        if pd.isna(off_cluster) or pd.isna(def_cluster):
                            continue

                        # OPTIMIZATION: Use cached cluster data if available
                        cache_key_off = f"off_{off_cluster}"
                        cache_key_def = f"def_{def_cluster}"

                        if cache_key_off not in cluster_stats_cache:
                            df_off_cluster = merged_filtered[merged_filtered['OffCluster'] == off_cluster]
                            df_off_valid = df_off_cluster[df_off_cluster['MIN'] >= MIN_THRESHOLD]
                            df_off_per36 = df_off_valid[offensive_stats].div(df_off_valid['MIN'], axis=0) * 36
                            season_avg_per36_off = df_off_per36.mean()
                            df_off_cluster['OPP_TEAM'] = df_off_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                            cluster_stats_cache[cache_key_off] = {
                                'df': df_off_cluster,
                                'season_avg': season_avg_per36_off
                            }

                        if cache_key_def not in cluster_stats_cache:
                            df_def_cluster = merged_filtered[merged_filtered['DefCluster'] == def_cluster]
                            df_def_valid = df_def_cluster[df_def_cluster['MIN'] >= MIN_THRESHOLD]
                            df_def_per36 = df_def_valid[defensive_stats].div(df_def_valid['MIN'], axis=0) * 36
                            season_avg_per36_def = df_def_per36.mean()
                            df_def_cluster['OPP_TEAM'] = df_def_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                            cluster_stats_cache[cache_key_def] = {
                                'df': df_def_cluster,
                                'season_avg': season_avg_per36_def
                            }

                        # Get from cache
                        df_off_cluster = cluster_stats_cache[cache_key_off]['df']
                        season_avg_per36_off = cluster_stats_cache[cache_key_off]['season_avg']
                        df_def_cluster = cluster_stats_cache[cache_key_def]['df']
                        season_avg_per36_def = cluster_stats_cache[cache_key_def]['season_avg']

                        if df_off_cluster.empty or df_def_cluster.empty:
                            continue

                        # Filter by opponent
                        df_off_vs_team = df_off_cluster[df_off_cluster['OPP_TEAM'] == opponent]
                        df_def_vs_team = df_def_cluster[df_def_cluster['OPP_TEAM'] == opponent]

                        df_off_vs_team_valid = df_off_vs_team[df_off_vs_team['MIN'] >= MIN_THRESHOLD]
                        df_def_vs_team_valid = df_def_vs_team[df_def_vs_team['MIN'] >= MIN_THRESHOLD]

                        # Calculate Bayesian percentage differences
                        prior_mean = 0.0
                        prior_weight = 3.0

                        pct_diff_off = pd.Series(0.0, index=offensive_stats)
                        pct_diff_def = pd.Series(0.0, index=defensive_stats)
                        has_off_history = False
                        has_def_history = False

                        # Offensive cluster Bayesian calculation
                        if not df_off_vs_team_valid.empty:
                            has_off_history = True
                            df_off_vs_team_valid_sorted = df_off_vs_team_valid.sort_values('GAME_DATE')
                            df_off_vs_team_per36 = df_off_vs_team_valid_sorted[offensive_stats].div(
                                df_off_vs_team_valid_sorted['MIN'], axis=0) * 36

                            game_pct_diffs_off = (
                                    (df_off_vs_team_per36 - season_avg_per36_off[offensive_stats]) /
                                    season_avg_per36_off[offensive_stats] * 100
                            )

                            for stat in offensive_stats:
                                bayesian_estimate = prior_mean
                                total_weight = prior_weight
                                for game_pct_diff in game_pct_diffs_off[stat].values:
                                    bayesian_estimate = (total_weight * bayesian_estimate + game_pct_diff) / (
                                                total_weight + 1)
                                    total_weight += 1
                                pct_diff_off[stat] = bayesian_estimate

                        # Defensive cluster Bayesian calculation
                        if not df_def_vs_team_valid.empty:
                            has_def_history = True
                            df_def_vs_team_valid_sorted = df_def_vs_team_valid.sort_values('GAME_DATE')
                            df_def_vs_team_per36 = df_def_vs_team_valid_sorted[defensive_stats].div(
                                df_def_vs_team_valid_sorted['MIN'], axis=0) * 36

                            game_pct_diffs_def = (
                                    (df_def_vs_team_per36 - season_avg_per36_def[defensive_stats]) /
                                    season_avg_per36_def[defensive_stats] * 100
                            )

                            for stat in defensive_stats:
                                bayesian_estimate = prior_mean
                                total_weight = prior_weight
                                for game_pct_diff in game_pct_diffs_def[stat].values:
                                    bayesian_estimate = (total_weight * bayesian_estimate + game_pct_diff) / (
                                                total_weight + 1)
                                    total_weight += 1
                                pct_diff_def[stat] = bayesian_estimate

                        # Only include players with historical data vs this opponent
                        if has_off_history or has_def_history:
                            matchup_scores.append({
                                'PLAYER_NAME': player_name,
                                'PLAYER_ID': player_id,
                                'Opponent': opponent,
                                'Home': '🏠' if is_home else '✈️',
                                'Off Cluster': int(off_cluster),
                                'Def Cluster': int(def_cluster),
                                'Has Off History': has_off_history,
                                'Has Def History': has_def_history,
                                'PTS %': pct_diff_off['PTS'] if has_off_history else 0,
                                'AST %': pct_diff_off['AST'] if has_off_history else 0,
                                'OREB %': pct_diff_off['OREB'] if has_off_history else 0,
                                'DREB %': pct_diff_def['DREB'] if has_def_history else 0,
                                'FG3A %': pct_diff_off['FG3A'] if has_off_history else 0,
                                'STL %': pct_diff_def['STL'] if has_def_history else 0,
                                'BLK %': pct_diff_def['BLK'] if has_def_history else 0,
                                'Off Games': len(df_off_vs_team_valid),
                                'Def Games': len(df_def_vs_team_valid),
                                # Store full pct_diff for later use
                                'pct_diff_off': pct_diff_off,
                                'pct_diff_def': pct_diff_def
                            })

                        progress_bar.progress((player_idx + 1) / len(todays_players_merged))

                    progress_bar.empty()
                    status_text.empty()

                    if not matchup_scores:
                        st.warning("No players found with historical matchup data vs today's opponents.")
                    else:
                        matchups_df = pd.DataFrame(matchup_scores)
                        # Sort by PTS % by default
                        matchups_df = matchups_df.sort_values('PTS %', ascending=False)

                        # Display filter options
                        col1, col2 = st.columns(2)
                        with col1:
                            show_negative = st.checkbox("Show negative matchups too", value=True)

                        # Apply filters
                        filtered_df = matchups_df.copy()

                        if not show_negative:
                            filtered_df = filtered_df[filtered_df['PTS %'] > 0]

                        if not show_negative:
                            filtered_df = filtered_df[filtered_df['PTS %'] > 0]

                        st.markdown(f"### 📊 Top Matchup Opportunities ({len(filtered_df)} players)")

                        # Display summary table
                        display_cols = ['PLAYER_NAME', 'Home', 'Opponent', 'Off Cluster', 'Def Cluster',
                                        'PTS %', 'AST %', 'DREB %', 'OREB %', 'FG3A %', 'STL %', 'BLK %',
                                        'Off Games', 'Def Games']

                        def highlight_score(val, col_name):
                            if '%' in col_name and col_name not in ['Home']:
                                try:
                                    if val > 10:
                                        return 'background-color: #28A745; color: white; font-weight: bold'
                                    elif val > 5:
                                        return 'background-color: #90EE90; color: black'
                                    elif val > 0:
                                        return 'background-color: #D4EDDA; color: black'
                                    elif val < -5:
                                        return 'background-color: #DC3545; color: white'
                                    elif val < 0:
                                        return 'background-color: #F8D7DA; color: black'
                                except:
                                    pass
                            return ''

                        styled_matchups = filtered_df[display_cols].style.format({
                            'PTS %': '{:.1f}%',
                            'AST %': '{:.1f}%',
                            'DREB %': '{:.1f}%',
                            'OREB %': '{:.1f}%',
                            'FG3A %': '{:.1f}%',
                            'STL %': '{:.1f}%',
                            'BLK %': '{:.1f}%'
                        }).apply(
                            lambda row: [highlight_score(val, col) for col, val in zip(display_cols, row)],
                            axis=1
                        )

                        st.dataframe(styled_matchups, use_container_width=True, height=400)

                        st.caption("🏠 = Home game | ✈️ = Away game")

                        # =====================================================
                        # 🎯 Detailed Projection Section
                        # =====================================================
                        st.markdown("---")
                        st.markdown("### 🎯 Detailed Projection")

                        selected_today_player = st.selectbox(
                            "Select player for full projection:",
                            options=filtered_df['PLAYER_NAME'].tolist(),
                            key="today_player_select"
                        )

                        if selected_today_player:
                            player_info = filtered_df[filtered_df['PLAYER_NAME'] == selected_today_player].iloc[0]
                            opponent = player_info['Opponent']
                            off_cluster = player_info['Off Cluster']
                            def_cluster = player_info['Def Cluster']
                            pct_diff_off = player_info['pct_diff_off']
                            pct_diff_def = player_info['pct_diff_def']

                            st.markdown(f"**{selected_today_player}** vs **{opponent}**")
                            st.caption(f"Off Cluster {off_cluster} | Def Cluster {def_cluster}")

                            # Get player's season stats
                            player_season_stats = merged_filtered[
                                (merged_filtered['PLAYER_NAME'] == selected_today_player) &
                                (merged_filtered['MIN'] >= MIN_THRESHOLD)
                                ].copy()

                            if not player_season_stats.empty:
                                # Calculate player's season averages
                                all_stats = offensive_stats + defensive_stats
                                player_avg_stats = player_season_stats[all_stats + ['MIN']].mean()

                                # Get median minutes from last 10 games
                                player_last_10 = player_season_stats.sort_values('GAME_DATE', ascending=False).head(10)
                                median_min = player_last_10['MIN'].median() if not player_last_10.empty else \
                                player_avg_stats['MIN']

                                # Minutes slider
                                selected_minutes = st.slider(
                                    "Minutes per game:",
                                    min_value=0.0,
                                    max_value=48.0,
                                    value=float(median_min),
                                    step=0.5,
                                    key=f"minutes_{selected_today_player}",
                                    help=f"Default is median of last 10 games ({median_min:.1f} min)"
                                )

                                minutes_multiplier = selected_minutes / player_avg_stats['MIN'] if player_avg_stats[
                                                                                                       'MIN'] > 0 else 1.0

                                # Build projection
                                projection_data = {'Stat': [], 'Season Avg': [], 'Projected': [], 'Difference': []}

                                display_stats_order = ['PTS', 'AST', 'OREB', 'DREB', 'REB', 'FG3A', 'FG3M', 'FTA',
                                                       'FTM', 'STL', 'BLK']

                                for stat in display_stats_order:
                                    if stat == 'REB':
                                        # Calculate REB from OREB + DREB
                                        season_avg = player_avg_stats['OREB'] + player_avg_stats['DREB']
                                        oreb_pct = pct_diff_off['OREB'] if player_info['Has Off History'] else 0
                                        dreb_pct = pct_diff_def['DREB'] if player_info['Has Def History'] else 0
                                        projected = (
                                                            player_avg_stats['OREB'] * (1 + oreb_pct / 100) +
                                                            player_avg_stats['DREB'] * (1 + dreb_pct / 100)
                                                    ) * minutes_multiplier
                                    elif stat in offensive_stats:
                                        season_avg = player_avg_stats[stat]
                                        pct_change = pct_diff_off[stat] if player_info['Has Off History'] else 0
                                        projected = season_avg * (1 + pct_change / 100) * minutes_multiplier
                                    elif stat in defensive_stats:
                                        season_avg = player_avg_stats[stat]
                                        pct_change = pct_diff_def[stat] if player_info['Has Def History'] else 0
                                        projected = season_avg * (1 + pct_change / 100) * minutes_multiplier
                                    else:
                                        continue

                                    projection_data['Stat'].append(stat)
                                    projection_data['Season Avg'].append(season_avg)
                                    projection_data['Projected'].append(projected)
                                    projection_data['Difference'].append(projected - season_avg)

                                proj_df = pd.DataFrame(projection_data)

                                def highlight_diff(val, col_name):
                                    if col_name == 'Difference':
                                        try:
                                            if val > 0:
                                                return 'background-color: #D4EDDA; color: #155724'
                                            elif val < 0:
                                                return 'background-color: #F8D7DA; color: #721C24'
                                        except:
                                            pass
                                    return ''

                                styled_proj = proj_df.style.format({
                                    'Season Avg': '{:.1f}',
                                    'Projected': '{:.1f}',
                                    'Difference': '{:.1f}'
                                }).apply(
                                    lambda row: [highlight_diff(val, col) for col, val in zip(proj_df.columns, row)],
                                    axis=1
                                )

                                st.dataframe(styled_proj, use_container_width=True)

                                st.caption(f"💡 Projections based on {selected_minutes:.1f} minutes")
                                st.caption(
                                    f"📊 Using Off Cluster {off_cluster} and Def Cluster {def_cluster} historical performance vs {opponent}")
                            else:
                                st.warning(f"No season stats found for {selected_today_player}")

        except Exception as e:
            st.error(f"Error loading today's games: {str(e)}")
            st.exception(e)
            st.info("Make sure the NBA API is accessible and there are games scheduled today.")
if __name__ == '__main__':
    #create_clusters()
    main()