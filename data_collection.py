import time
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV2

START_YEAR = 2001
CURRENT_YEAR = 2025
failed_games = []

# Function to get all NBA games for a given season
def get_games_by_season(season_year):
    print(f"Fetching games for season {season_year}-{season_year+1}")
    game_finder = LeagueGameFinder(season_nullable=f"{season_year}-{str(season_year+1)[-2:]}")
    games = game_finder.get_data_frames()[0]
    return games[['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 'MATCHUP']]

# Function to fetch play-by-play data for a given game
def get_play_by_play(game_id):
    time.sleep(1)  # To avoid rate limiting
    pbp = PlayByPlayV2(game_id=game_id)
    pbp_df = pbp.get_data_frames()[0]
    return pbp_df[['EVENTNUM', 'EVENTMSGTYPE', 'PLAYER1_ID', 'PLAYER2_ID', 'PLAYER3_ID', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION']]

# Function to extract relevant player interactions, including assists
def extract_interactions(pbp_df):
    interactions = []

    # Filter only relevant events: Shot Made (1) and Shot Missed (2)
    df = pbp_df[pbp_df['EVENTMSGTYPE'].isin([1, 2])]

    # Extract Shot Made events
    shot_made_df = df[df['EVENTMSGTYPE'] == 1][['PLAYER1_ID', 'PLAYER2_ID', 'PLAYER3_ID', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION']]
    shot_made_df.columns = ['Scorer', 'Assister', 'PLAYER3_ID', 'Home_Description', 'Visitor_Description']
    shot_made_df['Event_Type'] = 'Shot_Made'  # Add event type

    # Extract Shot Missed events
    shot_missed_df = df[df['EVENTMSGTYPE'] == 2][['PLAYER1_ID', 'PLAYER2_ID', 'PLAYER3_ID', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION']]
    shot_missed_df.columns = ['Scorer', 'Assister', 'PLAYER3_ID', 'Home_Description', 'Visitor_Description']
    shot_missed_df['Event_Type'] = 'Shot_Missed'  # Add event type

    # Fill NaN values in descriptions with an empty string to avoid misalignment
    shot_made_df.fillna('', inplace=True)
    shot_missed_df.fillna('', inplace=True)

    # Convert to list of tuples for efficiency (ensure correct order)
    interactions.extend(shot_made_df[['Scorer', 'Assister', 'PLAYER3_ID', 'Event_Type', 'Home_Description', 'Visitor_Description']].itertuples(index=False, name=None))
    interactions.extend(shot_missed_df[['Scorer', 'Assister', 'PLAYER3_ID', 'Event_Type', 'Home_Description', 'Visitor_Description']].itertuples(index=False, name=None))

    return interactions  # Returns (Scorer, Assister, PLAYER3_ID, Event_Type, Home_Description, Visitor_Description)

# Main function to collect data for multiple seasons
def collect_nba_data():
    all_data = []
    processed_games = set()  # Store processed GAME_IDs to avoid duplication

    for year in range(START_YEAR, CURRENT_YEAR + 1):
        print(f"Processing Season: {year}-{year+1}")
        try:
            games = get_games_by_season(year)
        except Exception as e:
            print(f"Error fetching games for {year}: {e}")
            continue  # Skip to the next season if an error occurs

        for _, game in games.iterrows():
            game_id = game['GAME_ID']

            if game_id in processed_games:
                continue  # Skip duplicate game ID

            processed_games.add(game_id)  # Mark game as processed

            print(f"Processing Game ID: {game_id}")

            try:
                pbp_df = get_play_by_play(game_id)
                interactions = extract_interactions(pbp_df)
            except Exception as e:
                print(f"Error processing Game ID {game_id}: {e}")
                failed_games.append(game_id)
                continue  # Skip to the next game if an error occurs

            for interaction in interactions:
                scorer, assister, player3, event_type, home_desc, visitor_desc = interaction
                all_data.append([year, game_id, game['GAME_DATE'], event_type, scorer, assister, player3, home_desc, visitor_desc])

    # Convert to DataFrame
    columns = ['Season', 'Game_ID', 'Game_Date', 'Event_Type', 'Player1_ID', 'Player2_ID', 'PLAYER3_ID', 'Home_Description', 'Visitor_Description']
    df = pd.DataFrame(all_data, columns=columns)

    # Save to CSV (optional)
    df.to_csv('nba_player_interactions_2001_2025.csv', index=False)
    print("Data collection complete. Saved as 'nba_player_interactions_2001_2024.csv'.")

    # Save failed game IDs for later retry
    if failed_games:
        with open("failed_games.txt", "w") as f:
            for game_id in failed_games:
                f.write(f"{game_id}\n")
        print(f"⚠️ Saved {len(failed_games)} failed game IDs to 'failed_games.txt'.")

# Run the data collection process
if __name__ == "__main__":
    collect_nba_data()
