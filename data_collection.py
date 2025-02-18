import time
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV2

# Define the range of seasons
START_YEAR = 2001
CURRENT_YEAR = 2024  # Update as needed

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
    return pbp_df[['EVENTNUM', 'EVENTMSGTYPE', 'SCORE', 'SCOREMARGIN',
                   'PERSON1TYPE', 'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ID', 'PLAYER1_TEAM_NICKNAME', 'PLAYER1_TEAM_ABBREVIATION', 
                   'PERSON2TYPE', 'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ID', 'PLAYER2_TEAM_NICKNAME', 'PLAYER2_TEAM_ABBREVIATION', 
                   'HOMEDESCRIPTION', 'VISITORDESCRIPTION']]

# Function to extract relevant player interactions, including assists
def extract_interactions(pbp_df):
    # Filter only relevant events: Shot Made (1)
    df = pbp_df[pbp_df['EVENTMSGTYPE'].isin([1])]

    return df

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

        for i, game in games.iterrows():
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

            for row in interactions.itertuples(index=False, name=None):
                all_data.append([year, game_id, game['GAME_DATE'], *row])

    # Convert to DataFrame
    columns = ['Season', 'Game_ID', 'Game_Date', 'EVENTNUM', 'EVENTMSGTYPE', 'SCORE', 'SCOREMARGIN',
                'PERSON1TYPE', 'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ID', 'PLAYER1_TEAM_NICKNAME', 'PLAYER1_TEAM_ABBREVIATION', 
                'PERSON2TYPE', 'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ID', 'PLAYER2_TEAM_NICKNAME', 'PLAYER2_TEAM_ABBREVIATION', 
                'HOMEDESCRIPTION', 'VISITORDESCRIPTION']
    df = pd.DataFrame(all_data, columns=columns)

    # Save to CSV (optional)
    df.to_csv(f'nba_player_interactions_{START_YEAR}_{CURRENT_YEAR}.csv', index=False)
    print(f"Data collection complete. Saved as 'nba_player_interactions_{START_YEAR}_{CURRENT_YEAR}.csv'.")

    # Save failed game IDs for later retry
    if failed_games:
        with open(f"failed_games_{START_YEAR}_{CURRENT_YEAR}.txt", "w") as f:
            for game_id in failed_games:
                f.write(f"{game_id}\n")
        print(f"⚠️ Saved {len(failed_games)} failed game IDs to 'failed_games_{START_YEAR}_{CURRENT_YEAR}.txt'.")

# Run the data collection process
if __name__ == "__main__":
    collect_nba_data()
