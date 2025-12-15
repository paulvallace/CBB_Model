import json
import csv
import os
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple


def get_date_range(start_date_str: str, end_date_str: str) -> List[str]:
    """
    Generate a list of dates between start and end date.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    return date_list


def fetch_odds_data(date_str: str) -> dict:
    url = f"https://www.oddsshark.com/api/scores/ncaab/{date_str}"
    
    headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.oddsshark.com/ncaab/scores",
    "Origin": "https://www.oddsshark.com",
    "Connection": "keep-alive"
}

    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code} for {date_str}")

    if not resp.text.strip():
        raise ValueError(f"Empty response for {date_str}")

    content_type = resp.headers.get("Content-Type", "")
    if "application/json" not in content_type.lower():
        raise ValueError(f"Non-JSON response for {date_str} ({content_type})")

    try:
        return resp.json()
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON for {date_str}")



def format_date(timestamp: int) -> str:
    """
    Format timestamp to 'Feb 1st' format
    """
    date = datetime.fromtimestamp(timestamp)
    day = date.day
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10 if day not in [11, 12, 13] else 0, 'th')
    return date.strftime(f'%b {day}{suffix}')


def format_spread(home_spread: float, is_home_team: bool) -> str:
    """
    Format spread with proper sign.
    For home team: negative means favorite, positive means underdog
    For away team: opposite of home team's spread
    """
    if is_home_team:
        return f"{home_spread:+.1f}"
    else:
        return f"{-home_spread:+.1f}"


def process_games(data: dict) -> List[Dict]:
    """
    Process OddsShark JSON data and format game results with spread coverage.
    Only one team per game is included (home team).
    """
    formatted_games = []
    
    for game in data.get('scores', []):
        try:
            home_team = game['teams']['home']
            away_team = game['teams']['away']
            
            game_date = format_date(game['date'])
            
            # Get team names and scores
            home_name = home_team['names']['display_name']
            away_name = away_team['names']['display_name']
            home_score = int(home_team['score'])
            away_score = int(away_team['score'])
            
            # Get spread (from home team's perspective)
            home_spread = float(home_team['spread'])
            
            # Calculate if spread was covered
            score_difference = home_score - away_score
            # For home team
            home_covered = 'Y' if score_difference > home_spread else 'N'
            if home_spread > 0:  # If home team is underdog
                home_covered = 'Y' if score_difference + home_spread > 0 else 'N'
            
            # Create single game entry (home team perspective)
            game_entry = {
                'Team': home_name,
                'Opponent': away_name,
                'Team_Score': home_score,
                'Opp_Score': away_score,
                'Spread': format_spread(home_spread, True),
                'Spread_Covered': home_covered,
                'Final_Score': f"{home_score}-{away_score}",
                'Date': game_date
            }
            
            formatted_games.append(game_entry)
            
        except (KeyError, TypeError) as e:
            print(f"Error processing game: {str(e)}")
            continue
    
    return formatted_games


def save_to_csv(games: List[Dict], output_file: str):
    """
    Save game results to a CSV file.
    """
    if not games:
        raise ValueError("No games to save")
        
    fieldnames = [
        'Team',
        'Opponent',
        'Team_Score',
        'Opp_Score',
        'Spread',
        'Spread_Covered',
        'Final_Score',
        'Date'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(games)


def update_master_file(new_data_file, master_file='/Users/PaulVallace/Desktop/College Basketball/historical data/Past Games/historical_games.csv'):
    """
    Update master file with new game data.
    
    Parameters:
    new_data_file (str): Path to the new data file
    master_file (str): Path to the master file (default: 'historical_games.csv')
    """
    try:
        # Read the new data
        new_data = pd.read_csv(new_data_file)
        
        # Check if master file exists
        if os.path.exists(master_file):
            # Read existing master file
            master_data = pd.read_csv(master_file)
            # Concatenate new data with master data
            combined_data = pd.concat([master_data, new_data], ignore_index=True)
        else:
            # If master file doesn't exist, use new data as master
            combined_data = new_data
            
        # Remove duplicates if any (modify the subset based on your unique identifiers)
        combined_data = combined_data.drop_duplicates(subset=['Date', 'Team', 'Opponent'], keep='last')
        
        # Sort by date properly - handling text month formats like "Feb 24th"
        if 'Date' in combined_data.columns:
            try:
                # First, check and print a sample of dates to understand the format
                print("Sample dates before conversion:", combined_data['Date'].head().tolist())
                
                # Clean the dates by removing ordinal suffixes (st, nd, rd, th)
                combined_data['Date'] = combined_data['Date'].astype(str).str.replace(r'(\d+)(st|nd|rd|th)', r'\1', regex=True)
                
                # Add the current year if not present (assumes current season's games)
                current_year = datetime.now().year
                
                def parse_date(date_str):
                    try:
                        # Try to parse with various formats
                        if len(date_str.split()) == 2:  # If only month and day (e.g., "Feb 24")
                            date_str = f"{date_str} {current_year}"
                        # Convert to datetime
                        return pd.to_datetime(date_str, errors='coerce')
                    except:
                        return pd.NaT
                
                # Apply the custom parsing function
                combined_data['Date'] = combined_data['Date'].apply(parse_date)
                
                # Check for any NaT (Not a Time) values that indicate conversion failures
                nat_count = combined_data['Date'].isna().sum()
                if nat_count > 0:
                    print(f"Warning: {nat_count} dates could not be converted properly.")
                
                # Sort by the datetime column
                combined_data = combined_data.sort_values('Date')
                
                print("Date conversion successful. Sample after conversion:",
                      combined_data['Date'].head().tolist())
                
            except Exception as e:
                print(f"Warning: Could not convert dates properly: {str(e)}")
                print("Attempting to sort without conversion...")
                # Still try to sort using string comparison as fallback
                combined_data = combined_data.sort_values('Date')
        
        # Save updated data to master file
        combined_data.to_csv(master_file, index=False)
        print(f"Successfully updated {master_file} with data from {new_data_file}")
        print(f"Total records in master file: {len(combined_data)}")
        
    except Exception as e:
        print(f"Error updating master file: {str(e)}")


def main():
    try:
        # User can choose to fetch new data, update the master file, or both
        action = input("What would you like to do? (1: Fetch new data, 2: Update master file, 3: Both): ")
        
        if action in ['1', '3']:
            # Specify your date range here
            start_date = input("Enter start date (YYYY-MM-DD): ")
            end_date = input("Enter end date (YYYY-MM-DD): ")
            
            print(f"Fetching data for dates between {start_date} and {end_date}...")
            
            dates = get_date_range(start_date, end_date)
            all_games = []
            
        for date_str in dates:
            print(f"\nFetching data for {date_str}...")
            try:
                data = fetch_odds_data(date_str)
            except Exception as e:
                print(f"  ⚠️ Error fetching data for {date_str}: {str(e)}")
                continue

            # If API returns valid JSON but no games
            if not data.get("scores"):
                print(f"  No games found for {date_str} — skipping.")
                continue

            # Process the games
            games = process_games(data)
            print(f"  Found {len(games)} games for {date_str}")
            all_games.extend(games)


            if all_games:
                output_file = f'/Users/PaulVallace/Desktop/College Basketball/historical data/Past Games/game_results_{start_date}_{end_date}.csv'
                save_to_csv(all_games, output_file)
                print(f"\nTotal games processed: {len(all_games)}")
                print(f"Results saved to {output_file}")
                
                if action == '3':
                    # Automatically update master file with new data
                    print("\nUpdating master file with new data...")
                    update_master_file(output_file)
            else:
                print("No games found for the specified date range")
        
        if action == '2':
            # Prompt for input file to update master file
            input_file = input("Enter path to new data file: ")
            update_master_file(input_file)
                
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()