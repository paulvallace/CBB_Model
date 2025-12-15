import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import re
from rapidfuzz import process, fuzz

# ======================================================
# TEAM NAME NORMALIZATION AND STANDARDIZATION
# ======================================================
TEAM_NAME_MAPPING = {
    # Existing examples preserved
    'St Marys': 'Saint Marys',
    'St. Marys': 'Saint Marys',
    'Saint Marys': 'Saint Marys',
    'Wisconsin-Green Bay': 'Green Bay',
    'UMass': 'Massachusetts',
    'UMKC': 'Kansas City',
    'UIC': 'Illinois Chicago',
    'UC Davis': 'UC Davis',
    'UC Irvine': 'UC Irvine',
    'UC Riverside': 'UC Riverside',
    'UC San Diego': 'UC San Diego',
    'UC Santa Barbara': 'UC Santa Barbara',
    'UNC Asheville': 'UNC Asheville',
    'UNC Greensboro': 'UNC Greensboro',
    'UNC Wilmington': 'UNC Wilmington',
    'UT Rio Grande Valley': 'UTRGV',
    'UT Arlington': 'UT Arlington',
    'St Thomas (MN)': 'St Thomas',
    'Arkansas-Little Rock': 'Little Rock',
    'Central Connecticut State': 'Central Connecticut',
    'Dixie State': 'Utah Tech',
    'Georgia Stte': 'Georgia State',
    'Alabama A&M': 'Alabama A&M',
    'Alabama St': 'Alabama State',
    'Alabama Stte': 'Alabama State',
    'Albany': 'Albany (NY)',
    'Alcorn St': 'Alcorn State',
    'Alcorn Stte': 'Alcorn State',
    'Appalachian St': 'Appalachian State',
    'Appalachian Stte': 'Appalachian State',
    'Arizona St': 'Arizona State',
    'Arizona Stte': 'Arizona State',
    'Arkansas St': 'Arkansas State',
    'Arkansas Stte': 'Arkansas State',
    'Ball St': 'Ball State',
    'Ball Stte': 'Ball State',
    'Boise St': 'Boise State',
    'Boise Stte': 'Boise State',
    'Boston Col': 'Boston College',
    'Cal St Bakersfield': 'Cal State Bakersfield',
    'Cal St Fullerton': 'Cal State Fullerton',
    'Cal Stte-Bakersfield': 'Cal State Bakersfield',
    'Cal Stte-Fullerton': 'Cal State Fullerton',
    'Cal Stte-Northridge': 'Cal State Northridge',
    'Central Connecticut': 'Central Connecticut State',
    'Central Connecticut Stte': 'Central Connecticut State',
    'Cleveland St': 'Cleveland State',
    'Cleveland Stte': 'Cleveland State',
    'Colorado St': 'Colorado State',
    'Colorado Stte': 'Colorado State',
    'Coppin St': 'Coppin State',
    'Coppin Stte': 'Coppin State',
    'Delaware St': 'Delaware State',
    'Delaware Stte': 'Delaware State',
    'Detroit': 'Detroit Mercy',
    'East Tennessee St': 'East Tennessee State',
    'East Tennessee Stte': 'East Tennessee State',
    'FIU': 'Florida International',
    'Florida St': 'Florida State',
    'Florida Stte': 'Florida State',
    'Fresno St': 'Fresno State',
    'Fresno Stte': 'Fresno State',
    'Georgia St': 'Georgia State',
    'Grambling St': 'Grambling State',
    'Grambling Stte': 'Grambling State',
    'Houston Baptist': 'Houston Christian',
    'Illinois St': 'Illinois State',
    'Illinois Stte': 'Illinois State',
    'Indiana St': 'Indiana State',
    'Indiana Stte': 'Indiana State',
    'Iowa St': 'Iowa State',
    'Iowa Stte': 'Iowa State',
    'Jackson St': 'Jackson State',
    'Jackson Stte': 'Jackson State',
    'Kansas St': 'Kansas State',
    'Kansas Stte': 'Kansas State',
    'Kennesaw St': 'Kennesaw State',
    'Kennesaw Stte': 'Kennesaw State',
    'Kent St': 'Kent State',
    'Kent Stte': 'Kent State',
    'Long Beach St': 'Long Beach State',
    'Louisiana-Lafayette': 'Louisiana',
    'Louisiana-Monroe': 'ULM',
    'Loyola-Chicago': 'Loyola Chicago',
    'Miami-Ohio': 'Miami (OH)',
    'Michigan St': 'Michigan State',
    'Michigan Stte': 'Michigan State',
    'Miss Stte': 'Mississippi State',
    'Mississippi St': 'Mississippi State',
    'Missouri St': 'Missouri State',
    'Missouri Stte': 'Missouri State',
    'Montana St': 'Montana State',
    'Montana Stte': 'Montana State',
    'Morehead St': 'Morehead State',
    'Morehead Stte': 'Morehead State',
    'Morgan St': 'Morgan State',
    'Morgan Stte': 'Morgan State',
    'Murray St': 'Murray State',
    'Murray Stte': 'Murray State',
    'NC State': 'North Carolina State',
    'NC Stte': 'North Carolina State',
    'New Mexico St': 'New Mexico State',
    'New Mexico Stte': 'New Mexico State',
    'Nicholls St': 'Nicholls State',
    'Nicholls Stte': 'Nicholls State',
    'Norfolk St': 'Norfolk State',
    'Norfolk Stte': 'Norfolk State',
    'North Dakota St': 'North Dakota State',
    'North Dakota Stte': 'North Dakota State',
    'Northwestern St': 'Northwestern State',
    'Northwestern Stte': 'Northwestern State',
    'Ohio St': 'Ohio State',
    'Ohio Stte': 'Ohio State',
    'Oklahoma St': 'Oklahoma State',
    'Oklahoma Stte': 'Oklahoma State',
    'Oregon St': 'Oregon State',
    'Oregon Stte': 'Oregon State',
    'Penn St': 'Penn State',
    'Penn Stte': 'Penn State',
    'Pittsburgh': 'Pitt',
    'San Diego St': 'San Diego State',
    'San Diego Stte': 'San Diego State',
    'San Jose St': 'San Jose State',
    'San Jose Stte': 'San Jose State',
    'South Dakota St': 'South Dakota State',
    'South Dakota Stte': 'South Dakota State',
    'TCU': 'TCU',
    'TX Christian': 'TCU',
    'Utah St': 'Utah State',
    'Utah Stte': 'Utah State',
    'VA Tech': 'Virginia Tech',
    'W Virginia': 'West Virginia',
    'Washington St': 'Washington State',
    'Washington Stte': 'Washington State',
    'Weber St': 'Weber State',
    'Weber Stte': 'Weber State',
    'Wichita St': 'Wichita State',
    'Wichita Stte': 'Wichita State',
    'Wright St': 'Wright State',
    'Wright Stte': 'Wright State',
    'Youngstown St': 'Youngstown State',
    'Youngstown Stte': 'Youngstown State',
}

# ======================================================
# SMART NAME STANDARDIZATION FUNCTION
# ======================================================
def standardize_team_name(name, valid_names=None):
    name = name.strip()
    # direct mapping first
    if name in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[name]

    # clean up common issues
    clean_name = re.sub(r'[^A-Za-z0-9&\s]', '', name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()

    if clean_name in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[clean_name]

    # fuzzy match if we have valid_names provided
    if valid_names is not None and len(valid_names) > 0:
        match, score, _ = process.extractOne(name, valid_names, scorer=fuzz.token_sort_ratio)
        if score >= 85:
            return match

    return name  # fallback
def audit_team_name_matches(df, column_name, source_name, valid_names=None):
    """
    Checks for team names that could not be matched/standardized.
    Logs the unmatched ones with their source file name.
    
    Parameters:
        df (pd.DataFrame): The dataframe to check
        column_name (str): The column containing team names
        source_name (str): Identifier for the file/source (e.g. 'kenpom_2025', 'historical_games')
        valid_names (list or set): Optional set of valid names for fuzzy matching
    """
    unmatched = set()

    standardized = []
    for name in df[column_name].unique():
        std_name = standardize_team_name(name, valid_names=valid_names)
        if std_name == name:  # if name didn't change, it likely wasn’t matched
            unmatched.add(name)
        standardized.append(std_name)

    if unmatched:
        print(f"\n⚠️ Unmatched teams found in {source_name}:")
        for u in sorted(unmatched):
            print(f"  '{u}': ''")
    else:
        print(f"✅ All team names in {source_name} matched successfully!")

    return unmatched

# ======================================================
# END OF INSERTED STANDARDIZATION SECTION
# ======================================================

def parse_date(date_str):
    """Convert date strings to datetime objects"""
    try:
        return pd.to_datetime(date_str)
    except:
        months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month = months[date_str[:3]]
        day = int(''.join(filter(str.isdigit, date_str[4:])))
        return datetime(2024, month, day)

def prepare_data():
    # Load the data
    games_df = pd.read_csv('/Users/PaulVallace/Desktop/College Basketball/historical data/Past Games/historical_games.csv')
    kenpom_df = pd.read_excel('/Users/PaulVallace/Desktop/College Basketball/historical data/Kenpom/historical_kenpom.xlsx')
    
    # Add diagnostic counters
    total_games = len(games_df)
    processed_games = 0
    matched_games = 0
    
    # Standardize team names in games dataset
    games_df['Team'] = games_df['Team'].apply(standardize_team_name)
    games_df['Opponent'] = games_df['Opponent'].apply(standardize_team_name)
    
    print("Sample of standardized team names from games data:")
    print(games_df[['Team', 'Opponent']].head())
    
    # Check KenPom data structure
    print("\nKenPom columns:", kenpom_df.columns.tolist())
    
    # Assuming KenPom data is already structured with separate columns
    # Standardize team names in KenPom data
    if 'Team' in kenpom_df.columns:
        kenpom_df['Team'] = kenpom_df['Team'].apply(standardize_team_name)
    
    print("\nSample of standardized team names from KenPom data:")
    print(kenpom_df.head())
    
    # Ensure required columns exist
    required_columns = ['Team', 'Date', 'NetRtg', 'ORtg', 'DRtg', 'AdjT']
    missing_columns = [col for col in required_columns if col not in kenpom_df.columns]
    
    if missing_columns:
        print(f"Missing required columns in KenPom data: {missing_columns}")
        print("Available columns:", kenpom_df.columns.tolist())
        # Attempt to map column names if they exist with different names
        column_mapping = {
            'Rk': 'Rank',
            'Team Name': 'Team',
            'Conference': 'Conf',
            'Net Rating': 'NetRtg',
            'Offensive Rating': 'ORtg',
            'Defensive Rating': 'DRtg',
            'Adjusted Tempo': 'AdjT',
            'Game Date': 'Date'
        }
        
        kenpom_df = kenpom_df.rename(columns={old: new for old, new in column_mapping.items() 
                                          if old in kenpom_df.columns and new not in kenpom_df.columns})
        
        # Check again after renaming
        missing_columns = [col for col in required_columns if col not in kenpom_df.columns]
        if missing_columns:
            raise ValueError(f"Still missing required columns after renaming: {missing_columns}")
    
    # Convert metrics to numeric if they're not already
    numeric_columns = ['NetRtg', 'ORtg', 'DRtg', 'AdjT']
    for col in numeric_columns:
        kenpom_df[col] = pd.to_numeric(kenpom_df[col], errors='coerce')
    
    # Convert dates
    games_df['Date'] = games_df['Date'].apply(parse_date)
    kenpom_df['Date'] = pd.to_datetime(kenpom_df['Date'])
    
    # Create feature DataFrame
    features = []
    unmatched_teams = set()
    
    for _, game in games_df.iterrows():
        processed_games += 1
        try:
            date = game['Date']
            team_name = game['Team']
            opp_name = game['Opponent']
            
            # Get team stats
            team_matches = kenpom_df[
                (kenpom_df['Team'] == team_name) & 
                (kenpom_df['Date'] == date)
            ]
            
            opp_matches = kenpom_df[
                (kenpom_df['Team'] == opp_name) & 
                (kenpom_df['Date'] == date)
            ]
            
            if len(team_matches) == 0 or len(opp_matches) == 0:
                if len(team_matches) == 0:
                    unmatched_teams.add(team_name)
                if len(opp_matches) == 0:
                    unmatched_teams.add(opp_name)
                continue
            
            team_stats = team_matches.iloc[0]
            opp_stats = opp_matches.iloc[0]
            
            features.append({
                'Spread': game['Spread'],
                'Team_ORtg': team_stats['ORtg'],
                'Team_DRtg': team_stats['DRtg'],
                'Team_AdjT': team_stats['AdjT'],
                'Opp_ORtg': opp_stats['ORtg'],
                'Opp_DRtg': opp_stats['DRtg'],
                'Opp_AdjT': opp_stats['AdjT'],
                'NetRtg_Diff': team_stats['NetRtg'] - opp_stats['NetRtg'],
                'Covered': 1 if game['Spread_Covered'] == 'Y' else 0
            })
            matched_games += 1
            
        except Exception as e:
            print(f"Error processing game: {team_name} vs {opp_name}")
            print(f"Error details: {str(e)}")
            continue
    
    if unmatched_teams:
        print("\nUnmatched teams (add these to TEAM_NAME_MAPPING):")
        for team in sorted(unmatched_teams):
            print(f"'{team}': '',")
    
    result_df = pd.DataFrame(features)
    print(f"\nTotal games in input: {total_games}")
    print(f"Games processed: {processed_games}")
    print(f"Games successfully matched: {matched_games}")
    print(f"Match rate: {matched_games/total_games:.2%}")
    print(f"\nFinal feature dataset shape: {result_df.shape}")
    return result_df

def train_model():
    # Prepare the data
    print("Preparing data...")
    df = prepare_data()
    
    if len(df) == 0:
        raise ValueError("No valid data after preparation. Please check the data processing steps.")
    
    print(f"\nPrepared dataset shape: {df.shape}")
    print("\nFeature statistics:")
    print(df.describe())
    
    # Split features and target
    X = df.drop('Covered', axis=1)
    y = df['Covered']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance
    X_train_scaled = scaler.fit_transform(X_train) # Fit to data, then transform it
    X_test_scaled = scaler.transform(X_test) # Perform standardization by centering and scaling
    
    # Train the model with optimal number of trees
    model = RandomForestClassifier(n_estimators=400, random_state=42) 
    model.fit(X_train_scaled, y_train) # Fit the model according to the given training data

    # Evaluate the model
    y_pred = model.predict(X_test_scaled) 
    accuracy = accuracy_score(y_test, y_pred) 
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, scaler

def predict_game(model, scaler, team_stats, opp_stats, spread):
    """
    Predict the outcome of a single game
    """
    # Ensure all required fields are present
    required_fields = ['ORtg', 'DRtg', 'AdjT', 'NetRtg']
    for field in required_fields:
        if field not in team_stats:
            raise ValueError(f"Missing required field '{field}' in team stats")
        if field not in opp_stats:
            raise ValueError(f"Missing required field '{field}' in opponent stats")
            
    features = pd.DataFrame([{
        'Spread': spread,
        'Team_ORtg': team_stats['ORtg'],
        'Team_DRtg': team_stats['DRtg'],
        'Team_AdjT': team_stats['AdjT'],
        'Opp_ORtg': opp_stats['ORtg'],
        'Opp_DRtg': opp_stats['DRtg'],
        'Opp_AdjT': opp_stats['AdjT'],
        'NetRtg_Diff': team_stats['NetRtg'] - opp_stats['NetRtg']
    }])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0]
    
    return {
        'will_cover': bool(prediction[0]),
        'confidence': float(max(probability))
    }

if __name__ == "__main__":
    try:
        # Train the model
        print("Training model...")
        model, scaler = train_model()
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        print("Please check the data files and their format.")