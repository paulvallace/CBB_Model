import os
import smtplib
from email.message import EmailMessage
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import re
from datetime import datetime, date, timedelta
from CBB_Model import standardize_team_name, predict_game, train_model

def parse_matchup(matchup: str) -> tuple:
    """Parse matchup string into team names and spread"""
    pattern = r'(.*?)\s+at\s+(.*?)\s+\(([-\d.]+)\)'
    match = re.match(pattern, matchup)
    if match:
        away_team = match.group(1).strip()
        home_team = match.group(2).strip()
        spread = float(match.group(3))
        return away_team, home_team, spread
    else:
        raise ValueError(f"Could not parse matchup: {matchup}")


import os
import smtplib
from email.message import EmailMessage

def send_predictions_email(attachment_path: str, predictions_count: int, run_date: str) -> None:
    """Email the predictions file if SMTP settings are available."""
    email_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    email_port = int(os.environ.get('SMTP_PORT', '587'))
    sender = os.environ.get('SMTP_USER')        # ✅ FIXED
    password = os.environ.get('SMTP_PASSWORD') # ✅ correct
    recipient = os.environ.get('SMTP_RECIPIENT', 'paul.vallace@rtspecialty.com')
    use_tls = os.environ.get('SMTP_USE_TLS', 'true').lower() in {'1', 'true', 'yes', 'on'}

    missing = [name for name, value in [
        ('SMTP_HOST', email_host),
        ('SMTP_USER', sender),
        ('SMTP_PASSWORD', password),
    ] if not value]


    if missing:
        print(f"Skipping email notification; missing settings: {', '.join(missing)}")
        return

    try:
        # Build the email
        msg = EmailMessage()
        msg['Subject'] = f"College Basketball predictions for {run_date}"
        msg['From'] = sender
        msg['To'] = recipient
        msg.set_content(
            f"Attachment contains {predictions_count} predictions generated on {run_date}.\n"
            "This email was sent automatically by train_model.py."
        )

        # Attach the Excel file
        with open(attachment_path, 'rb') as attachment_file:
            file_data = attachment_file.read()
            filename = os.path.basename(attachment_path)

        msg.add_attachment(
            file_data,
            maintype='application',
            subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=filename
        )

        # Send via SMTP
        with smtplib.SMTP(email_host, email_port) as server:
            if use_tls:
                server.starttls()
            server.login(sender, password)
            server.send_message(msg)

        print("✅ Email notification sent successfully.")

    except Exception as e:
        print(f"⚠️ Failed to send email notification: {e}")


def prepare_and_predict_today(model=None, scaler=None):
    """Generate predictions for today's games"""
    print("\n" + "="*80)
    print("STEP 4: GENERATING PREDICTIONS")
    print("="*80)
    
    # If model or scaler are not provided, train/load them using train_model()
    if model is None or scaler is None:
        model, scaler = train_model()
    
    # Try to find the most recent KenPom file
    yesterday = (datetime.now().date() - timedelta(days=1)).strftime('%Y-%m-%d')
    kenpom_file = f'/Users/PaulVallace/Desktop/College Basketball/historical data/Kenpom/kenpom_{yesterday}.xlsx'
    
    # If yesterday's file doesn't exist, try to find the most recent one
    if not os.path.exists(kenpom_file):
        kenpom_dir = '/Users/PaulVallace/Desktop/College Basketball/historical data/Kenpom'
        if os.path.exists(kenpom_dir):
            kenpom_files = [f for f in os.listdir(kenpom_dir) if f.startswith('kenpom_') and f.endswith('.xlsx')]
            if kenpom_files:
                kenpom_file = os.path.join(kenpom_dir, sorted(kenpom_files)[-1])
                print(f"⚠️  Yesterday's KenPom file not found. Using most recent: {os.path.basename(kenpom_file)}")
            else:
                raise FileNotFoundError(f"No KenPom files found in {kenpom_dir}")
        else:
            raise FileNotFoundError(f"KenPom directory not found: {kenpom_dir}")
    
    kenpom_today = pd.read_excel(kenpom_file)
    
    # Create dictionary of team stats
    team_stats = {}
    for _, row in kenpom_today.iterrows():
        team_stats[standardize_team_name(row['Team'])] = {
            'ORtg': row['ORtg'],
            'DRtg': row['DRtg'],
            'NetRtg': row['NetRtg'],
            'AdjT': row['AdjT']
        }
    
    # Load today's games
    today = datetime.now().date().strftime('%Y-%m-%d')
    games_file = f'/Users/PaulVallace/Desktop/College Basketball/Game Days/{today}.csv'
    
    if not os.path.exists(games_file):
        print(f"No games file found for today: {games_file}")
        return pd.DataFrame()
    
    games_df = pd.read_csv(games_file)
    
    predictions = []
    for _, row in games_df.iterrows():
        try:
            away_team, home_team, spread = parse_matchup(row['Matchups'])
            
            away_team = standardize_team_name(away_team)
            home_team = standardize_team_name(home_team)
            
            away_stats = team_stats.get(away_team)
            home_stats = team_stats.get(home_team)
            
            if not away_stats or not home_stats:
                print(f"Missing stats for: {away_team} at {home_team}")
                continue
            
            result = predict_game(model, scaler, home_stats, away_stats, spread)
            
            predictions.append({
                'Away_Team': away_team,
                'Home_Team': home_team,
                'Spread': spread,
                'Will_Cover': result['will_cover'],
                'Confidence': result['confidence'],
                'Home_ORtg': home_stats['ORtg'],
                'Home_DRtg': home_stats['DRtg'],
                'Home_AdjT': home_stats['AdjT'],
                'Away_ORtg': away_stats['ORtg'],
                'Away_DRtg': away_stats['DRtg'],
                'Away_AdjT': away_stats['AdjT'],
                'NetRtg_Diff': home_stats['NetRtg'] - away_stats['NetRtg'],
                'Date': datetime.now().date()
            })
            
        except Exception as e:
            print(f"Error processing game: {row['Matchups']}")
            print(f"Error details: {str(e)}")
            continue
    
    predictions_df = pd.DataFrame(predictions)
    
    # CHECK IF EMPTY BEFORE SORTING
    if len(predictions_df) == 0:
        print("\n⚠️  No valid predictions could be generated.")
        print("Possible issues:")
        print("  1. Game file format doesn't match 'Team1 at Team2 (spread)'")
        print("  2. Team names don't match between your game file and KenPom data")
        return predictions_df
    
    predictions_df = predictions_df.sort_values('Confidence', ascending=False)
    
    return predictions_df

def print_predictions(predictions_df: pd.DataFrame, confidence_threshold: float = 0.65):
    """Print predictions with analysis"""
    print("\nHigh Confidence Predictions:")
    print("=" * 80)
    
    high_conf_predictions = predictions_df[predictions_df['Confidence'] >= confidence_threshold]
    
    for _, game in high_conf_predictions.iterrows():
        print(f"\n{game['Away_Team']} at {game['Home_Team']}")
        print(f"Spread: {game['Spread']}")
        print(f"Prediction: {'Home' if game['Will_Cover'] else 'Away'} to cover")
        print(f"Confidence: {game['Confidence']:.2f}")
        print(f"Net Rating Difference: {game['NetRtg_Diff']:.1f}")
        print(f"Offensive Rating Comparison: {game['Home_ORtg']:.1f} vs {game['Away_ORtg']:.1f}")
        print(f"Defensive Rating Comparison: {game['Home_DRtg']:.1f} vs {game['Away_DRtg']:.1f}")
        print(f"Adjusted Tempo Comparison: {game['Home_AdjT']:.1f} vs {game['Away_AdjT']:.1f}")
    
    print("\nSummary:")
    print(f"Total Games Analyzed: {len(predictions_df)}")
    print(f"High Confidence Plays ({confidence_threshold*100}%+): {len(high_conf_predictions)}")

if __name__ == "__main__":
    try:
        print("Loading and training model...")
        predictions = prepare_and_predict_today()
        today = date.today().strftime('%Y-%m-%d')
        
        # 🔒 NEW: Handle case with no predictions
        if predictions is None or predictions.empty:
            print("\nNo predictions were generated. Skipping printing and saving.")
        else:
            # Print predictions
            print_predictions(predictions, confidence_threshold=0.65)
            
            # Save to Excel for further analysis
            output_path = f'/Users/PaulVallace/Desktop/College Basketball/Model/predictions_{today}.xlsx'
            predictions.to_excel(output_path, index=False)
            print(f"\nPredictions have been saved to 'predictions_{today}.xlsx'")

            # Email the predictions file if SMTP credentials are configured
            send_predictions_email(output_path, len(predictions), today)
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        print("Please check the data files and their format.")
        import traceback
        traceback.print_exc()
