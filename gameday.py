import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime

def scrape_matchups_to_excel(url, output_file):
    try:
        # Send an HTTP GET request
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Debug: Print the page title to verify the correct page
        print("Page title:", soup.title.string)

        # Locate all `<a>` tags within `<td>` elements
        matchup_links = soup.select('td > a')  # CSS selector to find <a> within <td>

        if not matchup_links:
            print("No matchups found.")
            return

        # Collect all matchups
        matchups = []
        for link in matchup_links:
            matchups.append(link.text.strip())  # Add the text of each matchup

        # Create a DataFrame
        df = pd.DataFrame(matchups, columns=["Matchups"])

        # Save DataFrame to Excel
        df.to_csv(output_file, index=False)
        print(f"Matchups successfully saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# Use today's date
today = datetime.date.today()
url = "https://www.teamrankings.com/ncb/"


output_file = f"/Users/PaulVallace/Desktop/College Basketball/Game Days/{today.strftime('%Y-%m-%d')}.csv"
scrape_matchups_to_excel(url, output_file)
