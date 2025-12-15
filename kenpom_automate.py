import os
import time
from datetime import datetime, timedelta

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def make_driver(headless: bool = False) -> webdriver.Chrome:
    """Create a Selenium 4 compatible Chrome driver."""
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    if headless:
        chrome_options.add_argument("--headless=new")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


from typing import List

def get_date_range(start_date_str: str, end_date_str: str) -> List[str]:
    ...

    """Generate a list of dates between start and end date (YYYY-MM-DD)."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return dates


def scrape_kenpom_data(driver: webdriver.Chrome, date_str: str) -> list[list]:
    """Scrape archived KenPom ratings table for a specific date."""
    url = f"https://kenpom.com/archive.php?d={date_str}"
    driver.get(url)

    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, "ratings-table"))
    )

    rows = driver.find_elements(By.XPATH, "//table[@id='ratings-table']//tr")

    data = []
    for row in rows[1:]:
        cells = row.find_elements(By.XPATH, ".//td")
        if len(cells) >= 7:
            # Expected columns: Rk, Team, Conf, NetRtg, ORtg, DRtg, AdjT (+ maybe extras)
            row_data = [
                cells[0].text.strip(),
                cells[1].text.strip(),
                cells[2].text.strip(),
                cells[3].text.strip(),
                cells[4].text.strip(),
                cells[5].text.strip(),
                cells[6].text.strip(),
                date_str,
            ]
            data.append(row_data)

    return data


def clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert numeric columns with + signs to floats."""
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(r"\+", "", regex=True)
                .replace("nan", pd.NA)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def login_and_scrape_yesterday(email: str, password: str, headless: bool = False) -> str:
    """Login to KenPom and scrape yesterday's archive. Returns the daily file path."""
    driver = make_driver(headless=headless)

    try:
        driver.get("https://kenpom.com/")

        email_field = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "input[type='email'][name='email']")
            )
        )
        password_field = driver.find_element(
            By.CSS_SELECTOR, "input[type='password'][name='password']"
        )
        submit_button = driver.find_element(
            By.CSS_SELECTOR, "input[type='submit'][name='submit']"
        )

        email_field.clear()
        email_field.send_keys("tgaston@wisc.edu") 
        password_field.clear()
        password_field.send_keys("Jakel123")
        submit_button.click()

        # Wait for a post-login element (site varies; this is a decent generic check)
        WebDriverWait(driver, 15).until(lambda d: "kenpom" in d.current_url.lower())
        time.sleep(1)

        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        dates = get_date_range(yesterday, yesterday)

        all_rows = []
        for d in dates:
            print(f"Scraping data for {d}")
            all_rows.extend(scrape_kenpom_data(driver, d))
            time.sleep(0.5)

        columns = ["Rk", "Team", "Conf", "NetRtg", "ORtg", "DRtg", "AdjT", "Date"]
        df = pd.DataFrame(all_rows, columns=columns)

        df = clean_numeric(df, ["Rk", "NetRtg", "ORtg", "DRtg", "AdjT"])

        # Paths
        base_path = "/Users/PaulVallace/Desktop/College Basketball/historical data/Kenpom"
        os.makedirs(base_path, exist_ok=True)

        daily_file = os.path.join(base_path, f"kenpom_{yesterday}.xlsx")
        historical_file = os.path.join(base_path, "historical_kenpom.xlsx")

        # Save daily file
        df.to_excel(daily_file, index=False)
        print(f"\n✅ Daily data saved to {daily_file}")

        # Append to historical
        if os.path.exists(historical_file):
            historical_df = pd.read_excel(historical_file)
            combined_df = pd.concat([historical_df, df], ignore_index=True)

            original_len = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["Team", "Date"], keep="last")
            removed = original_len - len(combined_df)

            combined_df = combined_df.sort_values(["Date", "Rk"]).reset_index(drop=True)
            combined_df.to_excel(historical_file, index=False)

            print(f"✅ Updated historical file: {historical_file}")
            print(f"   Total rows: {len(combined_df)} (removed {removed} duplicates)")
        else:
            df.to_excel(historical_file, index=False)
            print(f"📘 Created new historical file: {historical_file}")

        print("\n📋 First few rows:")
        print(df.head())

        return daily_file

    finally:
        driver.quit()


if __name__ == "__main__":
    # ⚠️ Better: use environment variables / Streamlit secrets instead of hardcoding.
    EMAIL = "tgaston@wisc.edu"
    PASSWORD = "Jakel123"

    try:
        output_file = login_and_scrape_yesterday(EMAIL, PASSWORD, headless=False)
        print("\n🎉 Script completed successfully!")
        print(f"Daily file: {output_file}")
    except Exception as e:
        print(f"❌ Script failed: {e}")
