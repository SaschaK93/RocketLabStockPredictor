# Import Libraries
import requests
from datetime import date, timedelta
from transformers import pipeline
from datetime import datetime
from pathlib import Path
import pandas as pd

# Set Up Finnhub API Key and Parameters for News Data Collection
FINNHUB_API_KEY = ""    # Replace with your actual Finnhub API key to access financial news data
news_symbol = "RKLB" 
url = "https://finnhub.io/api/v1/company-news" 

# Initialize an Empty List to Store All News Articles Collected from the API
all_news = []

# Define the Time Range for News Data Collection (Last 5 Years) and Collect News in 90-Day Blocks to Avoid API Limitations
end_date = date(2026, 6, 5)
start_date = end_date - timedelta(days=365 * 5)
current_to = end_date

# Loop to Collect News Data in 90-Day Blocks Until the Start Date is Reached
while current_to > start_date:

    current_from = current_to - timedelta(days=90)

    if current_from < start_date:
        current_from = start_date

    params = {
        "symbol": news_symbol,
        "from": current_from,
        "to": current_to,
        "token": FINNHUB_API_KEY
    }

    response = requests.get(url, params=params)
    block_news = response.json()

    print(
        current_from,
        "to",
        current_to,
        "News:",
        len(block_news)
    )

    all_news.extend(block_news)

    current_to = current_from - timedelta(days=1)

print("\n--- Total News Collected ---")
print(len(all_news))

news_data = all_news

# Save the Collected News Data to a CSV File for Further Processing and Analysis
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Convert the List of News Articles into a DataFrame for Easier Manipulation and Analysis
news_df = pd.DataFrame(news_data)

# Save the DataFrame to a CSV File in the Specified Data Folder with UTF-8 Encoding to Ensure Proper Handling of Special Characters
file_path = data_folder / "rklb_news_1year.csv"
news_df.to_csv(file_path, index=False, encoding="utf-8-sig")

print("Saved file here:")
print(file_path.resolve())

news_df = pd.read_csv("data/rklb_news_1year.csv")