# Import Libraries
import pandas as pd
from transformers import pipeline

# Load News Data
news_df = pd.read_csv(r"")  # Insert File Path

# Initialize FinBERT Sentiment Analysis Pipeline
classifier = pipeline("sentiment-analysis",model="ProsusAI/finbert")

# Create Lists to Store Sentiment Analysis Results
sentiment_labels = []
sentiment_scores = []
sentiment_values = []

# Perform Sentiment Analysis on Each News Article
for index, row in news_df.iterrows():

    # Combine Headline and Summary for Sentiment Analysis
    headline = str(row["headline"])
    summary = "" if pd.isna(row["summary"]) else str(row["summary"])
    text = headline + " " + summary

    # Analyze Sentiment of the Combined Text Using FinBERT
    result = classifier(text)[0]

    # Extract Sentiment Label and Score from the Result
    label = result["label"]
    score = result["score"]

    # Convert Sentiment Label and Score into a Single Sentiment Value for Easier Analysis (Positive = +Score, Negative = -Score, Neutral = 0)
    if label == "positive":
        sentiment_value = score
    elif label == "negative":
        sentiment_value = -score
    else:
        sentiment_value = 0
    
    # Append the Results to the Respective Lists
    sentiment_labels.append(label)
    sentiment_scores.append(score)
    sentiment_values.append(sentiment_value)

    # Print Progress Every 50 Articles to Monitor the Sentiment Analysis Process
    if index % 50 == 0:
        print("Processed:", index)

# Add Sentiment Analysis Results to the Original DataFrame
news_df["Sentiment_Label"] = sentiment_labels
news_df["Sentiment_Confidence"] = sentiment_scores
news_df["Sentiment_Value"] = sentiment_values

# Save the Updated DataFrame with Sentiment Analysis Results to a New CSV File
news_df.to_csv("rklb_news_with_sentiment.csv",index=False,encoding="utf-8-sig")


# Load the Updated News Data with Sentiment Analysis Results
# news_df = pd.read_csv(r"C:\Users\sasch\OneDrive\Desktop\Python\Projekte\AI Finance\rklb_news_with_sentiment.csv")
"""
# Analyze the Distribution of Sentiment Labels and Values to Understand the Overall Sentiment of the News Articles
print(news_df["Sentiment_Label"].value_counts())
print(news_df["Sentiment_Value"].describe())

# Convert the Datetime Column from Unix Timestamp to Readable Date Format and Extract Only the Date Part for Grouping
news_df["Date"] = pd.to_datetime(news_df["datetime"],unit="s").dt.date

# Group the News Data by Date and Calculate the Average Sentiment Value for Each Day to Create a Daily Sentiment Score that Can Be Merged with Stock Price Data for Analysis
daily_sentiment = (news_df.groupby("Date")["Sentiment_Value"].mean().reset_index())

# Convert the Date Column to Datetime Format and Set it as the Index of the DataFrame to Facilitate Merging with Stock Price Data Based on Date
daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"])

# Set the Date Column as the Index of the Daily Sentiment DataFrame to Prepare for Merging with Stock Price Data Based on Date
daily_sentiment = daily_sentiment.set_index("Date")

print(daily_sentiment.head())
"""