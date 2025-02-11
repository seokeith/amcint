import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import newspaper
from newspaper import Article
import nltk
import statistics
import collections
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.collocations import QuadgramAssocMeasures, QuadgramCollocationFinder
import time
import openai
import re
import streamlit as st
from apify_client import ApifyClient
from transformers import GPT2Tokenizer

import json

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('trigram_collocations')
nltk.download('quadgram_collocations')

# Set OpenAI API Key
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

@st.cache_data(show_spinner=False)
def scrape_google(search):
    APIFY_API_URL = 'https://api.apify.com/v2'
    ACTOR_NAME = 'lhotanok/google-news-scraper'
    APIFY_API_KEY = st.secrets.get("APIFY_API_KEY", "")
    
    if not APIFY_API_KEY:
        st.error("Apify API Key is missing!")
        return pd.DataFrame()
    
    client = ApifyClient(APIFY_API_KEY)
    run_input = {
        "csvFriendlyOutput": False,
        "includeUnfilteredResults": False,
        "maxPagesPerQuery": 1,
        "mobileResults": False,
        "queries": [search],
        "resultsPerPage": 10,
        "saveHtml": False
    }
    
    run = client.actor(ACTOR_NAME).call(run_input=run_input)
    results = [item for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
    
    urls = [result['url'] for item in results for result in item.get('organicResults', [])]
    return pd.DataFrame(urls, columns=['url'])

@st.cache_data(show_spinner=False)
def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error scraping article: {e}")
        return ""

@st.cache_data(show_spinner=False)
def truncate_to_token_length(input_string, max_tokens=1700):
    tokens = tokenizer.tokenize(input_string)
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(truncated_tokens)

@st.cache_data(show_spinner=False)
def analyze_serps(query):
    df = scrape_google(query)
    for index, row in df.iterrows():
        url = row['url']
        article_text = scrape_article(url)
        df.at[index, 'Article Text'] = article_text
    return df

@st.cache_data(show_spinner=False)
def generate_content(prompt, model="gpt-4", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt, 2500)
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simulate an exceptionally talented data-led news writer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message["content"].strip()

@st.cache_data(show_spinner=False)
def generate_article(topic):
    st.text('Analyzing SERPs...')
    results = analyze_serps(topic)
    article_texts = " ".join(results['Article Text'].dropna())
    
    if not article_texts:
        st.error("No valid articles found for analysis.")
        return ""
    
    st.text('Generating news article...')
    final_article = generate_content(article_texts)
    return final_article

def main():
    st.title('AI News Generator')
    topic = st.text_input("Enter topic:", "Add a keyword here")
    if st.button('Generate Content'):
        if openai.api_key:
            with st.spinner("Generating content..."):
                final_draft = generate_article(topic)
                st.markdown(final_draft)
        else:
            st.warning("Please enter your OpenAI API key in Streamlit secrets.")

if __name__ == "__main__":
    main()
