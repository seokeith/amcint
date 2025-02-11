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
openai_client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))

@st.cache_data(show_spinner=False)
def scrape_google(search):
    APIFY_API_KEY = st.secrets.get("APIFY_API_KEY", "")
    ACTOR_NAME = "lhotanok/google-news-scraper"

    if not APIFY_API_KEY:
        st.error("Apify API Key is missing! Please add it to Streamlit Secrets.")
        return pd.DataFrame()

    client = ApifyClient(APIFY_API_KEY)
    run_input = {
        "csvFriendlyOutput": False,
        "includeUnfilteredResults": False,
        "maxPagesPerQuery": 1,
        "mobileResults": False,
        "queries": [search],
        "resultsPerPage": 10,
        "saveHtml": False,
        "language": "US:en",
        "extractImages": True,
        "proxyConfiguration": {"useApifyProxy": True}
    }

    try:
        run = client.actor(ACTOR_NAME).call(run_input=run_input)
        results = [item for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
        urls = [result['url'] for item in results for result in item.get('organicResults', [])]
        return pd.DataFrame(urls, columns=['url'])
    except Exception as e:
        st.error(f"Apify API error: {e}")
        return pd.DataFrame()

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
def generate_content(prompt, model="gpt-4", max_tokens=1000, temperature=0.4):
    prompt = truncate_to_token_length(prompt, 2500)
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Simulate an exceptionally talented data-led news writer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.choices[0].message.content.strip()

@st.cache_data(show_spinner=False)
def generate_article(topic):
    st.text('Generating news article...')
    final_article = generate_content(topic)
    return final_article

def main():
    st.title('AI News Generator')
    topic = st.text_input("Enter topic:", "Add a keyword here")
    if st.button('Generate Content'):
        if st.secrets.get("OPENAI_API_KEY", ""):
            with st.spinner("Generating content..."):
                final_draft = generate_article(topic)
                st.markdown(final_draft)
        else:
            st.warning("Please enter your OpenAI API key in Streamlit secrets.")

if __name__ == "__main__":
    main()
