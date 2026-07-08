<h1 align="left">bubble-breaker</h1>

<p align="left">
    <strong>A custom feed that automatically extracts news items outside users' interest domains (beyond their "filter bubbles") from specified news media RSS feeds, then structures and delivers them using LLM (Gemini API)</strong>
</p>

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10">
  </a>
  <a href="https://github.com/ph-cookie/bubble-breaker/actions">
    <img src="https://github.com/ph-cookie//bubble-breaker/actions/workflows/update-rss.yml/badge.svg" alt="Build Status">
  </a>
  <a href="https://ph-cookie.github.io/bubble-breaker/">
    <img src="https://img.shields.io/badge/Hostedon-GitHubPages-brightgreen.svg" alt="GitHub Pages">
  </a>
  <a href="https://aistudio.google.com/">
    <img src="https://img.shields.io/badge/Poweredby-Gemini3.1FlashLite-orange.svg" alt="Gemini API">
  </a>
  <a href="https://huggingface.co/intfloat/multilingual-e5-small">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-multilingual--e5--small-yellow.svg" alt="Hugging Face">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

日本語版はこちら → [README.ja.md](README.ja.md)


## 1. System Overview

This system is designed to break through the filter bubble effect (the algorithmic bias toward users' existing interests) in modern information consumption by implementing an inverse filtering approach to news delivery.

Utilizing user-defined "interest clusters" and "non-interest clusters," the system employs a three-stage filtering process: ① `negative cluster difference scoring`, ② `NMF topic modeling`, and ③ `statistical dynamic thresholding`, to identify and extract news articles that fall outside users' areas of interest.

The selected articles are then processed using LLM technology to: rewrite titles, structure the content into four sections ("What Happened," "Background," "Implications," and "Relevance to Interests"), and distribute them via GitHub Pages as both a new RSS feed (XML format) and an index page.

> [!WARNING]
> The LLM-generated explanations and summaries produced by this system are intended primarily as "introductions" and "supplements" to help users develop interest or understanding in fields outside their expertise.
> While efforts have been made to improve factual accuracy through techniques like Google Search Grounding, there may be instances of hallucination characteristic of LLM outputs.
> **For precise factual information and details, please always verify by reading the original news articles (source materials) linked within the feed.**

## 2. System flow

<details>
<summary style="color: #666; font-size: 0.9em; cursor: pointer;">
🔍 <b>Click to view the system flow diagram</b>
</summary>

```mermaid
graph TD
    A[GitHub Actions Scheduled Run] --> B[Cache Load: Read URLs & Saved Articles]
    B --> C[Parse Multiple RSS Feeds]
    C --> D{Read URL or Excluded Keywords?}
    D -- Yes --> E[Skip / Log Excluded Items]
    D -- No --> F[Accumulate New Articles]
    
    F --> G{Are There New Articles?}
    G -- No --> K
    G -- Yes --> H[Thorough HTML Sanitization & Bulk Vectorization using SentenceTransformer]
    
    H --> I[3-Stage Filtering:<br/>1. Negative Example Difference Score<br/>2. NMF Score<br/>3. Statistical Dynamic Threshold]
    I -- Threshold Exceeded --> J[Extract Non-Relevant News]
    I -- No Cases --> J2[Force Extraction of Top 3 Ensemble Score Items]
    
    J --> L[Structured Generation via LLM & Title Rewriting]
    J2 --> L
    L -- API Limits, etc.: Automatic Retries using tenacity & Fallback Mechanism --> M[Dynamic MIME Type Detection / HTML Format Generation]
    
    M --> K[Combine New and Existing Articles, Extract Latest 30 Items]
    K --> N[Cache Save: Latest Read URLs & Article Data]
    N --> O[Generate Custom RSS File using feedgen]
    O --> Q[Automatic Generation of index.html Guide Page]
    Q --> P[Automated Deployment to GitHub Pages]
```

</details>

## 3. Key Features

### Advanced 3-Stage Inverse Filtering Algorithm

Rather than relying on simple text comparison, the system achieves high-precision filtering of irrelevant content by ensembling three metrics: the similarity difference between "interest" and "non-interest" clusters (Negative Example Difference), Latent Topic Analysis using NMF (Non-negative Matrix Factorization), and a statistical dynamic threshold based on the formula `mean + K_SIGMA*std`

### Contextual Supplementation via Google Search Grounding

By leveraging the Gemini API's search integration, the system avoids reliance solely on the LLM’s pre-trained knowledge. It fetches the latest real-time information and structural background context to generate accurate and up-to-date explanations.

### Persistent Feed Maintenance via Caching

Utilizing `actions/cache`, the system persists processed URLs (up to 500) and generated article data in JSON format. This prevents redundant processing while maintaining a constant stream of the latest 30 articles, ensuring stable feed delivery even when no new content is ingested.

### Robust Error Handling & API Rate Limit Management

To accommodate free-tier API constraints (e.g., 15 RPM), the LLM generation module implements a 5-second pre-emptive throttling mechanism. Additionally, it employs exponential backoff via `tenacity` to automatically retry failed requests up to five times. In the event of total generation failure, the system falls back to summarizing the original article, ensuring continuous operation.

### Optimized RSS Presentation

Each article summary explicitly includes the source name and similarity score. The system features robust `enclosure` support with dynamic MIME type detection for image URLs and utilizes inline CSS to optimize HTML spacing and readability.

## 4. Technical Stack

* Programming Language: Python 3.10
* LLM SDK: google-genai
* Generation Model: gemini-3.1-flash-lite (with Google Search Grounding enabled)
* Embedding Model: sentence-transformers (intfloat/multilingual-e5-small)
* Retry Control: tenacity
* RSS Parsing/Generation: feedparser, feedgen
* Infrastructure: GitHub Actions for CI/CD and GitHub Pages for static hosting

## 5. Repository Structure

* `main.py`: The main script that handles the entire pipeline from RSS retrieval, filtering, LLM explanation generation, and file output
* `processed_urls.json`: A cache file automatically generated to store a list of read URLs and the most recent output articles
* `requirements.txt`: List of dependency packages
* `.github/workflows/generate-rss.yml`: GitHub Actions configuration file for scheduled execution and cache management

## 6. Setup Instructions

1. **Repository Preparation**

   Clone or fork this repository to your own GitHub account.

2. **Obtaining API Keys and Tokens**

   * Retrieve your Gemini API key from [Google AI Studio](https://aistudio.google.com/).
   * Obtain an Access Token (with Read permissions) from [Hugging Face](https://huggingface.co/settings/tokens).

3. **Configuring GitHub Secrets**

   In your GitHub repository, go to `Settings` > `Secrets and variables` > `Actions` and add the following environment variables:
   * `API_KEY1`: Your obtained Gemini API key
   * `HF_TOKEN1`: Your obtained Hugging Face token

4. **Configuring GitHub Variables/Environment Variables (Optional)**

    * `USE_GROUNDING`: Setting this to "true" enables Google Search Grounding in LLM generation (note API usage limits).

5. **Enabling GitHub Pages**

    In your GitHub repository, navigate to `Settings` > `Pages` and properly configure the Build and deployment Source to "GitHub Actions" or similar settings.

6. **Customizing Source Code**

    Please customize the following variables in `main.py` according to your needs:
    * `SOURCE_RSS_URLS`: List of RSS URLs for news sources you want to fetch
    * `INTEREST_TEXTS`: Your current areas of interest (interest clusters)
    * `DISINTEREST_TEXTS`: Areas you want to avoid (non-interest clusters)
    * `EXCLUDE_KEYWORDS`: List of keywords to filter out paid articles, etc.
    * `K_SIGMA` / `N_TOPICS`: Adjust statistical thresholds and number of topics for filtering

## 7. Usage Instructions

Upon successful completion of the GitHub Actions workflow, the project will be automatically deployed to the GitHub Pages environment, generating an index page (index.html) accessible at the following URL:

`https://[your-github-username].github.io/[repository-name]/`

Please subscribe to this project by registering the rss.xml file in the same directory with any RSS reader application, such as Feedly or NetNewsWire.

## LICENSE

MIT
