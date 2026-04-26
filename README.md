# 💳 FinSight AI — Financial Transaction Categorizer

An AI-powered web app that automatically categorizes bank transactions, visualizes spending patterns, and detects anomalies using Groq (LLaMA 3.3 70B).

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red) ![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange)
## Features

- **AI Categorization** — Groq LLaMA 3.3 70B tags every transaction into 12 categories (Food, Shopping, Transport, etc.)
- **Spending Dashboard** — Interactive pie chart, bar chart, and daily trend line
- **Anomaly Detection** — Flags transactions that spike >2σ above your category average
- **CSV Export** — Download the categorized data for further analysis
- **Dark UI** — Clean, modern dark theme built with Streamlit

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Plotly |
| AI Engine | Groq LLaMA 3.3 70B |
| Data Processing | Pandas |
| Deployment | Streamlit Cloud |

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-finance-categorizer
cd ai-finance-categorizer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
Go to [console.groq.com/keys](console.groq.com/keys) — free, no credit card needed.

### 4. Run the app
```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Paste your Groq API key in the sidebar
3. Upload a CSV with columns: `Date, Description, Amount, Type`
4. Click **Categorize with AI**
5. Explore your spending dashboard!

## CSV Format

```
Date,Description,Amount,Type
2026-04-01,Swiggy Order,450.00,Debit
2026-04-01,Salary Credit,85000.00,Credit
```

A `sample_transactions.csv` is included for testing.

## Deployment on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set main file as `app.py`
4. Deploy — live URL in 2 minutes!

## Architecture

```
CSV Upload → Pandas (parse & clean)
                ↓
           Groq LLaMA 3.3 70B
         (batch categorization via prompt)
                ↓
         Anomaly Detection (statistical z-score)
                ↓
         Plotly Charts + Streamlit Dashboard
```

## Author

Satakshi Anand Srivastava  
B.Tech Information Technology, KIIT University  
