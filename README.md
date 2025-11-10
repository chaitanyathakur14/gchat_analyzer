

# ChatGPT Conversation Analyzer

## Overview

ChatGPT Conversation Analyzer is a Django + Machine Learning project designed to analyze ChatGPT’s responses for sentiment, repetition, topic trends, and overall quality.
It helps understand how AI communicates — identifying tone, clarity, and recurring patterns using multiple ML techniques.



## Preprocessing

* Exported ChatGPT data is parsed from JSON to CSV using `json.load()` and `pandas`.
* Extracts role, message, and conversation title.
* Final dataset: **2,416 messages from 271 conversations (shape: 2416 × 3)** — ready for analysis.



## Core ML Workflow

1. **Sentiment Analysis:** SVM + KNN classify assistant responses using VADER sentiment scores.
2. **Topic Modeling:** LDA identifies recurring conversation topics.
3. **Clustering:** K-Means and Hierarchical clustering group similar responses.
4. **Quality Scoring:** Weighted ANN-inspired model rates responses on readability and tone.
5. **Repetition Analysis:** TF-IDF + cosine similarity detect duplicate or near-similar messages.
6. **Dimensionality Reduction:** PCA and LDA visualize feature and sentiment separation.



## Django Setup

* `django-admin startproject foml_project` → Create project
* `python manage.py startapp analyzer` → Add ML app
* Includes views for sentiment, topics, quality, repetition, and similarity search
* Frontend built with HTML templates and static assets (CSS/JS)


## Deployment Notes

* Hosted via Render using **Gunicorn + Whitenoise** for static file handling.
* `DEBUG=False` and `CSRF_TRUSTED_ORIGINS` set for production.
* Database configured with `dj_database_url` (default: SQLite).


## Uniqueness

* Integrates over six ML algorithms in one workflow
* Works on real ChatGPT data for explainable insights
* Extendable for AI transparency and response quality monitoring





**GitHub Repository:** [https://github.com/chaitanyathakur14/gchat_analyzer](https://github.com/chaitanyathakur14/gchat_analyzer)  
**Live Demo (Render):** [https://gchat-analyzer.onrender.com](https://gchat-analyzer.onrender.com)


Would you like me to include a **short “How to run”** section (setup + runserver) at the end for clarity?
