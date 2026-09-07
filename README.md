# Tourism Experience Analytics

Classification, prediction, and recommendation system for tourism experiences.

## Streamlit app
The application entrypoint is `app.py` and dependencies are declared in `requirements.txt`.

The app is designed to deploy without storing large binary model files in GitHub. It first looks for the project's dataset files in `datasets/`; if they are not present, it downloads the normalized source tables at startup, builds the analytical dataset, trains compact Random Forest and Extra Trees models, and caches the results for the session.

## Features
- Interactive tourism analytics dashboard
- Attraction rating prediction using Random Forest Regression
- Visit mode prediction using Extra Trees Classification
- Content-based attraction recommendations using cosine similarity
- Attraction popularity and rating analysis
- Responsive Streamlit interface

## Data
The underlying tourism dataset contains 52,930 transactions, 33,530 users, and 30 attraction items. The published dataset describes normalized tables for transactions, users, attractions, visit modes, cities, countries, regions, continents, and attraction types.

## Deployment
Streamlit Community Cloud settings:
- Repository: `kashish-alt0786/Tourism-Experience-Analytics`
- Branch: `main`
- Main file: `app.py`

The repository now contains the code needed for deployment. Local dataset files can optionally be added under `datasets/`; otherwise the app uses its public fallback source and trains the compact models at startup.

## Project structure
```text
Tourism-Experience-Analytics/
├── app.py
├── requirements.txt
├── README.md
├── FINAL_REPORT.md
├── Tourism_Narration_Script.md
└── datasets/                 # optional local copies of the source tables
```

## Main results from the project analysis
- Rating regression: Random Forest, RMSE 0.922, R² 0.097
- Visit mode classification: Extra Trees, accuracy 41.8%, weighted F1 43.1%
- Recommendation evaluation: content-based Precision@5 about 7.4%, Recall@5 about 37.1%

The deployed app retrains compact runtime models for portability; the reported evaluation metrics above refer to the finalized project analysis and should not be interpreted as a guarantee of live prediction accuracy.
