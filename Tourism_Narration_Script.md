# Narration Script — Tourism Experience Analytics

**~1,150 words → ~7 minutes at a natural AI-voice pace (~150–160 wpm).**

---

### Slide 1 — Title (~25 sec)
Good morning everyone. I'm presenting my project: Tourism Experience Analytics — a system for classification, prediction, and recommendation, built for the tourism domain. The core idea is to turn raw traveler and attraction data into three useful outputs: recommendations, rating predictions, and visit-mode classification. Over the next few minutes I'll walk through the problem, the three modeling objectives, the methodology, the data, and the final results.

### Slide 2 — The Problem (~35 sec)
Tourism platforms can use historical traveler and attraction data to improve personalization and understand visitor behavior. This project combines these goals in one integrated pipeline. It focuses on personalized attraction recommendations, tourism analytics that identify popular attractions, prediction of visitor ratings, and classification of visit behavior. A shared cleaned dataset supports all three modeling tasks.

### Slide 3 — Three Modeling Objectives (~35 sec)
The project has three modeling objectives. First, a regression task predicts the rating a user may give a tourist attraction on a 1-to-5 scale. Second, a classification task predicts VisitMode from structured visit and geographic information. Third, a recommendation system suggests attractions using content-based similarity and historical attraction ratings.

### Slide 4 — Objective 1: Regression (~40 sec)
The first objective is rating prediction. A Random Forest regression model uses visit year, visit month, visit mode, geographic identifiers, attraction identifiers, attraction type, and attraction city. The model was evaluated with an 80/20 train-test split. The final RMSE is 0.922 and the R-squared value is 0.097. This means the available structured variables have limited explanatory power for rating variation, which is an important finding and a clear area for improvement.

### Slide 5 — Objective 2: Classification (~35 sec)
The second objective is VisitMode classification. An Extra Trees classifier with balanced class weighting was trained using visit timing, geographic identifiers, and attraction information. With a stratified 80/20 split, the model achieved 41.8 percent accuracy and a weighted F1-score of 43.1 percent. The result shows that the structured variables contain useful behavioral information, but additional user-history and contextual features could improve performance.

### Slide 6 — Objective 3: Recommendation (~35 sec)
The third objective is attraction recommendation. The implemented system is content-based rather than collaborative. It represents attractions using attraction type and attraction city, calculates cosine similarity, and uses historical average rating as a secondary ranking signal. For evaluation, one attraction was held out from eligible user histories and recommendations were generated from the remaining history. Precision@5 was 7.4 percent and Recall@5 was 37.1 percent.

### Slide 7 — The Data (~35 sec)
The supplied data is relational and spread across linked tables. Transaction data contains visit-level information and ratings. User data provides geographic information. The attraction, type, city, country, region, continent, and visit-mode tables provide the related attributes needed for analysis. These tables were joined using their supplied identifiers to create one analytical dataset containing 52,930 transactions, 33,530 users, and 30 canonical attractions.

### Slide 8 — Approach & Methodology (~40 sec)
The workflow followed several stages. First, the relational tables were cleaned and joined. Missing CityId values were handled, duplicate transactions were checked, and the canonical attraction table was identified. Next, exploratory analysis examined visit modes, ratings, and attraction popularity. Then the regression, classification, and recommendation models were trained and evaluated. Finally, the trained artifacts were connected to a Streamlit application for interactive use.

### Slide 9 — Model Evaluation (~35 sec)
Each task uses metrics suited to its purpose. Regression uses RMSE and R-squared. Classification uses accuracy, precision, recall, and F1-score. The recommendation system uses Precision@5 and Recall@5. These results are reported directly rather than presenting only expected or planned outcomes.

### Slide 10 — Deployment & Deliverables (~35 sec)
The project includes a Streamlit application with four sections: a dashboard, rating prediction, visit-mode prediction, and attraction recommendations. The repository also contains the project report, requirements file, source code, cleaned analytical data, trained model artifacts, and supporting datasets when uploaded. Streamlit Community Cloud can use the GitHub repository as the deployment source once all required runtime files are present.

### Slide 11 — Results & Insights (~40 sec)
The final results show three different levels of performance. The rating model has modest explanatory power, the VisitMode classifier provides moderate predictive performance, and the recommender retrieves a useful portion of held-out attractions but has low Precision@5. The analysis also shows uneven VisitMode behavior, ratings concentrated toward the upper end of the scale, and repeated user-attraction interactions that support recommendation modeling.

### Slide 12 — Conclusion & Next Steps (~40 sec)
In conclusion, this project turns a relational tourism dataset into an integrated analytics application covering prediction, classification, and recommendation. The current system is functional, but there are clear opportunities to improve it. Future work could use time-based and user-grouped validation, richer user-history features, stronger gradient-boosting models, review-text information, hybrid recommendation, and additional ranking metrics such as MAP@K and NDCG@K. Thank you.
