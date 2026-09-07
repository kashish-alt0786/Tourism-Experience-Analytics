# Final Project Report — Tourism Experience Analytics

## 1. Project Overview
This project develops an end-to-end tourism analytics system covering regression, classification, and attraction recommendation using the supplied tourism transaction dataset.

## 2. Data Preparation
The source tables were joined using the supplied identifiers:
Transaction → User → Mode → Item → Type → City → Country → Region → Continent.
The final analytical dataset contains 52,930 transactions, 33,530 users, and 30 attractions. Duplicate transaction rows were checked, and missing CityId values were handled before modeling.

`Updated_Item.xlsx` was preserved in the source-data folder but was not used for the canonical transaction join because its structure does not match the 30-attraction Item table used by the transactions.

## 3. Regression
**Goal:** predict attraction ratings from visit, geographic, attraction, and visit-mode variables.

**Model:** RandomForestRegressor.

**Features:** VisitYear, VisitMonth, VisitMode, ContinentId, RegionId, CountryId, CityId, AttractionId, AttractionTypeId, AttractionCityId.

**Evaluation:** 80/20 train-test split, random_state=42.

**Results:** RMSE = **0.922** and R² = **0.097**.

The R² indicates that the available structured variables explain only a modest share of rating variation. This is an important limitation rather than something to hide.

## 4. Classification
**Goal:** predict VisitMode.

**Model:** ExtraTreesClassifier with balanced class weighting.

**Evaluation:** stratified 80/20 split, random_state=42.

**Results:** Accuracy = **41.8%**, weighted Precision = **45.1%**, weighted Recall = **41.8%**, weighted F1 = **43.1%**.

## 5. Recommendation
A content-based recommendation approach was implemented using attraction type and attraction city, with cosine similarity and historical average rating used for ranking.

For users with at least two distinct attractions, one attraction was held out and recommendations were generated from the remaining history. The evaluation was capped at 3,000 eligible users.

**Precision@5 = 7.4%**  
**Recall@5 = 37.1%**

The recommender retrieves a useful portion of held-out attractions, while low Precision@5 shows that the current small catalog and limited item features leave room for improvement.

## 6. Key Insights
- Visit-mode behavior is unevenly distributed, so class imbalance matters.
- Ratings are concentrated toward the upper end of the rating scale.
- The dataset contains repeated user-attraction interactions that support recommendation modeling.
- The current structured variables provide limited explanatory power for ratings, suggesting that richer behavioral and review-level features would improve prediction.

## 7. Deployment
The project includes a Streamlit application with four sections:
1. Dashboard
2. Rating prediction
3. Visit-mode prediction
4. Attraction recommendations

The app is designed for deployment through Streamlit Community Cloud using GitHub.

## 8. Limitations
- The canonical attraction catalog contains 30 attractions.
- Rating prediction has modest R².
- VisitMode classes are not perfectly balanced.
- Random train-test splits can be optimistic for temporal/user-history scenarios.
- Recommendation evaluation uses a relatively small item catalog and Precision@5/Recall@5 rather than a full ranking-metric suite.

## 9. Future Improvements
- Time-based and user-grouped validation.
- Richer user-history features.
- XGBoost/LightGBM comparison.
- Review-text sentiment and embeddings.
- Hybrid collaborative + content recommendation.
- MAP@K, NDCG@K, and coverage metrics.
- More granular travel context and attraction metadata.
