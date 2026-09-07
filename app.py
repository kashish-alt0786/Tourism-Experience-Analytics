import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "cleaned_tourism_data.csv")
reg = joblib.load(ROOT / "models/rating_regression_model.joblib")
clf = joblib.load(ROOT / "models/visit_mode_classifier.joblib")
rec_art = joblib.load(ROOT / "models/recommendation_artifacts.joblib")
sim = rec_art["similarity"]
avg_rating = rec_art["avg_rating"]

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🌍", layout="wide")
st.title("🌍 Tourism Experience Analytics")
st.caption("Classification, prediction, and recommendation system built from the supplied tourism transaction data.")

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Predict Rating", "Predict Visit Mode", "Recommendations"])

with tab1:
    st.subheader("Dataset & Model Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Users", f"{df.UserId.nunique():,}")
    c3.metric("Attractions", f"{df.AttractionId.nunique():,}")
    c4.metric("Average Rating", f"{df['Rating'].mean():.2f}")
    st.divider()
    a, b = st.columns(2)
    with a:
        st.write("Visit Mode Distribution")
        st.bar_chart(df["VisitMode"].value_counts())
    with b:
        st.write("Rating Distribution")
        st.bar_chart(df["Rating"].value_counts().sort_index())
    st.write("Top Attractions by Transaction Volume")
    top = df.groupby("AttractionId").size().sort_values(ascending=False).head(10)
    st.bar_chart(top)

with tab2:
    st.subheader("Predict Attraction Rating")
    attraction = st.selectbox("Attraction ID", sorted(df.AttractionId.unique()))
    visit_mode = st.selectbox("Visit Mode", sorted(df.VisitMode.unique()))
    year = st.number_input("Visit Year", int(df.VisitYear.min()), int(df.VisitYear.max()), int(df.VisitYear.median()))
    month = st.number_input("Visit Month", 1, 12, 6)
    continent = st.number_input("Continent ID", int(df.ContinentId.min()), int(df.ContinentId.max()), int(df.ContinentId.median()))
    region = st.number_input("Region ID", int(df.RegionId.min()), int(df.RegionId.max()), int(df.RegionId.median()))
    country = st.number_input("Country ID", int(df.CountryId.min()), int(df.CountryId.max()), int(df.CountryId.median()))
    city = st.number_input("City ID", int(df.CityId.min()), int(df.CityId.max()), int(df.CityId.median()))
    atype = int(df.loc[df.AttractionId == attraction, "AttractionTypeId"].iloc[0])
    acity = int(df.loc[df.AttractionId == attraction, "AttractionCityId"].iloc[0])
    if st.button("Predict Rating"):
        X = pd.DataFrame([{
            "VisitYear": year, "VisitMonth": month, "VisitMode": visit_mode,
            "ContinentId": continent, "RegionId": region, "CountryId": country, "CityId": city,
            "AttractionId": attraction, "AttractionTypeId": atype, "AttractionCityId": acity
        }])
        st.success(f"Predicted rating: {float(reg.predict(X)[0]):.2f} / 5")

with tab3:
    st.subheader("Predict Visit Mode")
    attraction = st.selectbox("Attraction ID", sorted(df.AttractionId.unique()), key="clf_attr")
    year = st.number_input("Visit Year", int(df.VisitYear.min()), int(df.VisitYear.max()), int(df.VisitYear.median()), key="clf_year")
    month = st.number_input("Visit Month", 1, 12, 6, key="clf_month")
    continent = st.number_input("Continent ID", int(df.ContinentId.min()), int(df.ContinentId.max()), int(df.ContinentId.median()), key="clf_cont")
    region = st.number_input("Region ID", int(df.RegionId.min()), int(df.RegionId.max()), int(df.RegionId.median()), key="clf_region")
    country = st.number_input("Country ID", int(df.CountryId.min()), int(df.CountryId.max()), int(df.CountryId.median()), key="clf_country")
    city = st.number_input("City ID", int(df.CityId.min()), int(df.CityId.max()), int(df.CityId.median()), key="clf_city")
    atype = int(df.loc[df.AttractionId == attraction, "AttractionTypeId"].iloc[0])
    acity = int(df.loc[df.AttractionId == attraction, "AttractionCityId"].iloc[0])
    if st.button("Predict Visit Mode"):
        X = pd.DataFrame([{
            "VisitYear": year, "VisitMonth": month, "ContinentId": continent, "RegionId": region,
            "CountryId": country, "CityId": city, "AttractionId": attraction,
            "AttractionTypeId": atype, "AttractionCityId": acity
        }])
        st.success(f"Predicted visit mode: {clf.predict(X)[0]}")

with tab4:
    st.subheader("Attraction Recommendations")
    selected = st.selectbox("Choose an attraction you liked", sorted(sim.index.tolist()))
    k = st.slider("Number of recommendations", 3, 10, 5)
    if st.button("Recommend"):
        scores = sim.loc[selected].copy()
        scores.loc[selected] = -1
        out = pd.DataFrame({"AttractionId": scores.index, "Similarity": scores.values})
        out["AverageRating"] = out["AttractionId"].map(avg_rating).fillna(df["Rating"].mean())
        out = out.sort_values(["Similarity", "AverageRating"], ascending=False).head(k)
        st.dataframe(out, use_container_width=True, hide_index=True)
