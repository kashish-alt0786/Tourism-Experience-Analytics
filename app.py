import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🌍", layout="wide")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "datasets"

# Public fallback source for the normalized tourism dataset used by this project.
# Local files are always preferred when they exist in the repository.
SOURCE_BASE = "https://raw.githubusercontent.com/dharmindersinghvirk25-creator/Tourism_Experience_Analytics/main"
FILES = {
    "Transaction": "transaction.xlsx",
    "User": "user.xlsx",
    "Mode": "mode.xlsx",
    "Item": "item.xlsx",
    "Type": "type.xlsx",
    "City": "city.xlsx",
    "Country": "country.xlsx",
    "Region": "region.xlsx",
    "Continent": "continent.xlsx",
}

REG_FEATURES = [
    "VisitYear", "VisitMonth", "VisitMode", "ContinentId", "RegionId",
    "CountryId", "CityId", "AttractionId", "AttractionTypeId", "AttractionCityId"
]
CLF_FEATURES = [
    "VisitYear", "VisitMonth", "ContinentId", "RegionId", "CountryId", "CityId",
    "AttractionId", "AttractionTypeId", "AttractionCityId"
]


def _read_excel(name: str) -> pd.DataFrame:
    local = DATA_DIR / FILES[name]
    if not local.exists():
        candidates = list(DATA_DIR.glob("*")) if DATA_DIR.exists() else []
        wanted = FILES[name].lower()
        match = next((p for p in candidates if p.name.lower() == wanted), None)
        local = match if match is not None else local
    if local.exists():
        return pd.read_excel(local)
    url = f"{SOURCE_BASE}/{FILES[name]}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return pd.read_excel(io.BytesIO(response.read()))


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    tx = _read_excel("Transaction")
    user = _read_excel("User")
    mode = _read_excel("Mode")
    item = _read_excel("Item")
    typ = _read_excel("Type")
    city = _read_excel("City")
    country = _read_excel("Country")
    region = _read_excel("Region")
    continent = _read_excel("Continent")

    user = user.copy()
    user["CityId"] = user["CityId"].fillna(0)

    df = tx.merge(user, on="UserId", how="left")
    df = df.merge(mode, left_on="VisitMode", right_on="VisitModeId", how="left", suffixes=("", "_mode"))
    df = df.merge(item, on="AttractionId", how="left")
    df = df.merge(typ, on="AttractionTypeId", how="left")
    df = df.merge(city, left_on="AttractionCityId", right_on="CityId", how="left", suffixes=("", "_attraction_city"))
    df = df.merge(country, on="CountryId", how="left", suffixes=("", "_country"))
    df = df.merge(region, on="RegionId", how="left", suffixes=("", "_region"))
    df = df.merge(continent, on="ContinentId", how="left", suffixes=("", "_continent"))

    keep = [
        "TransactionId", "UserId", "VisitYear", "VisitMonth", "VisitMode", "VisitMode_mode",
        "AttractionId", "Rating", "ContinentId", "RegionId", "CountryId", "CityId",
        "AttractionCityId", "AttractionTypeId", "Attraction", "AttractionAddress",
        "AttractionType", "CityName", "Country", "Region", "Continent"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["CityId"] = pd.to_numeric(df["CityId"], errors="coerce").fillna(0).astype(int)
    for c in ["VisitYear", "VisitMonth", "VisitMode", "ContinentId", "RegionId", "CountryId",
              "AttractionId", "AttractionCityId", "AttractionTypeId", "Rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Rating", "AttractionId", "AttractionTypeId", "AttractionCityId"])
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    # Compact models are trained at runtime so the GitHub repository does not need
    # to store large binary joblib files. Local pre-trained models can be added later.
    reg = RandomForestRegressor(
        n_estimators=25, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    reg.fit(df[REG_FEATURES], df["Rating"])

    clf = ExtraTreesClassifier(
        n_estimators=30, max_depth=24, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    clf.fit(df[CLF_FEATURES], df["VisitMode"])
    return reg, clf


@st.cache_data(show_spinner=False)
def build_recommender(df: pd.DataFrame):
    item_features = (
        df.groupby("AttractionId")[["AttractionTypeId", "AttractionCityId"]]
        .first()
        .sort_index()
    )
    encoded = pd.get_dummies(item_features.astype(str))
    sim = pd.DataFrame(
        cosine_similarity(encoded), index=encoded.index, columns=encoded.index
    )
    avg_rating = df.groupby("AttractionId")["Rating"].mean()
    names = df.groupby("AttractionId")["Attraction"].first()
    return sim, avg_rating, names


try:
    with st.spinner("Loading tourism data and preparing the analytics engine..."):
        df = load_data()
    with st.spinner("Preparing prediction models..."):
        reg, clf = train_models(df)
    sim, avg_rating, attraction_names = build_recommender(df)
except Exception as exc:
    st.error("The app could not load the tourism dataset.")
    st.exception(exc)
    st.stop()

st.title("🌍 Tourism Experience Analytics")
st.caption(
    "Classification, rating prediction, exploratory analytics, and attraction recommendation "
    "built from the normalized tourism transaction dataset."
)

with st.sidebar:
    st.header("Project")
    st.write("**52,930+ tourism transactions**")
    st.write("**30 attractions**")
    st.write("**33,530 users in the source dataset**")
    st.divider()
    st.info("Models are trained automatically when the app starts. Cached results make later interactions faster.")


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "⭐ Predict Rating", "🧳 Predict Visit Mode", "📍 Recommendations"
])

with tab1:
    st.subheader("Dataset & Analytics Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Users", f"{df['UserId'].nunique():,}")
    c3.metric("Attractions", f"{df['AttractionId'].nunique():,}")
    c4.metric("Average Rating", f"{df['Rating'].mean():.2f} / 5")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Visit Mode Distribution")
        mode_counts = df["VisitMode_mode"].fillna(df["VisitMode"].astype(str)).value_counts()
        st.bar_chart(mode_counts)
    with right:
        st.markdown("#### Rating Distribution")
        st.bar_chart(df["Rating"].value_counts().sort_index())

    st.markdown("#### Most Visited Attractions")
    top = df.groupby(["AttractionId", "Attraction"]).size().reset_index(name="Transactions")
    top = top.sort_values("Transactions", ascending=False).head(10).set_index("Attraction")
    st.bar_chart(top["Transactions"])

    st.markdown("#### Average Rating by Attraction")
    rating_table = (
        df.groupby(["AttractionId", "Attraction"])["Rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "AverageRating", "count": "Ratings"})
        .sort_values("AverageRating", ascending=False)
    )
    st.dataframe(rating_table, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Predict Attraction Rating")
    st.write("Estimate the rating a visitor may give an attraction from visit and geographic context.")

    attractions = sorted(df["AttractionId"].unique().tolist())
    attraction = st.selectbox(
        "Attraction", attractions,
        format_func=lambda x: f"{int(x)} — {attraction_names.get(x, 'Unknown attraction')}"
    )
    visit_mode = st.selectbox(
        "Visit Mode", sorted(df["VisitMode"].dropna().unique().tolist()),
        format_func=lambda x: f"{int(x)} — {df.loc[df['VisitMode'].eq(x), 'VisitMode_mode'].dropna().iloc[0] if not df.loc[df['VisitMode'].eq(x), 'VisitMode_mode'].dropna().empty else 'Unknown'}"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("Visit Year", int(df["VisitYear"].min()), int(df["VisitYear"].max()), int(df["VisitYear"].median()), key="reg_year")
        month = st.number_input("Visit Month", 1, 12, 6, key="reg_month")
        continent = st.number_input("Continent ID", int(df["ContinentId"].min()), int(df["ContinentId"].max()), int(df["ContinentId"].median()), key="reg_cont")
    with col2:
        region = st.number_input("Region ID", int(df["RegionId"].min()), int(df["RegionId"].max()), int(df["RegionId"].median()), key="reg_region")
        country = st.number_input("Country ID", int(df["CountryId"].min()), int(df["CountryId"].max()), int(df["CountryId"].median()), key="reg_country")
        city = st.number_input("City ID", int(df["CityId"].min()), int(df["CityId"].max()), int(df["CityId"].median()), key="reg_city")
    with col3:
        row = df.loc[df["AttractionId"].eq(attraction)].iloc[0]
        st.metric("Attraction Type", int(row["AttractionTypeId"]))
        st.metric("Attraction City", int(row["AttractionCityId"]))

    if st.button("Predict Rating", type="primary"):
        X = pd.DataFrame([{
            "VisitYear": year, "VisitMonth": month, "VisitMode": visit_mode,
            "ContinentId": continent, "RegionId": region, "CountryId": country, "CityId": city,
            "AttractionId": attraction, "AttractionTypeId": int(row["AttractionTypeId"]),
            "AttractionCityId": int(row["AttractionCityId"])
        }])
        prediction = float(np.clip(reg.predict(X)[0], 1, 5))
        st.success(f"Predicted rating: **{prediction:.2f} / 5**")

with tab3:
    st.subheader("Predict Visit Mode")
    st.write("Estimate the likely visit mode from timing, location, and attraction context.")
    attraction = st.selectbox(
        "Attraction", attractions, key="clf_attr",
        format_func=lambda x: f"{int(x)} — {attraction_names.get(x, 'Unknown attraction')}"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("Visit Year", int(df["VisitYear"].min()), int(df["VisitYear"].max()), int(df["VisitYear"].median()), key="clf_year")
        month = st.number_input("Visit Month", 1, 12, 6, key="clf_month")
        continent = st.number_input("Continent ID", int(df["ContinentId"].min()), int(df["ContinentId"].max()), int(df["ContinentId"].median()), key="clf_cont")
    with col2:
        region = st.number_input("Region ID", int(df["RegionId"].min()), int(df["RegionId"].max()), int(df["RegionId"].median()), key="clf_region")
        country = st.number_input("Country ID", int(df["CountryId"].min()), int(df["CountryId"].max()), int(df["CountryId"].median()), key="clf_country")
        city = st.number_input("City ID", int(df["CityId"].min()), int(df["CityId"].max()), int(df["CityId"].median()), key="clf_city")
    with col3:
        row = df.loc[df["AttractionId"].eq(attraction)].iloc[0]
        st.metric("Attraction Type", int(row["AttractionTypeId"]))
        st.metric("Attraction City", int(row["AttractionCityId"]))

    if st.button("Predict Visit Mode", type="primary"):
        X = pd.DataFrame([{
            "VisitYear": year, "VisitMonth": month, "ContinentId": continent, "RegionId": region,
            "CountryId": country, "CityId": city, "AttractionId": attraction,
            "AttractionTypeId": int(row["AttractionTypeId"]), "AttractionCityId": int(row["AttractionCityId"])
        }])
        predicted = clf.predict(X)[0]
        label = df.loc[df["VisitMode"].eq(predicted), "VisitMode_mode"].dropna()
        label = label.iloc[0] if not label.empty else str(predicted)
        st.success(f"Predicted visit mode: **{label}**")

with tab4:
    st.subheader("Attraction Recommendations")
    st.write("Content-based recommendations use attraction type and attraction city, with historical rating as a secondary ranking signal.")
    selected = st.selectbox(
        "Choose an attraction you liked", sim.index.tolist(),
        format_func=lambda x: f"{int(x)} — {attraction_names.get(x, 'Unknown attraction')}"
    )
    k = st.slider("Number of recommendations", 3, min(10, max(3, len(sim.index) - 1)), 5)

    if st.button("Recommend Attractions", type="primary"):
        scores = sim.loc[selected].copy()
        scores.loc[selected] = -1
        out = pd.DataFrame({"AttractionId": scores.index, "Similarity": scores.values})
        out["AverageRating"] = out["AttractionId"].map(avg_rating).fillna(df["Rating"].mean())
        out["Attraction"] = out["AttractionId"].map(attraction_names)
        out = out.sort_values(["Similarity", "AverageRating"], ascending=False).head(k)
        out["Similarity"] = out["Similarity"].round(3)
        out["AverageRating"] = out["AverageRating"].round(2)
        st.dataframe(
            out[["AttractionId", "Attraction", "Similarity", "AverageRating"]],
            use_container_width=True, hide_index=True
        )

st.divider()
st.caption("Tourism Experience Analytics • Streamlit • Pandas • Scikit-learn")
