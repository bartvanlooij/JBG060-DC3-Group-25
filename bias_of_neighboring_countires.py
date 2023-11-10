import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


data = pd.read_csv("data/articles_topics.csv")

world = gpd.read_file("data/countries.geojson")


def get_country(lat, lng):
    point = Point(lng, lat)
    country = world[world.geometry.contains(point)]["ADMIN"].values
    return country[0] if country.size > 0 else None


data["country"] = data.apply(lambda row: get_country(row["lat"], row["lng"]), axis=1)

# calculate the proportion of positive sentiments to negative ones per neighboring country
data["positive_sentiment"] = data["sentiment_paragraph"] == "POSITIVE"
positive_proportion = data.groupby("country")["positive_sentiment"].mean()

# neighboring countries
target_countries = [
    "Ethiopia",
    "Sudan",
    "Central African Republic",
    "Democratic Republic of the Congo",
    "Uganda",
    "Kenya",
]


bias_per_country = positive_proportion.loc[target_countries]

print(bias_per_country)
