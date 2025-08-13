import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Streamlit: Charts & Maps", page_icon="📈", layout="wide")


@st.cache_data
def make_data(seed: int = 42, days: int = 120):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")

    # City centers (approx) — India focus
    cities = {
        "Pune": (18.5204, 73.8567),
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Bengaluru": (12.9716, 77.5946),
    }

    rows = []
    for city, (lat, lon) in cities.items():
        base_orders = rng.integers(80, 200)  # baseline orders
        noise = rng.normal(0, 20, size=len(dates))  # some daily variation
        trend = np.linspace(-10, 10, len(dates))  # small trend
        orders = np.clip(base_orders + noise + trend, 10, None).round().astype(int)

        # Price per order (varies by city)
        price = {"Pune": 350.0, "Mumbai": 420.0, "Delhi": 390.0, "Bengaluru": 410.0}[
            city
        ]
        revenue = orders * (price + rng.normal(0, 15, size=len(dates)))

        for d, o, r in zip(dates, orders, revenue):
            rows.append((d, city, o, float(r), lat, lon))

    df = pd.DataFrame(rows, columns=["date", "city", "orders", "revenue", "lat", "lon"])
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name()

    # Also make a point cloud around city centers for map layers (scatter/hex)
    pts = []
    for city, (lat, lon) in cities.items():
        # 400 random points around the city center
        for _ in range(400):
            # ~1-3 km jitter
            jitter_lat = lat + (rng.normal(0, 0.01))
            jitter_lon = lon + (rng.normal(0, 0.01))
            vol = float(max(0, rng.normal(100, 40)))  # some volume
            pts.append((city, jitter_lat, jitter_lon, vol))
    geo = pd.DataFrame(pts, columns=["city", "lat", "lon", "volume"])

    return df, geo


df, geo = make_data()

st.title("📈 Charts & 🗺️ Maps with Streamlit for Bajaj")
st.caption(
    "A guided tour of quick charts, Altair/Plotly/Matplotlib, and mapping with st.map & pydeck."
)

st.sidebar.header("Data Overview")
city_filter = st.sidebar.multiselect(
    "Cities", sorted(df["city"].unique()), default=sorted(df["city"].unique())
)
metric = st.sidebar.selectbox("Metric", ["orders", "revenue"], index=1)
days_back = st.sidebar.slider(
    "Days to show", min_value=14, max_value=120, value=60, step=7
)
smooth = st.sidebar.checkbox("Show 7D rolling mean (Altair & Plotly demos)")

df_view = df[df["city"].isin(city_filter)].copy()
cutoff = df_view["date"].max() - pd.Timedelta(days=days_back-1)
df_view = df_view[df_view["date"] >= cutoff]

if smooth:
    df_view["smoothed"] = (
        df_view
        .sort_values(["city", "date"])
        .groupby("city")[metric]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )

st.divider()

# ------------------------------------------------------------
st.header("A) Quick built-in charts")

st.write("**line_chart / area_chart / bar_chart / scatter_chart** accept DataFrames directly.")

pivot = (
    df_view.pivot_table(index="date", columns="city", values=metric, aggfunc="sum")
    .sort_index()
)

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Line Chart")
    st.line_chart(pivot, use_container_width=True)

with c2:
    st.subheader("Area Chart")
    st.area_chart(pivot, use_container_width=True)

with c3:
    st.subheader("Bar Chart")
    st.bar_chart(pivot, use_container_width=True)

st.caption("Tip: Put cities on columns and dates on the index for quick multi-series charts.")

st.divider()

# ------------------------------------------------------------
st.header("B) Altair (customizable)")

show_points = st.checkbox("Show data points (Altair)", value=False)

alt_df = df_view if not smooth else df_view.assign(value=df_view["smoothed"])
if not smooth:
    alt_df = df_view.assign(value=df_view[metric])

alt_chart = (
    alt.Chart(alt_df)
    .mark_line(point=show_points)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title=metric.capitalize()),
        color=alt.Color("city:N", title="City"),
        tooltip=["date:T", "city:N", alt.Tooltip("value:Q", title=metric)],
    )
    .properties(height=320)
    .interactive()  # zoom & pan
)

st.altair_chart(alt_chart, use_container_width=True)
st.caption("Use `interactive()` for pan/zoom, tooltips for details-on-demand.")

st.divider()

# ------------------------------------------------------------
st.header("C) Matplotlib & Plotly")

left, right = st.columns(2)

with left:
    st.subheader("Matplotlib (static image)")
    # Build separate lines per city
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    for city in city_filter:
        series = df_view[df_view["city"] == city].sort_values("date")
        y = series["smoothed"] if smooth else series[metric]
        ax.plot(series["date"], y, label=city)
    ax.set_xlabel("Date")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} by City")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Plotly (interactive)")
    plotly_df = df_view.assign(value=df_view["smoothed"] if smooth else df_view[metric])
    fig = px.line(
        plotly_df.sort_values("date"),
        x="date", y="value",
        color="city",
        title=f"{metric.capitalize()} by City",
        markers=False,
        hover_data={"value": ":.2f"} if metric == "revenue" or smooth else None,
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------------------------------------------------------
st.header("D) Maps")
last_day = df_view["date"].max()
points = df[df["date"] == last_day]  # original df has city centers repeated per day
points = points[points["city"].isin(city_filter)][["lat", "lon"]]
st.map(points, use_container_width=True)
st.caption("`st.map` expects columns named **lat** and **lon**. It's great for quick plots.")

# ------------------------------------------------------------
st.header("E) Small analytics helpers")

k1, k2, k3, k4 = st.columns(4)
total_metric = df_view[metric].sum()
latest_day = df_view["date"].max()
prev_day = latest_day - timedelta(days=1)

today_metric = df_view[df_view["date"] == latest_day][metric].sum()
yday_metric = df_view[df_view["date"] == prev_day][metric].sum()

delta_val = today_metric - yday_metric
pct = (delta_val / yday_metric * 100.0) if yday_metric else 0.0

with k1:
    st.metric(f"Total {metric.capitalize()} (last {days_back}d)", f"{total_metric:,.0f}" if metric=="orders" else f"{total_metric:,.2f}")
with k2:
    st.metric(f"Today {metric.capitalize()}", f"{today_metric:,.0f}" if metric=="orders" else f"{today_metric:,.2f}",
              delta=f"{delta_val:,.0f}" if metric=="orders" else f"{delta_val:,.2f}")
with k3:
    st.metric("Day-over-day %", f"{pct:+.1f}%")
with k4:
    best_city = (
        df_view[df_view["date"] == latest_day]
        .groupby("city")[metric].sum().sort_values(ascending=False).index[0]
    )
    st.metric("Top city today", best_city)

st.caption("Use `st.metric` when you want compact KPIs with optional delta.")

st.divider()

# ------------------------------------------------------------
# 🧩 Mini-exercise
# ------------------------------------------------------------
st.header("🧩 Mini-exercise")

st.markdown(
    """
**Goal:** Build a small **city comparer** with a chart + map.

**Requirements**
1) Add a `selectbox` named **Compare city** (choose one from the sidebar selection).  
2) Show a **Plotly bar chart** of **weekday-wise totals** (Mon..Sun) for the selected city, using the chosen `metric`.  
3) Add a **slider** called **Hex radius (m)** to control the `HexagonLayer.radius` (between 200 and 1200).  
4) Render a **pydeck** map for *only that city* using the chosen radius.

*(Hints:)*  
- Group by `weekday` (order by calendar, not alphabet).  
- Filter `geo` by the selected city for the map.  
- You can reuse `view_state` with the city’s mean lat/lon.
"""
)