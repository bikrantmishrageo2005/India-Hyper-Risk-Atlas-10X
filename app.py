import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="India HyperRisk Atlas 10X", layout="wide")


@st.cache_data
def load_data():
    # CSV file should be in the same repo: india_state_multi_hazard.csv
    df = pd.read_csv("india_state_multi_hazard.csv")
    return df


@st.cache_resource
def train_vulnerability_model(df, hazard_cols):
    # Create a simple label from overall index (for demo)
    overall = df[hazard_cols].mean(axis=1)
    labels = []
    for v in overall:
        if v < 35:
            labels.append("Low")
        elif v < 70:
            labels.append("Moderate")
        else:
            labels.append("High")
    df = df.copy()
    df["vulnerability_label"] = labels

    X = df[hazard_cols].values
    y = df["vulnerability_label"].values

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X, y)
    return model


def plot_bar(labels, scores):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, scores)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hazard index (0–100)")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_radar(labels, scores):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    scores = np.array(scores)
    scores = np.concatenate((scores, [scores[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, scores, linewidth=2)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


df = load_data()

st.title("India HyperRisk Atlas 10X")
st.write(
    "This application provides a multi-hazard profile for each Indian state using ten normalized "
    "hazard indices (0–100). It combines them into a simple multi-hazard index and a machine learning "
    "based vulnerability category. Data values should be prepared from national hazard sources such as "
    "NDMA, IMD, GSI, FSI, CPCB and related open datasets."
)

hazard_cols = [
    "eq_risk",
    "flood_risk",
    "cyclone_risk",
    "tsunami_risk",
    "landslide_risk",
    "heatwave_risk",
    "drought_risk",
    "forestfire_risk",
    "airpollution_risk",
    "lightning_risk",
]

hazard_labels = [
    "Earthquake",
    "Flood",
    "Cyclone",
    "Tsunami",
    "Landslide",
    "Heatwave",
    "Drought",
    "Forest Fire",
    "Air Pollution",
    "Lightning",
]

states = sorted(df["state"].unique())
state = st.sidebar.selectbox("Select state", states)

model = train_vulnerability_model(df, hazard_cols)

row = df[df["state"] == state].iloc[0]
scores = [row[c] for c in hazard_cols]
overall_index = float(np.mean(scores))

st.subheader(f"Hazard profile: {state}")

col1, col2 = st.columns([2, 1])

with col1:
    fig_bar = plot_bar(hazard_labels, scores)
    st.pyplot(fig_bar)

with col2:
    st.metric("Multi-hazard index", f"{overall_index:.1f} / 100")

    input_array = np.array(scores).reshape(1, -1)
    predicted_label = model.predict(input_array)[0]
    st.write("Estimated vulnerability category:", predicted_label)

    sorted_pairs = sorted(zip(hazard_labels, scores), key=lambda x: x[1], reverse=True)
    top3 = sorted_pairs[:3]
    st.write("Top contributing hazards:")
    for name, val in top3:
        st.write(f"- {name}: {val:.1f}")

st.subheader("Radar view of hazard fingerprint")
fig_radar = plot_radar(hazard_labels, scores)
st.pyplot(fig_radar)

st.subheader(f"Detailed indices for {state}")
table = pd.DataFrame({"Hazard": hazard_labels, "Index": scores})
st.dataframe(table)

st.subheader("State comparison by multi-hazard index")
df["overall_index"] = df[hazard_cols].mean(axis=1)
df_sorted = df[["state", "overall_index"]].sort_values("overall_index", ascending=False)
st.dataframe(df_sorted.reset_index(drop=True))

st.caption(
    "Note: This is a prototype research dashboard. Hazard scores are normalized indices "
    "and should be derived carefully from official maps and time-series data."
)
