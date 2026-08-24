import streamlit as st
import sqlite3
import os
import pandas as pd

st.set_page_config(
    page_title="Traffic AI Smart Camera",
    page_icon="🚦",
    layout="wide"
)

DB = "traffic_ai.db"
VIDEO = "results/Traffic_AI_Final.mp4"
RESULTS = "results"

st.title("🚦 TRAFFIC AI SMART CAMERA")
st.caption("AI Traffic Monitoring and Violation Detection System")

def get_data():
    if not os.path.exists(DB):
        return pd.DataFrame()

    conn = sqlite3.connect(DB)

    data = pd.read_sql_query(
        "SELECT * FROM violations ORDER BY id DESC",
        conn
    )

    conn.close()

    return data

data = get_data()

if not data.empty:

    data["violation_type"] = data["violation_type"].astype(str)

    total = len(data)

    overspeed = len(
        data[data["violation_type"].str.lower() == "overspeed"]
    )

    wrong_direction = len(
        data[
            data["violation_type"].str.lower()
            == "wrong direction"
        ]
    )

    triple = len(
        data[
            data["violation_type"].str.lower()
            == "triple riding"
        ]
    )

    nohelmet = len(
        data[
            data["violation_type"].str.lower()
            == "no helmet"
        ]
    )

else:

    total = 0
    overspeed = 0
    wrong_direction = 0
    triple = 0
    nohelmet = 0

st.subheader("📊 Central Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Violations", total)
c2.metric("Overspeed", overspeed)
c3.metric("Wrong Direction", wrong_direction)
c4.metric("Triple Riding", triple)
c5.metric("No Helmet", nohelmet)

st.divider()

left, right = st.columns([2, 1])

with left:

    st.subheader("📹 Camera / AI Output")

    if os.path.exists(VIDEO):

        with open(VIDEO, "rb") as video_file:
            video_bytes = video_file.read()

        st.video(video_bytes)

    else:

        st.error(
            "Traffic_AI_Final.mp4 not found inside results folder."
        )

with right:

    st.subheader("🚨 Violation Summary")

    if not data.empty:

        summary = (
            data["violation_type"]
            .value_counts()
            .rename_axis("Violation")
            .reset_index(name="Count")
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No violation records found.")

st.divider()

st.subheader("📋 Violation Records")

if not data.empty:

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No records available.")

st.divider()

st.subheader("🖼️ Violation Evidence")

images = []

if os.path.exists(RESULTS):

    for filename in os.listdir(RESULTS):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            images.append(filename)

if images:

    cols = st.columns(4)

    for i, filename in enumerate(images[:20]):

        with cols[i % 4]:

            st.image(
                os.path.join(RESULTS, filename),
                caption=filename,
                use_container_width=True
            )

else:

    st.info("No evidence images found.")

st.divider()

st.subheader("🔌 System Status")

s1, s2, s3 = st.columns(3)

with s1:

    if os.path.exists(DB):
        st.success("SQLite Database Connected")
    else:
        st.error("Database Not Found")

with s2:

    if os.path.exists(VIDEO):
        st.success("AI Output Video Available")
    else:
        st.error("AI Output Video Not Found")

with s3:

    st.success("Streamlit Dashboard Running")

st.divider()

st.caption(
    "Traffic AI | YOLO11 | ByteTrack | OpenCV | "
    "FastAPI | SQLite | Streamlit"
)