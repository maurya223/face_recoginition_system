"""Flask dashboard for attendance visualization."""
import os
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.utils
from flask import Flask, render_template, jsonify

app = Flask(__name__)

CSV_FILE = "attendance.csv"


@app.route("/")
def dashboard():
    """Render main dashboard."""
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    """Get attendance data and summary stats."""
    if not os.path.exists(CSV_FILE):
        return jsonify({"error": "No attendance data"})

    df = pd.read_csv(CSV_FILE)
    if df.empty:
        return jsonify({"data": [], "summary": {}})

    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Summary stats
    summary = {
        "total_entries": len(df),
        "unique_persons": df["Name"].nunique() if "Name" in df else 0,
        "today_count": len(df[df["Timestamp"].dt.date == datetime.now().date()]),
        "persons_today": df[df["Timestamp"].dt.date == datetime.now().date()]["Name"].nunique(),
    }

    # Recent 20
    recent = df.tail(20).to_dict("records")

    # Charts JSON
    fig1 = px.bar(df, x="Name", color="Name", title="Attendance Count per Person")
    fig1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    if len(df) > 1:
        fig2 = px.line(df, x="Timestamp", color="Name", title="Attendance Timeline")
        fig2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        fig2_json = "{}"

    return jsonify(
        {
            "data": recent,
            "summary": summary,
            "chart1": fig1_json,
            "chart2": fig2_json,
            "full_csv_url": "/api/full_csv",
        }
    )


@app.route("/api/full_csv")
def full_csv():
    """Get full CSV data as JSON."""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        return df.to_json(orient="records")
    return jsonify([])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
