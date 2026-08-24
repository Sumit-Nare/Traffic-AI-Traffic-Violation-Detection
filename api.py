import sqlite3
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Traffic AI REST API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DB = "traffic_ai.db"

class Violation(BaseModel):
    vehicle_id: str
    violation_type: str
    plate_number: str = ""
    confidence: float = 0.0

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT,
            violation_type TEXT,
            plate_number TEXT,
            timestamp TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_data():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM violations ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/")
def home():
    return {
        "project": "Traffic AI",
        "status": "Running",
        "api": "/violations",
        "dashboard": "/dashboard"
    }

@app.get("/violations")
def violations():
    return get_data()

@app.post("/violations")
def add_violation(data: Violation):
    conn = sqlite3.connect(DB)
    conn.execute(
        """INSERT INTO violations
        (vehicle_id, violation_type, plate_number, timestamp, confidence)
        VALUES (?, ?, ?, ?, ?)""",
        (
            data.vehicle_id,
            data.violation_type,
            data.plate_number,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.confidence
        )
    )
    conn.commit()
    conn.close()
    return {"status": "saved"}

@app.get("/violations/{violation_id}")
def violation(violation_id: int):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM violations WHERE id=?",
        (violation_id,)
    ).fetchone()
    conn.close()

    if row is None:
        return {"error": "Violation not found"}

    return dict(row)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    data = get_data()
    total = len(data)
    types = {}

    for item in data:
        v = item["violation_type"]
        types[v] = types.get(v, 0) + 1

    cards = ""

    for key, value in types.items():
        cards += f"""
        <div class="card">
            <h3>{key}</h3>
            <p>{value}</p>
        </div>
        """

    rows = ""

    for item in data:
        rows += f"""
        <tr>
            <td>{item["id"]}</td>
            <td>{item["vehicle_id"]}</td>
            <td>{item["violation_type"]}</td>
            <td>{item["plate_number"]}</td>
            <td>{item["timestamp"]}</td>
            <td>{item["confidence"]:.2f}</td>
        </tr>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>Traffic AI Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
    body {{
        font-family: Arial;
        background: #111827;
        color: white;
        margin: 0;
        padding: 30px;
    }}
    h1 {{
        font-size: 36px;
    }}
    .summary {{
        font-size: 24px;
        margin-bottom: 25px;
    }}
    .cards {{
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 30px;
    }}
    .card {{
        background: #1f2937;
        padding: 20px;
        border-radius: 12px;
        min-width: 160px;
    }}
    .card p {{
        font-size: 30px;
        margin: 10px 0 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        background: #1f2937;
    }}
    th, td {{
        padding: 12px;
        border-bottom: 1px solid #374151;
        text-align: left;
    }}
    th {{
        background: #374151;
    }}
    </style>
    </head>
    <body>
    <h1>TRAFFIC AI</h1>
    <div class="summary">Total Violations: {total}</div>
    <div class="cards">{cards}</div>
    <h2>Violation Records</h2>
    <table>
    <tr>
    <th>ID</th>
    <th>Vehicle ID</th>
    <th>Violation</th>
    <th>Plate</th>
    <th>Timestamp</th>
    <th>Confidence</th>
    </tr>
    {rows}
    </table>
    </body>
    </html>
    """