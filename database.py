import sqlite3

conn = sqlite3.connect("traffic_ai.db")

conn.execute("""
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT,
    violation_type TEXT,
    plate_number TEXT,
    timestamp TEXT,
    confidence REAL,
    evidence TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully")