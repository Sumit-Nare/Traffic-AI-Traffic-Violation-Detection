# Traffic AI - Traffic Violation Detection

## Overview

Traffic AI is an AI-based traffic monitoring system that analyzes traffic videos and detects vehicles and traffic violations.

## Features

- Vehicle Detection
- Vehicle Classification
- Vehicle Counting
- Helmet Detection
- No Helmet Detection
- Triple Riding Detection
- Wrong Direction Detection
- Overspeed Detection
- Number Plate Detection
- Violation Counting
- Violation Evidence Generation
- SQLite Database
- REST API
- Central Dashboard

## Technologies

- Python
- YOLO
- OpenCV
- ByteTrack
- FastAPI
- SQLite

## Project Pipeline

Video Input → YOLO Detection → ByteTrack Tracking → Violation Detection → SQLite Database → REST API → Dashboard

## How to Run

```bash
python main.py
