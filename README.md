# Traffic AI - Smart Traffic Monitoring and Violation Detection

## Overview

Traffic AI is an AI-based smart traffic monitoring system that analyzes traffic video and detects vehicles and traffic violations using computer vision and deep learning.

The system combines AI detection, object tracking, violation analysis, SQLite database storage, REST API services and a Streamlit central dashboard in one project.

## Features

- Vehicle Detection
- Car Detection
- Motorcycle Detection
- Bus Detection
- Truck Detection
- Vehicle Counting
- Helmet Detection
- No Helmet Detection
- Triple Riding Detection
- Wrong Direction Detection
- Overspeed Detection
- Number Plate Detection
- Violation Counting
- Violation Evidence Images
- SQLite Database
- REST API
- FastAPI
- Streamlit Central Dashboard
- AI Output Video

## Technologies Used

- Python
- YOLO11
- OpenCV
- ByteTrack
- FastAPI
- SQLite
- Streamlit
- Pandas

## Project Architecture

Traffic Video

↓

YOLO Object Detection

↓

ByteTrack Object Tracking

↓

Vehicle Counting and Violation Detection

↓

SQLite Database

↓

FastAPI REST API

↓

Streamlit Central Dashboard

## Traffic Violations

The current prototype detects and records:

- No Helmet
- Triple Riding
- Wrong Direction
- Overspeed

Vehicle and number-plate detection are also included.

## Dashboard

The Streamlit dashboard provides a single web page containing:

- AI processed video
- Vehicle and violation counts
- Violation summary
- Violation records
- Evidence images
- Database status
- AI output status

## REST API

The project provides REST API endpoints for accessing violation records.

API documentation:

http://127.0.0.1:8000/docs

Dashboard:

http://localhost:8501

## How to Install

Install the required packages:

```bash
pip install -r requirements.txt
