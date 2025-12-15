# Cricket Ball Analysis & Prediction System

_An AI-powered cricket ball analysis tool that tracks the ball from match videos, extracts motion features, analyzes pitch behavior, classifies delivery types, and predicts possible next deliveries.  
Built using **Python, OpenCV, Machine Learning, and Tkinter GUI**._

---

## Project Overview

This project processes cricket match videos to:
- Track ball movement frame-by-frame
- Calculate ball speed, bounce height, and trajectory variation
- Classify the type of delivery (Fast / Spin)
- Analyze pitch conditions
- Predict the next possible ball using a trained ML model
- Display results through a user-friendly Tkinter GUI

---

## Features

-  Video-based cricket ball tracking
-  Ball speed calculation
-  Trajectory variation analysis
-  Pitch condition detection
-  Machine Learning–based delivery classification
- Next-ball prediction logic
-  Graph visualization (speed, bounce, trajectory)
-  Interactive Tkinter GUI
-  CSV export of analysis data

---

##  Tech Stack

| Component | Technology |
|---------|------------|
| Language | Python |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Data Handling | Pandas |
| Machine Learning | Scikit-learn (Random Forest) |
| Visualization | Matplotlib |
| GUI | Tkinter |
| Model Storage | Joblib |

---
**Project Structure**
cricket-ball-analysis/
│
├── cricket_analysis_tool.py # Main ML-based analysis tool
├── tkaikav2.py # Ball tracking & analysis GUI
├── tk3.py # Trajectory & bounce analysis
├── TKAIKAV.py # Speed & delivery classification
├── requirements.txt # Python dependencies
└── README.md

** How It Works
**
1. User uploads a cricket video
2. OpenCV detects the ball using circular Hough Transform
3. Ball movement is tracked across frames
4. Speed, bounce, and trajectory features are extracted
5. ML model classifies delivery type
6. System predicts the next possible ball
7. Results and graphs are displayed via GUI

**Sample Output**

Delivery Type: Fast / Spin

Pitch Condition: Hard / Soft / Normal

Ball Rating: ⭐⭐⭐⭐☆

Graphs:

Ball speed vs frames

Trajectory variation

Bounce height

