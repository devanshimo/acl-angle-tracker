# ACL Angle Tracker 

A real-time computer vision system that measures **knee flexion angle** and **range of motion (ROM)** using a webcam, aligned with **orthopedic goniometry** principles.

This project focuses on **robust biomechanics-aware tracking**, not just pose estimation.

---

##  Features

- Real-time knee flexion angle (ortho convention)
- Range of Motion (ROM) tracking per session
- Persistent high-score (best ROM achieved)
- Progress bar for visual feedback
- Leg identity locking (prevents left/right switching)
- Temporal consistency filtering (no knee teleporting)
- Geometry-based rejection of bad frames

---

##  How It Works

1. **Pose Estimation**
   - Uses MediaPipe Pose to extract body landmarks.

2. **Biomechanical Constraint**
   - Computes the knee angle using hip–knee–ankle geometry.
   - Converts CV angle to orthopedic knee flexion angle.

3. **Tracking Stability**
   - Locks onto one leg per session.
   - Rejects implausible landmark jumps.
   - Filters geometrically invalid detections.

4. **Performance Scoring**
   - Computes ROM as `max_flexion - min_flexion`.
   - Tracks and persists best ROM across sessions.
   - Displays a live progress bar.

---

##  Recommended Camera Setup

- Side-on view (sagittal plane)
- Full leg visible (hip → ankle)
- Camera ~2–3 meters away
- Avoid occlusion by hands or clothing

---

##  Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/acl-angle-tracker.git
cd acl-angle-tracker
```
### 2. Create a virtual environment
### 3. Activate the virtual environment
### 4. Install dependencies
```
pip install -r requirements.txt
```
---
## Usage

Run the application:
```
python main.py
```
---
## Controls

q → Quit the application

r → Reset current session (min/max/ROM)

---
## Output

The application displays:

-Current knee flexion angle (degrees).

-Minimum and maximum flexion achieved.

-Range of Motion (ROM).

-Best ROM achieved across sessions.

-Visual progress bar comparing current performance to best.
