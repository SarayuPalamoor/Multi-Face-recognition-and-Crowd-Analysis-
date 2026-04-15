# How to Run Multilevelface Locally

## Architecture Overview
This project has 5 services running simultaneously:

| Service | Port | Command | What it does |
|---------|------|---------|--------------|
| React frontend | 5173 | `npm run dev` in `frontend/` | Main UI |
| Node.js server | 4000 | `node server.js` in `backend/server/` | Login + image upload + training |
| Multi-face Flask | 5000 | `python face_recognition_and_attendance.py` in `backend/backendMulti/` | Recognize multiple faces in a photo |
| Crowd analysis Flask | 5001 | `python a.py` in `backend/backendcrowd/` | Count people using YOLO |
| Single-face Flask | 5003 | `python match.py` in `backend/backSingle/` | Compare 2 faces |

---

## Step 1 — Install Python dependencies (do this once)
```bash
pip install -r requirements.txt
```

---

## Step 2 — Install Node.js dependencies (do this once)

```bash
cd backend/server
npm install

cd ../../frontend
npm install
```

---

## Step 3 — Make sure MongoDB is running
```bash
# On Windows:
net start MongoDB

# On Mac:
brew services start mongodb-community

# On Linux:
sudo systemctl start mongod
```

---

## Step 4 — Start all services (open 5 separate terminals)

**Terminal 1 — Frontend:**
```bash
cd frontend
npm run dev
```
Open http://localhost:5173 in your browser.

**Terminal 2 — Node.js server (login + training):**
```bash
cd backend/server
node server.js
```

**Terminal 3 — Multi-face recognition:**
```bash
cd backend/backendMulti
python face_recognition_and_attendance.py
```

**Terminal 4 — Crowd analysis (downloads yolov8n.pt automatically on first run):**
```bash
cd backend/backendcrowd
python a.py
```

**Terminal 5 — Single-face comparison (downloads model.h5 automatically on first run):**
```bash
cd backend/backSingle
python match.py
```

---

## Step 5 — Register a person and generate encodings

Before multi-face recognition works, you need to train it with some face photos:

1. Open the React UI at http://localhost:5173
2. Go to Login → register a new person (scan QR or fill details + upload photos)
3. Then run the training:
```bash
cd backend/server
node server.js  # must be running
# training triggers automatically when you click "Train" in the UI
# OR manually:
python generate_encodings.py dataset/<rollNo>
```

---

## What downloads automatically on first run
- `backend/backSingle/models/model.h5` — Siamese network (~100MB, from Google Drive)
- `backend/backendcrowd/yolov8n.pt` — YOLOv8 nano weights (~6MB, from Ultralytics)

---

## Ports summary
- http://localhost:5173 → Main React UI  
- http://localhost:4000 → Node.js API  
- http://localhost:5000 → Multi-face recognition  
- http://localhost:5001 → Crowd analysis  
- http://localhost:5003 → Single-face comparison
