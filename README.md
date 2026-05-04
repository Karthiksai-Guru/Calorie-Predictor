# Calorie Predictor

A single-service web demo that predicts calories burned from workout metrics. The backend is FastAPI with an XGBoost model, and the frontend is a React/Vite form that uses ranges derived from the training data.

## Features
- FastAPI prediction API with input validation
- Model artifacts committed for out-of-the-box inference
- Frontend form driven by `/metadata` ranges
- Single deploy: FastAPI serves the built frontend

## Project structure
- `backend/` - FastAPI app, preprocessing, training script, and artifacts
- `frontend/` - React/Vite UI

## Local development
### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Optional: set `VITE_API_BASE_URL` for local dev (see `frontend/.env.example`).

## API usage
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Sex": "female",
    "Age": 35,
    "Height": 165,
    "Weight": 62,
    "Duration": 20,
    "Heart_Rate": 102,
    "Body_Temp": 40.1
  }'
```

## Single deploy build
```bash
cd frontend
npm install
npm run build
cd ../backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Notes
- CSV datasets and notebooks are intentionally excluded from this repository.
- Model artifacts live in `backend/artifacts` and are used by the API at startup.
