@echo off
echo Starting AI Candidate Ranker...
echo.
echo [1/2] Starting FastAPI backend on http://localhost:8000
start "FastAPI Backend" cmd /k "uvicorn api:app --reload --port 8000"
timeout /t 3 /nobreak >nul
echo [2/2] Starting Streamlit frontend on http://localhost:8501
start "Streamlit Frontend" cmd /k "streamlit run app.py"
echo.
echo Both services started. Open http://localhost:8501 in your browser.
