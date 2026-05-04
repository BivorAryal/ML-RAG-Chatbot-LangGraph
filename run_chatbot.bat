@echo off
echo.
echo ========================================================
echo   ML RAG Chatbot — Setup ^& Launch
echo   Author: Bivor
echo ========================================================
echo.

REM Check if .venv exists
if not exist ".venv\" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    echo       Done!
) else (
    echo [1/3] Virtual environment already exists. Skipping.
)

echo.
echo [2/3] Installing dependencies from pyproject.toml...
call .venv\Scripts\activate.bat

REM Install using pip with pyproject.toml (since you don't have requirements.txt)
pip install -e .
echo       Done!

echo.
echo [3/3] Launching ML Guru RAG Chatbot...
echo.
echo  Open your browser at: http://localhost:8501
echo  Press Ctrl+C to stop the server.
echo.
streamlit run rag_frontend.py

pause