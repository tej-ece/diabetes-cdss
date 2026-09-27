@echo off
title Launching Diabetes CDSS Web App
echo Activating local venv environment...

cd /d "%~dp0"
call "%~dp0venv\Scripts\activate.bat"

echo Starting Streamlit CDSS Application...
streamlit run app.py

pause