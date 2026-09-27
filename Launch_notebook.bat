@echo off
title Launching Jupyter Notebook
echo Activating local venv environment...

cd /d "%~dp0"
call "%~dp0venv\Scripts\activate.bat"

echo Opening cdss_random_forest.ipynb in Jupyter...
jupyter notebook cdss_random_forest.ipynb

pause