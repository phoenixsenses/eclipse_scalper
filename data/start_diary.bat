@echo off
REM Start event diary if not already running
REM Used by Windows Task Scheduler for auto-restart on boot

tasklist /FI "WINDOWTITLE eq EventDiary" 2>NUL | find /I "python" >NUL
if %ERRORLEVEL% EQU 0 (
    echo Diary already running, skipping.
    exit /b 0
)

cd /d "C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper"
start "EventDiary" /MIN python -W ignore -u -m data.event_diary --db-path data/microstructure.db --csv-path data/event_diary.csv
echo Diary started at %date% %time% >> logs\diary_restarts.log
