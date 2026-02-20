@echo off
REM Start microstructure collector if not already running
REM Used by Windows Task Scheduler for auto-restart on boot

tasklist /FI "WINDOWTITLE eq MicroCollector" 2>NUL | find /I "python" >NUL
if %ERRORLEVEL% EQU 0 (
    echo Collector already running, skipping.
    exit /b 0
)

cd /d "C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper"
start "MicroCollector" /MIN python -W ignore -u -m data.microstructure_collector --symbols BTCUSDT,ETHUSDT --db-path data/microstructure.db --stats-interval 300
echo Collector started at %date% %time% >> logs\collector_restarts.log
