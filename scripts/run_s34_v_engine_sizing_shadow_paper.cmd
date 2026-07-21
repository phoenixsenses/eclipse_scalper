@echo off
cd /d D:\eclipse_scalper
echo begin %DATE% %TIME% >> logs\s34_v_engine_sizing_shadow_paper.launch.log
"C:\Users\Windows 11\AppData\Local\Programs\Python\Python313\python.exe" -W ignore -u -m tools.s34_v_engine_sizing_shadow_paper --loop --interval-sec 60 >> logs\s34_v_engine_sizing_shadow_paper.out.log 2>> logs\s34_v_engine_sizing_shadow_paper.err.log
echo exit %ERRORLEVEL% %DATE% %TIME% >> logs\s34_v_engine_sizing_shadow_paper.launch.log
