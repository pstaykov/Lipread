@echo off
rem Double-click entry point for install.ps1 -- runs it with a permissive
rem execution policy scoped to just this process, so it works even if
rem PowerShell scripts are blocked by default on this machine.
setlocal
set SCRIPT_DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1" %*
if errorlevel 1 (
    echo.
    echo Setup fehlgeschlagen -- siehe Meldungen oben.
    pause
)
