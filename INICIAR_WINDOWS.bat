@echo off
setlocal
cd /d "%~dp0"
title Visao Computacional Interativa - Aula 7

if exist ".venv\Scripts\python.exe" goto RUN

echo [1/3] Criando ambiente Python... 
where py >nul 2>&1
if errorlevel 1 goto PYTHON
py -3.13 --version >nul 2>&1
if not errorlevel 1 (py -3.13 -m venv .venv & goto INSTALL)
py -3.12 --version >nul 2>&1
if not errorlevel 1 (py -3.12 -m venv .venv & goto INSTALL)
py -3 -m venv .venv
goto INSTALL
:PYTHON
python -m venv .venv
:INSTALL
if not exist ".venv\Scripts\python.exe" (
 echo ERRO: Nao foi possivel criar .venv. Instale Python 3.12 ou 3.13.
 pause
 exit /b 1
)
echo [2/3] Instalando pacotes; pode requerer internet e alguns minutos...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto ERRO
:RUN
echo [3/3] Iniciando aplicativo...
".venv\Scripts\python.exe" visao_computacional.py
if errorlevel 1 goto ERRO
exit /b 0
:ERRO
echo.
echo Verifique a mensagem de erro acima.
pause
exit /b 1
