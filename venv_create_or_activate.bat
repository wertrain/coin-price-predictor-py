::---------------------------------------------------------
:: 仮想環境を を作成するか activate する
::---------------------------------------------------------
echo off
setlocal
cd /d %~dp0
:: 仮想環境名
set VENV_NAME=.venv

if not exist %VENV_NAME% goto CREATEVENV
goto ACTIVATE

:CREATEVENV
py -m venv %VENV_NAME%

:ACTIVATE
cmd /k .venv\Scripts\activate.bat

endlocal
