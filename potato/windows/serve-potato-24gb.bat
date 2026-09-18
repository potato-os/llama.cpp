@echo off
rem ============================================================================================
rem  Potato-CODER-24GB: starts a LOCAL llama.cpp server on this PC.
rem   - It is ready when this window shows "listening on http://127.0.0.1:8080".
rem   - Point your agent (OpenCode, Cline, ...) at http://127.0.0.1:8080/v1 (OpenAI-compatible API).
rem   - Keep this window open while you work. Ctrl+C or closing the window stops the server.
rem   - After each request this window prints its speed (prompt and generation tokens per second).
rem  Put everything in ONE folder, next to this file (no edits needed):
rem     serve-potato.bat                          (this file)
rem     IQ4_XS\Potato-CODER-24GB-IQ4_XS.gguf        (the model, keep the IQ4_XS subfolder)
rem     mmproj-Potato-CODER-24GB-Q8_0.gguf          (vision projector)
rem     llama-server.exe and its DLLs               (unzip llama-potato-bin-win-cuda-*.zip + cudart-*.zip here)
rem  Stock llama.cpp does not work with this script (no --mmproj-swap-draft); use github.com/potato-os/llama.cpp
rem  To keep the model somewhere else, change the MODELS line below (no trailing backslash).
rem ============================================================================================
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "MODELS=%HERE%"
set "LLAMA=%HERE%\llama-server.exe"

if not exist "%LLAMA%" goto :no_llama
set "F=%MODELS%\IQ4_XS\Potato-CODER-24GB-IQ4_XS.gguf"
if not exist "%F%" goto :no_model
set "F=%MODELS%\mmproj-Potato-CODER-24GB-Q8_0.gguf"
if not exist "%F%" goto :no_model

echo Starting Potato-CODER-24GB on http://127.0.0.1:8080 ... keep this window open.
"%LLAMA%" -m "%MODELS%\IQ4_XS\Potato-CODER-24GB-IQ4_XS.gguf" ^
  -c 196608 -ngl 99 -fa on -ctk q8_0 -ctv q5_0 -ub 256 -b 2048 --no-context-shift ^
  --mmproj "%MODELS%\mmproj-Potato-CODER-24GB-Q8_0.gguf" --no-mmproj-offload --mmproj-swap-draft ^
  --spec-type draft-mtp --spec-draft-n-max 3 ^
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --chat-template-kwargs "{\"preserve_thinking\": false, \"reasoning_effort\": \"medium\"}"

echo.
echo The server has stopped. If you did not press Ctrl+C, read the messages above.
pause
exit /b 0

:no_llama
echo.
echo ERROR: llama-server.exe not found in %HERE%
echo Unzip the Windows llama.cpp release AND its cudart- zip into this folder, then run this file again.
echo Download both from https://github.com/potato-os/llama.cpp/releases
echo.
pause
exit /b 1

:no_model
echo.
echo ERROR: model file not found: %F%
echo Keep the files and subfolders exactly as downloaded from Hugging Face.
echo.
pause
exit /b 1
