@echo off
rem ============================================================================================
rem  Potato-CODER-48GB (two 24 GB NVIDIA cards): starts a LOCAL llama.cpp server on this PC.
rem   - It is ready when this window shows "listening on http://127.0.0.1:8080".
rem   - Point your agent (OpenCode, Cline, ...) at http://127.0.0.1:8080/v1 (OpenAI-compatible API).
rem   - Keep this window open while you work. Ctrl+C or closing the window stops the server.
rem   - After each request this window prints its speed (prompt and generation tokens per second).
rem  Put everything in ONE folder, next to this file (no edits needed):
rem     serve-potato.bat                                        (this file)
rem     IQ4_XS\Potato-CODER-48GB-IQ4_XS-00001-of-00002.gguf       (the model, two parts, keep the IQ4_XS subfolder)
rem     IQ4_XS\Potato-CODER-48GB-IQ4_XS-00002-of-00002.gguf
rem     MTP\mtp-Potato-CODER-48GB-shared-Q3_K_M.gguf             (speculative-decoding head, keep the MTP subfolder)
rem     mmproj-Potato-CODER-48GB-Q8_0.gguf                        (vision projector)
rem     llama-server.exe and its DLLs                             (unzip llama-potato-bin-win-cuda-*.zip + cudart-*.zip here)
rem  Stock llama.cpp does not work with this script (no --mmproj-swap-draft); use github.com/potato-os/llama.cpp
rem  To keep the model somewhere else, change the MODELS line below (no trailing backslash).
rem ============================================================================================
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "MODELS=%HERE%"
set "LLAMA=%HERE%\llama-server.exe"

if not exist "%LLAMA%" goto :no_llama
set "F=%MODELS%\IQ4_XS\Potato-CODER-48GB-IQ4_XS-00001-of-00002.gguf"
if not exist "%F%" goto :no_model
set "F=%MODELS%\IQ4_XS\Potato-CODER-48GB-IQ4_XS-00002-of-00002.gguf"
if not exist "%F%" goto :no_model
set "F=%MODELS%\MTP\mtp-Potato-CODER-48GB-shared-Q3_K_M.gguf"
if not exist "%F%" goto :no_model
set "F=%MODELS%\mmproj-Potato-CODER-48GB-Q8_0.gguf"
if not exist "%F%" goto :no_model

rem Thinking budget: 16k tokens per thinking block, then a one-sentence wrap-up. The message needs real line breaks,
rem which cmd cannot type: build them in a variable and hand the message to llama.cpp through its environment variable.
setlocal EnableDelayedExpansion
(set LF=^
%=empty=%
)
set "LLAMA_ARG_THINK_BUDGET_MESSAGE=!LF!!LF!Thinking budget reached. Stop deliberating now: state the approach in one or two sentences and write the complete final code.!LF!"
setlocal DisableDelayedExpansion

echo Starting Potato-CODER-48GB on http://127.0.0.1:8080 ... keep this window open.
"%LLAMA%" -m "%MODELS%\IQ4_XS\Potato-CODER-48GB-IQ4_XS-00001-of-00002.gguf" ^
  -c 229376 -ngl 99 -ts 1.1,1 -ub 512 -b 4096 -fa on -ctk f16 -ctv f16 ^
  --mmproj "%MODELS%\mmproj-Potato-CODER-48GB-Q8_0.gguf" --no-mmproj-offload --mmproj-swap-draft --image-max-tokens 2048 ^
  --spec-type draft-mtp -md "%MODELS%\MTP\mtp-Potato-CODER-48GB-shared-Q3_K_M.gguf" --spec-draft-n-max 2 -devd CUDA1 ^
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --chat-template-kwargs "{\"preserve_thinking\": false, \"reasoning_effort\": \"medium\"}" ^
  --reasoning-budget 16384

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
