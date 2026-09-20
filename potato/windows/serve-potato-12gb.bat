@echo off
rem ============================================================================================
rem  Potato-CODER-12GB: starts a LOCAL llama.cpp server on this PC (RTX 4070 / 3060 12 GB class).
rem   - It is ready when this window shows "listening on http://127.0.0.1:8080".
rem   - Point your agent (OpenCode, Cline, ...) at http://127.0.0.1:8080/v1 (OpenAI-compatible API).
rem   - Keep this window open while you work. Ctrl+C or closing the window stops the server.
rem  Put everything in ONE folder, next to this file (no edits needed):
rem     serve-potato.bat                            (this file)
rem     Potato-CODER-12GB-UD-IQ3_XXS.gguf           (the model)
rem     mmproj-Potato-CODER-12GB-Q8_0.gguf          (vision projector)
rem     llama-server.exe and its DLLs               (unzip llama-potato-bin-win-cuda-*.zip + cudart-*.zip here)
rem  This build keeps 26 of the expert layers in system RAM (--n-cpu-moe 26): that is what makes a
rem  131,072-token context fit in 12 GB. It needs about 16 GB of system RAM. If the model does not load,
rem  lower the context to 114688 first, then raise --n-cpu-moe.
rem  To keep the model somewhere else, change the MODELS line below (no trailing backslash).
rem ============================================================================================
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
set "MODELS=%HERE%"
set "LLAMA=%HERE%\llama-server.exe"

if not exist "%LLAMA%" goto :no_llama
set "F=%MODELS%\Potato-CODER-12GB-UD-IQ3_XXS.gguf"
if not exist "%F%" goto :no_model
set "F=%MODELS%\mmproj-Potato-CODER-12GB-Q8_0.gguf"
if not exist "%F%" goto :no_model

rem  The thinking budget needs its wrap-up message, or a capped-out chain is truncated instead of finished.
rem  cmd.exe cannot put line feeds in a variable directly; this is the standard caret idiom for it.
setlocal EnableDelayedExpansion
(set LF=^
%=empty=%
)
set "LLAMA_ARG_THINK_BUDGET_MESSAGE=!LF!!LF!Thinking budget reached. Stop deliberating now: state the approach in one or two sentences and write the complete final code.!LF!"
setlocal DisableDelayedExpansion

echo Starting Potato-CODER-12GB on http://127.0.0.1:8080 ... keep this window open.
"%LLAMA%" -m "%MODELS%\Potato-CODER-12GB-UD-IQ3_XXS.gguf" ^
  -c 131072 -ngl 99 --n-cpu-moe 26 -fa on -ctk q8_0 -ctv q8_0 -ub 512 -b 2048 --no-context-shift ^
  --mmproj "%MODELS%\mmproj-Potato-CODER-12GB-Q8_0.gguf" ^
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --reasoning-budget 16384 ^
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
