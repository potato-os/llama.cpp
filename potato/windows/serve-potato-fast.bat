@echo off
rem Potato-CODER-12GB-Fast: local OpenAI-compatible server at http://127.0.0.1:8080/v1
rem Put this file, the Fast GGUF, projector, chat template, and a Fast-capable
rem Windows CUDA llama-server.exe with its DLLs in one folder.
rem Regular Potato Windows binaries cannot load this ternary model.
set "HERE=%~dp0"
set "LLAMA=%HERE%llama-server.exe"
if defined LLAMA_SERVER set "LLAMA=%LLAMA_SERVER%"

if not exist "%LLAMA%" goto :no_llama
set "F=%HERE%Potato-CODER-12GB-Fast-Ternary.gguf"
if not exist "%F%" goto :no_file
set "F=%HERE%mmproj-Potato-CODER-12GB-Fast-Q8_0.gguf"
if not exist "%F%" goto :no_file
set "F=%HERE%chat-template-sharp-potato.jinja"
if not exist "%F%" goto :no_file

rem Preserve the line feeds in the thinking-budget wrap-up message.
setlocal EnableDelayedExpansion
(set LF=^
%=empty=%
)
set "LLAMA_ARG_THINK_BUDGET_MESSAGE=!LF!!LF!Thinking budget reached. Stop deliberating now: state the approach in one or two sentences and write the complete final code.!LF!"
setlocal DisableDelayedExpansion

echo Starting Potato-CODER-12GB-Fast on http://127.0.0.1:8080 ... keep this window open.
"%LLAMA%" -m "%HERE%Potato-CODER-12GB-Fast-Ternary.gguf" ^
  -c 131072 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 512 -b 2048 --no-context-shift ^
  --host 127.0.0.1 --port 8080 --alias potato-coder-12gb-fast ^
  --mmproj "%HERE%mmproj-Potato-CODER-12GB-Fast-Q8_0.gguf" ^
  --chat-template-file "%HERE%chat-template-sharp-potato.jinja" ^
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 ^
  --reasoning-budget 16384 ^
  --chat-template-kwargs "{\"preserve_thinking\": false, \"reasoning_effort\": \"medium\"}"
set "SERVER_EXIT=%ERRORLEVEL%"
echo.
echo The server has stopped. If you did not press Ctrl+C, read the messages above.
pause
exit /b %SERVER_EXIT%

:no_llama
echo.
echo ERROR: llama-server.exe not found at %LLAMA%
echo This model needs a Fast-capable Windows CUDA build of potato-millie-cuda.
echo Download the matching Windows CUDA build linked from the Fast model card.
echo.
pause
exit /b 1

:no_file
echo.
echo ERROR: file not found: %F%
echo Keep the Fast model, projector, template, and this script in one folder.
echo.
pause
exit /b 1
