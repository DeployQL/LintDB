cmake -B _build ^
      -T v141 ^
      -A x64 ^
      -G "Visual Studio 16 2019" ^
      .
if %errorlevel% neq 0 exit /b %errorlevel%

cmake --build _build --config Release -j %CPU_COUNT%
if %errorlevel% neq 0 exit /b %errorlevel%

cmake --install _build --config Release --prefix %PREFIX%
if %errorlevel% neq 0 exit /b %errorlevel%