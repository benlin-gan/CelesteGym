CELESTE_ROOT=../../..
# pkill Celeste
taskkill /IM Celeste.exe
dotnet build
$CELESTE_ROOT/Celeste
