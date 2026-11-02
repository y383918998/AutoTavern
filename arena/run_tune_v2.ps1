param([int]$Seed=(Get-Random -Minimum 1 -Maximum 400000000),[int]$Rounds=3,[int]$SearchGames=32,[int]$ValidGames=128)
$ErrorActionPreference='Stop'
$dir=Join-Path 'artifacts' "tune_v2_$Seed"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
python tune_v2.py --seed $Seed --rounds $Rounds --search-games $SearchGames --valid-games $ValidGames --out "$dir\report.json"
Write-Output "Report: $dir\report.json"
