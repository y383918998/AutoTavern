param([int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000), [int]$SearchGames = 16, [int]$ValidGames = 64)
$ErrorActionPreference = 'Stop'
$dir = Join-Path 'artifacts' "rule_search_$Seed"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
python rule_search.py --seed $Seed --search-games $SearchGames --valid-games $ValidGames --out "$dir\result.json"
Get-Content "$dir\result.json"
