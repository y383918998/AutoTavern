param([int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000), [int]$Games = 512)
$ErrorActionPreference = 'Stop'
$dir = Join-Path 'artifacts' "fixed_candidate_$Seed"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$config = '{"upgrade_turn_factor":2.4,"pair_bonus":15.0,"refresh_min_gold":3}'
python validate_candidate.py --seed $Seed --games $Games --config $config --out "$dir\report.json"
Get-Content "$dir\report.json"
