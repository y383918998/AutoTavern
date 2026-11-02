param([int]$Seed=(Get-Random -Minimum 1 -Maximum 1800000000),[int]$Games=512)
$ErrorActionPreference='Stop';$dir=Join-Path 'artifacts' "v3_candidate_$Seed";New-Item -ItemType Directory -Force -Path $dir|Out-Null
python validate_v3_candidate.py --seed $Seed --games $Games --out "$dir\report.json"
Get-Content "$dir\report.json"
