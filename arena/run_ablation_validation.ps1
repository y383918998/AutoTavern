param([ValidateSet('choice','position')][string]$Mode,[int]$Seed=(Get-Random -Minimum 1 -Maximum 1800000000),[int]$Games=512)
$ErrorActionPreference='Stop';$dir=Join-Path 'artifacts' "ablation_${Mode}_$Seed";New-Item -ItemType Directory -Force -Path $dir|Out-Null
$policy=if($Mode -eq 'choice'){'smart_agent_v3_choice:act'}else{'smart_agent_v3_position:act'}
python validate_ablation.py --policy $policy --seed $Seed --games $Games --out "$dir\report.json"
Get-Content "$dir\report.json"
