param(
    [int]$Games = 512,
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000)
)

$out = "artifacts\v7_candidate_$Seed\report.json"
python validate_ablation.py `
  --policy smart_agent_v7_candidate:act `
  --seed $Seed `
  --games $Games `
  --out $out
Write-Host "Report: $out"
