param(
    [int]$Games = 512,
    [int]$Jobs = 4,
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000)
)

$out = "artifacts\v8_candidate_$Seed\report.json"
python validate_ablation_parallel.py `
  --baseline smart_agent_v7:act `
  --policy smart_agent_v8_candidate:act `
  --seed $Seed `
  --games $Games `
  --jobs $Jobs `
  --out $out
Write-Host "Report: $out"
