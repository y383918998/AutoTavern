param(
    [int]$Games = 1024,
    [int]$Jobs = 4,
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000)
)

$out = "artifacts\v7_parallel_$Seed\report.json"
python validate_ablation_parallel.py `
  --policy smart_agent_v7_candidate:act `
  --seed $Seed `
  --games $Games `
  --jobs $Jobs `
  --out $out
Write-Host "Report: $out"
