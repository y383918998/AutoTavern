param(
    [ValidateSet('upgrade', 'refresh', 'tempo')]
    [string]$Mode = 'upgrade',
    [int]$Games = 256,
    [int]$Jobs = 4,
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000)
)

$policy = "smart_agent_v8_${Mode}_only:act"
$out = "artifacts\v8_${Mode}_$Seed\report.json"
python validate_ablation_parallel.py `
  --baseline smart_agent_v7:act `
  --policy $policy `
  --seed $Seed `
  --games $Games `
  --jobs $Jobs `
  --out $out
Write-Host "Report: $out"
