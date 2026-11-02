param(
    [int]$Games = 8,
    [int]$SamplesPerGame = 4,
    [int]$Jobs = 4,
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000)
)

$out = "artifacts\rollout_preferences_$Seed.npz"
python collect_rollout_preferences.py `
  --policy smart_agent_v8:act `
  --games $Games `
  --seed $Seed `
  --samples-per-game $SamplesPerGame `
  --jobs $Jobs `
  --out $out
python audit_rollout_preferences.py --input $out
Write-Host "Dataset: $out"
