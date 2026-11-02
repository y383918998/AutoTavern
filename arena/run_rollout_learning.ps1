param(
    [int]$Games = 16,
    [int]$SamplesPerGame = 4,
    [int]$Continuations = 4,
    [int]$MaxActions = 5,
    [int]$Jobs = 4,
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000),
    [switch]$Train
)

$dir = "artifacts\rollout_learning_$Seed"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
python collect_rollout_preferences.py `
  --policy smart_agent_v8:act `
  --games $Games `
  --seed $Seed `
  --samples-per-game $SamplesPerGame `
  --continuations $Continuations `
  --max-actions $MaxActions `
  --jobs $Jobs `
  --out "$dir\preferences.npz"
python audit_rollout_preferences.py --input "$dir\preferences.npz"
if ($Train) {
  Write-Warning "The current high-dimensional ranker is experimental; do not deploy its output."
  python train_rollout_ranker.py `
    --train "$dir\preferences.npz" `
    --out "$dir\ranker.npz" `
    --seed $Seed
}
Write-Host "Artifacts: $dir"
