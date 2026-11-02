param(
    [int]$Games = 64,
    [int]$SamplesPerGame = 4,
    [int]$Continuations = 6,
    [int]$Jobs = 6,
    [string]$Families = 'discover,trinket',
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000),
    [switch]$Train
)

$dir = "artifacts\choice_learning_$Seed"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
python collect_choice_preferences.py `
  --policy smart_agent_v8:act `
  --games $Games `
  --seed $Seed `
  --samples-per-game $SamplesPerGame `
  --continuations $Continuations `
  --families $Families `
  --jobs $Jobs `
  --out "$dir\preferences.npz"
python audit_choice_preferences.py --input "$dir\preferences.npz"
if ($Train) {
  python train_choice_ranker.py --train "$dir\preferences.npz" --out "$dir\ranker.npz" --seed $Seed
}
Write-Host "Artifacts: $dir"
