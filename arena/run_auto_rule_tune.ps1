param(
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 400000000),
    [int]$Rounds = 3,
    [int]$SearchGames = 32,
    [int]$ValidGames = 128
)
$ErrorActionPreference = 'Stop'
$dir = Join-Path 'artifacts' "auto_rule_tune_$Seed"
python auto_rule_tune.py --seed $Seed --rounds $Rounds --search-games $SearchGames --valid-games $ValidGames --out-dir $dir
if (Test-Path "$dir\promoted_config.json") {
  Write-Output "PROMOTED CONFIG: $dir\promoted_config.json"
} else {
  Write-Output "No promotion. Existing smart_agent remains the active stable policy."
}
