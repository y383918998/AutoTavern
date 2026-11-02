param(
    [ValidateSet('discover', 'trinket')]
    [string]$Family = 'discover',
    [Parameter(Mandatory = $true)]
    [string]$Train,
    [Parameter(Mandatory = $true)]
    [string]$Valid,
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 1800000000),
    [int]$MinStates = 128
)

$out = "artifacts\choice_${Family}_$Seed.npz"
python train_choice_ranker.py `
  --train $Train `
  --valid $Valid `
  --family $Family `
  --min-states $MinStates `
  --out $out `
  --seed $Seed
Write-Host "Model: $out"
