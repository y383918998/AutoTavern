param(
    [int]$Seed = (Get-Random -Minimum 1 -Maximum 1900000000),
    [int]$TrainGames = 32,
    [int]$ValidGames = 8,
    [int]$EvalGames = 32,
    [int]$Epochs = 12
)
$ErrorActionPreference = 'Stop'
$run = "run_$Seed"
$dir = Join-Path 'artifacts' $run
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$validSeed = $Seed + 20000000
$evalSeed = $Seed + 40000000
python collect_bot_data.py --games $TrainGames --seed $Seed --out "$dir\train.npz"
python collect_bot_data.py --games $ValidGames --seed $validSeed --out "$dir\valid.npz"
python train_imitation.py --train "$dir\train.npz" --valid "$dir\valid.npz" --out "$dir\model.npz" --epochs $Epochs
$env:TAVERN_POLICY_MODEL = (Resolve-Path "$dir\model.npz")
python local_evaluate.py --games $EvalGames --seed $evalSeed --policy learned_agent:act
python local_evaluate.py --games $EvalGames --seed $evalSeed --policy hybrid_agent:act
python local_evaluate.py --games $EvalGames --seed $evalSeed --policy smart_agent:act
