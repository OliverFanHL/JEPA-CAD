param(
    [string]$Python = "python",
    [string]$Config = "configs/full_dataset.yaml"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python "$ProjectRoot/run.py" --config "$ProjectRoot/$Config"
