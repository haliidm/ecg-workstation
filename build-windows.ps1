param([string]$Python = 'python', [string]$Output = 'dist')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& $Python -m PyInstaller --noconfirm --onedir --windowed --noupx --name ECGWorkstation `
    --distpath $Output --workpath build/windows --specpath build `
    --paths vendor/beat --add-data '../founder_labels_zh.json:.' --add-data '../vendor/beat/checkpoints:vendor/beat/checkpoints' --add-data '../vendor/beat/LICENSE:vendor/beat' --add-data '../vendor/beat/NOTICE:vendor/beat' --add-data '../vendor/ecgfounder/net1d.py:vendor/ecgfounder' --add-data '../vendor/ecgfounder/tasks.txt:vendor/ecgfounder' --add-data '../vendor/ecgfounder/LICENSE:vendor/ecgfounder' `
    --hidden-import mitbih_aami.cnn_transformer `
    --hidden-import mitbih_aami.ecg_data --hidden-import mitbih_aami.record_inference_metrics `
    --hidden-import wfdb.processing --collect-data neurokit2 windows_launcher.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Copy-Item README.md, LICENSE (Join-Path $Output ECGWorkstation)
