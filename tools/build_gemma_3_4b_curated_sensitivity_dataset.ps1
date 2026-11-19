[CmdletBinding()]
param(
    [int]$EvaluationBatchSize = 1,
    [double]$TeacherGpuCacheGiB = 0.0,
    [int]$TeacherPasses = 1,
    [switch]$TeacherOnly
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repository '.venv/Scripts/python.exe'
$builder = Join-Path $PSScriptRoot 'build_gemma_3_4b_sensitivity_dataset.py'
$corpus = (Resolve-Path -LiteralPath (Join-Path $repository 'Datasets/gemma-3-imatrix-calibration-v1/calibration.txt')).Path
$dataset = Join-Path $repository 'Datasets/gemma-3-4b-it-fp16-sensitivity-curated-160x256'

$builderArgs = @(
    '--calibration-text', $corpus,
    '--samples', '160',
    '--sequence-length', '256',
    '--calibration-seed', '0',
    '--teacher-dataset', $dataset,
    '--granularity', 'projection',
    '--error-levels', '0.01', '0.025', '0.05', '0.10', '0.20', '0.35',
    '--seeds', '0', '1', '2',
    '--evaluation-batch-size', "$EvaluationBatchSize",
    '--teacher-gpu-cache-gib', $TeacherGpuCacheGiB.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--teacher-passes', "$TeacherPasses",
    '--local-files-only'
)
if ($TeacherOnly) {
    $builderArgs += '--teacher-only'
}

& $python $builder @builderArgs
exit $LASTEXITCODE
