param(
    [string]$Kernel = "thanhnguyenvinh/notebook2d422470cf",
    [string]$Destination = ".\\artifacts\\kaggle_notebook2d422470cf"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "Kaggle CLI is not installed. Install it with: py -m pip install kaggle"
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

# The notebook's output contains retriever_ckpt/result_3.  The pattern avoids
# downloading unrelated output artifacts when Kaggle exposes nested paths.
kaggle kernels output $Kernel --path $Destination --file-pattern ".*result_3.*"

Write-Host "Downloaded result_3 output under: $(Resolve-Path -LiteralPath $Destination)"
