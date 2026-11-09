<#
.SYNOPSIS
    Downloads official llama.cpp Windows CUDA 12.x release binaries (with cuBLAS + Flash Attention)
    into .\bin\llama-cuda and provides a helper to launch llama-server.
#>

param(
    [string]$InstallDir = "$PSScriptRoot\..\bin\llama-cuda"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Write-Host "[llama.cpp] Fetching latest release metadata from GitHub..." -ForegroundColor Cyan
$Releases = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
$CudaBinAsset = $null
$CudartAsset  = $null

foreach ($Rel in $Releases) {
    $Bin = $Rel.assets | Where-Object { $_.name -match "^llama-.*-bin-win-cuda-12\.4-x64\.zip$" } | Select-Object -First 1
    $Rt  = $Rel.assets | Where-Object { $_.name -match "^cudart-llama-bin-win-cuda-12\.4-x64\.zip$" } | Select-Object -First 1
    if ($Bin -and $Rt) {
        $CudaBinAsset = $Bin
        $CudartAsset  = $Rt
        Write-Host "[llama.cpp] Selected release $($Rel.tag_name)" -ForegroundColor Cyan
        break
    }
}

if (-not $CudaBinAsset) {
    throw "Could not find a Windows CUDA 12 x64 asset in the latest llama.cpp release."
}

foreach ($Asset in @($CudaBinAsset, $CudartAsset)) {
    if ($null -ne $Asset) {
        $ZipPath = Join-Path $InstallDir $Asset.name
        if (-not (Test-Path $ZipPath)) {
            Write-Host "[llama.cpp] Downloading $($Asset.name)..." -ForegroundColor Yellow
            Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $ZipPath
        }
        Write-Host "[llama.cpp] Extracting $($Asset.name)..." -ForegroundColor Green
        Expand-Archive -Path $ZipPath -DestinationPath $InstallDir -Force
    }
}

Write-Host "[llama.cpp] Installed to $InstallDir" -ForegroundColor Cyan
Get-ChildItem -Path $InstallDir -Filter "llama-server.exe" -Recurse | Select-Object FullName
