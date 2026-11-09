<#
.SYNOPSIS
    Configures Windows Graphics Preferences so desktop/browser apps run on the AMD Radeon 780M iGPU
    (GpuPreference=1) and verifies that the NVIDIA GeForce RTX 4060 Laptop GPU sits at 0 MiB VRAM.
#>

param(
    [switch]$ApplyRegistryPreferences,
    [switch]$RestartDesktopApps
)

$ErrorActionPreference = "Stop"
$RegPath = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"

if (-not (Test-Path $RegPath)) {
    New-Item -Path $RegPath -Force | Out-Null
}

$CandidateApps = @(
    "$env:LOCALAPPDATA\BraveSoftware\Brave-Browser\Application\brave.exe",
    "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    "$env:LOCALAPPDATA\Thorium\Application\thorium.exe"
)

if ($ApplyRegistryPreferences) {
    foreach ($app in $CandidateApps) {
        if (Test-Path $app) {
            Set-ItemProperty -Path $RegPath -Name $app -Value "GpuPreference=1;" -Type String
            Write-Host "[GPU Isolation] Bound $app -> AMD Radeon 780M iGPU (GpuPreference=1)" -ForegroundColor Green
        }
    }
}

Write-Host "`n=== Current NVIDIA RTX 4060 Laptop GPU Status ===" -ForegroundColor Cyan
nvidia-smi
