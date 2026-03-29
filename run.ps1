#Requires -Version 5.1

param(
    [string]$CondaEnv = "spatial_env",
    [string]$CondaRoot = "C:\ProgramData\Miniconda3",
    [switch]$SkipPip,
    [switch]$SkipVcpkg,
    [switch]$SkipTests,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Assert-Path {
    param(
        [string]$PathValue,
        [string]$Description
    )

    if (-not (Test-Path $PathValue)) {
        throw "$Description not found: $PathValue"
    }
}

function Invoke-Checked {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Command
    )

    if ($Command.Length -eq 0) {
        throw "Invoke-Checked requires a command"
    }

    if ($Command.Length -eq 1) {
        & $Command[0]
    } else {
        & $Command[0] $Command[1..($Command.Length - 1)]
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$condaExe = Join-Path $CondaRoot "Scripts\conda.exe"
$envPython = Join-Path $CondaRoot "envs\$CondaEnv\python.exe"
$vcpkgRoot = Join-Path $projectRoot "dependencies\vcpkg"
$vcpkgExe = Join-Path $vcpkgRoot "vcpkg.exe"
$toolchainFile = Join-Path $vcpkgRoot "scripts\buildsystems\vcpkg.cmake"
$physxBin = Join-Path $vcpkgRoot "installed\x64-windows\bin"
$buildDir = Join-Path $projectRoot "build"

Assert-Path $condaExe "conda.exe"
Assert-Path $envPython "Conda environment Python"

Write-Step "Environment"
Write-Host "Project:    $projectRoot"
Write-Host "Conda env:  $CondaEnv"
Write-Host "Python:     $envPython"

if (-not $SkipPip) {
    Write-Step "Install Python dependencies"
    Invoke-Checked $condaExe run -n $CondaEnv python -m pip install -r requirements.txt
}

if (-not $SkipVcpkg) {
    Write-Step "Prepare vcpkg"
    if (-not (Test-Path $vcpkgRoot)) {
        if (-not (Test-Path (Join-Path $projectRoot "dependencies"))) {
            New-Item -ItemType Directory -Path (Join-Path $projectRoot "dependencies") | Out-Null
        }
        git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
    }

    if (-not (Test-Path $vcpkgExe)) {
        Invoke-Checked (Join-Path $vcpkgRoot "bootstrap-vcpkg.bat")
    }

    Write-Step "Install PhysX"
    Invoke-Checked $vcpkgExe install physx:x64-windows
}

Assert-Path $vcpkgExe "vcpkg executable"
Assert-Path $toolchainFile "vcpkg toolchain file"

if ($Clean -and (Test-Path $buildDir)) {
    Write-Step "Clean build directory"
    Remove-Item $buildDir -Recurse -Force
}

Write-Step "Configure CMake"
Invoke-Checked cmake -S $projectRoot -B $buildDir -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_TOOLCHAIN_FILE="$toolchainFile" `
    -DPython_EXECUTABLE="$envPython" `
    -DPython_ROOT_DIR="$(Split-Path $envPython -Parent)" `
    -DPYTHON_ROOT="$(Split-Path $envPython -Parent)" `
    -DCONDA_ENV_PATH="$(Split-Path $envPython -Parent)"

Write-Step "Build"
Invoke-Checked cmake --build $buildDir --config Release --parallel 16

$env:PATH = "$physxBin;$(Split-Path $envPython -Parent);$(Join-Path (Split-Path $envPython -Parent) 'Library\bin');$(Join-Path (Split-Path $envPython -Parent) 'Scripts');$env:PATH"

Write-Step "Smoke import"
@'
import pathlib
import sys

root = pathlib.Path.cwd()
sys.path.insert(0, str(root / "python"))

import spatial_db

print("HAS_NATIVE_MODULE =", spatial_db.HAS_NATIVE_MODULE)
print("native =", getattr(spatial_db.native, "__name__", type(spatial_db.native).__name__))
print("init_physx =", spatial_db.native.init_physx(0))
'@ | & $envPython -
if ($LASTEXITCODE -ne 0) {
    throw "Smoke import failed with exit code ${LASTEXITCODE}"
}

if (-not $SkipTests) {
    Write-Step "Run tests"
    Invoke-Checked $condaExe run -n $CondaEnv python -m pytest tests -q
}

Write-Step "Done"
