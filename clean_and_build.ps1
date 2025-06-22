<#
.SYNOPSIS
Build script for SpatialDB project with GPU acceleration
.DESCRIPTION
Cleans, builds and copies dependencies for the project
#>

# Environment setup
Write-Host "`n==================== ENVIRONMENT SETUP ====================" -ForegroundColor Cyan
Write-Host "Setting execution policy..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-Command {Set-ExecutionPolicy RemoteSigned -Scope Process -Force}" -Verb RunAs

# CMake version check
Write-Host "`n==================== DEPENDENCY CHECK ====================" -ForegroundColor Cyan
Write-Host "Checking CMake version:" -ForegroundColor Yellow
$cmake_version_output = cmake --version
$cmake_version = $cmake_version_output.Split()[2]  # Extract version number
Write-Host "Detected CMake version: $cmake_version"

# Check if version meets requirements
$required_version = [System.Version]"3.15.0"
$current_version = [System.Version]$cmake_version

if ($current_version -lt $required_version) {
    Write-Host "`n[ERROR] CMake $required_version or higher required!" -ForegroundColor Red
    Write-Host "Installed version: $cmake_version" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "`n[SUCCESS] CMake version meets requirements" -ForegroundColor Green
}

# Dependency paths
$project_dir = "C:\Users\lolol\PycharmProjects\PhysX_projects\spatialdb"
$python_venv = "C:\Users\lolol\PycharmProjects\PhysX_projects\.venv"
$physx_bin_dir = "C:\PhysX-107.0-physx-5.6.0\physx\bin\win.x86_64.vc143.mt\release"
$miniforge_bin_dir = "C:\ProgramData\miniforge3\envs\spatial_env\Library\bin"

# Console encoding
Write-Host "`nSetting console encoding..." -ForegroundColor Yellow
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

# Project cleanup
Write-Host "`n==================== PROJECT CLEANUP ====================" -ForegroundColor Cyan
Write-Host "Removing previous build artifacts..." -ForegroundColor Yellow
$cleanItems = @(
    "build",
    "CMakeCache.txt",
    "CMakeFiles",
    "*.sln",
    "*.vcxproj*",
    "*.pyd",
    "*.dll"
)

foreach ($item in $cleanItems) {
    Write-Host "Deleting: $item" -ForegroundColor DarkGray
    Remove-Item -Path $item -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "`n[SUCCESS] Project cleaned" -ForegroundColor Green

# Build directory setup
Write-Host "`n==================== BUILD SETUP ====================" -ForegroundColor Cyan
Write-Host "Creating build directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "build" -Force -ErrorAction Stop | Out-Null
Set-Location "build" -ErrorAction Stop
Write-Host "Current directory: $(Get-Location)" -ForegroundColor DarkGray

# CMake project generation
Write-Host "`n==================== PROJECT GENERATION ====================" -ForegroundColor Cyan
Write-Host "Generating project with CMake..." -ForegroundColor Yellow
Write-Host "Using generator: Visual Studio 17 2022" -ForegroundColor DarkGray
Write-Host "Architecture: x64" -ForegroundColor DarkGray
Write-Host "Build type: Release" -ForegroundColor DarkGray

$generateOutput = cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_DEBUG_POSTFIX="" -DCMAKE_RELEASE_POSTFIX="" -DPython_ROOT_DIR="C:\Path\To\Python\3.13"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] CMake generation failed!" -ForegroundColor Red
    Write-Host "Error details:" -ForegroundColor Red
    Write-Host $generateOutput -ForegroundColor Red

    # Save error log
    $logPath = "$project_dir\build_error.log"
    $generateOutput | Out-File -Encoding UTF8 $logPath
    Write-Host "`nFull error log saved to: $logPath" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "CMakeCache.txt")) {
    Write-Host "`n[ERROR] CMakeCache.txt not found after generation!" -ForegroundColor Red
    Write-Host "Directory contents:" -ForegroundColor Red
    Get-ChildItem | Format-Table Name, Length, LastWriteTime
    exit 1
}

Write-Host "`n[SUCCESS] Project generated successfully" -ForegroundColor Green

# Project build
Write-Host "`n==================== PROJECT BUILD ====================" -ForegroundColor Cyan
Write-Host "Building native module..." -ForegroundColor Yellow
Write-Host "Build target: spatialdb_core_pybind" -ForegroundColor DarkGray
Write-Host "Configuration: Release" -ForegroundColor DarkGray

$buildOutput = cmake --build . --config Release --target spatialdb_core_pybind

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Build failed!" -ForegroundColor Red
    Write-Host "Error details:" -ForegroundColor Red
    Write-Host $buildOutput -ForegroundColor Red

    # Save error log
    $logPath = "$project_dir\build_error.log"
    $buildOutput | Out-File -Encoding UTF8 $logPath
    Write-Host "`nFull build log saved to: $logPath" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n[SUCCESS] Project built successfully!" -ForegroundColor Green

# Dependency copying
Write-Host "`n==================== COPYING DEPENDENCIES ====================" -ForegroundColor Cyan
$dest_dir = "$project_dir\python\spatial_db\lib"
Write-Host "Target directory: $dest_dir" -ForegroundColor Yellow

# Create target directory
New-Item -ItemType Directory -Path $dest_dir -Force -ErrorAction SilentlyContinue | Out-Null

# Files to copy
$filesToCopy = @(
    @{Source = "$physx_bin_dir\PhysXCommon_64.dll"; Dest = "$dest_dir\PhysXCommon_64.dll"},
    @{Source = "$physx_bin_dir\PhysXCooking_64.dll"; Dest = "$dest_dir\PhysXCooking_64.dll"},
    @{Source = "$physx_bin_dir\PhysX_64.dll"; Dest = "$dest_dir\PhysX_64.dll"},
    @{Source = "$physx_bin_dir\PhysXFoundation_64.dll"; Dest = "$dest_dir\PhysXFoundation_64.dll"},
    @{Source = "$miniforge_bin_dir\libcurl.dll"; Dest = "$dest_dir\libcurl.dll"},
    @{Source = "$miniforge_bin_dir\proj_9_2.dll"; Dest = "$dest_dir\proj.dll"}
)

# Copy files
foreach ($file in $filesToCopy) {
    $source = $file.Source
    $dest = $file.Dest

    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $dest -Force -ErrorAction Stop
        Write-Host "Copied: $(Split-Path $source -Leaf) -> $dest" -ForegroundColor DarkGray
    }
    else {
        Write-Host "[WARNING] File not found: $source" -ForegroundColor Yellow
    }
}

# Find and copy PYD file
Write-Host "`nSearching for compiled module..." -ForegroundColor Yellow
$pyd_path = Get-ChildItem -Path . -Recurse -Filter "spatialdb_core_pybind.pyd" |
            Select-Object -First 1 -ExpandProperty FullName

if ($pyd_path) {
    Copy-Item -Path $pyd_path -Destination $dest_dir -Force -ErrorAction Stop
    Write-Host "`n[SUCCESS] Native module copied: $(Split-Path $pyd_path -Leaf)" -ForegroundColor Green
    Write-Host "Location: $dest_dir" -ForegroundColor DarkGray
}
else {
    Write-Host "`n[ERROR] spatialdb_core_pybind.pyd not found!" -ForegroundColor Red
    Write-Host "Search locations:" -ForegroundColor Red
    Get-ChildItem -Recurse -Include *.pyd | Format-Table FullName
    exit 1
}

# Completion
Write-Host "`n==================== BUILD COMPLETE ====================" -ForegroundColor Cyan
Write-Host "[SUCCESS] Project built and ready to use!" -ForegroundColor Green
Write-Host "Completion time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor DarkGray
Write-Host "`nNext steps:"
Write-Host "1. Activate Python virtual environment" -ForegroundColor DarkGray
Write-Host "2.1. Run the script as an example: python .\gis_terrain.py" -ForegroundColor DarkGray
Write-Host "2.2. Run the script as an example: python .\lidar_medical.py" -ForegroundColor DarkGray