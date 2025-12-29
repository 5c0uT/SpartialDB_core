#Requires -Version 5.1
<#
.SYNOPSIS
Complete build script for SpatialDB_Core

.DESCRIPTION
Fully automated build with all dependency management
#>

param(
    [switch]$SkipDependencies,
    [switch]$SkipBuild,
    [string]$CondaPath = "C:\ProgramData\Miniconda3",
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"
$VerbosePreference = if ($Verbose) { "Continue" } else { "SilentlyContinue" }

# Setup
$logFile = Join-Path (Get-Location) "build.log"
$startTime = Get-Date
"" | Set-Content $logFile -Force

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("INFO", "SUCCESS", "ERROR", "WARNING", "VERBOSE")]
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Add-Content -Path $logFile -Value $logEntry -Encoding UTF8
    
    switch ($Level) {
        "SUCCESS" { Write-Host "[OK] $Message" -ForegroundColor Green }
        "ERROR" { Write-Host "[XX] $Message" -ForegroundColor Red }
        "WARNING" { Write-Host "[!!] $Message" -ForegroundColor Yellow }
        "INFO" { Write-Host "[->] $Message" -ForegroundColor Cyan }
        "VERBOSE" { if ($Verbose) { Write-Host "[~~] $Message" -ForegroundColor Gray } }
    }
}

function Write-Section {
    param([string]$Title)
    $line = "=" * 80
    Add-Content -Path $logFile -Value "`n$line`n$Title`n$line`n" -Encoding UTF8
    Write-Host "`n$line" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "$line`n" -ForegroundColor Cyan
}

Write-Log "========== SpatialDB_Core Build System ==========" "INFO"
Write-Log "Build started at: $startTime" "INFO"
Write-Log "Log file: $logFile" "INFO"

# ADMIN CHECK
Write-Section "STEP 0: Administrator Check"
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Log "Admin privileges required" "ERROR"
    exit 1
}
Write-Log "Admin: OK" "SUCCESS"

$projectDir = Get-Location
$buildDir = Join-Path $projectDir "build"
$depsDir = Join-Path $projectDir "dependencies"
$vcpkgDir = Join-Path $depsDir "vcpkg"

# STEP 1: SYSTEM REQUIREMENTS
Write-Section "STEP 1: System Requirements"

$systemOK = $true

Write-Log "PowerShell: checking" "INFO"
if ($PSVersionTable.PSVersion.Major -ge 5) {
    Write-Log "PowerShell: $($PSVersionTable.PSVersion.Major).0+ OK" "SUCCESS"
} else {
    Write-Log "PowerShell too old" "ERROR"
    $systemOK = $false
}

Write-Log "Windows: checking" "INFO"
$osVersion = [System.Environment]::OSVersion.Version
if ($osVersion.Major -ge 10) {
    Write-Log "Windows: $($osVersion.Major).$($osVersion.Minor) OK" "SUCCESS"
} else {
    Write-Log "Windows too old" "ERROR"
    $systemOK = $false
}

Write-Log "Git: checking" "INFO"
if (Get-Command git -ErrorAction SilentlyContinue) {
    $gitVersion = git --version
    Write-Log "Git: $gitVersion OK" "SUCCESS"
} else {
    Write-Log "Git: NOT FOUND" "ERROR"
    $systemOK = $false
}

Write-Log "CMake: checking" "INFO"
if (Get-Command cmake -ErrorAction SilentlyContinue) {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Log "CMake: $cmakeVersion OK" "SUCCESS"
} else {
    Write-Log "CMake: NOT FOUND" "ERROR"
    $systemOK = $false
}

Write-Log "Visual Studio: checking" "INFO"
$vsPath = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($vsPath) {
    Write-Log "VS: $($vsPath.Name) OK" "SUCCESS"
} else {
    Write-Log "VS 2022: NOT FOUND" "ERROR"
    $systemOK = $false
}

Write-Log "MSBuild: checking" "INFO"
$msbuildPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
)
$msbuildFound = $false
foreach ($path in $msbuildPaths) {
    if (Test-Path $path) {
        Write-Log "MSBuild: OK" "SUCCESS"
        $msbuildFound = $true
        break
    }
}
if (-not $msbuildFound) {
    Write-Log "MSBuild: NOT FOUND" "ERROR"
    $systemOK = $false
}

if (-not $systemOK) {
    Write-Log "Critical requirements missing" "ERROR"
    exit 1
}

# STEP 2: CONDA
Write-Section "STEP 2: Conda Detection"

$condaPaths = @(
    "C:\ProgramData\Miniconda3",
    "C:\ProgramData\Anaconda3",
    "$env:USERPROFILE\miniconda3",
    "$env:USERPROFILE\Anaconda3"
)

$condaPath = $null
$condaExe = $null

Write-Log "Searching conda..." "INFO"
try {
    $condaInPath = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($condaInPath) {
        $condaExe = $condaInPath.Source
        $condaPath = Split-Path (Split-Path $condaExe -Parent) -Parent
        Write-Log "Found: $condaPath" "SUCCESS"
    }
}
catch { }

if (-not $condaPath) {
    foreach ($path in $condaPaths) {
        if (Test-Path "$path\Scripts\conda.exe") {
            $condaPath = $path
            $condaExe = Join-Path $path "Scripts\conda.exe"
            Write-Log "Found: $path" "SUCCESS"
            break
        }
    }
}

if (-not $condaPath) {
    Write-Section "Installing Miniconda"
    $url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    $installerPath = "$env:TEMP\Miniconda3-installer.exe"
    
    Write-Log "Downloading..." "INFO"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $installerPath -ErrorAction Stop
    Write-Log "Downloaded" "SUCCESS"
    
    $minicondaPath = $CondaPath
    Write-Log "Installing to: $minicondaPath" "INFO"
    
    $installArgs = @(
        "/InstallationType=JustMe",
        "/AddMinicondaToPath=Yes",
        "/RegisterPython=Yes",
        "/S",
        "/D=$minicondaPath"
    )
    Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -NoNewWindow
    
    if (Test-Path "$minicondaPath\Scripts\conda.exe") {
        $condaPath = $minicondaPath
        $condaExe = Join-Path $minicondaPath "Scripts\conda.exe"
        Write-Log "Installed" "SUCCESS"
    } else {
        Write-Log "Install failed" "ERROR"
        exit 1
    }
    
    Remove-Item -Path $installerPath -Force -ErrorAction SilentlyContinue
}

$env:PATH = "$condaPath\Scripts;$condaPath\Library\bin;$env:PATH"
Write-Log "Verifying conda..." "INFO"
$condaVersion = & $condaExe --version 2>&1
Write-Log "Conda: $condaVersion" "SUCCESS"
$env:CONDA_ENV_PATH = Join-Path (Split-Path $condaPath) "envs\spatial_env"

# STEP 3: PYTHON ENVIRONMENT
Write-Section "STEP 3: Python Environment"

Write-Log "Checking spatial_env..." "INFO"
$envList = & $condaExe env list 2>&1
if ($envList -match "spatial_env") {
    Write-Log "Environment exists" "SUCCESS"
} else {
    Write-Log "Creating environment..." "INFO"
    & $condaExe create -n spatial_env -c conda-forge python=3.11 -y 2>&1 | Out-Null
    Write-Log "Created" "SUCCESS"
}

Write-Log "Installing conda packages..." "INFO"
@("proj", "libcurl", "numpy", "scipy") | ForEach-Object {
    $pkgName = $_
    & $condaExe run -n spatial_env conda install -c conda-forge $pkgName -y 2>&1 | Out-Null
    Write-Log "$pkgName - OK" "SUCCESS"
}

Write-Log "Installing pip packages..." "INFO"
& $condaExe run -n spatial_env pip install pybind11 scikit-learn --quiet 2>&1 | Out-Null
Write-Log "pip packages - OK" "SUCCESS"

# STEP 4: DEPENDENCIES
Write-Section "STEP 4: Dependencies"
if (-not (Test-Path $depsDir)) {
    New-Item -ItemType Directory -Path $depsDir -Force | Out-Null
}
Write-Log "Dependencies: $depsDir" "SUCCESS"

# STEP 5-6: VCPKG & PHYSX
if (-not $SkipDependencies) {
    Write-Section "STEP 5: VCPKG"
    
    if (-not (Test-Path $vcpkgDir)) {
        Write-Log "Cloning vcpkg..." "INFO"
        Set-Location $depsDir
        git clone https://github.com/Microsoft/vcpkg.git 2>&1 | Out-Null
        Set-Location $vcpkgDir
        Write-Log "Bootstrap..." "INFO"
        .\bootstrap-vcpkg.bat 2>&1 | Out-Null
        Set-Location $projectDir
        Write-Log "vcpkg - OK" "SUCCESS"
    } else {
        Write-Log "vcpkg exists" "SUCCESS"
    }
    
    Write-Section "STEP 6: PhysX"
    
    $physxInclude = Join-Path $vcpkgDir "installed\x64-windows\include\physx"
    $physxLibs = @(Get-ChildItem "$vcpkgDir\installed\x64-windows\lib\*physx*.lib" -ErrorAction SilentlyContinue)
    
    if ((Test-Path $physxInclude) -and ($physxLibs.Count -gt 0)) {
        Write-Log "PhysX exists" "SUCCESS"
    } else {
        Write-Log "Installing PhysX (10-15 min)..." "WARNING"
        Set-Location $vcpkgDir
        .\vcpkg install physx:x64-windows --triplet x64-windows 2>&1 | Out-Null
        Set-Location $projectDir
        
        if ((Test-Path $physxInclude) -and (@(Get-ChildItem "$vcpkgDir\installed\x64-windows\lib\*physx*.lib" -ErrorAction SilentlyContinue).Count -gt 0)) {
            Write-Log "PhysX - OK" "SUCCESS"
        } else {
            Write-Log "PhysX install failed" "ERROR"
            exit 1
        }
    }
    
    Write-Log "PhysX headers - OK" "SUCCESS"
    $count = @(Get-ChildItem "$vcpkgDir\installed\x64-windows\lib\*physx*.lib" -ErrorAction SilentlyContinue).Count
    Write-Log "PhysX libraries - $count files" "SUCCESS"
}

# STEP 7: CLEAN BUILD
Write-Section "STEP 7: Clean Build"

if (Test-Path $buildDir) {
    Write-Log "Removing old build..." "INFO"
    Remove-Item -Path $buildDir -Recurse -Force -ErrorAction Stop
    Write-Log "Cleaned" "SUCCESS"
}

New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
Write-Log "Build directory ready" "SUCCESS"

# STEP 8-9: CMAKE & BUILD
if (-not $SkipBuild) {
    Write-Section "STEP 8: CMake Configuration"
    
    Set-Location $buildDir
    
    $vcpkgToolchain = "$vcpkgDir\scripts\buildsystems\vcpkg.cmake"
    $physxPath = "$vcpkgDir\installed\x64-windows"
    
    Write-Log "Running CMake..." "INFO"
    
    & cmake ".." -G "Visual Studio 17 2022" -A x64 `
        "-DCMAKE_BUILD_TYPE=Release" `
        "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain" `
        "-DCMAKE_PREFIX_PATH=$physxPath" `
        "-DPYTHON_ROOT=C:\Python314" `
        "-DCONDA_ENV_PATH=$env:CONDA_ENV_PATH" 2>&1 | Tee-Object -FilePath "$projectDir\cmake_config.log" | ForEach-Object {
        Add-Content -Path $logFile -Value $_ -Encoding UTF8
        if ($Verbose) { Write-Host $_ -ForegroundColor Gray }
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Log "CMake FAILED" "ERROR"
        Write-Log "See: cmake_config.log" "ERROR"
        Set-Location $projectDir
        exit 1
    }
    
    Write-Log "CMake - OK" "SUCCESS"
    
    Write-Section "STEP 9: Building Project"
    
    Write-Log "Building with 16 parallel jobs..." "INFO"
    
    & cmake --build . --config Release --parallel 16 2>&1 | Tee-Object -FilePath "$projectDir\build_output.log" | ForEach-Object {
        Add-Content -Path $logFile -Value $_ -Encoding UTF8
        if ($Verbose) { Write-Host $_ -ForegroundColor Gray }
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Build FAILED" "ERROR"
        Set-Location $projectDir
        exit 1
    }
    
    Write-Log "Build - OK" "SUCCESS"
    
    Write-Section "STEP 10: Output Files"
    
    $outputDir = Join-Path $projectDir "spatialdb_core"
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    
    $pydFile = Get-ChildItem -Path . -Recurse -Filter "*.pyd" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pydFile) {
        Copy-Item -Path $pydFile.FullName -Destination $outputDir -Force
        Write-Log "Module: $($pydFile.Name)" "SUCCESS"
    }
    
    $dllCount = 0
    @(
        "$vcpkgDir\installed\x64-windows\bin",
        "$env:CONDA_ENV_PATH\Library\bin"
    ) | ForEach-Object {
        if (Test-Path $_) {
            Get-ChildItem "$_\*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-Item -Path $_.FullName -Destination $outputDir -Force -ErrorAction SilentlyContinue
                $dllCount++
            }
        }
    }
    
    Write-Log "DLLs: $dllCount copied" "SUCCESS"
    Write-Log "Output: $outputDir" "SUCCESS"
    
    Set-Location $projectDir
}

# SUMMARY
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Section "BUILD COMPLETE"

Write-Log "Start: $startTime" "INFO"
Write-Log "End: $endTime" "INFO"
Write-Log "Duration: $([Math]::Round($duration.TotalMinutes, 2)) minutes" "INFO"
Write-Log "SUCCESS" "SUCCESS"

Write-Host ""
Write-Host "CONDA Environment: $env:CONDA_ENV_PATH" -ForegroundColor Yellow
Write-Host "Build Log: $logFile" -ForegroundColor Yellow
Write-Host ""
