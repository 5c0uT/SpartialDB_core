# Check PhysX in vcpkg
$vcpkgPath = "D:\SpartialDB_core\dependencies\vcpkg\installed\x64-windows"

Write-Host "=== Checking for PhysX in vcpkg ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "Include directory:" -ForegroundColor Yellow
if (Test-Path "$vcpkgPath\include\physx") {
    ls "$vcpkgPath\include\physx" | Select-Object Name
    Write-Host "✓ PhysX headers found" -ForegroundColor Green
} else {
    Write-Host "✗ PhysX headers NOT found" -ForegroundColor Red
    Write-Host "Available include dirs:"
    ls "$vcpkgPath\include" | Select-Object Name | head -20
}

Write-Host ""
Write-Host "Library files:" -ForegroundColor Yellow
Write-Host "Looking for .lib files with 'physx' in name:"
ls "$vcpkgPath\lib\*physx*.lib" -ErrorAction SilentlyContinue | Select-Object Name

if ((ls "$vcpkgPath\lib\*physx*.lib" -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
    Write-Host "✗ PhysX libraries NOT found" -ForegroundColor Red
    Write-Host "All .lib files in lib directory:"
    ls "$vcpkgPath\lib\*.lib" -ErrorAction SilentlyContinue | Select-Object Name | head -30
} else {
    Write-Host "✓ PhysX libraries found" -ForegroundColor Green
}

Write-Host ""
Write-Host "DLL files:" -ForegroundColor Yellow
Write-Host "Looking for .dll files with 'physx' in name:"
ls "$vcpkgPath\bin\*physx*.dll" -ErrorAction SilentlyContinue | Select-Object Name

if ((ls "$vcpkgPath\bin\*physx*.dll" -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
    Write-Host "✗ PhysX DLLs NOT found" -ForegroundColor Red
    Write-Host "All files in bin directory:"
    ls "$vcpkgPath\bin\*" -ErrorAction SilentlyContinue | Select-Object Name | head -30
} else {
    Write-Host "✓ PhysX DLLs found" -ForegroundColor Green
}
