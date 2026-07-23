[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BuildDirectory,
    [Parameter(Mandatory = $true)][string]$EnginePath,
    [Parameter(Mandatory = $true)][string]$PluginPath,
    [Parameter(Mandatory = $true)][string]$QtRoot,
    [Parameter(Mandatory = $true)][string]$TensorRTRoot,
    [Parameter(Mandatory = $true)][string]$CudaRoot,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$PackageName = "",
    [string]$SampleCloudPath = ""
)

$ErrorActionPreference = "Stop"
$version = "0.1.1"
$resolvedOutputRoot = [IO.Path]::GetFullPath($OutputRoot).TrimEnd("\")
$resolvedPackageName = if ([string]::IsNullOrWhiteSpace($PackageName)) {
    "PTV2_Weld_App_$version"
} else {
    $PackageName.Trim()
}
if ($resolvedPackageName -ne [IO.Path]::GetFileName($resolvedPackageName)) {
    throw "UNSAFE_PACKAGE_NAME: $resolvedPackageName"
}
$packageRoot = Join-Path $resolvedOutputRoot $resolvedPackageName
if ([IO.Path]::GetDirectoryName($packageRoot).TrimEnd("\") -ne $resolvedOutputRoot) {
    throw "UNSAFE_PACKAGE_TARGET: $packageRoot"
}
$exe = Join-Path $BuildDirectory "Release\ptv2_weld_qt_smoke.exe"
$windeployqt = Join-Path $QtRoot "bin\windeployqt.exe"

$required = @(
    $exe,
    $EnginePath,
    $PluginPath,
    $windeployqt,
    (Join-Path $TensorRTRoot "bin\nvinfer_11.dll"),
    (Join-Path $TensorRTRoot "bin\nvinfer_plugin_11.dll"),
    (Join-Path $CudaRoot "bin\cudart64_12.dll")
)
if (-not [string]::IsNullOrWhiteSpace($SampleCloudPath)) {
    $required += $SampleCloudPath
}
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "PACKAGE_REQUIRED_DEPENDENCY_MISSING: $path"
    }
}

if (Test-Path -LiteralPath $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $packageRoot | Out-Null
@("config", "engine", "plugins", "logs", "exports", "sample") | ForEach-Object {
    New-Item -ItemType Directory -Path (Join-Path $packageRoot $_) | Out-Null
}

Copy-Item -LiteralPath $exe -Destination (Join-Path $packageRoot "ptv2_weld_qt.exe")
Copy-Item -LiteralPath $EnginePath -Destination (
    Join-Path $packageRoot "engine\strict_fp32_voxelunique_cub.plan")
Copy-Item -LiteralPath $PluginPath -Destination (
    Join-Path $packageRoot "plugins\VoxelUniqueCubPlugin.dll")
Copy-Item -LiteralPath (Join-Path $TensorRTRoot "bin\nvinfer_11.dll") -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $TensorRTRoot "bin\nvinfer_plugin_11.dll") -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $CudaRoot "bin\cudart64_12.dll") -Destination $packageRoot
if (-not [string]::IsNullOrWhiteSpace($SampleCloudPath)) {
    Copy-Item -LiteralPath $SampleCloudPath -Destination (
        Join-Path $packageRoot "sample\weld_65.txt")
}

& $windeployqt --release --no-translations --no-system-d3d-compiler `
    --dir $packageRoot (Join-Path $packageRoot "ptv2_weld_qt.exe")
if ($LASTEXITCODE -ne 0) {
    throw "WINDEPLOYQT_FAILED: exit=$LASTEXITCODE"
}

$engineSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $EnginePath).Hash.ToLowerInvariant()
$pluginSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $PluginPath).Hash.ToLowerInvariant()
$ini = @"
[Runtime]
engine_path=../engine/strict_fp32_voxelunique_cub.plan
plugin_path=../plugins/VoxelUniqueCubPlugin.dll
engine_sha256=$engineSha
plugin_sha256=$pluginSha

[Application]
default_cloud_directory=
default_export_directory=../exports
remember_last_cloud=true
remember_window_geometry=true
auto_initialize=true

[Visualization]
show_bbox=true
show_centroid=true
show_pca=true
point_size=3.0

[Logging]
log_directory=../logs
maximum_log_files=20
"@
Set-Content -LiteralPath (Join-Path $packageRoot "config\qt_weld_app.ini") `
    -Value $ini -Encoding UTF8

$launcher = @'
@echo off
setlocal
set "ROOT=%~dp0"
set "PATH=%ROOT%;%ROOT%plugins;%PATH%"
"%ROOT%ptv2_weld_qt.exe" --config "%ROOT%config\qt_weld_app.ini" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo PTV2 Weld Segmentation exited with code %EXIT_CODE%
exit /b %EXIT_CODE%
'@
Set-Content -LiteralPath (Join-Path $packageRoot "launch.bat") -Value $launcher -Encoding ASCII

$readme = @"
PTV2 Weld Segmentation 0.1.1

Launch: double-click launch.bat
Requirements: NVIDIA RTX 5060-compatible driver and Windows x64.
The Engine and VoxelUniqueCub Plugin are package-local and validated by SHA-256.
Input: weld point-cloud TXT with x y z label columns and at least 2048 valid points.
Labels: class 0 = weld_seam; class 1 = background.
Sample: sample\weld_65.txt (when included by the qualification package).
"@
Set-Content -LiteralPath (Join-Path $packageRoot "README.txt") -Value $readme -Encoding UTF8
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "..\QUICK_START.md") `
    -Destination (Join-Path $packageRoot "QUICK_START.md")

$files = Get-ChildItem -LiteralPath $packageRoot -Recurse -File | Sort-Object FullName
$inventory = [ordered]@{
    application = "PTV2 Weld Segmentation"
    version = $version
    generated_at = (Get-Date).ToString("o")
    precision = "Strict FP32"
    tf32_enabled = $false
    fp16_enabled = $false
    int8_enabled = $false
    engine_sha256 = $engineSha
    plugin_sha256 = $pluginSha
    files = @($files | ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($packageRoot.Length).TrimStart("\").Replace("\", "/")
            size_bytes = $_.Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        }
    })
}
$inventory | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (
    Join-Path $packageRoot "runtime_inventory.json") -Encoding UTF8

$checksumLines = Get-ChildItem -LiteralPath $packageRoot -Recurse -File |
    Where-Object { $_.Name -ne "checksums.sha256" } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($packageRoot.Length).TrimStart("\").Replace("\", "/")
        "$((Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant())  $relative"
    }
Set-Content -LiteralPath (Join-Path $packageRoot "checksums.sha256") `
    -Value $checksumLines -Encoding ASCII

$mandatory = @(
    "ptv2_weld_qt.exe",
    "platforms\qwindows.dll",
    "engine\strict_fp32_voxelunique_cub.plan",
    "plugins\VoxelUniqueCubPlugin.dll",
    "config\qt_weld_app.ini",
    "launch.bat",
    "QUICK_START.md",
    "runtime_inventory.json",
    "checksums.sha256"
)
foreach ($relative in $mandatory) {
    if (-not (Test-Path -LiteralPath (Join-Path $packageRoot $relative) -PathType Leaf)) {
        throw "PACKAGE_CONTRACT_FAILED: $relative"
    }
}
Write-Output "PACKAGE_RELEASE_COMPLETED=$packageRoot"
