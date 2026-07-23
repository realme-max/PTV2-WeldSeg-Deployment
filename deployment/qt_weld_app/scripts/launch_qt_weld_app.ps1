[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ApplicationArguments
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "ptv2_weld_qt.exe"
$config = Join-Path $root "config\qt_weld_app.ini"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "APPLICATION_NOT_FOUND: $exe"
}
if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "CONFIGURATION_NOT_FOUND: $config"
}
$env:PATH = "$root;$(Join-Path $root 'plugins');$env:PATH"
& $exe --config $config @ApplicationArguments
exit $LASTEXITCODE
