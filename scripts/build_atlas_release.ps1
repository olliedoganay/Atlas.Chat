param(
  [switch]$CleanBundle
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$atlasDir = Join-Path $repoRoot "apps\atlas"
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$bundleDir = Join-Path $atlasDir "src-tauri\target\release\bundle"

if (-not (Test-Path -LiteralPath $atlasDir)) {
  throw "Atlas desktop directory not found: $atlasDir"
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
  throw "Python environment not found: $pythonExe"
}

if (-not (Test-Path -LiteralPath (Join-Path $atlasDir "node_modules"))) {
  throw "Frontend dependencies not found. Run 'npm install' in apps\\atlas first."
}

$version = (& $pythonExe scripts\check_atlas_version.py).Trim()
if ($LASTEXITCODE -ne 0 -or -not $version) {
  throw "Atlas version check failed."
}

if ($CleanBundle -and (Test-Path -LiteralPath $bundleDir)) {
  Remove-Item -LiteralPath $bundleDir -Recurse -Force
}

Write-Host "Building Atlas MSI release bundle for Atlas v$version..." -ForegroundColor Cyan
Push-Location $atlasDir
try {
  npm.cmd run tauri build -- --bundles msi
  if ($LASTEXITCODE -ne 0) {
    throw "Atlas release build failed with exit code $LASTEXITCODE."
  }
} finally {
  Pop-Location
}

if (-not (Test-Path -LiteralPath $bundleDir)) {
  throw "Atlas release bundle directory was not created: $bundleDir"
}

$artifacts = Get-ChildItem -Path $bundleDir -Recurse -File |
  Where-Object { $_.Extension -eq ".msi" }

Write-Host ""
Write-Host "Release artifacts:" -ForegroundColor Green
foreach ($artifact in $artifacts) {
  $relative = $artifact.FullName.Substring($repoRoot.Length).TrimStart("\")
  Write-Host " - $relative"
}

if (@($artifacts).Count -ne 1) {
  throw "Expected exactly one MSI installer artifact under $bundleDir; found $(@($artifacts).Count)."
}
