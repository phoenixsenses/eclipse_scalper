param(
    [string]$ShortcutName = "EclipseScalperDataStack.lnk",
    [string]$Symbols = "BTCUSDT,ETHUSDT,SOLUSDT"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StartScript = Join-Path $RepoRoot "start_eclipse.ps1"

if (-not (Test-Path -LiteralPath $StartScript)) {
    throw "Missing start script: $StartScript"
}

$StartupDir = [Environment]::GetFolderPath("Startup")
if (-not (Test-Path -LiteralPath $StartupDir)) {
    throw "Startup directory not found: $StartupDir"
}

$ShortcutPath = Join-Path $StartupDir $ShortcutName
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$StartScript`" -Symbols `"$Symbols`""
$Shortcut.WorkingDirectory = $RepoRoot
$Shortcut.WindowStyle = 7
$Shortcut.Description = "Start Eclipse Scalper data stack"
$Shortcut.Save()

Write-Output "INSTALLED startup_shortcut=$ShortcutPath"
Write-Output "target=powershell.exe $($Shortcut.Arguments)"
