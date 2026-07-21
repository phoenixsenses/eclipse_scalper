param(
    [string]$ShortcutName = "EclipseScalperDataStack.lnk"
)

$ErrorActionPreference = "Stop"
$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir $ShortcutName

if (Test-Path -LiteralPath $ShortcutPath) {
    Remove-Item -LiteralPath $ShortcutPath -Force
    Write-Output "UNINSTALLED startup_shortcut=$ShortcutPath"
} else {
    Write-Output "NOT_INSTALLED startup_shortcut=$ShortcutPath"
}
