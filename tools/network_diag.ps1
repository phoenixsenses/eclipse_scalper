param()

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogsDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path -LiteralPath $LogsDir)) {
    New-Item -Path $LogsDir -ItemType Directory | Out-Null
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outPath = Join-Path $LogsDir ("network_diag_" + $ts + ".txt")

function Write-Section([string]$title) {
    Add-Content -LiteralPath $outPath -Value ""
    Add-Content -LiteralPath $outPath -Value ("=" * 80)
    Add-Content -LiteralPath $outPath -Value $title
    Add-Content -LiteralPath $outPath -Value ("=" * 80)
}

function Write-CommandOutput([scriptblock]$Block) {
    try {
        & $Block | Out-String | Add-Content -LiteralPath $outPath
    } catch {
        Add-Content -LiteralPath $outPath -Value ("ERROR: " + $_.Exception.Message)
    }
}

Add-Content -LiteralPath $outPath -Value ("ts_utc=" + [DateTime]::UtcNow.ToString("o"))
Add-Content -LiteralPath $outPath -Value ("computer=" + $env:COMPUTERNAME)
Add-Content -LiteralPath $outPath -Value ("user=" + $env:USERNAME)

Write-Section "Get-NetConnectionProfile"
Write-CommandOutput { Get-NetConnectionProfile -ErrorAction Stop | Format-List * }

Write-Section "Get-NetFirewallProfile"
Write-CommandOutput { Get-NetFirewallProfile -ErrorAction Stop | Select-Object Name,Enabled,DefaultInboundAction,DefaultOutboundAction | Format-Table -AutoSize }

Write-Section "netsh advfirewall show allprofiles"
Write-CommandOutput { cmd /c "netsh advfirewall show allprofiles" }

Write-Section "Test-NetConnection fstream.binance.com -Port 443"
Write-CommandOutput { Test-NetConnection fstream.binance.com -Port 443 -ErrorAction Stop | Format-List * }

Write-Section "nslookup fstream.binance.com"
Write-CommandOutput { cmd /c "nslookup fstream.binance.com" }

Write-Section "Get-MpComputerStatus (if available)"
if (Get-Command Get-MpComputerStatus -ErrorAction SilentlyContinue) {
    Write-CommandOutput { Get-MpComputerStatus -ErrorAction Stop | Format-List * }
} else {
    Add-Content -LiteralPath $outPath -Value "Get-MpComputerStatus not available"
}

Write-Output "WROTE network diagnostics: $outPath"
