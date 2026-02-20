param(
    [switch]$Apply,
    [switch]$Public
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunsLog = Join-Path $RepoRoot "logs\data_layer_runs.jsonl"

function Resolve-PythonExe() {
    if (Test-Path -LiteralPath $RunsLog) {
        $lines = Get-Content -LiteralPath $RunsLog -ErrorAction SilentlyContinue
        for ($i = $lines.Count - 1; $i -ge 0; $i--) {
            $line = $lines[$i]
            if (-not $line) { continue }
            try {
                $obj = $line | ConvertFrom-Json
            } catch {
                continue
            }
            if ($obj.cmd) {
                $cmd = [string]$obj.cmd
                $first = $cmd.Split(" ")[0]
                if (Test-Path -LiteralPath $first) {
                    return $first
                }
            }
        }
    }
    $gc = Get-Command python -ErrorAction Stop
    return [string]$gc.Source
}

$pythonExe = Resolve-PythonExe
$profiles = @("Private")
if ($Public) { $profiles += "Public" }

Write-Output "python_exe=$pythonExe"
Write-Output "profiles=$($profiles -join ',')"

foreach ($profile in $profiles) {
    $ruleName = "EclipseScalper Python Outbound $profile"
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Output "rule_exists name=$ruleName"
        continue
    }
    if (-not $Apply) {
        Write-Output "dry_run add_rule name=$ruleName profile=$profile program=$pythonExe direction=Outbound action=Allow protocol=TCP remote_port=443"
        continue
    }
    New-NetFirewallRule `
        -DisplayName $ruleName `
        -Direction Outbound `
        -Action Allow `
        -Profile $profile `
        -Program $pythonExe `
        -Protocol TCP `
        -RemotePort 443 | Out-Null
    Write-Output "added_rule name=$ruleName profile=$profile"
}

if (-not $Apply) {
    Write-Output "Dry run complete. Re-run with -Apply to create rules."
}
