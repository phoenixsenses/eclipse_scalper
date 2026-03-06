[CmdletBinding()]
param(
    [string]$Owner = "phoenixsenses",
    [string]$Repo = "eclipse_scalper",
    [string]$Branch = "main",
    [string]$GhExe = "C:\Program Files\GitHub CLI\gh.exe",
    [int]$RequiredApprovals = 1,
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Command {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "gh executable not found at: $Path"
    }
}

function Invoke-GhJson {
    param(
        [string]$Exe,
        [string]$Method,
        [string]$Endpoint,
        [string]$InputPath
    )
    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        & $Exe api --method $Method -H "Accept: application/vnd.github+json" $Endpoint
        return
    }
    & $Exe api --method $Method -H "Accept: application/vnd.github+json" $Endpoint --input $InputPath
}

Assert-Command -Path $GhExe

Write-Host "Checking gh authentication..."
& $GhExe auth status | Out-Null

$checks = @(
    "Chaos Required - ack-after-fill-recovery",
    "Chaos Required - cancel-unknown-idempotent",
    "Chaos Required - replace-race-single-exposure",
    "Execution Invariants and Gate"
)

if (-not $VerifyOnly) {
    $payload = @{
        required_status_checks = @{
            strict = $true
            contexts = $checks
        }
        enforce_admins = $true
        required_pull_request_reviews = @{
            required_approving_review_count = [Math]::Max(1, [int]$RequiredApprovals)
            dismiss_stale_reviews = $true
            require_code_owner_reviews = $false
        }
        restrictions = $null
        required_linear_history = $false
        allow_force_pushes = $false
        allow_deletions = $false
        block_creations = $false
        required_conversation_resolution = $true
        lock_branch = $false
        allow_fork_syncing = $true
    }

    $tmp = New-TemporaryFile
    try {
        Set-Content -Path $tmp -Value ($payload | ConvertTo-Json -Depth 10) -Encoding Ascii
        Write-Host "Applying branch protection to $Owner/${Repo}:$Branch ..."
        Invoke-GhJson -Exe $GhExe -Method "PUT" -Endpoint "/repos/$Owner/$Repo/branches/$Branch/protection" -InputPath $tmp
    }
    finally {
        if (Test-Path -LiteralPath $tmp) {
            Remove-Item -LiteralPath $tmp -Force
        }
    }
}

Write-Host "Verifying required check contexts..."
$rawProtection = & $GhExe api "/repos/$Owner/$Repo/branches/$Branch/protection"
$contextsText = ""
try {
    $payload = $rawProtection | ConvertFrom-Json
    if ($null -ne $payload.required_status_checks.contexts) {
        $contextsText = (($payload.required_status_checks.contexts | ForEach-Object { [string]$_ }) -join "`n")
    }
    elseif ($null -ne $payload.required_status_checks.checks) {
        $contextsText = (($payload.required_status_checks.checks | ForEach-Object { [string]$_.context }) -join "`n")
    }
}
catch {
    $contextsText = [string]($rawProtection | Out-String)
}

foreach ($check in $checks) {
    $ctx = [string]$check
    if ($contextsText -notmatch [Regex]::Escape($ctx)) {
        throw "Missing required check context: $ctx"
    }
}

Write-Host "Branch protection is configured correctly for $Owner/${Repo}:$Branch"
