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
    @{ context = "Chaos Required - ack-after-fill-recovery" },
    @{ context = "Chaos Required - cancel-unknown-idempotent" },
    @{ context = "Chaos Required - replace-race-single-exposure" },
    @{ context = "Execution Invariants and Gate" }
)

if (-not $VerifyOnly) {
    $payload = @{
        required_status_checks = @{
            strict = $true
            checks = $checks
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
        Set-Content -Path $tmp -Value ($payload | ConvertTo-Json -Depth 10) -Encoding UTF8
        Write-Host "Applying branch protection to $Owner/$Repo:$Branch ..."
        Invoke-GhJson -Exe $GhExe -Method "PUT" -Endpoint "/repos/$Owner/$Repo/branches/$Branch/protection" -InputPath $tmp
    }
    finally {
        if (Test-Path -LiteralPath $tmp) {
            Remove-Item -LiteralPath $tmp -Force
        }
    }
}

Write-Host "Verifying required check contexts..."
$contexts = & $GhExe api "/repos/$Owner/$Repo/branches/$Branch/protection" --jq ".required_status_checks.checks[].context"
$contextsText = ($contexts | Out-String)

foreach ($check in $checks) {
    $ctx = [string]$check.context
    if ($contextsText -notmatch [Regex]::Escape($ctx)) {
        throw "Missing required check context: $ctx"
    }
}

Write-Host "Branch protection is configured correctly for $Owner/$Repo:$Branch"
