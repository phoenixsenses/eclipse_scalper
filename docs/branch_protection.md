# Branch Protection Setup

Use this to require the new execution reliability checks before merges.

## Prerequisites

- Repo admin access on `phoenixsenses/eclipse_scalper`
- Personal access token with repo admin rights (classic: `repo`, or fine-grained with Administration write)

## Required check contexts

Set these as required status checks on `main`:

- `Chaos Required - ack-after-fill-recovery`
- `Chaos Required - cancel-unknown-idempotent`
- `Chaos Required - replace-race-single-exposure`
- `Execution Invariants and Gate`

## One-command helper

```powershell
.\tools\apply_branch_protection.ps1
```

Verify only (no changes):

```powershell
.\tools\apply_branch_protection.ps1 -VerifyOnly
```

## PowerShell command (`gh api`)

```powershell
$owner = "phoenixsenses"
$repo = "eclipse_scalper"
$branch = "main"

# Requires: gh auth login (admin on repo)
 $json = @'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      { "context": "Chaos Required - ack-after-fill-recovery" },
      { "context": "Chaos Required - cancel-unknown-idempotent" },
      { "context": "Chaos Required - replace-race-single-exposure" },
      { "context": "Execution Invariants and Gate" }
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
'@
$tmp = New-TemporaryFile
Set-Content -Path $tmp -Value $json -Encoding UTF8
& "C:\Program Files\GitHub CLI\gh.exe" api `
  --method PUT `
  -H "Accept: application/vnd.github+json" `
  "/repos/$owner/$repo/branches/$branch/protection" `
  --input $tmp
Remove-Item $tmp -Force
```

## PowerShell command (GitHub REST API fallback)

```powershell
$env:GITHUB_TOKEN="YOUR_TOKEN_WITH_REPO_ADMIN"
$owner="phoenixsenses"
$repo="eclipse_scalper"
$branch="main"

$body = @{
  required_status_checks = @{
    strict = $true
    checks = @(
      @{ context = "Chaos Required - ack-after-fill-recovery"; app_id = -1 }
      @{ context = "Chaos Required - cancel-unknown-idempotent"; app_id = -1 }
      @{ context = "Chaos Required - replace-race-single-exposure"; app_id = -1 }
      @{ context = "Execution Invariants and Gate"; app_id = -1 }
    )
  }
  enforce_admins = $true
  required_pull_request_reviews = @{
    required_approving_review_count = 1
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
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Method Put `
  -Uri "https://api.github.com/repos/$owner/$repo/branches/$branch/protection" `
  -Headers @{
    Authorization = "Bearer $env:GITHUB_TOKEN"
    Accept = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
  } `
  -ContentType "application/json" `
  -Body $body
```

## Verify (`gh api`)

```powershell
& "C:\Program Files\GitHub CLI\gh.exe" api `
  "/repos/phoenixsenses/eclipse_scalper/branches/main/protection" `
  --jq ".required_status_checks.checks[].context"
```

## Verify (REST API)

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "https://api.github.com/repos/phoenixsenses/eclipse_scalper/branches/main/protection" `
  -Headers @{
    Authorization = "Bearer $env:GITHUB_TOKEN"
    Accept = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
  }
```

The response should include all four required contexts listed above.
