param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("research", "runtime", "shared")]
    [string]$Lane,

    [Parameter(Mandatory = $true)]
    [string]$Topic
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$normalizedTopic = $Topic.Trim().ToLowerInvariant()
$normalizedTopic = $normalizedTopic -replace "[^a-z0-9]+", "-"
$normalizedTopic = $normalizedTopic.Trim("-")

if (-not $normalizedTopic) {
    throw "Topic must contain at least one alphanumeric character."
}

$baseBranch = "codex/$Lane-mainline"
$newBranch = "codex/$Lane/$normalizedTopic"

Push-Location $repoRoot
try {
    $baseExists = git branch --list $baseBranch
    if (-not $baseExists) {
        throw "Base branch '$baseBranch' does not exist. Create the lane branches first."
    }

    $existing = git branch --list $newBranch
    if ($existing) {
        git checkout $newBranch
    } else {
        git checkout $baseBranch
        git checkout -b $newBranch
    }

    Write-Host ("Active branch: {0}" -f $newBranch)
}
finally {
    Pop-Location
}
