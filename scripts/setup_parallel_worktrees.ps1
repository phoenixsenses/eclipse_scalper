param(
    [string]$BaseDir = (Split-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path -Parent),
    [switch]$CreateShared
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$laneSpecs = @(
    @{
        Branch = "codex/research-mainline"
        Path = Join-Path $BaseDir "eclipse_scalper-research"
    },
    @{
        Branch = "codex/runtime-mainline"
        Path = Join-Path $BaseDir "eclipse_scalper-runtime"
    }
)

if ($CreateShared) {
    $laneSpecs += @{
        Branch = "codex/shared-mainline"
        Path = Join-Path $BaseDir "eclipse_scalper-shared"
    }
}

Push-Location $repoRoot
try {
    foreach ($lane in $laneSpecs) {
        $branch = $lane.Branch
        $path = $lane.Path

        $branchExists = (git branch --list $branch)
        if (-not $branchExists) {
            git branch $branch | Out-Null
        }

        $worktreeExists = (git worktree list --porcelain | Select-String -SimpleMatch $path)
        if (-not $worktreeExists) {
            git worktree add $path $branch
        }
    }

    Write-Host "Parallel worktrees are ready:"
    foreach ($lane in $laneSpecs) {
        Write-Host ("- {0} -> {1}" -f $lane.Branch, $lane.Path)
    }
}
finally {
    Pop-Location
}
