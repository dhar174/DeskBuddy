[CmdletBinding()]
param(
    [switch] $Headless
)

$ErrorActionPreference = "Stop"

# LEGACY launcher for the existing Picoh desktop server. This script only makes
# startup portable; it does not configure or validate robot transport security.
Write-Host "Legacy Picoh startup: Issue #4 owns robot transport security."

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install uv and run this script again."
}

Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Headless) {
        # Lightweight configuration/bootstrap check; does not start Picoh.
        & uv run --locked python start_bot.py --headless
    }
    else {
        # Picoh and its speech stack are optional project dependencies.
        & uv run --locked --extra speech --extra picoh python buddy1.py
    }
    $nativeExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($nativeExitCode -ne 0) {
    exit $nativeExitCode
}
