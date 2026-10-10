param(
    [string]$Image = "shenji-agent-test:latest",
    [int]$Port = 17862
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$name = "shenji-demo-image-test"

if (docker ps -a --format "{{.Names}}" | Select-String -SimpleMatch $name) {
    docker rm -f $name | Out-Null
}

try {
    $required = @(
        "/app/web/main.py",
        "/app/web/templates/index.html",
        "/app/data/inference_data.json",
        "/app/input/replay.zip",
        "/app/input/test_001/scenario.json",
        "/app/weights/intention_lstm.pth",
        "/app/weights/situation_gcn.pth"
    )

    $check = ($required | ForEach-Object { "test -f '$_'" }) -join " && "
    docker run --rm --entrypoint /bin/bash $Image -lc $check
    if ($LASTEXITCODE -ne 0) { throw "required files missing from image" }

    docker run -d --name $name -p "${Port}:7862" $Image | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "container failed to start" }

    $response = $null
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $response = Invoke-RestMethod "http://127.0.0.1:${Port}/api/data" -TimeoutSec 2
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if ($null -eq $response) {
        docker logs $name
        throw "API did not become ready"
    }
    if ([int]$response.frame_count -le 0) { throw "invalid frame_count" }
    if (-not $response.map_info) { throw "map_info missing" }

    Write-Host "PASS image=$Image frames=$($response.frame_count)"
} finally {
    if (docker ps -a --format "{{.Names}}" | Select-String -SimpleMatch $name) {
        docker rm -f $name | Out-Null
    }
}
