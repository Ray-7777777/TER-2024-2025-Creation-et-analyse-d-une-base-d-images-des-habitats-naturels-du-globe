# ────────────────────────────────────────────
# detect.ps1
# ────────────────────────────────────────────

# 1) On se place dans /Codes/Dependances
Set-Location -Path $PSScriptRoot

# 2) Chemins ABSOLUS Linux du dataset
$datasetPath  = "/Donnees/birds_dataset"
$dataYamlPath = "/Donnees/birds_dataset/data.yaml"

# Debug préliminaire
Write-Host "Dataset path    =" $datasetPath
Write-Host "Dataset exists? =" $(Test-Path $datasetPath)
Write-Host "Detect exists?  =" $(Test-Path 'yolov5/detect.py')
Write-Host "Weights exists? =" $(Test-Path 'yolov5/best.pt')

# 3) (Re)créer data.yaml
"names:" | Set-Content -Path $dataYamlPath -Force

# 4) Lister les classes
$classNames = Get-ChildItem -Path $datasetPath -Directory | ForEach-Object { $_.Name }

# 5) Écrire noms/id dans le YAML
for ($i = 0; $i -lt $classNames.Count; $i++) {
    "$($i): $($classNames[$i])" |
      Out-File -Append -FilePath $dataYamlPath -Encoding UTF8
}

# 6) Boucle de détection YOLOv5
Get-ChildItem -Path $datasetPath -Directory | ForEach-Object {
    $className   = $_.Name
    $src         = $_.FullName           # ex. /Donnees/birds_dataset/Anas_platyrhynchos
    $detectPy    = "yolov5/detect.py"    # relatif à /Codes/Dependances
    $weightsPath = "yolov5/best.pt"

    Write-Host "Processing directory: $src"

    $command = "python '$detectPy' --weights '$weightsPath' --source '$src' --save-txt --save-conf --project runs/detect --name $className --nosave"
    Write-Host "Running: $command"
    Invoke-Expression $command

    # 7) Post-traitement des labels
    $labelDir = "runs/detect/$className/labels"
    if (Test-Path $labelDir) {
        Get-ChildItem -Path $labelDir -Filter *.txt | ForEach-Object {
            (Get-Content $_.FullName) | ForEach-Object {
                if ($_ -match "^14\s") {
                    $_ -replace "^14", [array]::IndexOf($classNames, $className)
                }
            } | Set-Content $_.FullName
        }
    }
}
