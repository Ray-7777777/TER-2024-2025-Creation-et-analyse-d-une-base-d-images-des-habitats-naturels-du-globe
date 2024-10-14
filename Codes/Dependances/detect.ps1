# Dossier contenant les ensembles de données
$datasetPath = "..\Donnees\birds_dataset"
$dataYamlPath = "..\Donnees\birds_dataset\data.yaml"

# Créer ou vider le fichier data.yaml
"names:" | Set-Content $dataYamlPath

# Créer une liste pour stocker les noms
$classNames = @()

# Parcourir les dossiers pour remplir la liste des noms avant l'exécution de la détection
Get-ChildItem -Path $datasetPath -Directory | ForEach-Object {
    $className = $_.Name
    $classNames += $className
}

# Écrire les noms et leurs identifiants dans le fichier YAML
for ($i = 0; $i -lt $classNames.Count; $i++) {
    # Utiliser des accolades pour délimiter la variable i
    "$($i): $($classNames[$i])" | Out-File -Append -FilePath $dataYamlPath -Encoding UTF8
}

# Exécuter la détection en utilisant chaque dossier
Get-ChildItem -Path $datasetPath -Directory | ForEach-Object {
    $className = $_.Name
    Write-Host "Processing directory: $($_.FullName)"

    # Exécuter la détection
    $command = "python Dependances/yolov5/detect.py --weights yolov5s.pt --source `"$($_.FullName)`" --save-txt --save-conf --project runs/detect --name $className --nosave"

    Write-Host "Running command: $command"
    Invoke-Expression $command
	
    # Parcourir chaque fichier de labels généré et remplacer 'bird' par le nom du dossier
    $labelFiles = Get-ChildItem -Path "runs/detect/$className/labels" -Filter *.txt
    foreach ($labelFile in $labelFiles) {
        (Get-Content $labelFile.PSPath) | ForEach-Object {
            # Remplacer la classe 14 (bird) par le nom du dossier (className)
            if ($_ -match "^14\s") {
                $_ = $_ -replace "^14", [array]::IndexOf($classNames, $className)
            } else {
        	# La ligne ne correspond pas au motif, on la "supprime"
        	$_ = $null
		}
            $_
        } | Set-Content $labelFile.PSPath
    }
}
