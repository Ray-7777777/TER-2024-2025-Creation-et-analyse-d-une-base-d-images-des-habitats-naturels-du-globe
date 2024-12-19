# Variables principales
$datasetPath = "..\Donnees\birds_dataset"
$dataYamlPath = "..\Donnees\birds_dataset\data.yaml"
$modelGoogleDriveLink = "https://drive.google.com/file/d/11C3rlEcJdcO27XVfHsAISI-UMYBS8n5A/view?usp=drive_link"  # Remplacez FILE_ID par l'identifiant du fichier
$yoloRepo = "https://github.com/ultralytics/yolov5.git"
$yoloPath = "Dependances/yolov5"
$modelPath = "Dependances/yolov5/yolov5s.pt"  # Chemin par défaut pour YOLOv5 modèle léger

if (-Not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git n'est pas installé ou accessible. Installez Git et réessayez."
    exit
}

if (-Not (Get-Command gdown -ErrorAction SilentlyContinue)) {
    Write-Error "`gdown` n'est pas installé. Installez-le via pip : `pip install gdown`."
    exit
}


# Télécharger YOLOv5 depuis GitHub si le dossier n'existe pas
if (-Not (Test-Path -Path $yoloPath)) {
    Write-Host "Téléchargement de YOLOv5 depuis GitHub..."
    git clone $yoloRepo $yoloPath
    if (-Not (Test-Path -Path $yoloPath)) {
        Write-Error "Échec du téléchargement de YOLOv5. Vérifiez votre connexion internet et réessayez."
        exit
    }
    Write-Host "YOLOv5 téléchargé avec succès."
}

# Télécharger le modèle YOLO depuis Google Drive si non présent
if (-Not (Test-Path -Path $modelPath)) {
    Write-Host "Téléchargement du modèle YOLO depuis Google Drive..."
    $gdownPath = "gdown"  # Assurez-vous que `gdown` est installé sur votre système
    $command = "$gdownPath $modelGoogleDriveLink -O $modelPath"
    Invoke-Expression $command
    if (-Not (Test-Path -Path $modelPath)) {
        Write-Error "Échec du téléchargement du modèle YOLO. Vérifiez le lien et réessayez."
        exit
    }
    Write-Host "Modèle YOLO téléchargé avec succès."
}

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
    "$($i): $($classNames[$i])" | Out-File -Append -FilePath $dataYamlPath -Encoding UTF8
}

# Exécuter la détection en utilisant chaque dossier
Get-ChildItem -Path $datasetPath -Directory | ForEach-Object {
    $className = $_.Name
    Write-Host "Processing directory: $($_.FullName)"

    # Exécuter la détection avec le modèle téléchargé
    $command = "python $yoloPath/detect.py --weights $modelPath --source `"$($_.FullName)`" --save-txt --save-conf --project runs/detect --name $className --nosave"

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
                $_ = $null  # Supprimer la ligne si elle ne correspond pas
            }
            $_
        } | Set-Content $labelFile.PSPath
    }
}
