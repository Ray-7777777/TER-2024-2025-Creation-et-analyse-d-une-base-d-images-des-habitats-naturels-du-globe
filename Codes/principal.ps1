Remove-Item -Path "..\Donnees\birds_dataset\*" -Recurse -Force
Remove-Item -Path "..\Donnees\oiseaux_extraits\*" -Recurse -Force
Remove-Item -Path "runs\detect\*" -Recurse -Force

try {
    # Exécuter le script Python
    python3 Dependances/Extraction_photo_oiseaux_originalformat.py
} catch {
    Write-Error "Erreur lors de l'exécution de Extraction_photo_oiseaux_originalformat.py: $_"
}

try {
    # Exécuter le script PowerShell
    ./Dependances/detect.ps1
} catch {
    Write-Error "Erreur lors de l'exécution de detect.ps1: $_"
}

try {
    # Exécuter le script Bash
    ./Dependances/extraire_oiseaux.sh
} catch {
    Write-Error "Erreur lors de l'exécution de extraire_oiseaux.sh: $_"
}
