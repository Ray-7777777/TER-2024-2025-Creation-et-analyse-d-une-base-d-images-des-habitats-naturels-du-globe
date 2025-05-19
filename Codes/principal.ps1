# ────────────────────────────────────────────
# principal.ps1
# ────────────────────────────────────────────

# 1) On se place dans /Codes (où Docker a monté votre repo)
Set-Location -Path $PSScriptRoot

# Debug rapide
Write-Host "Root folder     =" (Get-Location)
Write-Host "Dependances dir =" (Test-Path 'Dependances')

# 2) Nettoyer les anciens résultats
Remove-Item -Path "/Donnees/birds_dataset/*"           -Recurse -Force
Remove-Item -Path "/Donnees/oiseaux_extraits/*"        -Recurse -Force
if (Test-Path 'Dependances/runs/detect') {
    Remove-Item -Path 'Dependances/runs/detect/*' -Recurse -Force
}
Remove-Item -Path "/Donnees/backgrounds_extracted/*"  -Recurse -Force

# 3) Extraction des photos (Python)
try {
    python3 Dependances/Extraction_photo_oiseaux_originalformat.py
} catch {
    Write-Error "Erreur Extraction_photo_oiseaux_originalformat.py: $_"
}

# 3.1) Classification flou/nette et suppression des images floues
try {
    Write-Host "🛠️  Classification flou/nette…"
    python3 Dependances/verifFlou.py "/Donnees/birds_dataset" "Dependances/yolov5/best_flou.pt"
    Write-Host "✅ Images floues supprimées."
} catch {
    Write-Error "Erreur verifFlou.py: $_"
}

# 4) Détection YOLOv5 (PowerShell)
try {
    pwsh ./Dependances/detect.ps1
    Write-Host "✅ detect.ps1 terminé, on passe au Bash"
} catch {
    Write-Error "Erreur detect.ps1: $_"
}

Set-Location -Path $PSScriptRoot

# 5) Extraction des oiseaux (Bash)
Write-Host "🛠️  Lancement de extraire_oiseaux.sh…"
& "./Dependances/extraire_oiseaux.sh"
Write-Host "✅ Script Bash terminé"

# 6) Extraction des backgrounds (Python)
try {
    # Vérifie d'abord que le script existe
    Write-Host "Background script exists? =" (Test-Path './extraction_bckgd.py')
    
    # Puis lance-le
    python3 ./extraction_bckgd.py
} catch {
    Write-Error "Erreur extraction_bckgd.py: $_"
}

# 7) Comparaison features (Python)
try {
    Write-Host "🛠️  Lancement du pipeline comparaison_features…"
    python3 ./Features/comparaison_features.py
    Write-Host "✅ comparaison_features.py terminé"
} catch {
    Write-Error "Erreur comparaison_features.py: $_"
}