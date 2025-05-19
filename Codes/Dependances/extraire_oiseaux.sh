#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "🔧 extraire_oiseaux.sh démarré dans $(pwd)"

output_dir="/Donnees/oiseaux_extraits"
mkdir -p "$output_dir"

# active le "nullglob" pour que *.txt vide ne renvoie pas le motif brut
shopt -s nullglob

for run_dir in runs/detect/*/; do
  # ex. run_dir="runs/detect/Anas_platyrhynchos5/"
  runName=$(basename "$run_dir")
  # enlève tout suffixe numérique
  className="${runName%%[0-9]*}"
  label_files=( "$run_dir/labels/"*.txt )
  if (( ${#label_files[@]} == 0 )); then
    echo "⚠️  Aucun label pour la classe '$runName', on passe."
    continue
  fi

  for label_file in "${label_files[@]}"; do
    base=$(basename "$label_file" .txt)
    # cherche l'image avec différentes extensions/casse
    img=""
    for ext in jpg jpeg JPG JPEG png PNG; do
      candidate="/Donnees/birds_dataset/$className/$base.$ext"
      if [[ -f "$candidate" ]]; then
        img="$candidate"
        break
      fi
    done

    if [[ -z "$img" ]]; then
      echo "⚠️  Image introuvable pour '$className/$base' (.jpg/.jpeg/.png)"
      continue
    fi

    echo "✅  Image trouvée : $img"
    # lit chaque ligne de label
    while read -r class_id x_center y_center width height _conf; do
      img_w=$(magick identify -format "%w" "$img")
      img_h=$(magick identify -format "%h" "$img")
      x_min=$(awk "BEGIN{printf \"%d\",($x_center*$img_w-($width*$img_w/2))}")
      y_min=$(awk "BEGIN{printf \"%d\",($y_center*$img_h-($height*$img_h/2))}")
      w_box=$(awk "BEGIN{printf \"%d\",($width*$img_w)}")
      h_box=$(awk "BEGIN{printf \"%d\",($height*$img_h)}")

      echo "Traitement de l'image : $img"

      if (( x_min>=0 && y_min>=0 && x_min+w_box<=img_w && y_min+h_box<=img_h )); then
        out="$output_dir/${base}_$RANDOM.jpg"
        magick "$img" -crop "${w_box}x${h_box}+${x_min}+${y_min}" "$out" \
          && echo "  ➡️ recadrage OK : $out"
      else
        echo "  ⚠️ Valeurs invalides pour cropping : x_min=$x_min y_min=$y_min w=$w_box h=$h_box"
      fi
    done < "$label_file"
  done
done

echo "🏁 Extraction terminée : $(ls -1 "$output_dir" | wc -l) fichiers extraits."```