#!/bin/bash

# Dossier de sortie pour les images recadrées
output_dir="../Donnees/oiseaux_extraits"

# Créer le dossier de sortie s'il n'existe pas déjà
mkdir -p "$output_dir"

# Lire chaque dossier dans le dossier 'detect'
for dir in runs/detect/*; do
    label_dir="$dir/labels"

    # Lire chaque fichier de labels dans le dossier correspondant
    for label_file in "$label_dir"/*.txt; do
        # Tenter de trouver l'image avec .jpg puis .jpeg
        image_file="../Donnees/birds_dataset/${dir##*/}/$(basename "$label_file" .txt).jpg"
        if [[ ! -f "$image_file" ]]; then
            image_file="../Donnees/birds_dataset/${dir##*/}/$(basename "$label_file" .txt).jpeg"
        fi

        # Lire chaque ligne du fichier de labels
        while read -r line; do
            # Extraire les informations de la bounding box
            class_id=$(echo "$line" | awk '{print $1}')
            x_center=$(echo "$line" | awk '{print $2}')
            y_center=$(echo "$line" | awk '{print $3}')
            width=$(echo "$line" | awk '{print $4}')
            height=$(echo "$line" | awk '{print $5}')

            # Obtenir les dimensions de l'image originale
            img_width=$(magick identify -format "%w" "$image_file")
            img_height=$(magick identify -format "%h" "$image_file")

            # Vérification de la réussite de la commande 'magick'
            if [[ -z "$img_width" || -z "$img_height" ]]; then
                echo "Erreur lors de la récupération des dimensions de l'image : $image_file"
                continue
            fi

            # Calculs en utilisant awk pour maintenir la précision
            x_min=$(awk "BEGIN {printf \"%d\", ($x_center * $img_width - ($width * $img_width / 2));}")
            y_min=$(awk "BEGIN {printf \"%d\", ($y_center * $img_height - ($height * $img_height / 2));}")
            box_width=$(awk "BEGIN {printf \"%d\", ($width * $img_width);}")
            box_height=$(awk "BEGIN {printf \"%d\", ($height * $img_height);}")

            echo "Traitement de l'image : $image_file"

            # Vérifier si les valeurs pour le recadrage sont valides
            if (( x_min >= 0 && y_min >= 0 && (x_min + box_width) <= img_width && (y_min + box_height) <= img_height )); then
                output_file="$output_dir/$(basename "$label_file" .txt)_$RANDOM.jpg"
                if magick "$image_file" -crop "${box_width}x${box_height}+${x_min}+${y_min}" "$output_file"; then
                    echo "Image recadrée sauvegardée : $output_file"
                else
                    echo "Erreur lors du recadrage de l'image : $image_file"
                fi
            else
                echo "Valeurs invalides pour cropping : x_min=$x_min, y_min=$y_min, box_width=$box_width, box_height=$box_height"
            fi
        done < "$label_file"
    done
done

echo "Extraction des cadres terminée."




