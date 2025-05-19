#!/usr/bin/env python3
import os
import requests
import geopandas as gpd
import climatsEtHabitats
from climatsEtHabitats import parse_legend

# ────────────────────────────────────────────────────────────────
# Configuration globale
# ────────────────────────────────────────────────────────────────

# Dossier parent pour toutes les espèces d'oiseaux
dataset_dir = '/Donnees/birds_dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Shapefiles vectoriels
shapefile_climats    = gpd.read_file("/Donnees/climates/climates.shp")
shapefile_climats = shapefile_climats.to_crs(epsg=4326)
shapefile_ecoregions = gpd.read_file("/Donnees/Ecoregions/wwf_terr_ecos.shp")

# Raster multi-couches de biomes + légende
RASTER_PATH  = "/Donnees/Raster_habitats/Biome_Inventory_RasterStack.tif"
LEGEND_PATH  = "/Donnees/Raster_habitats/Biome_Inventory_Legends.txt"
# 4 bandes d’intérêt et leurs noms
BANDS       = [26, 9, 18, 23]
BAND_NAMES  = ["Leemans", "Higgins", "Friedl", "Olson"]

# Charger la légende une seule fois
legend_map = parse_legend(LEGEND_PATH, BANDS)

# Dossier contenant les rasters d'écosystèmes (unicouche : 1=Major,2=Minor)
raster_ecosystemes = "/Donnees/Ecosystemes/raster"

# Fichier AVONET
avonet     = "/Donnees/avonet/AVONET2_eBird.xlsx"
sheet_name = "AVONET2_eBird"

# Paramètres de téléchargement
num_species             = 1
num_images_per_species  = 1

# Dictionnaire pour la carte globale
all_coords = {}

# ────────────────────────────────────────────────────────────────
# Fonctions existantes (copiées de ton code)
# ────────────────────────────────────────────────────────────────

def get_bird_species():
    species_list, page = [], 1
    while len(species_list) < num_species:
        url = "https://api.inaturalist.org/v1/taxa"
        params = {'taxon_id':3,'rank':'species','per_page':100,'page':page}
        r = requests.get(url, params=params)
        if r.status_code!=200:
            print(f"[WARN] iNaturalist taxa HTTP {r.status_code}")
            break
        data = r.json().get('results',[])
        species_list.extend(data)
        if len(data)<100:
            break
        page+=1
    return species_list[:num_species]

def download_images_for_species(taxon_id, species_name):
    observations, page = [], 1
    while len(observations) < num_images_per_species:
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            'taxon_id': taxon_id,
            'per_page': 100,
            'page': page,
            'order_by': 'created_at'
        }
        r = requests.get(url, params=params)
        if r.status_code!=200:
            print(f"[WARN] iNaturalist obs HTTP {r.status_code} pour {species_name}")
            break
        data = r.json().get('results',[])
        observations.extend(data)
        if len(data)<100:
            break
        page+=1

    observations = observations[:num_images_per_species]
    photo_urls, coordinates = [], []

    for obs in observations:
        if isinstance(obs, dict):
            geojson = obs.get('geojson',{})
            coords = geojson.get('coordinates')
            if coords:
                lon, lat = coords
                coordinates.append((lat, lon))
            else:
                print(f"[WARN] Pas de coord pour obs {obs.get('id')}")
        for photo in obs.get('photos', []):
            url0 = photo.get('url','')
            base = url0.rsplit('/',1)[0]
            if url0.lower().endswith(('.jpg','.jpeg')):
                photo_urls.append(f"{base}/original.jpg")

    species_dir = os.path.join(dataset_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)
    for i, url in enumerate(photo_urls[:num_images_per_species], start=1):
        ext = url.split('.')[-1]
        try:
            resp = requests.get(url); resp.raise_for_status()
            with open(f"{species_dir}/{species_name}_{i}.{ext}", "wb") as f:
                f.write(resp.content)
        except Exception as e:
            print(f"[WARN] Erreur download {url}: {e}")

    if coordinates:
        all_coords[species_name] = coordinates
        coords_txt = os.path.join(species_dir, "coordinates.txt")
        with open(coords_txt, "w", encoding="utf-8") as f:
            for lat, lon in coordinates:
                f.write(f"Coordinates: {lat},{lon}\n")
        print(f"[INFO] Coordonnées pour {species_name} → {coords_txt}")
    else:
        print(f"[INFO] Aucune coordonnée pour {species_name}")

    # ────────────────────────────────────────────────────────────────
    # Appels aux fonctions de climatsEtHabitats
    # ────────────────────────────────────────────────────────────────
    if coordinates:
        climatsEtHabitats.climats(
            coordinates, shapefile_climats,
            species_name, dataset_dir
        )
        climatsEtHabitats.ecoregions(
            coordinates, shapefile_ecoregions,
            species_name, dataset_dir
        )
        climatsEtHabitats.ecosystemes(
            coordinates, raster_ecosystemes,
            species_name, dataset_dir
        )
        climatsEtHabitats.raster_classifications(
            coordinates,
            RASTER_PATH, BANDS, BAND_NAMES,
            species_name, dataset_dir,
            legend_map=legend_map,
            src_epsg="EPSG:4326"
        )
        climatsEtHabitats.carte(dataset_dir)

    climatsEtHabitats.avonet_habitats(
        avonet, sheet_name,
        species_name, dataset_dir
    )

# ────────────────────────────────────────────────────────────────
# Exécution principale
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bird_species = get_bird_species()
    for sp in bird_species:
        name = sp['name'].replace(" ", "_")
        print(f"[INFO] Traitement espèce {name} (ID {sp['id']})")
        download_images_for_species(sp['id'], name)

    # ────────────────────────────────────────────────────────────────
    # Enfin : carte globale de toutes les espèces
    # ────────────────────────────────────────────────────────────────
    global_map = os.path.join(dataset_dir, "carte_toutes_especes.html")
    climatsEtHabitats.carte_globale(all_coords, global_map)
    print(f"[SUCCESS] Carte globale → {global_map}")
