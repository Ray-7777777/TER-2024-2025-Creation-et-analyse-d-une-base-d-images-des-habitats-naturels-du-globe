document.addEventListener("DOMContentLoaded", function() {
    const uploadForm = document.getElementById("upload-form");
    const imageInput = document.getElementById("image-input");
    const previewContainer = document.getElementById("image-preview-container");
    const previewElement = document.getElementById("image-preview");

    const detectionResultsDiv = document.getElementById("detection-results");
    const resultsList = document.getElementById("results-list");

    uploadForm.addEventListener("submit", function(event) {
        event.preventDefault();

        const files = imageInput.files;
        if (files.length === 0) {
            alert("Veuillez sélectionner au moins une image.");
            return;
        }

        const formData = new FormData();
        // Ajouter chaque fichier sous le même champ "images"
        for (let i = 0; i < files.length; i++) {
            formData.append("images", files[i]);
        }

        // Désactiver le bouton
        const submitButton = document.querySelector(".upload-btn");
        submitButton.disabled = true;
        submitButton.innerText = "Envoi en cours...";

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            submitButton.disabled = false;
            submitButton.innerText = "Envoyer";

            // 'data' est un tableau de résultats
            console.log("Réponse du serveur:", data);

            // Vider l'ancien contenu
            resultsList.innerHTML = "";

            // Si data est un tableau d'objets
            data.forEach(item => {
                // Créer un conteneur pour chaque résultat
                const resultItem = document.createElement("div");
                resultItem.style.margin = "20px";
                resultItem.style.border = "1px solid #ccc";
                resultItem.style.padding = "10px";

                // Nom du fichier
                const filenameEl = document.createElement("h3");
                filenameEl.textContent = `Fichier : ${item.filename}`;
                resultItem.appendChild(filenameEl);

                if (item.success) {
                    // Afficher l'image annotée
                    const img = document.createElement("img");
                    img.src = item.result_image;
                    img.style.maxWidth = "400px";
                    img.style.display = "block";
                    resultItem.appendChild(img);
                } else {
                    // Afficher le message d'erreur
                    const errorEl = document.createElement("p");
                    errorEl.style.color = "red";
                    errorEl.textContent = `Erreur : ${item.error}`;
                    resultItem.appendChild(errorEl);
                    // Ajouter l'image d'origine:
                    const img = document.createElement("img");
                    img.src = item.result_image;
                    // facultatif : style
                    img.style.maxWidth = "400px";
                    resultItem.appendChild(img);
                }

                resultsList.appendChild(resultItem);
            });

            // Rendre le bloc de résultats visible
            detectionResultsDiv.classList.remove("content");
            detectionResultsDiv.classList.add("show-content");
        })
        .catch(error => {
            console.error("Erreur:", error);
            submitButton.disabled = false;
            submitButton.innerText = "Envoyer";
        });
    });

    // Prévisualisation de l’image avant upload
    imageInput.addEventListener("change", function(event) {
        const files = event.target.files;
        if (files.length === 0) {
            previewContainer.style.display = "none";
            return;
        }

        previewContainer.style.display = "block";
        previewElement.innerHTML = "";

        Array.from(files).forEach(file => {
            const reader = new FileReader();
            reader.onload = function() {
                const imgWrapper = document.createElement("div");
                imgWrapper.style.position = "relative";
                const img = document.createElement("img");
                img.src = reader.result;
                img.style.maxWidth = "100%";
                img.style.maxHeight = "200px";

                previewElement.appendChild(imgWrapper);
                imgWrapper.appendChild(img);
            };
            reader.readAsDataURL(file);
        });
    });
});
