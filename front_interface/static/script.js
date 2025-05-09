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
        for (let i = 0; i < files.length; i++) {
            formData.append("images", files[i]);
        }

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

            resultsList.innerHTML = "";
            data.forEach(item => {
                const resultItem = document.createElement("div");
                resultItem.style.margin = "20px";
                resultItem.style.border = "1px solid #ccc";
                resultItem.style.padding = "10px";

                const filenameEl = document.createElement("h3");
                filenameEl.textContent = `Fichier : ${item.filename}`;
                resultItem.appendChild(filenameEl);

                if (item.success) {
                    const img = document.createElement("img");
                    img.src = item.result_image;
                    img.style.maxWidth = "400px";
                    img.style.display = "block";
                    resultItem.appendChild(img);

                    const metricsEl = document.createElement("div");
                    metricsEl.innerHTML = `<h4>Métriques</h4>`;
                    item.metrics.forEach(metric => {
                        metricsEl.innerHTML += `<p>Classe : ${metric.label} (Confiance : ${metric.confidence})</p>
                                                <p>Bbox : [${metric.bbox.join(", ")}]</p>`;
                    });
                    resultItem.appendChild(metricsEl);
                } else {
                    const errorEl = document.createElement("p");
                    errorEl.style.color = "red";
                    errorEl.textContent = `Erreur : ${item.error}`;
                    resultItem.appendChild(errorEl);

                    const img = document.createElement("img");
                    img.src = item.result_image;
                    img.style.maxWidth = "400px";
                    resultItem.appendChild(img);
                }

                resultsList.appendChild(resultItem);
            });

            detectionResultsDiv.classList.remove("content");
            detectionResultsDiv.classList.add("show-content");
        })
        .catch(error => {
            console.error("Erreur:", error);
            submitButton.disabled = false;
            submitButton.innerText = "Envoyer";
        });
    });
});
