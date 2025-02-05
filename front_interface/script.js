// Handle file selection
document.getElementById("select-images-btn").addEventListener("click", function() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.multiple = true;

    input.addEventListener("change", function(event) {
        handleFileSelection(event.target.files);
    });

    input.click();
});

// Handle drag and drop
const dropZone = document.querySelector(".drop-zone");
dropZone.addEventListener("dragover", function(event) {
    event.preventDefault();
    dropZone.style.backgroundColor = "#e1eaff";
});
dropZone.addEventListener("dragleave", function() {
    dropZone.style.backgroundColor = "white";
});
dropZone.addEventListener("drop", function(event) {
    event.preventDefault();
    dropZone.style.backgroundColor = "white";
    handleFileSelection(event.dataTransfer.files);
});

// Function to display selected images
function handleFileSelection(files) {
    const previewContainer = document.getElementById("image-preview-container");
    const previewElement = document.getElementById("image-preview");

    previewContainer.style.display = "block"; // Show the preview container
    previewElement.innerHTML = ""; // Clear previous previews

    Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onload = function() {
            const imgWrapper = document.createElement("div");
            imgWrapper.style.position = "relative";
            const img = document.createElement("img");
            img.src = reader.result;

            // Add delete button
            const deleteButton = document.createElement("button");
            deleteButton.classList.add("delete-btn");
            deleteButton.innerText = "Supprimer";
            deleteButton.addEventListener("click", function() {
                imgWrapper.remove(); // Remove the image and the button
                previewContainer.style.display = "none"; // Hide preview container when image is removed
                showContent("background"); // Optionally reset the content of the tab (or any other tab)
            });

            imgWrapper.appendChild(img);
            imgWrapper.appendChild(deleteButton);
            previewElement.appendChild(imgWrapper);
        };
        reader.readAsDataURL(file);
    });
}

// Fonction pour afficher le contenu associé à l'onglet cliqué
function showContent(tab) {
    // Cacher tous les contenus
    const allContents = document.querySelectorAll('.content');
    allContents.forEach(content => {
        content.style.visibility = 'hidden';
        content.style.opacity = '0';
        content.style.display = 'none';  // Cacher tous les contenus au départ
    });

    // Afficher le contenu correspondant
    const content = document.getElementById(tab);
    content.style.visibility = 'visible';
    content.style.opacity = '1';
    content.style.display = 'block';  // Afficher le contenu sélectionné
}
