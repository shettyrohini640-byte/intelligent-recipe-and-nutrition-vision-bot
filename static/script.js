
function previewImage() {
    const input = document.getElementById("foodImage");
    const preview = document.getElementById("preview");

    if (!input.files || !input.files[0]) {
        alert("Please choose a food image first.");
        return;
    }

    const file = input.files[0];

    if (!file.type.startsWith("image/")) {
        alert("Please select a valid image file.");
        return;
    }

    if (preview.dataset.objectUrl) {
        URL.revokeObjectURL(preview.dataset.objectUrl);
    }

    const imageUrl = URL.createObjectURL(file);

    preview.dataset.objectUrl = imageUrl;
    preview.src = imageUrl;
    preview.style.display = "block";
}

// Show the preview automatically after choosing an image.
document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("foodImage");

    if (input) {
        input.addEventListener("change", previewImage);
    }
});

async function askQuestion() {
    const questionInput = document.getElementById("question");
    const answer = document.getElementById("answer");

    const question = questionInput.value.trim();

    if (!question) {
        answer.textContent = "Please type a recipe question.";
        questionInput.focus();
        return;
    }

    answer.textContent = "🌿 Finding your answer...";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        if (!response.ok) {
            answer.textContent =
                data.answer || "Sorry, I couldn't find an answer.";
            return;
        }

        answer.textContent =
            data.answer || "No answer was returned. Please try again.";

    } catch (error) {
        console.error("Recipe assistant error:", error);

        answer.textContent =
            "Unable to connect to the recipe assistant. Please check that your Flask website is running and try again.";
    }
}
