
import os
import json
import uuid
from flask import (
    Flask, request, render_template_string,
    url_for, redirect, flash
)
from werkzeug.utils import secure_filename
from food_classifier import predict_food, class_names
from chatbot import answer_question

app = Flask(__name__)
app.secret_key = "recipe-vision-bot-local-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RECIPES_FILE = os.path.join(BASE_DIR, "data", "recipes.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

FOOD_NAMES = {
    "aloo_gobi": "Aloo Gobi",
    "idli": "Idli",
    "kadai_paneer": "Kadai Paneer",
    "masala_dosa": "Masala Dosa",
    "tomato_curry": "Tomato Curry",
    "vegetable_pulao": "Vegetable Pulao",
    "vegetable_upma": "Vegetable Upma",
}


def display_food_name(name):
    """Convert a model label into a readable food name."""
    normalized = str(name).strip().lower().replace(" ", "_")
    return FOOD_NAMES.get(normalized, str(name).replace("_", " ").title())


def load_recipes():
    """Load recipe data safely."""
    if not os.path.isfile(RECIPES_FILE):
        print("Recipe file not found:", RECIPES_FILE, flush=True)
        return []

    try:
        with open(RECIPES_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            recipes = data.get("recipes", data)
            if isinstance(recipes, list):
                return recipes
            if isinstance(recipes, dict):
                result = []
                for key, value in recipes.items():
                    if isinstance(value, dict):
                        item = value.copy()
                        item.setdefault("name", key)
                        result.append(item)
                return result

        return []

    except (json.JSONDecodeError, OSError) as error:
        print("Could not load recipes:", error, flush=True)
        return []


def find_recipe_by_name(food_name, recipes):
    """Find the closest recipe name match."""
    wanted = str(food_name).strip().lower().replace("_", " ")

    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue

        name = str(
            recipe.get("name")
            or recipe.get("title")
            or recipe.get("food_name")
            or ""
        ).strip().lower().replace("_", " ")

        if name == wanted:
            return recipe

    # Try a partial match only if an exact match is unavailable.
    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue

        name = str(
            recipe.get("name")
            or recipe.get("title")
            or recipe.get("food_name")
            or ""
        ).strip().lower().replace("_", " ")

        if wanted in name or name in wanted:
            return recipe

    return None


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intelligent Recipe & Nutrition Vision Bot</title>
<style>
:root {
    --purple: #7661b5;
    --deep: #4e3d83;
    --lavender: #f0ebff;
    --ink: #302b40;
    --muted: #716b80;
    --white: #ffffff;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: "Segoe UI", Arial, sans-serif;
    color: var(--ink);
    background: linear-gradient(135deg, #faf8ff, #eee8ff);
    min-height: 100vh;
}
header {
    text-align: center;
    padding: 38px 18px 28px;
    background: linear-gradient(120deg, #6d58a8, #a292dc);
    color: white;
}
header h1 { margin: 0 0 10px; font-size: clamp(25px, 4vw, 38px); }
header p { margin: 0; opacity: .95; }
main { width: min(1000px, 92%); margin: 28px auto 50px; }
.card {
    background: white;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(68, 47, 110, .09);
}
h2 { color: var(--deep); margin-top: 0; }
label { display: block; font-weight: 600; margin: 12px 0 8px; }
input[type=file], select {
    width: 100%;
    padding: 13px;
    border: 1px solid #d9d0f1;
    border-radius: 10px;
    background: #fcfbff;
    color: var(--ink);
}
button {
    border: 0;
    background: var(--purple);
    color: white;
    padding: 13px 20px;
    border-radius: 10px;
    font-weight: 700;
    cursor: pointer;
    margin-top: 16px;
}
button:hover { background: var(--deep); }
button.secondary { background: #8c80aa; }
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
}
.food-image {
    width: 100%;
    max-height: 350px;
    object-fit: contain;
    border-radius: 12px;
    background: #f7f4ff;
}
.result-name { font-size: 27px; color: var(--deep); font-weight: 750; }
.muted { color: var(--muted); }
.confidence {
    display: inline-block;
    background: var(--lavender);
    color: var(--deep);
    border-radius: 20px;
    padding: 7px 12px;
    font-size: 14px;
    margin: 8px 0 15px;
}
.warning {
    background: #fff4db;
    color: #775318;
    border-radius: 10px;
    padding: 12px;
    margin: 12px 0;
}
.prediction-row {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    padding: 10px 0;
    border-bottom: 1px solid #eeeaf7;
}
ul, ol { padding-left: 23px; line-height: 1.7; }
.nutrition {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}
.nutrition div {
    background: #f5f1ff;
    padding: 13px;
    border-radius: 10px;
}
.nutrition strong { display: block; color: var(--deep); }
.alert {
    padding: 12px;
    border-radius: 10px;
    background: #fcecec;
    color: #8d2929;
    margin-bottom: 15px;
}
footer { text-align: center; color: var(--muted); padding: 0 20px 30px; }
</style>
</head>
<body>
<header>
    <h1>Intelligent Recipe & Nutrition Vision Bot</h1>
    <p>Discover food, explore recipes and learn about nutrition.</p>
</header>

<main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <div class="alert">{{ message }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <section class="card">
        <h2>Identify your food</h2>
        <p class="muted">Upload a clear food photo to get a prediction and recipe suggestions.</p>
        <form action="{{ url_for('analyze') }}" method="post"
              enctype="multipart/form-data">
            <label for="food_image">Choose a food image</label>
            <input id="food_image" type="file" name="food_image"
                   accept=".png,.jpg,.jpeg,.webp" required>
            <button type="submit">Analyze Image</button>
        </form>
    </section>
        <section class="card" id="chatbot">
        <h2>Recipe & Nutrition Chatbot</h2>
        <p class="muted">
            Ask questions about ingredients, cooking instructions,
            and nutrition information.
        </p>

        <form action="{{ url_for('chat') }}" method="post">
            <label for="question">Your question</label>
            <input
                id="question"
                type="text"
                name="question"
                placeholder="How do I make Vegetable Pulao?"
                required
                style="width:100%;padding:13px;border:1px solid #d9d0f1;border-radius:10px;"
            >
            <button type="submit">Ask Chatbot</button>
        </form>

        {% if chat_question %}
            <div style="margin-top:20px;padding:16px;background:#f5f1ff;border-radius:12px;">
                <p><strong>You asked:</strong> {{ chat_question }}</p>
                <p><strong>Chatbot:</strong> {{ chat_answer }}</p>
            </div>
        {% endif %}
    </section>

    {% if result %}
    <section class="card">
        <h2>Analysis result</h2>
        <div class="grid">
            <div>
                <img class="food-image"
                     src="{{ url_for('static', filename=result.image) }}"
                     alt="Uploaded food image">
            </div>
            <div>
                <p class="muted">Model's predicted food</p>
                <div class="result-name">{{ result.food }}</div>
                <div class="confidence">
                    Model confidence: {{ result.confidence }}%
                </div>

                {% if result.uncertain %}
                <div class="warning">
                    The model is uncertain. Please check the prediction before
                    using the recipe or nutrition information.
                </div>
                {% endif %}

                <h3>Other predictions</h3>
                {% for item in result.alternatives %}
                <div class="prediction-row">
                    <span>{{ item.label }}</span>
                    <strong>{{ item.confidence }}%</strong>
                </div>
                {% endfor %}
            </div>
        </div>
    </section>

    {% if result.recipe %}
    <section class="card">
        <h2>Recipe: {{ result.recipe_name }}</h2>
        {% if result.recipe.description %}
        <p>{{ result.recipe.description }}</p>
        {% endif %}

        {% if result.ingredients %}
        <h3>Ingredients</h3>
        <ul>
        {% for ingredient in result.ingredients %}
            <li>{{ ingredient }}</li>
        {% endfor %}
        </ul>
        {% endif %}

        {% if result.instructions %}
        <h3>Instructions</h3>
        <ol>
        {% for instruction in result.instructions %}
            <li>{{ instruction }}</li>
        {% endfor %}
        </ol>
        {% endif %}

        {% if result.nutrition %}
        <h3>Nutrition information</h3>
        <p class="muted">Values depend on the recipe and serving size.</p>
        <div class="nutrition">
        {% for key, value in result.nutrition.items() %}
            <div>
                <strong>{{ key.replace('_', ' ').title() }}</strong>
                {{ value }}
            </div>
        {% endfor %}
        </div>
        {% endif %}

        {% if result.recipe.diet %}
        <h3>Diet information</h3>
        <p>{{ result.recipe.diet }}</p>
        {% endif %}
    </section>
    {% else %}
    <section class="card">
        <h2>Recipe information</h2>
        <p>No matching recipe was found in <code>data/recipes.json</code>
        for {{ result.food }}. Add a recipe with that food's name to your
        recipe file to display its details.</p>
    </section>
    {% endif %}
    {% endif %}
</main>

<footer>
    BCA Final-Year Project · ML · Neural Networks · Digital Image Processing · NLP
</footer>
</body>
</html>
"""


def prepare_result(image_relative_path, prediction, selected_label=None):
    """Build the page result from a model prediction or manual correction."""
    predicted_label = prediction.get("prediction", "")
    label_key = str(predicted_label).strip().lower().replace(" ", "_")

    confidence = prediction.get("confidence", 0)
    alternatives = prediction.get("alternatives", [])
    uncertain = prediction.get("uncertain", False)

    if selected_label:
        label_key = selected_label
        predicted_label = display_food_name(selected_label)
        uncertain = False
        # Manual correction is a user selection, not model confidence.
        confidence = None

    food = display_food_name(label_key)
    recipes = load_recipes()
    recipe = find_recipe_by_name(food, recipes)

    ingredients = []
    instructions = []
    nutrition = {}

    if recipe:
        ingredients = recipe.get("ingredients", [])
        instructions = (
            recipe.get("instructions")
            or recipe.get("steps")
            or []
        )
        nutrition = recipe.get("nutrition", {}) or {}

        if isinstance(ingredients, str):
            ingredients = [ingredients]
        if isinstance(instructions, str):
            instructions = [instructions]

    # If the user corrected the result, show that food first in alternatives.
    if selected_label:
        alternatives = [
            {"label": food, "confidence": "Manually selected"}
        ] + [
            item for item in alternatives
            if str(item.get("label", "")).lower().replace(" ", "_")
            != selected_label
        ]

    return {
        "image": image_relative_path,
        "food": food,
        "label_key": label_key,
        "confidence": confidence if confidence is not None else "Not applicable",
        "alternatives": alternatives,
        "uncertain": uncertain,
        "recipe": recipe,
        "recipe_name": (
            recipe.get("name") or recipe.get("title") or food
            if recipe else food
        ),
        "ingredients": ingredients,
        "instructions": instructions,
        "nutrition": nutrition,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        PAGE,
        result=None,
        class_names=class_names,
        food_names=FOOD_NAMES,
        chat_question=None,
        chat_answer=None
    )
@app.route("/chat", methods=["POST"])
def chat():
    question = request.form.get("question", "").strip()

    if not question:
        flash("Please enter a question.")
        return redirect(url_for("index"))

    answer = answer_question(question)

    return render_template_string(
        PAGE,
        result=None,
        class_names=class_names,
        food_names=FOOD_NAMES,
        chat_question=question,
        chat_answer=answer
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_file = request.files.get("food_image")

    if not uploaded_file or not uploaded_file.filename:
        flash("Please choose an image first.")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Use a PNG, JPG, JPEG or WEBP image.")
        return redirect(url_for("index"))

    safe_name = secure_filename(uploaded_file.filename)
    extension = safe_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"
    absolute_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        uploaded_file.save(absolute_path)

        print("STEP 1: CALLING predict_food()", flush=True)
        prediction = predict_food(absolute_path)
        print("TOP FOOD PREDICTIONS:", prediction.get("alternatives"), flush=True)
        print("MODEL RESULT:", prediction, flush=True)

        if not isinstance(prediction, dict) or not prediction.get("prediction"):
            raise ValueError("The model did not return a food prediction.")

        result = prepare_result(
            os.path.join("uploads", unique_name).replace("\\", "/"),
            prediction,
        )

        return render_template_string(
            PAGE,
            result=result,
            class_names=class_names,
            food_names=FOOD_NAMES,
        )

    except Exception as error:
        print("IMAGE ANALYSIS ERROR:", repr(error), flush=True)
        if os.path.exists(absolute_path):
            try:
                os.remove(absolute_path)
            except OSError:
                pass
        flash(f"Could not analyze the image: {error}")
        return redirect(url_for("index"))


@app.route("/correct", methods=["POST"])
def correct_food():
    image_relative_path = request.form.get("image", "")
    selected_label = request.form.get("corrected_food", "")

    # Only allow known model classes and images in the upload folder.
    if selected_label not in class_names:
        flash("Please select a valid food category.")
        return redirect(url_for("index"))

    if not image_relative_path.startswith("uploads/"):
        flash("The image reference is invalid. Please upload the image again.")
        return redirect(url_for("index"))

    image_name = os.path.basename(image_relative_path)
    absolute_path = os.path.join(UPLOAD_FOLDER, image_name)

    if not os.path.isfile(absolute_path):
        flash("The uploaded image is no longer available. Please upload it again.")
        return redirect(url_for("index"))

    try:
        # Keep the model's actual predictions available for comparison.
        prediction = predict_food(absolute_path)
        result = prepare_result(
            image_relative_path,
            prediction,
            selected_label=selected_label,
        )

        return render_template_string(
            PAGE,
            result=result,
            class_names=class_names,
            food_names=FOOD_NAMES,
        )

    except Exception as error:
        print("CORRECTION ERROR:", repr(error), flush=True)
        flash(f"Could not display the corrected result: {error}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    print("Starting Intelligent Recipe & Nutrition Vision Bot...", flush=True)
    app.run(debug=True)
