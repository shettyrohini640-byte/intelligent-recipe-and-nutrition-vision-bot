
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
/* ===== RecipeVision header and food banner ===== */

.site-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    padding: 20px 6%;
    background: #ffffff;
    color: #39275f;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #39275f;
    font-size: 25px;
    font-weight: 800;
    text-decoration: none;
}

.brand-icon {
    background: #f0e8ff;
    padding: 10px;
    border-radius: 15px;
    font-size: 28px;
}

.brand-highlight {
    color: #8b5cf6;
}

.main-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
}

.main-nav a {
    color: #5d5078;
    text-decoration: none;
    font-weight: 600;
}

.main-nav a:hover {
    color: #8b5cf6;
}

.healthy-badge {
    color: #6b568e;
    font-size: 14px;
}

.hero-banner {
    display: grid;
    grid-template-columns: 1fr 1fr;
    align-items: center;
    gap: 30px;
    width: min(1100px, 92%);
    margin: 30px auto;
    padding: 35px;
    background: linear-gradient(135deg, #f4edff, #ffffff);
    border: 1px solid #e8ddff;
    border-radius: 25px;
}

.hero-banner h1 {
    color: #30204f;
    font-size: clamp(28px, 4vw, 42px);
    line-height: 1.2;
}

.hero-banner p {
    color: #716780;
    line-height: 1.8;
}

.hero-label {
    color: #8052cf;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.hero-button {
    display: inline-block;
    padding: 13px 20px;
    background: #8052cf;
    color: white;
    border-radius: 12px;
    text-decoration: none;
    font-weight: 700;
}

.hero-banner img {
    width: 100%;
    height: 300px;
    object-fit: cover;
    border-radius: 20px;
}

@media (max-width: 700px) {
    .hero-banner {
        grid-template-columns: 1fr;
        padding: 22px;
    }

    .hero-banner img {
        height: 230px;
    }

    .main-nav {
        gap: 10px;
    }
}
/* ===== RecipeVision Premium Design ===== */

:root {
    --primary: #8060c7;
    --primary-dark: #6244a3;
    --lavender-light: #f5f0ff;
    --text-dark: #302747;
    --text-muted: #777087;
    --border: #e9e1f6;
    --white: #ffffff;
}

body {
    background: linear-gradient(135deg, #fbf9ff, #f2edff);
    color: var(--text-dark);
    font-family: "Segoe UI", Arial, sans-serif;
    line-height: 1.7;
}

.site-header {
    position: relative;
    padding: 20px 5%;
    box-shadow: 0 4px 20px rgba(65, 43, 100, 0.06);
}

.brand {
    font-size: 26px;
    letter-spacing: -0.7px;
}

.brand-icon {
    box-shadow: 0 4px 12px rgba(128, 96, 199, 0.15);
}

.main-nav a {
    transition: color 0.25s ease;
}

.main-nav a:hover {
    color: var(--primary);
}

.hero-banner {
    overflow: hidden;
    box-shadow: 0 15px 45px rgba(91, 65, 145, 0.09);
    transition: box-shadow 0.3s ease;
}

.hero-banner:hover {
    box-shadow: 0 20px 55px rgba(91, 65, 145, 0.14);
}

.hero-banner h1 {
    letter-spacing: -1.2px;
}

.hero-banner img {
    box-shadow: 0 10px 28px rgba(48, 39, 71, 0.12);
    transition: transform 0.4s ease;
}

.hero-banner img:hover {
    transform: scale(1.02);
}

.card {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 26px;
    box-shadow: 0 8px 30px rgba(65, 43, 100, 0.055);
    transition: transform 0.25s ease,
                box-shadow 0.25s ease;
}

.card:hover {
    box-shadow: 0 14px 38px rgba(65, 43, 100, 0.10);
}

.card h2 {
    color: var(--text-dark);
    margin-top: 0;
    font-size: 24px;
}

button,
.hero-button {
    background: linear-gradient(
        135deg,
        var(--primary),
        var(--primary-dark)
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 13px 23px;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.2s ease,
                box-shadow 0.2s ease;
}

button:hover,
.hero-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(98, 68, 163, 0.22);
}

input[type="file"],
input[type="text"],
input[type="number"],
textarea,
select {
    border: 1px solid #ddd3f0;
    border-radius: 12px;
    padding: 12px;
    background: #fff;
    max-width: 100%;
}

input:focus,
textarea:focus,
select:focus {
    outline: 2px solid #c7b5f2;
    border-color: var(--primary);
}

.food-image {
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(48, 39, 71, 0.12);
    max-width: 100%;
}

.result-name {
    color: var(--primary-dark);
    font-size: 28px;
    font-weight: 800;
}

.nutrition div {
    background: var(--lavender-light);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 15px;
}

.prediction-row {
    background: #faf8ff;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 8px 0;
}

.alert {
    border-radius: 12px;
    background: #f3edff;
    border: 1px solid #dfd1fa;
    padding: 14px;
}

footer {
    padding: 25px 15px;
    color: var(--text-muted);
    text-align: center;
}

@media (max-width: 700px) {
    .site-header {
        padding: 18px;
    }

    .brand {
        font-size: 22px;
    }

    .hero-banner {
        margin: 20px auto;
        padding: 22px;
    }

    .card {
        padding: 20px;
    }

    .card h2 {
        font-size: 21px;
    }
}
/* ===== RecipeVision Animation Effects ===== */

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(22px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes gentleFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

.hero-banner {
    animation: fadeInUp 0.8s ease-out both;
}

.hero-banner img {
    animation: fadeIn 1.1s ease-out both;
}

.card {
    animation: fadeInUp 0.65s ease-out both;
}

.card:nth-of-type(2) {
    animation-delay: 0.12s;
}

.brand-icon {
    animation: gentleFloat 3s ease-in-out infinite;
}

.hero-button,
button {
    transition: transform 0.25s ease,
                box-shadow 0.25s ease,
                background 0.25s ease;
}

.hero-button:hover,
button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 24px rgba(98, 68, 163, 0.25);
}

.food-image {
    transition: transform 0.35s ease,
                box-shadow 0.35s ease;
}

.food-image:hover {
    transform: scale(1.025);
    box-shadow: 0 14px 30px rgba(48, 39, 71, 0.16);
}

.main-nav a {
    transition: color 0.25s ease;
}

.main-nav a:hover {
    color: #a17be8;
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}
/* ===== AI Food Analysis Loading Indicator ===== */

.analysis-loading {
    margin-top: 25px;
    padding: 28px 20px;
    text-align: center;
    background: linear-gradient(135deg, #f8f4ff, #ffffff);
    border: 1px solid #e5d9ff;
    border-radius: 20px;
    animation: fadeInUp 0.4s ease-out;
}

.analysis-loading[hidden] {
    display: none !important;
}

.loading-spinner {
    width: 52px;
    height: 52px;
    margin: 0 auto 16px;
    border: 5px solid #e8ddff;
    border-top-color: #8060c7;
    border-radius: 50%;
    animation: analysisSpin 0.8s linear infinite;
}

@keyframes analysisSpin {
    to {
        transform: rotate(360deg);
    }
}

.analysis-loading h3 {
    margin: 8px 0;
    color: #6244a3;
}

.analysis-loading p,
.loading-note {
    color: #777087;
    font-size: 14px;
}

.loading-progress {
    width: min(280px, 90%);
    height: 7px;
    margin: 18px auto;
    overflow: hidden;
    background: #e8ddff;
    border-radius: 20px;
}

.loading-progress-bar {
    width: 35%;
    height: 100%;
    background: linear-gradient(90deg, #8060c7, #b69bea);
    border-radius: 20px;
    animation: loadingProgress 1.5s ease-in-out infinite alternate;
}

@keyframes loadingProgress {
    from {
        transform: translateX(0);
    }
    to {
        transform: translateX(185%);
    }
}
/* ===== Nutrition Dashboard ===== */

.nutrition {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    margin: 24px 0;
}

.nutrition div {
    position: relative;
    overflow: hidden;
    min-width: 0;
    padding: 22px 14px;
    text-align: center;
    background: linear-gradient(145deg, #ffffff, #f5f0ff);
    border: 1px solid #e6dcfa;
    border-radius: 20px;
    box-shadow: 0 7px 20px rgba(90, 65, 145, 0.07);
    transition: transform 0.25s ease,
                box-shadow 0.25s ease;
}

.nutrition div:hover {
    transform: translateY(-5px);
    box-shadow: 0 13px 28px rgba(90, 65, 145, 0.14);
}

.nutrition div strong {
    display: block;
    margin-bottom: 8px;
    color: #74668e;
    font-size: 14px;
    font-weight: 600;
}

.nutrition div span {
    display: block;
    color: #6244a3;
    font-size: 25px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.nutrition div small {
    display: block;
    margin-top: 5px;
    color: #888098;
    font-size: 12px;
}

@media (max-width: 700px) {
    .nutrition {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }

    .nutrition div {
        padding: 18px 10px;
    }

    .nutrition div span {
        font-size: 22px;
    }
}
/* Nutrition Dashboard Cards */

.nutrition .nutrition-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 18px 12px;
    background: #ffffff;
    border: 1px solid #eee6ff;
    border-radius: 18px;
    box-shadow: 0 5px 18px rgba(98, 68, 163, 0.08);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.nutrition .nutrition-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(98, 68, 163, 0.15);
}

.nutrition-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: #eee6ff;
    border-radius: 15px;
    font-size: 25px;
}

.nutrition .nutrition-card strong {
    text-align: center;
    color: #514078;
}

.nutrition .nutrition-card span {
    color: #6244a3;
    font-size: 25px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.nutrition .nutrition-card small {
    color: #888098;
    font-size: 12px;
}
/* =================================
   MOBILE-FRIENDLY RESPONSIVE DESIGN
   ================================= */

/* Make images and form elements fit their containers */
img {
    max-width: 100%;
    height: auto;
}

input,
select,
textarea,
button {
    max-width: 100%;
    box-sizing: border-box;
}

/* Tablet screens */
@media (max-width: 900px) {
    .container {
        width: 92%;
        max-width: 100%;
    }

    .nutrition {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
    }

    .hero-banner {
        padding: 35px 24px;
    }
}

/* Mobile screens */
@media (max-width: 600px) {
    body {
        font-size: 15px;
        line-height: 1.6;
    }

    .container {
        width: 92%;
        margin: 0 auto;
        padding-left: 0;
        padding-right: 0;
    }

    /* Header */
    .site-header {
        padding: 14px 0;
    }

    .site-header .container {
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
    }

    .site-header nav {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 16px;
        width: 100%;
    }

    .site-header nav a {
        font-size: 14px;
    }

    /* Hero banner */
    .hero-banner {
        padding: 25px 18px;
        border-radius: 18px;
        background-position: center;
    }

    .hero-banner h1 {
        font-size: clamp(25px, 7vw, 34px);
        line-height: 1.2;
    }

    .hero-banner p {
        font-size: 14px;
    }

    /* Main content cards */
    .card {
        padding: 18px;
        margin-bottom: 20px;
        border-radius: 18px;
    }

    h1 {
        overflow-wrap: anywhere;
    }

    h2 {
        font-size: 23px;
    }

    h3 {
        font-size: 18px;
    }

    /* Image upload form */
    input[type="file"] {
        display: block;
        width: 100%;
        padding: 10px;
        font-size: 13px;
    }

    button,
    input[type="submit"] {
        min-height: 44px;
        width: 100%;
        margin-top: 8px;
    }

    /* Nutrition dashboard */
    .nutrition {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }

    .nutrition .nutrition-card {
        min-width: 0;
        padding: 14px 8px;
        overflow-wrap: anywhere;
    }

    .nutrition-icon {
        width: 42px;
        height: 42px;
        font-size: 22px;
    }

    .nutrition .nutrition-card span {
        font-size: 21px;
    }

    .nutrition .nutrition-card strong {
        font-size: 13px;
    }

    /* Food images */
    .food-image,
    .result-image {
        width: 100%;
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }

    /* Lists and chatbot responses */
    ul,
    ol {
        padding-left: 22px;
    }

    pre,
    code {
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }
}

/* Small phones */
@media (max-width: 360px) {
    .nutrition {
        grid-template-columns: 1fr;
    }

    .card {
        padding: 14px;
    }

    .hero-banner {
        padding: 20px 14px;
    }
}

/* Respect users who prefer reduced motion */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}
/* Chatbot quick-question buttons */

.quick-questions {
    margin: 18px 0;
}

.quick-questions p {
    color: #6244a3;
    margin-bottom: 12px;
}

.quick-question-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.quick-question-buttons .quick-question {
    margin-top: 0;
    padding: 10px 14px;
    background: #f0e8ff;
    color: #6244a3;
    border: 1px solid #dfd1fa;
    border-radius: 12px;
    font-size: 14px;
    transition: background 0.2s ease, transform 0.2s ease;
}

.quick-question-buttons .quick-question:hover {
    background: #e3d5ff;
    transform: translateY(-2px);
}

@media (max-width: 600px) {
    .quick-question-buttons .quick-question {
        flex: 1 1 100%;
        width: 100%;
    }
}
</style>
</head>
<body>
<header class="site-header">
    <a href="/" class="brand">
        <span class="brand-icon">🥗</span>
        <span>Recipe<span class="brand-highlight">Vision</span></span>
    </a>

    <nav class="main-nav">
        <a href="/">Home</a>
        <a href="#upload">Image Upload</a>
        <a href="#chatbot">AI Assistant</a>
    </nav>

    <span class="healthy-badge">🌿 Eat Smart, Live Well</span>
</header>

<section class="hero-banner">
    <div>
        <span class="hero-label">YOUR SMART KITCHEN ASSISTANT</span>
        <h1>Discover the Story Behind Your Food</h1>
        <p>Identify food, explore delicious recipes, and discover nutrition information with AI.</p>
        <a class="hero-button" href="#upload">Explore Your Food ↓</a>
    </div>

    <div>
        <img src="{{ url_for('static', filename='food-banner1.jpg') }}"
             alt="Delicious food">
    </div>
</section>
<main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <div class="alert">{{ message }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}

   <section class="card" id="upload">
    <h2>Identify your food</h2>
    <p class="muted">Upload a clear food photo to get a prediction and recipe suggestions.</p>

    <form action="{{ url_for('analyze') }}" method="post"
          enctype="multipart/form-data">
        <label for="food_image">Choose a food image</label>
        <input id="food_image" type="file" name="food_image"
               accept=".png,.jpg,.jpeg,.webp" required>
        <button type="submit">Analyze Image</button>
    </form>

    <!-- AI food analysis loading indicator -->
    <div id="analysis-loading" class="analysis-loading"
         role="status" aria-live="polite" hidden>
        <div class="loading-spinner"></div>
        <h3>Analyzing your food...</h3>
        <p>Our AI is examining your image and finding matching recipes.</p>
        <div class="loading-progress">
            <div class="loading-progress-bar"></div>
        </div>
        <span class="loading-note">Please wait while we process your image.</span>
    </div>
</section>
        <section class="card" id="chatbot">
        <h2>Recipe & Nutrition Chatbot</h2>
        <p class="muted">
            Ask questions about ingredients, cooking instructions,
            and nutrition information.
        </p>
        <div class="quick-questions">
    <p><strong>Quick questions</strong></p>

    <div class="quick-question-buttons">
        <button type="button" class="quick-question"
                data-question="Show calories and nutrition">
            🔥 Show Nutrition
        </button>

        <button type="button" class="quick-question"
                data-question="List the ingredients">
            🥗 Ingredients
        </button>

        <button type="button" class="quick-question"
                data-question="Give me the cooking instructions">
            👩‍🍳 Cooking Instructions
        </button>

        <button type="button" class="quick-question"
                data-question="What are the health benefits?">
            💚 Health Benefits
        </button>
    </div>
</div>

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
        <div class="nutrition-card">
            <div class="nutrition-icon">
                {% if 'calorie' in key.lower() %}
                    🔥
                {% elif 'protein' in key.lower() %}
                    💪
                {% elif 'carb' in key.lower() %}
                    🌾
                {% elif 'fat' in key.lower() %}
                    🥑
                {% elif 'fibre' in key.lower() or 'fiber' in key.lower() %}
                    🥗
                {% else %}
                    🍽️
                {% endif %}
            </div>

            <strong>{{ key.replace('_', ' ').title() }}</strong>
            <span>{{ value }}</span>
            <small>Per serving</small>
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
<script>
document.addEventListener("DOMContentLoaded", function () {

    // Food image analysis loading indicator
    const form = document.querySelector('form[action*="/analyze"]');
    const imageInput = document.getElementById("food_image");
    const loading = document.getElementById("analysis-loading");

    if (form && imageInput && loading) {
        form.addEventListener("submit", function (event) {
            if (!imageInput.files || imageInput.files.length === 0) {
                event.preventDefault();
                alert("Please choose a food image first.");
                return;
            }

            loading.hidden = false;

            const button = form.querySelector('button[type="submit"]');

            if (button) {
                button.disabled = true;
                button.textContent = "Analyzing...";
            }
        });
    }

    // Chatbot quick-question buttons
    const questionInput = document.getElementById("question");
    const quickButtons = document.querySelectorAll(".quick-question");

    quickButtons.forEach(function (button) {
        button.addEventListener("click", function () {
            if (!questionInput) {
                return;
            }

            questionInput.value = button.dataset.question || "";
            questionInput.focus();

            questionInput.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });
        });
    });

});
</script>
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
