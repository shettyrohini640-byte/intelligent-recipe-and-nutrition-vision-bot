
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPES_PATH = os.path.join(BASE_DIR, "data", "recipes.json")


def load_recipes():
    try:
        with open(RECIPES_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return list(data.values())

        return []

    except (FileNotFoundError, json.JSONDecodeError) as error:
        print("Could not load recipes:", error)
        return []


def normalize(text):
    text = str(text).lower().replace("_", " ")
    return re.sub(r"[^a-z0-9\s]", " ", text).strip()


def format_list(items):
    if not isinstance(items, list):
        return str(items)

    lines = []

    for item in items:
        if isinstance(item, dict):
            name = item.get("name", "Item")
            quantity = item.get("quantity", "")
            lines.append(f"• {name}: {quantity}" if quantity else f"• {name}")
        else:
            lines.append(f"• {item}")

    return "\n".join(lines)


def find_recipe(question, recipes):
    question = normalize(question)

    aliases = {
        "pulav": "vegetable pulao",
        "karahi paneer": "kadai paneer",
        "rava upma": "vegetable upma"
    }

    for alias, recipe_name in aliases.items():
        if alias in question:
            for recipe in recipes:
                if normalize(recipe.get("name", "")) == recipe_name:
                    return recipe

    best_recipe = None
    best_score = 0

    for recipe in recipes:
        name = normalize(recipe.get("name", ""))
        tags = recipe.get("tags", [])

        if not isinstance(tags, list):
            tags = []

        terms = [name] + [normalize(tag) for tag in tags]

        for term in terms:
            if term and term in question:
                score = len(term.split())

                if score > best_score:
                    best_score = score
                    best_recipe = recipe

    return best_recipe


def get_nutrition(recipe):
    nutrition = recipe.get("nutrition", {})

    if not isinstance(nutrition, dict) or not nutrition:
        return "Nutrition information is not available."

    labels = {
        "calories": ("Calories", " kcal"),
        "protein": ("Protein", " g"),
        "carbs": ("Carbohydrates", " g"),
        "fat": ("Fat", " g"),
        "fibre": ("Fibre", " g")
    }

    lines = []

    for key, (label, unit) in labels.items():
        if key in nutrition:
            lines.append(f"• {label}: {nutrition[key]}{unit}")

    return "\n".join(lines) if lines else "Nutrition information is unavailable."


def answer_question(message):
    question = normalize(message)
    recipes = load_recipes()

    if not question:
        return "Please type a question about a recipe."

    if not recipes:
        return "I couldn't load the recipes. Check data/recipes.json."

    if question in [
        "hi", "hello", "hey", "good morning",
        "good afternoon", "good evening"
    ]:
        return (
            "Hello! Welcome to the Recipe and Nutrition Chatbot.\n"
            "Ask me about ingredients, cooking instructions, cooking time, "
            "nutrition, health benefits, allergens, or substitutions."
        )

    if any(phrase in question for phrase in [
        "what recipes", "list recipes", "available recipes", "show recipes"
    ]):
        names = [recipe.get("name", "Unnamed recipe") for recipe in recipes]
        return "Available recipes:\n" + "\n".join(f"• {name}" for name in names)

    recipe = find_recipe(message, recipes)

    if recipe is None:
        return (
            "I couldn't identify the recipe in your question.\n"
            "Try Vegetable Pulao, Masala Dosa, Idli, Paneer Butter Masala, "
            "Aloo Gobi, Vegetable Upma, Kadai Paneer, or Tomato Curry."
        )

    name = recipe.get("name", "This recipe")

    # Nutrition questions must be handled before general description questions.
    nutrition_words = [
        "nutrition", "nutritional", "calorie", "calories",
        "protein", "carbs", "carbohydrate", "fat", "fibre", "fiber"
    ]

    if any(word in question for word in nutrition_words):
        nutrition = recipe.get("nutrition", {})

        if isinstance(nutrition, dict):
            for key, label in [
                ("calories", "Calories"),
                ("protein", "Protein"),
                ("carbs", "Carbohydrates"),
                ("fat", "Fat"),
                ("fibre", "Fibre")
            ]:
                if key in question and key in nutrition:
                    value = nutrition[key]
                    unit = " kcal" if key == "calories" else " g"
                    return (
                        f"{label} in {name}: {value}{unit} "
                        "(estimated per serving)."
                    )

        return (
            f"Estimated nutrition for {name} per serving:\n"
            f"{get_nutrition(recipe)}"
        )

    # Ingredients
    if any(word in question for word in [
        "ingredient", "items needed", "what do i need"
    ]):
        return (
            f"Ingredients for {name}:\n"
            f"{format_list(recipe.get('ingredients', []))}"
        )

    # Cooking instructions
    if any(word in question for word in [
        "how to", "how do i", "cook", "prepare", "recipe",
        "steps", "instructions", "method", "make"
    ]):
        instructions = recipe.get("instructions", [])

        if isinstance(instructions, list):
            steps = "\n".join(
                f"{index}. {step}"
                for index, step in enumerate(instructions, start=1)
            )
        else:
            steps = str(instructions)

        return f"How to prepare {name}:\n\n{steps}"

    # Preparation and cooking time
    if any(word in question for word in [
        "time", "how long", "duration", "minutes"
    ]):
        prep = recipe.get("prep_time", "Not available")
        cook = recipe.get("cook_time", "Not available")

        if isinstance(prep, (int, float)) and isinstance(cook, (int, float)):
            total = prep + cook
        else:
            total = "Not available"

        return (
            f"Time required for {name}:\n"
            f"• Preparation: {prep} minutes\n"
            f"• Cooking: {cook} minutes\n"
            f"• Total: {total} minutes"
        )

    # Servings
    if any(word in question for word in [
        "serving", "servings", "portions", "how many people"
    ]):
        return (
            f"{name} serves approximately "
            f"{recipe.get('servings', 'an unspecified number of')} people."
        )

    # Health benefits
    if any(word in question for word in [
        "health", "benefit", "healthy", "good for"
    ]):
        benefits = recipe.get("health_benefits", [])

        if benefits:
            return (
                f"General health information about {name}:\n"
                f"{format_list(benefits)}"
            )

        return "Health benefit information is not available for this recipe."

    # Dietary information
    if any(word in question for word in [
        "vegetarian", "vegan", "diet", "dietary", "cuisine", "category"
    ]):
        return (
            f"Dietary information for {name}:\n"
            f"• Diet: {recipe.get('diet', 'Not specified')}\n"
            f"• Cuisine: {recipe.get('cuisine', 'Not specified')}\n"
            f"• Category: {recipe.get('category', 'Not specified')}"
        )

    # Allergens
    if any(word in question for word in [
        "allergen", "allergy", "allergies"
    ]):
        allergens = recipe.get("allergens", [])

        if allergens:
            return f"Listed allergens in {name}:\n{format_list(allergens)}"

        return (
            f"No allergens are listed in the database for {name}. "
            "Always check the actual ingredients and their packaging."
        )

    # Ingredient substitutions
    if any(word in question for word in [
        "substitute", "substitution", "replace",
        "instead of", "alternative"
    ]):
        substitutions = recipe.get("substitutions", [])

        if substitutions:
            return (
                f"Possible substitutions for {name}:\n"
                f"{format_list(substitutions)}"
            )

        return "No substitutions are listed for this recipe."

    # Cooking difficulty
    if any(word in question for word in [
        "difficulty", "difficult", "easy", "hard"
    ]):
        return (
            f"Cooking difficulty for {name}: "
            f"{recipe.get('difficulty', 'Not specified')}."
        )

    # Recipe description
    return (
        f"{name}\n\n"
        f"{recipe.get('description', 'No description available.')}\n\n"
        "You can ask me about its ingredients, cooking instructions, "
        "nutrition, preparation time, health benefits, allergens, "
        "dietary information, or substitutions."
    )
