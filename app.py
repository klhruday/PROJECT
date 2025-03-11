# first run the command in terminal : pip install -r requirements.txt

from flask import Flask, request, jsonify, send_from_directory, render_template, session,redirect,url_for
import google.generativeai as genai
import tensorflow as tf
import numpy as np
import base64
import os
import re
from PIL import Image

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session management

# Configure Gemini API securely using environment variables
GEMINI_API_KEY = "AIzaSyCBD3QbPd5NIUvZ3CIiQpR8eYk-uRA38vY"
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-1.5-flash") # Renamed to avoid confusion with TF model

model_tf = tf.keras.models.load_model("waste_classifier_fixed.h5", compile=False)  # Avoid metric warning # Renamed to avoid confusion

def predict(image):
    """
    Preprocesses the uploaded image, runs the model, and returns the predicted class and confidence.
    """

    class_names = [
        'aerosol_cans - hazardous',
        'aluminum_food_cans-non_biodegradable',
        'aluminum_soda_cans-non_biodegradable',
        'cardboard_boxes-biodegradable',
        'cardboard_packaging-biodegradable',
        'clothing-non_biodegradable',
        'coffee_grounds-biodegradable',
        'disposable_plastic_cutlery-non_biodegradable',
        'eggshells-biodegradable',
        'food_waste-biodegradable',
        'plastic_beverage_bottles-non_biodegradable',
        'glass_cosmetic_containers-non_biodegradable',
        'glass_food_jars-non_biodegradable',
        'magazines-biodegradable',
        'newspaper-biodegradable',
        'office_paper-biodegradable',
        'paper_cups-biodegradable',
        'plastic_cup_lids-non_biodegradable',
        'plastic_detergent_bottles-non_biodegradable',
        'plastic_food_containers-non_biodegradable',
        'plastic_shopping_bags-non_biodegradable',
        'plastic_soda_bottles-non_biodegradable',
        'plastic_straws-non_biodegradable',
        'plastic_trash_bags-non_biodegradable',
        'plastic_water_bottles-non_biode{radable', # corrected typo
        'shoes-non_biodegradable',
        'steel_food_cans-non_biodegradable',
        'styrofoam_cups-non_biodegradable',
        'styrofoam_food_containers-non_biodegradable',
        'tea_bags-biodegradable'
    ]
    # Convert image to RGB (handles cases where image is grayscale)
    image = image.convert("RGB")
    # Resize image to match model input size
    image = image.resize((224, 224))
    # Convert image to numpy array and normalize pixel values
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Run prediction
    preds = model_tf.predict(img_array) # using model_tf here
    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    predicted_class = class_names[pred_idx]

    # Return a dictionary with class names and probabilities
    print(predicted_class)
    return predicted_class

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/clear")
def clear():
    session.clear()
    return redirect(url_for('index'))

@app.route("/")
@app.route('/index')
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Validate file format
    allowed_extensions = {"jpg", "jpeg", "png","webp"}
    if file.filename.split(".")[-1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400

    # Save and resize the image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    image = Image.open(file)
    image = image.resize((300, 300))  # Resize
    image.save(file_path)

    # Convert image to base64 (still needed to display image)
    with open(file_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    try:
        predicted_class = predict(image) # Get classification from TF model

        # Construct prompt for Gemini using the classification output
        gemini_prompt_text = f"The image contains: {predicted_class}. Based on this waste type, ask a question to the user in one short line to initiate a conversation about waste management."

        response = model_gemini.generate_content([ # Using model_gemini here
            {"text": gemini_prompt_text} # Passing classification as text prompt
        ])

        caption = response.text.strip() if response.text else "Could not generate a description."

        # Store image context in session -  Now storing the classification
        session["image_context"] = predicted_class # Storing predicted class instead of Gemini caption

        return jsonify({
            "history": [{"message": caption}],
            "image_url": f"/uploads/{file.filename}"
        })

    except Exception as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500

# Serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def clean_response(text):
    """Format AI response for better readability with enhanced styling"""
    text = re.sub(r"\*\*(.*?)\*\*", r'<span class="bot-bold">\1</span>', text)  # Bold text
    text = re.sub(r"\*(.*?)\*", r'<span class="bot-bullet">• \1</span>', text)  # Bullet points
    text = re.sub(r"[_`]", "", text)  # Remove underscores and backticks
    text = text.replace("\n", "<br>")  # Add line breaks

    return f'<div class="bot-message-content">{text.strip()}</div>'


prompt = """You're a helpful AI assistant who assists users with waste/trash-related questions.
Be polite and stick to the topic of waste treatment and disposal, Also answer if the user asks queries that are related to waste recycling, reusing and reducing.
Please refrain from answering any other queries besides the ones i've mentioned above.
explain in detail only when required. If the user asks for some suggestions, mention them point wise. The user query: """

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message received"}), 400

    # Retrieve stored image context if available (now the classification)
    image_context = session.get("image_context", "")

    # Modify the prompt to continue based on the uploaded image (classification)
    full_prompt = prompt
    if image_context:
        full_prompt += f"\nContext from image classification: {image_context}\n" # Changed context description

    full_prompt += user_message

    try:
        response = model_gemini.generate_content(full_prompt) # Using model_gemini here
        bot_reply = response.text if response.text else "Sorry, I couldn't generate a response."

        bot_reply = clean_response(bot_reply)

        return jsonify({"history": [{"message": bot_reply}]})

    except Exception as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True,port=5001)

# Removed Gradio Interface block as requested. It's not part of the Flask application logic.