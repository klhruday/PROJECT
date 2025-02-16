# first run the command in terminal : pip install -r requirements.txt


from flask import Flask, request, jsonify, send_from_directory, render_template
import google.generativeai as genai
import base64
import os

app = Flask(__name__)

# Configure Gemini API securely using environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCBD3QbPd5NIUvZ3CIiQpR8eYk-uRA38vY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

UPLOAD_FOLDER = "static\\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
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
    allowed_extensions = {"jpg", "jpeg", "png"}
    if file.filename.split(".")[-1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file format. Only JPG, JPEG, and PNG are allowed."}), 400

    # Save the image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    from PIL import Image

# Open image from file bytes
    image = Image.open(file)

    # Resize to desired dimensions (e.g., 300x300)
    image = image.resize((300, 300))

    # Save resized image
    image.save(file_path)

    # Convert image to base64
    with open(file_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Proper Gemini API request with image handling
    try:
        response = model.generate_content([
            {"text": "Identify waste/trash in this image. Provide type, environmental impact, and disposal method in 3 lines. Strictly stick to the topic of waste treatment, and disposal."},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
        ])

        # Extract response text
        caption = response.text.strip() if response.text else "Could not generate a description."

        return jsonify({
            "history": [{"message": caption}],  # Maintain structure like chat response
            "image_url": f"/uploads/{file.filename}"
        })

    except Exception as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500


# Serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


import re

def clean_response(text):
    """ Remove Markdown-like formatting from response """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # Remove bullet point styling
    text = re.sub(r"[_`]", "", text)  # Remove underscores and backticks
    return text.strip()


prompt ="""You're a Helpful ai assistant who assists user on waste/trash asked by the user. 
be polite and stick to the topic of waste treatment, and disposal. Please refrain from answering any other queries. 
Keep your responses short and simple of max length 3 lines. The user query : """

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message received"}), 400

    try:
        response = model.generate_content(prompt+user_message)
        bot_reply = response.text if response.text else "Sorry, I couldn't generate a response."
        
        # Clean up formatting
        bot_reply = clean_response(bot_reply)

        return jsonify({"history": [{"message": bot_reply}]})

    except Exception as e:
        return jsonify({"error": f"API request failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
