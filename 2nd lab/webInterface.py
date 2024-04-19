import os
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import PIL
from io import BytesIO
import torch
from torchvision import transforms
from trainedModel import CNN


app = Flask(__name__)
CORS(app)

model = CNN()
model.load_state_dict(torch.load("model.pth"))
print(model)

model.eval()
classes = ["Banana", "Duck", "Sea lion"]


def preprocess_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4455, 0.4473, 0.3930], std=[0.2707, 0.2667, 0.2731]),
        ]
    )

    img = PIL.Image.open(image)
    img = preprocess(img).unsqueeze(0)
    return img


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected for uploading."}), 400

    try:
        img = preprocess_image(BytesIO(file.read()))

        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, preds = torch.max(probabilities, 1)
            pred_label = classes[preds.item()]
            pred_prob = max_prob.item()

        return jsonify({"prediction": pred_label, "probability": pred_prob})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
