import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel
from PIL import Image

class VisionAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Image processor
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )

        # ViT backbone
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"
        )

        # Lung disease classes
        self.labels = [
            "Normal",
            "Pneumonia",
            "Tuberculosis",
            "COVID-19",
            "Lung Cancer"
        ]

        # Classification head (trainable)
        self.classifier = nn.Linear(
            self.vit.config.hidden_size,
            len(self.labels)
        )

        # Evaluation mode
        self.vit.eval()
        self.classifier.eval()

        self.vit.to(self.device)
        self.classifier.to(self.device)

    def analyze_image(self, image: Image.Image):
        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.vit(**inputs)

            # CLS token embedding
            features = outputs.last_hidden_state[:, 0, :]

            # Classifier
            logits = self.classifier(features)
            probs = F.softmax(logits, dim=1)

        confidence, idx = torch.max(probs, dim=1)

        return {
            "disease": self.labels[idx.item()],
            "confidence": round(confidence.item() * 100, 2),
            "all_probabilities": {
                self.labels[i]: round(probs[0][i].item() * 100, 2)
                for i in range(len(self.labels))
            }
        }
    