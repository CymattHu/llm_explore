import json
import os

class KnowledgeBaseEntry:
    def __init__(self, image_embedding, incorrect_bboxes, correct_bboxes):
        self.image_embedding = image_embedding
        self.incorrect_bboxes = incorrect_bboxes
        self.correct_bboxes = correct_bboxes

    def to_dict(self):
        return {
            "image_embedding": self.image_embedding,
            "incorrect_bboxes": self.incorrect_bboxes,
            "correct_bboxes": self.correct_bboxes,
        }

    def save_to_json(self, file_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the dictionary to a JSON file
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"File saved to {file_path}")