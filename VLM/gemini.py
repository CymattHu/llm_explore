from google import genai
from google.genai import types


class Gemini:
    """
    Gemini class for interacting with the Gemini API.
    """
    def __init__(self):
        self.client = genai.Client(api_key="AIzaSyBUWpKqJ8HoXWga_V_ZGOiUVKaSVrbVIOQ")
        # self.model_name = "gemini-robotics-er"
        self.model_name = "gemini-2.5-pro-exp-03-25"
        self.bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 16 objects.
    Each object should be represented as a dictionary with the following keys:
    - "label": The label of the object should be unique with sequence number from the top left corner to the bottom right corner.
    - "box_2d": A list of four values representing the bounding box in the format [y1, x1, y2, x2].
    - "confidence": A float value representing the confidence score of the bounding box.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
    If user indicate the color or shape, please strictly follow the user's instruction.  
    """
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]
    def generate_response(self,prompt,image) -> types.GenerateContentResponse:
        return self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image],
            config = types.GenerateContentConfig(
                system_instruction=self.bounding_box_system_instructions,
                temperature=0.5,
                safety_settings=self.safety_settings,
            )
        )