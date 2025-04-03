from google import genai
from google.genai import types


class Gemini:
    """
    Gemini class for interacting with the Gemini API.
    """
    def __init__(self):
        self.client = genai.Client(api_key="AIzaSyBUWpKqJ8HoXWga_V_ZGOiUVKaSVrbVIOQ")
        self.model_name = "gemini-2.0-flash"
        self.bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 1 object and cover whole object.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
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