import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ChatbotService:
    @staticmethod
    def get_response(user_message):
        user_message_lower = user_message.lower().strip()
        
        # 1. Try Gemini AI first if API Key is available
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here" and len(api_key) > 5:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                # System context to guide the AI
                system_context = (
                    "You are Deep learning classification of fracture bones using ViT Assistant, a specialized AI for a bone fracture detection system. "
                    "The system uses ResNet50 and Vision Transformers to detect fractures in Elbow, Hand, and Shoulder X-rays. "
                    "Accuracy is 92%. You can answer questions about the system, its usage, and general bone health. "
                    "You should also be able to engage in normal conversation while maintaining your persona. "
                    "Always include a disclaimer that you are not a doctor and results should be verified by a professional."
                )
                
                response = model.generate_content(f"{system_context}\n\nUser: {user_message}")
                return response.text
            except Exception as e:
                print(f"Gemini API Error: {e}")
                # Fall through to hardcoded rules if AI fails
        
        # 2. Intelligent Fallback Logic (Pseudo-AI)
        # This uses a comprehensive dictionary for "proper" replies without an LLM
        
        knowledge_base = {
            "greetings": {
                "keywords": ["hi", "hello", "hey", "hola", "greetings"],
                "response": "Hello! I'm your Deep learning classification of fracture bones using ViT Assistant. I can help you analyze X-rays, explain results, or chat about bone health. How can I assist you today?"
            },
            "wellbeing": {
                "keywords": ["how are you", "how's it going", "how are things"],
                "response": "I'm functioning perfectly! Ready to help you with your bone fracture analysis. How about you?"
            },
            "identity": {
                "keywords": ["who are you", "what are you", "your name"],
                "response": "I am Deep learning classification of fracture bones using ViT Assistant, a specialized digital companion designed to help medical professionals detect and document bone fractures using deep learning."
            },
            "upload_process": {
                "keywords": ["upload", "how to start", "analyze", "image"],
                "response": "To start an analysis, go to the 'Overview' tab, drag and drop an X-ray image (JPG, PNG) into the box, and click 'Analyze Fracture'. Our ResNet50 models will then process the image."
            },
            "accuracy": {
                "keywords": ["accurate", "reliable", "precision", "accuracy"],
                "response": "Our system achieves a 92.4% accuracy rate on benchmark datasets like MURA. However, it is a diagnostic aid and not a final medical diagnosis. Clinical correlation is always required."
            },
            "results_info": {
                "keywords": ["confidence", "score", "percentage", "what does it mean"],
                "response": "The confidence score shows how certain the AI is. Scores above 75% indicate strong evidence, while lower scores (like 'Uncertain') suggest a manual review is strongly recommended."
            },
            "fracture_definition": {
                "keywords": ["what is a fracture", "broken bone", "bone break"],
                "response": "A fracture is a medical condition where there is a break in the continuity of the bone. It can range from subtle hairline fractures to complete displacements."
            },
            "supported_parts": {
                "keywords": ["which bones", "elbow", "hand", "shoulder", "parts"],
                "response": "Currently, Deep learning classification of fracture bones using ViT is optimized for Elbow, Hand, and Shoulder X-rays. We plan to add support for wrists and ankles in future updates."
            },
            "report_generation": {
                "keywords": ["report", "pdf", "generate", "download"],
                "response": "After analysis, click the 'Download Report' button in the results panel to get a professional PDF summing up the AI's findings and confidence scores."
            },
            "thanks": {
                "keywords": ["thanks", "thank you", "great", "awesome", "perfect"],
                "response": "You're very welcome! I'm glad I could help. Is there anything else you'd like to know about Deep learning classification of fracture bones using ViT?"
            }
        }

        # Find the best match
        best_match = None
        for category, data in knowledge_base.items():
            if any(keyword in user_message_lower for keyword in data["keywords"]):
                return data["response"]

        # If no specific match, try a more conversational generic response
        if len(user_message_lower) < 15:
            return "I'm here to help! You can ask me about how to upload images, what our accuracy is, or how to interpret your results. How can I assist you right now?"

        return (
            "That's an interesting question. While I'm currently in 'Optimized Mode' (waiting for a full AI API key), "
            "I'm specifically trained on Deep learning classification of fracture bones using ViT documentation. I can tell you about our ResNet50/ViT architecture, "
            "how to generate reports, or explain confidence scores. Try asking something like 'How accurate is the system?'"
        )
