import nltk
from textblob import TextBlob
import random
import string

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')

class EmotionAwareBot:
    def __init__(self):
        # Define our intents based on the assignment requirements
        self.intents = {
            "greetings": {
                "keywords": ["hello", "hi", "hey", "greetings", "morning"],
                "responses": ["Hello! I am ready to talk about AI and NLP.", "Hi there! How can I help you with your studies today?"]
            },
            "nlp_concepts": {
                "keywords": ["what is nlp", "define nlp", "natural language processing", "explain nlp", "nlp concept"],
                "responses": [
                    "NLP (Natural Language Processing) is a field of AI focused on the interaction between computers and humans through natural language. It helps machines process and analyze large amounts of natural language data.",
                    "Think of NLP as giving computers the ability to read, understand, and derive meaning from human languages. Tasks include translation, sentiment analysis, and speech recognition."
                ]
            },
            "ai_applications": {
                "keywords": ["ai applications", "use of ai", "where is ai used", "artificial intelligence application", "give me an example of ai"],
                "responses": [
                    "AI has amazing applications! It's used in healthcare for disease prediction, in finance for algorithmic trading, and in autonomous vehicles for navigation.",
                    "Common AI applications you might see every day include recommendation engines (like Netflix or Spotify), facial recognition on your phone, and virtual assistants like Siri."
                ]
            },
            "farewell": {
                "keywords": ["bye", "goodbye", "see you", "exit", "quit"],
                "responses": ["Goodbye! Good luck with your assignment!", "See you later! Keep coding!"]
            }
        }

    def preprocess_text(self, user_input):
        """NLP Preprocessing: Lowercase, remove punctuation, and tokenize."""
        # Lowercase the text
        text = user_input.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize using NLTK
        tokens = nltk.word_tokenize(text)
        return tokens

    def analyze_emotion(self, text):
        """Detects sentiment polarity using TextBlob to modify the bot's response."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.4:
            return "I love your enthusiasm! 😊 "
        elif polarity < -0.4:
            return "I sense a bit of frustration. Don't worry, learning these concepts takes time. Let me help. 🤖 "
        else:
            return "" # Neutral, no prefix needed

    def get_response(self, user_input):
        # 1. Preprocess the input
        tokens = self.preprocess_text(user_input)
        processed_input = " ".join(tokens) # Rejoin for simple keyword matching

        # 2. Analyze sentiment for emotional response
        emotion_prefix = self.analyze_emotion(user_input)

        # 3. Intent Matching
        for intent_name, intent_data in self.intents.items():
            for keyword in intent_data["keywords"]:
                # Simple substring match for basic intent recognition
                if keyword in processed_input:
                    base_response = random.choice(intent_data["responses"])
                    return emotion_prefix + base_response

        # Fallback response
        return emotion_prefix + "I'm a simple bot. I mostly know about NLP concepts and AI applications. Could you ask me about those?"

    def chat(self):
        print("Bot: Hi! I'm an NLP & AI explainer bot. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Bot: " + random.choice(self.intents["farewell"]["responses"]))
                break
            
            response = self.get_response(user_input)
            print("Bot:", response)

# Run the chatbot
if __name__ == "__main__":
    bot = EmotionAwareBot()
    bot.chat()