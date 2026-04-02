import joblib
import re
import random
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up required NLTK corpus silently
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# Initialize NLP processing objects
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Keep negations as they carry important emotional signal
negation_words = {'no', 'not', 'nor', 'never', 'neither', "n't"}
stop_words -= negation_words

def preprocess_text(text):
    """Clean and normalize a text string exactly as the model expects."""
    text = str(text).lower()
    # Contract Expansions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    
    # Remove URLs, Mentions, numbers, etc.
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize, remove stopwords, lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    
    return ' '.join(tokens)

# Load the Pre-Trained Model Pipeline
try:
    # Ensure this runs in the same directory as the .pkl file
    pipeline = joblib.load('emotion_classifier.pkl')
except FileNotFoundError:
    print("❌ Error: 'emotion_classifier.pkl' not found.")
    print("Please make sure you have run the notebook to train the model first.")
    print("You also need to run this script from the same directory where the model was saved.")
    exit(1)

# Emotion-Contextual Responses Dictionary
EMOTION_RESPONSES = {
    "joy": [
        "😊 That's wonderful! Your happiness is contagious — keep spreading those good vibes!",
        "🌟 I can feel your excitement! It sounds like things are going really well for you.",
        "🎉 Amazing! Moments of joy like this are worth cherishing. Hope it lasts long!",
        "✨ You're radiating positive energy! What a great thing to be experiencing.",
        "😄 Life is beautiful, isn't it? So glad you're feeling this way!",
        "🌈 That spark of happiness you have right now? Hold onto it tight!",
        "🎊 Celebrations are in order! You sound absolutely thrilled — and rightfully so.",
        "💛 Pure joy is rare and precious. Sounds like you've found a good reason to smile today!"
    ],
    "sadness": [
        "💙 I'm sorry you're feeling this way. It's okay to feel sad — your emotions are valid.",
        "🌧️ Tough times don't last. Allow yourself to feel this, and know that better days are ahead.",
        "🤗 Sending you a virtual hug. Whatever you're going through, you don't have to face it alone.",
        "💔 It's okay to not be okay sometimes. Take it one step at a time — you've got this.",
        "🕊️ Sadness is a part of life, but it's not the whole story. Hang in there.",
        "🫂 Your feelings matter. Give yourself permission to rest and heal at your own pace.",
        "🌱 Even in sadness, there is growth. You are stronger than you think.",
        "🌙 Not every night lasts forever. The sun will rise again — keep going."
    ],
    "anger": [
        "🔥 I can sense that frustration. Take a deep breath — it's okay to feel angry sometimes.",
        "😤 That sounds really frustrating. Try to step away for a moment if you can.",
        "⚡ Your anger is valid. Channel that energy into something constructive when you're ready.",
        "🧘 It's completely normal to feel this way. A few deep breaths can help reset your mind.",
        "🌊 Strong emotions like anger can be overwhelming. Give yourself space to process it.",
        "🛡️ Sometimes things feel unfair and it's reasonable to be upset. You deserve to be heard.",
        "💢 Recognized your anger — now try to understand what's behind it. That might help.",
        "🏃 If you can, go for a walk or do something physical. It really helps release that tension."
    ],
    "fear": [
        "😨 Fear is the mind's way of trying to protect you. You are safe, and you can handle this.",
        "🌟 Courage isn't the absence of fear — it's moving forward despite it. You can do this.",
        "🫶 Whatever is worrying you, remember: you've overcome challenges before. You will again.",
        "🌿 Take a slow, deep breath. Grounding yourself in the present can help ease anxiety.",
        "🔒 It's natural to be afraid. But don't let fear make your decisions for you.",
        "🌅 Beyond every fear lies the opportunity to grow. Trust yourself a little more.",
        "🤝 You're not alone in feeling this way. Many people share these fears — and they persevere.",
        "🧭 Feel the fear, acknowledge it, and then take one small step forward anyway."
    ],
    "love": [
        "❤️ Love is truly one of the most beautiful human experiences. Cherish it!",
        "💕 It sounds like your heart is full. That warmth is something very special.",
        "🌸 Love in any form — romantic, friendly, familial — is a gift. Hold it close.",
        "💖 There's something magical about feeling deeply connected to someone or something.",
        "🥰 That feeling of love you have? It's one of the most powerful forces in the world.",
        "🕊️ Love transforms us and the world around us. What a beautiful thing to feel!",
        "💌 Whether it's new love or something long-lasting, it's always worth celebrating.",
        "✨ You deserve all the love you feel — and just as much in return."
    ],
    "surprise": [
        "😲 Wow, sounds like something unexpected happened! How are you feeling about it?",
        "🎉 Life is full of surprises — some of the best things are completely unexpected!",
        "😮 That must have caught you completely off guard! Take a moment to process it.",
        "🌠 Sometimes the universe throws curveballs — good or bad, they shape our story.",
        "🤩 Surprises can be thrilling! How exciting to have something unexpected happen.",
        "🎊 Whether it was a pleasant or shocking surprise, life just got a lot more interesting!",
        "🔮 The unexpected is what makes life unpredictable and exciting. Embrace it!",
        "👀 Sounds like quite the twist! Hope it turns out to be a good one for you."
    ]
}

EMOTION_EMOJIS = {
    'joy': '😊 Joy',
    'sadness': '😢 Sadness',
    'anger': '😠 Anger',
    'fear': '😨 Fear',
    'love': '❤️ Love',
    'surprise': '😲 Surprise'
}

def predict_emotion(text):
    """Predict the emotion label and confidence for a given user text."""
    cleaned = preprocess_text(text)
    if not cleaned.strip():
        # Edge case default fallback
        return 'joy', 0.0  
        
    pred_label = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    confidence = proba[list(classes).index(pred_label)] * 100
    
    return pred_label, confidence

def chatbot_response(user_input):
    """Process user input against model predicting the emotion + response."""
    if not user_input.strip():
        return {
            'emotion': 'unknown',
            'confidence': 0.0,
            'response': "Please type something so I can understand how you're feeling! 😊"
        }

    emotion, confidence = predict_emotion(user_input)
    # Pick a random response from corresponding bucket
    response = random.choice(EMOTION_RESPONSES.get(emotion, ["I'm here for you! 😊"]))

    return {
        'emotion': emotion,
        'confidence': round(confidence, 2),
        'response': response
    }

def display_chat(user_input):
    """Pretty-print the result of checking."""
    result = chatbot_response(user_input)
    emotion_display = EMOTION_EMOJIS.get(result['emotion'], result['emotion'].title())
    
    print(f"{'='*60}")
    print(f"  🧑 You   : {user_input}")
    print(f"{'─'*60}")
    print(f"  🏷️  Detected Emotion : {emotion_display} ({result['confidence']}% confidence)")
    print(f"{'─'*60}")
    print(f"  🤖 Bot   : {result['response']}")
    print(f"{'='*60}\n")

def main():
    print("\n🤖 Emotion Chatbot — Interactive Mode")
    print("Tell me how you're feeling. Type 'quit', 'exit', or 'bye' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {'quit', 'exit', 'bye', 'goodbye'}:
                print("\n🤖 Bot: Take care! Wishing you joy and positivity ahead. Goodbye! 👋\n")
                break
            display_chat(user_input)
            
        except (KeyboardInterrupt, EOFError):
             # Escape gracefully on Ctrl+C or Ctrl+D
             print("\n\n🤖 Bot: Exiting... Goodbye! 👋\n")
             break

if __name__ == "__main__":
    main()
