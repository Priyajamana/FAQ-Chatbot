import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_data import faqs


# Get questions and answers
questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# Convert FAQ questions into numerical vectors
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)


def get_answer():
    user_question = entry.get().strip()

    if not user_question:
        return

    # Convert user's question into a vector
    user_vector = vectorizer.transform([user_question])

    # Calculate similarity with all FAQ questions
    similarities = cosine_similarity(user_vector, faq_vectors)

    # Find the most similar question
    best_match = similarities.argmax()
    best_score = similarities[0][best_match]

    # Minimum similarity required
    if best_score >= 0.25:
        answer = answers[best_match]
    else:
        answer = "Sorry, I don't know the answer to that question."

    # Display conversation
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "You: " + user_question + "\n")
    chat_area.insert(tk.END, "Bot: " + answer + "\n\n")
    chat_area.config(state=tk.DISABLED)

    # Clear input
    entry.delete(0, tk.END)


# Create window
window = tk.Tk()
window.title("FAQ Chatbot")
window.geometry("700x600")


# Title
title = tk.Label(
    window,
    text="🤖 FAQ Chatbot",
    font=("Arial", 24, "bold")
)
title.pack(pady=15)


# Chat area
chat_area = scrolledtext.ScrolledText( 
    window,
    width=75,
    height=25,
    wrap=tk.WORD,
    state=tk.DISABLED
)
chat_area.pack(padx=15, pady=10)


# Input area
entry = tk.Entry(
    window,
    width=60,
    font=("Arial", 12)
)
entry.pack(side=tk.LEFT, padx=(30, 5), pady=15)


# Ask button
ask_button = tk.Button(
    window,
    text="Ask",
    command=get_answer,
    font=("Arial", 12, "bold")
)
ask_button.pack(side=tk.LEFT, pady=15)


# Start application
window.mainloop()  