from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize knowledge base
KNOWLEDGE_FILE = 'knowledge_base.json'

class KnowledgeBase:
    def __init__(self):
        self.knowledge_file = KNOWLEDGE_FILE
        self.load_knowledge()
        self.update_vectorizer()
    
    def load_knowledge(self):
        if os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            # Default knowledge
            self.data = {
                "entries": [
                    {
                        "id": 1,
                        "content": "SolAI is your personal assistant designed to help you with various tasks and information.",
                        "category": "general",
                        "timestamp": "2024-01-01"
                    },
                    {
                        "id": 2,
                        "content": "You can ask me questions about general knowledge, technology, science, or any topic you're interested in.",
                        "category": "general",
                        "timestamp": "2024-01-01"
                    },
                    {
                        "id": 3,
                        "content": "The weather is typically determined by atmospheric conditions including temperature, humidity, and pressure.",
                        "category": "science",
                        "timestamp": "2024-01-01"
                    },
                    {
                        "id": 4,
                        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                        "category": "technology",
                        "timestamp": "2024-01-01"
                    }
                ],
                "suggestions": [],
                "next_id": 5
            }
            self.save_knowledge()
    
    def save_knowledge(self):
        with open(self.knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def update_vectorizer(self):
        contents = [entry['content'] for entry in self.data['entries']]
        if contents:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.vectors = self.vectorizer.fit_transform(contents)
        else:
            self.vectorizer = None
            self.vectors = None
    
    def add_knowledge(self, content, category="general"):
        new_entry = {
            "id": self.data['next_id'],
            "content": content,
            "category": category,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.data['entries'].append(new_entry)
        self.data['next_id'] += 1
        self.save_knowledge()
        self.update_vectorizer()
        return new_entry
    
    def add_suggestion(self, query):
        suggestion = {
            "query": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.data['suggestions'].append(suggestion)
        self.save_knowledge()
        return suggestion
    
    def find_best_match(self, query, threshold=0.3):
        if not self.data['entries'] or self.vectorizer is None:
            return None
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)
        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[0, max_sim_idx]
        
        if max_sim >= threshold:
            return self.data['entries'][max_sim_idx]
        return None
    
    def get_all_entries(self):
        return self.data['entries']
    
    def get_suggestions(self):
        return self.data['suggestions']

knowledge_base = KnowledgeBase()

# Basic responses for common greetings
GREETING_RESPONSES = {
    'hello': "Hello! I'm SolAI, your personal assistant. How can I help you today?",
    'hi': "Hi there! I'm SolAI. What would you like to know?",
    'hey': "Hey! Nice to see you. What can I assist you with?",
    'good morning': "Good morning! Ready to start the day? How can I help?",
    'good afternoon': "Good afternoon! How's your day going?",
    'good evening': "Good evening! Hope you had a great day. What can I do for you?",
    'how are you': "I'm functioning perfectly, thank you! How can I assist you today?",
    'whats up': "Not much, just here to help you! What's on your mind?"
}

def is_greeting(message):
    message_lower = message.lower().strip()
    for greeting in GREETING_RESPONSES:
        if greeting in message_lower:
            return greeting
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'response': 'Please enter a message.'})
        
        # Check for greetings
        greeting = is_greeting(message)
        if greeting:
            return jsonify({'response': GREETING_RESPONSES[greeting]})
        
        # Search knowledge base
        best_match = knowledge_base.find_best_match(message)
        
        if best_match:
            response = best_match['content']
        else:
            # Add to suggestions for learning
            knowledge_base.add_suggestion(message)
            response = "I'm still learning about this topic. I've noted your question and will improve my knowledge base. Is there anything else I can help you with?"
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'})

@app.route('/admin/verify', methods=['POST'])
def verify_admin():
    data = request.get_json()
    password = data.get('password', '')
    
    if password == 'Henley@2003':
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Invalid password'})

@app.route('/admin/knowledge', methods=['GET', 'POST', 'DELETE'])
def manage_knowledge():
    if request.method == 'GET':
        entries = knowledge_base.get_all_entries()
        suggestions = knowledge_base.get_suggestions()
        return jsonify({
            'entries': entries,
            'suggestions': suggestions
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        content = data.get('content', '').strip()
        category = data.get('category', 'general').strip()
        
        if not content:
            return jsonify({'success': False, 'error': 'Content cannot be empty'})
        
        entry = knowledge_base.add_knowledge(content, category)
        return jsonify({'success': True, 'entry': entry})
    
    elif request.method == 'DELETE':
        data = request.get_json()
        entry_id = data.get('id')
        
        knowledge_base.data['entries'] = [entry for entry in knowledge_base.data['entries'] if entry['id'] != entry_id]
        knowledge_base.save_knowledge()
        knowledge_base.update_vectorizer()
        
        return jsonify({'success': True})

@app.route('/admin/suggestions', methods=['DELETE'])
def clear_suggestions():
    knowledge_base.data['suggestions'] = []
    knowledge_base.save_knowledge()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
