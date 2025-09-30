from flask import Flask, request, jsonify
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

app = Flask(__name__)

# Initialize the semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')

# File paths
KNOWLEDGE_FILE = 'knowledge_base.json'
SUGGESTIONS_FILE = 'suggestions.json'

class KnowledgeManager:
    def __init__(self):
        self.knowledge_base = self.load_knowledge()
        self.suggestions = self.load_suggestions()
        self.update_embeddings()
    
    def load_knowledge(self):
        if os.path.exists(KNOWLEDGE_FILE):
            try:
                with open(KNOWLEDGE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return self.get_default_knowledge()
        return self.get_default_knowledge()
    
    def get_default_knowledge(self):
        return [
            {
                "id": 1,
                "title": "Welcome",
                "content": "Hello! I'm SolAI, your personal assistant. How can I help you today?",
                "tags": ["greeting", "welcome"],
                "created_at": datetime.now().isoformat()
            },
            {
                "id": 2,
                "title": "About SolAI",
                "content": "SolAI is a smart chatbot that learns from our conversations and helps you find information quickly.",
                "tags": ["about", "purpose"],
                "created_at": datetime.now().isoformat()
            },
            {
                "id": 3,
                "title": "Goodbye",
                "content": "Goodbye! Feel free to come back anytime you need help.",
                "tags": ["farewell", "closing"],
                "created_at": datetime.now().isoformat()
            },
            {
                "id": 4,
                "title": "Thanks",
                "content": "You're welcome! I'm happy to help.",
                "tags": ["gratitude", "response"],
                "created_at": datetime.now().isoformat()
            }
        ]
    
    def load_suggestions(self):
        if os.path.exists(SUGGESTIONS_FILE):
            try:
                with open(SUGGESTIONS_FILE, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_knowledge(self):
        with open(KNOWLEDGE_FILE, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
    
    def save_suggestions(self):
        with open(SUGGESTIONS_FILE, 'w') as f:
            json.dump(self.suggestions, f, indent=2)
    
    def update_embeddings(self):
        texts = [f"{item['title']} {item['content']}" for item in self.knowledge_base]
        if texts:
            self.embeddings = model.encode(texts)
        else:
            self.embeddings = np.array([])
    
    def find_best_match(self, query, threshold=0.3):
        if not self.knowledge_base:
            return None, 0
        
        query_embedding = model.encode([query])[0]
        similarities = []
        
        for emb in self.embeddings:
            similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append(similarity)
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity > threshold:
            return self.knowledge_base[best_idx], best_similarity
        return None, best_similarity
    
    def add_knowledge(self, title, content, tags=None):
        new_id = max([item.get('id', 0) for item in self.knowledge_base], default=0) + 1
        new_item = {
            "id": new_id,
            "title": title,
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now().isoformat()
        }
        self.knowledge_base.append(new_item)
        self.save_knowledge()
        self.update_embeddings()
        return new_item
    
    def add_suggestion(self, query):
        new_id = max([item.get('id', 0) for item in self.suggestions], default=0) + 1
        suggestion = {
            "id": new_id,
            "query": query,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        self.suggestions.append(suggestion)
        self.save_suggestions()
        return suggestion
    
    def delete_suggestion(self, suggestion_id):
        self.suggestions = [s for s in self.suggestions if s['id'] != suggestion_id]
        self.save_suggestions()

# Initialize knowledge manager
knowledge_manager = KnowledgeManager()

# Password for admin access
ADMIN_PASSWORD = "Henley@2003"

@app.route('/')
def index():
    # Serve the HTML file
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Check for exact matches first
        lower_message = message.lower()
        if any(word in lower_message for word in ['hello', 'hi', 'hey']):
            return jsonify({
                'response': "Hello! How can I help you today?",
                'match_found': True,
                'similarity': 1.0
            })
        
        if any(word in lower_message for word in ['thank', 'thanks']):
            return jsonify({
                'response': "You're welcome! Is there anything else I can help with?",
                'match_found': True,
                'similarity': 1.0
            })
        
        if any(word in lower_message for word in ['bye', 'goodbye']):
            return jsonify({
                'response': "Goodbye! Feel free to come back if you have more questions.",
                'match_found': True,
                'similarity': 1.0
            })
        
        # Semantic search
        best_match, similarity = knowledge_manager.find_best_match(message)
        
        if best_match:
            return jsonify({
                'response': best_match['content'],
                'match_found': True,
                'similarity': float(similarity),
                'source': best_match['title']
            })
        else:
            # Add to suggestions
            knowledge_manager.add_suggestion(message)
            return jsonify({
                'response': "I'm still learning about this topic. I've noted your question and will try to learn more about it soon!",
                'match_found': False,
                'similarity': float(similarity)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    password = data.get('password', '')
    
    if password == ADMIN_PASSWORD:
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid password'}), 401

@app.route('/api/admin/knowledge', methods=['GET'])
def get_knowledge():
    return jsonify(knowledge_manager.knowledge_base)

@app.route('/api/admin/knowledge', methods=['POST'])
def add_knowledge():
    data = request.get_json()
    title = data.get('title', '').strip()
    content = data.get('content', '').strip()
    
    if not title or not content:
        return jsonify({'error': 'Title and content are required'}), 400
    
    new_item = knowledge_manager.add_knowledge(title, content)
    return jsonify(new_item)

@app.route('/api/admin/suggestions', methods=['GET'])
def get_suggestions():
    return jsonify(knowledge_manager.suggestions)

@app.route('/api/admin/suggestions/<int:suggestion_id>', methods=['DELETE'])
def delete_suggestion(suggestion_id):
    knowledge_manager.delete_suggestion(suggestion_id)
    return jsonify({'success': True})

if __name__ == '__main__':
    print("Starting SolAI Chat Bot...")
    print("Access the application at: http://localhost:5000")
    print("Admin password: Henley@2003")
    app.run(debug=True, host='0.0.0.0', port=5000)
