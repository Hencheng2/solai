from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import threading

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
SECRET_PASSWORD = "Henley@2003"
DATABASE_FILE = "knowledge_base.json"
EMBEDDINGS_FILE = "embeddings.npy"

# Initialize knowledge base
if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'w') as f:
        json.dump([], f)

# Load or initialize embeddings
if os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
else:
    embeddings = np.array([])

# Load knowledge base
def load_knowledge_base():
    with open(DATABASE_FILE, 'r') as f:
        return json.load(f)

def save_knowledge_base(data):
    with open(DATABASE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def update_embeddings():
    global embeddings
    knowledge_base = load_knowledge_base()
    if knowledge_base:
        texts = [item['text'] for item in knowledge_base]
        embeddings = model.encode(texts)
        np.save(EMBEDDINGS_FILE, embeddings)
    else:
        embeddings = np.array([])

# Update embeddings on startup
update_embeddings()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'response': "I didn't get that. Could you please rephrase?"})
    
    knowledge_base = load_knowledge_base()
    
    if not knowledge_base or len(embeddings) == 0:
        # Save suggestion if no knowledge base
        suggestions = load_suggestions()
        suggestions.append({"query": user_message, "status": "pending"})
        save_suggestions(suggestions)
        return jsonify({'response': "I'm still learning! Your question has been noted for future improvement."})
    
    # Get embedding for user message
    query_embedding = model.encode([user_message])
    
    # Calculate similarities
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    # Threshold for considering a match
    if best_similarity > 0.5:
        response = knowledge_base[best_match_idx]['response']
    else:
        # Save as suggestion
        suggestions = load_suggestions()
        suggestions.append({"query": user_message, "status": "pending"})
        save_suggestions(suggestions)
        response = "I'm still learning! Your question has been noted for future improvement."
    
    return jsonify({'response': response})

@app.route('/api/admin/verify', methods=['POST'])
def verify_password():
    data = request.json
    password = data.get('password', '')
    if password == SECRET_PASSWORD:
        return jsonify({'success': True})
    return jsonify({'success': False}), 401

@app.route('/api/admin/knowledge', methods=['GET', 'POST'])
def manage_knowledge():
    if request.method == 'GET':
        knowledge_base = load_knowledge_base()
        suggestions = load_suggestions()
        return jsonify({
            'knowledge': knowledge_base,
            'suggestions': suggestions
        })
    
    elif request.method == 'POST':
        data = request.json
        text = data.get('text', '').strip()
        response = data.get('response', '').strip()
        
        if not text or not response:
            return jsonify({'success': False, 'error': 'Both text and response are required'}), 400
        
        knowledge_base = load_knowledge_base()
        knowledge_base.append({'text': text, 'response': response})
        save_knowledge_base(knowledge_base)
        
        # Update embeddings in background
        threading.Thread(target=update_embeddings).start()
        
        return jsonify({'success': True})

@app.route('/api/admin/suggestions/<int:index>', methods=['DELETE'])
def delete_suggestion(index):
    suggestions = load_suggestions()
    if 0 <= index < len(suggestions):
        suggestions.pop(index)
        save_suggestions(suggestions)
        return jsonify({'success': True})
    return jsonify({'success': False}), 404

def load_suggestions():
    if not os.path.exists('suggestions.json'):
        return []
    with open('suggestions.json', 'r') as f:
        return json.load(f)

def save_suggestions(data):
    with open('suggestions.json', 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
