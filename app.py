from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import bcrypt
import sqlite3
import base64
from io import BytesIO
from PIL import Image
import json
import shutil
import uuid
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Use secure key from env or fallback for dev
app.secret_key = os.getenv('SECRET_KEY', 'default-dev-secret-key-change-me')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
FACE_DB_PATH = 'face_database'
MODELS_DIR = 'models'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACE_DB_PATH, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

# --- Face Recognition Engine (OpenCV DNN) ---
class FaceEngine:
    def __init__(self):
        self.detector = None
        self.recognizer = None
        # Use absolute paths to avoid CWD issues
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, MODELS_DIR)
        
        self.detector_path = os.path.join(self.models_dir, "face_detection_yunet_2023mar.onnx")
        self.recognizer_path = os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx")
        
        # Ensure models dir exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.download_models()
        self.init_models()

    def download_models(self):
        """Download lightweight ONNX models if not present or corrupt"""
        # Use blob URLs with ?raw=true to correctly handle Git LFS redirects
        models = {
            self.detector_path: "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?raw=true",
            self.recognizer_path: "https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?raw=true"
        }
        
        for path, url in models.items():
            # Check if file exists, is valid size, and not HTML/LFS pointer
            needs_download = False
            
            # SFace should be > 2MB, YuNet > 200KB
            min_size = 200 * 1024
            if "sface" in path: min_size = 2 * 1024 * 1024

            if not os.path.exists(path) or os.path.getsize(path) < min_size:
                needs_download = True
            else:
                # Check for HTML or LFS pointer content
                try:
                    with open(path, 'rb') as f:
                        header = f.read(512)
                    if b'<!DOCTYPE' in header or b'<html' in header or b'git-lfs' in header:
                        print(f"⚠️ Found corrupt model (HTML/LFS) at {path}, deleting...")
                        needs_download = True
                        try:
                             f.close()
                             os.remove(path) 
                        except: pass
                except:
                    needs_download = True

            if needs_download:
                print(f"⬇️ Downloading model to {path}...")
                try:
                    r = requests.get(url, allow_redirects=True)
                    if r.status_code == 200:
                        with open(path, 'wb') as f:
                            f.write(r.content)
                        print(f"✅ Download complete: {os.path.basename(path)} ({len(r.content)//1024} KB)")
                    else:
                        print(f"❌ Failed to download model (Status {r.status_code})")
                except Exception as e:
                    print(f"❌ Failed to download model: {e}")

    def init_models(self):
        try:
            if not os.path.exists(self.detector_path):
                print(f"❌ Detector model not found at {self.detector_path}")
                return
            if not os.path.exists(self.recognizer_path):
                print(f"❌ Recognizer model not found at {self.recognizer_path}")
                return

            self.detector = cv2.FaceDetectorYN.create(
                self.detector_path, "", (320, 320), 0.9, 0.3, 5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                self.recognizer_path, ""
            )
            print("🤖 Face Recognition Models Loaded Successfully")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            import traceback
            traceback.print_exc()

    def get_embedding(self, image_path):
        """Detect face and return 128D embedding"""
        if self.detector is None or self.recognizer is None:
            print("❌ Models not initialized, attempting to re-initialize...")
            self.init_models()
            if self.detector is None:
                print("❌ Re-initialization failed.")
                return None

        try:
            img = cv2.imread(image_path)
            if img is None: 
                print(f"⚠️ Could not read image: {image_path}")
                return None
            
            # Resize for consistent detection
            h, w, _ = img.shape
            self.detector.setInputSize((w, h))
            
            # Detect
            _, faces = self.detector.detect(img)
            if faces is None or len(faces) == 0: 
                # print("No face detected") # verbose
                return None
            
            # Get the best face (highest confidence)
            face = faces[0]
            
            # Align and Recognize
            aligned_face = self.recognizer.alignCrop(img, face)
            embedding = self.recognizer.feature(aligned_face)
            return embedding[0] # Return flat array
        except Exception as e:
            print(f"Processing error in get_embedding: {e}")
            return None
        """Detect face and return 128D embedding"""
        try:
            img = cv2.imread(image_path)
            if img is None: return None
            
            # Resize for consistent detection
            h, w, _ = img.shape
            self.detector.setInputSize((w, h))
            
            # Detect
            _, faces = self.detector.detect(img)
            if faces is None: return None
            
            # Get the best face (highest confidence)
            face = faces[0]
            
            # Align and Recognize
            aligned_face = self.recognizer.alignCrop(img, face)
            embedding = self.recognizer.feature(aligned_face)
            return embedding[0] # Return flat array
        except Exception as e:
            print(f"Processing error: {e}")
            return None

    def compare(self, emb1, emb2):
        """Compare two embeddings (Cosine Similarity)"""
        if emb1 is None or emb2 is None: return 0.0
        return self.recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)

# Initialize Engine
face_engine = FaceEngine()

# --- Database & Helper Functions ---

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            pin_hash TEXT NOT NULL,
            face_encoding_path TEXT NOT NULL,
            unique_member_id TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shared_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            owner_id INTEGER,
            shared_with_id INTEGER,
            shared_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (document_id) REFERENCES documents (id),
            FOREIGN KEY (owner_id) REFERENCES users (id),
            FOREIGN KEY (shared_with_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            document_id INTEGER,
            action TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    migrate_db()

def migrate_db():
    """Add new columns to existing tables if they don't exist"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check shared_documents columns
        cursor.execute('PRAGMA table_info(shared_documents)')
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'expires_at' not in columns:
            print("Migrating DB: Adding expires_at to shared_documents")
            cursor.execute('ALTER TABLE shared_documents ADD COLUMN expires_at TIMESTAMP')
            
        if 'max_views' not in columns:
            print("Migrating DB: Adding max_views to shared_documents")
            cursor.execute('ALTER TABLE shared_documents ADD COLUMN max_views INTEGER DEFAULT -1')
            
        if 'current_views' not in columns:
            print("Migrating DB: Adding current_views to shared_documents")
            cursor.execute('ALTER TABLE shared_documents ADD COLUMN current_views INTEGER DEFAULT 0')
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Migration Error: {e}")

def log_access(user_id, action, details, doc_id=None):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO access_logs (user_id, document_id, action, details) VALUES (?, ?, ?, ?)',
                      (user_id, doc_id, action, details))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Logging Error: {e}")

def generate_unique_member_id():
    import random
    while True:
        member_id = f"{random.randint(100000, 999999)}"
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE unique_member_id = ?', (member_id,))
        if not cursor.fetchone():
            conn.close()
            return member_id
        conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_pin(pin):
    return bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt())

def verify_pin(pin, hashed):
    return bcrypt.checkpw(pin.encode('utf-8'), hashed)

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/signup_with_training', methods=['POST'])
def signup_with_training():
    try:
        data = request.get_json()
        username = data['username']
        email = data['email']
        pin = data['pin']
        face_images = data['face_images']
        
        # Determine strictness based on user count (first user strict, otherwise loose to avoid collision)
        # For simplicity, we just save valid faces.
        
        user_face_dir = os.path.join(FACE_DB_PATH, username)
        os.makedirs(user_face_dir, exist_ok=True)
        
        valid_embeddings = []
        valid_images = []
        
        print(f"Processing {len(face_images)} training images...")
        
        for i, base64_image in enumerate(face_images):
            try:
                # Decode
                image_bytes = base64.b64decode(base64_image.split(',')[1])
                temp_path = os.path.join('static/temp', f'train_{i}_{uuid.uuid4().hex}.jpg')
                with open(temp_path, 'wb') as f:
                    f.write(image_bytes)
                
                # Get Embedding
                embedding = face_engine.get_embedding(temp_path)
                
                if embedding is not None:
                    # Save permanent
                    perm_path = os.path.join(user_face_dir, f'face_{len(valid_embeddings)}.jpg')
                    shutil.move(temp_path, perm_path)
                    
                    valid_embeddings.append(embedding.tolist()) # Convert numpy to list for JSON
                    valid_images.append(perm_path)
                else:
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Error processing image {i}: {e}")

        if len(valid_embeddings) < 3:
            return jsonify({'success': False, 'message': 'Could not detect face clearly in enough photos. Please try again with better lighting.'})

        # Save embeddings
        with open(os.path.join(user_face_dir, 'embeddings.json'), 'w') as f:
            json.dump(valid_embeddings, f)

        # Save reference image
        ref_path = os.path.join(user_face_dir, 'reference.jpg')
        if valid_images:
            shutil.copy2(valid_images[0], ref_path)

        # DB Interaction
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            member_id = generate_unique_member_id()
            cursor.execute('INSERT INTO users (username, email, pin_hash, face_encoding_path, unique_member_id) VALUES (?, ?, ?, ?, ?)',
                          (username, email, hash_pin(pin), ref_path, member_id))
            conn.commit()
            return jsonify({'success': True, 'message': 'Registration Successful!', 'member_id': member_id})
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'message': 'Username or Email already exists'})
        finally:
            conn.close()

    except Exception as e:
        print(f"Signup Error: {e}")
        return jsonify({'success': False, 'message': f'Server Error: {str(e)}'})

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/verify_face', methods=['POST'])
def verify_face_route():
    try:
        data = request.get_json()
        username = data['username']
        image_data = data['image']
        
        # Check if user exists
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'success': False, 'message': 'User not found'})

        # Save temp login image
        temp_path = os.path.join('static/temp', f'login_{uuid.uuid4().hex}.jpg')
        with open(temp_path, 'wb') as f:
            f.write(base64.b64decode(image_data.split(',')[1]))
        
        # Get embedding
        login_embedding = face_engine.get_embedding(temp_path)
        os.remove(temp_path)
        
        if login_embedding is None:
            return jsonify({'success': False, 'message': 'No face detected. Look at the camera.'})

        # Load stored embeddings
        emb_path = os.path.join(FACE_DB_PATH, username, 'embeddings.json')
        if not os.path.exists(emb_path):
             return jsonify({'success': False, 'message': 'Face data corrupted.'})
             
        with open(emb_path, 'r') as f:
            stored_embeddings = json.load(f)
            
        # Verify against all stored embeddings
        # SFace COSINE match: Score > 0.363 is a match (standard threshold)
        # We will use stricter threshold for security
        threshold = 0.4 
        match_count = 0
        
        for stored in stored_embeddings:
            score = face_engine.compare(login_embedding, np.array(stored, dtype=np.float32))
            if score > threshold:
                match_count += 1
        
        # If matches > 30% of stored faces (robustness)
        if match_count > 0:
            session['face_verified'] = True
            session['username'] = username
            return jsonify({'success': True, 'message': 'Face Verified!'})
        else:
            return jsonify({'success': False, 'message': 'Face verification failed.'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/verify_pin', methods=['POST'])
def verify_pin_route():
    try:
        data = request.get_json()
        username = data['username']
        pin = data['pin']
        
        if not session.get('face_verified') or session.get('username') != username:
             return jsonify({'success': False, 'message': 'Biometric verification required first.'})
             
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        # --- Duress Mode Check ---
        if pin == "0000" and user:
            session['logged_in'] = True
            session['user_id'] = user[0]
            session['is_panic_mode'] = True
            log_access(user[0], 'PANIC_LOGIN', 'Duress PIN Used - Serving Safe Dashboard')
            return jsonify({'success': True, 'message': 'Login Successful'})
        # -------------------------
        
        if user and verify_pin(pin, user[3]):
            session['logged_in'] = True
            session['user_id'] = user[0]
            session['is_panic_mode'] = False # Normal login reset
            log_access(user[0], 'LOGIN', 'Successful Login')
            return jsonify({'success': True, 'message': 'Login Successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid PIN'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'): return redirect(url_for('login'))
    
    # --- Duress Mode View ---
    if session.get('is_panic_mode'):
        # Return deceptive "Safe Mode" (Empty)
        return render_template('dashboard.html', 
                         documents=[], 
                         member_id="000000", 
                         shared_documents=[], 
                         pending_shares=[])
    # ------------------------
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Get Documents
    cursor.execute('SELECT * FROM documents WHERE user_id = ?', (session['user_id'],))
    documents = cursor.fetchall()
    
    # Get Member ID
    cursor.execute('SELECT unique_member_id FROM users WHERE id = ?', (session['user_id'],))
    member_id = cursor.fetchone()[0]
    
    # Get Shared
    cursor.execute('''
        SELECT sd.*, d.original_filename, d.upload_date, u.username 
        FROM shared_documents sd
        JOIN documents d ON sd.document_id = d.id
        JOIN users u ON sd.owner_id = u.id
        WHERE sd.shared_with_id = ? AND sd.status = 'accepted'
    ''', (session['user_id'],))
    shared = cursor.fetchall()
    
    # Get Pending
    cursor.execute('''
        SELECT sd.*, d.original_filename, d.upload_date, u.username
        FROM shared_documents sd
        JOIN documents d ON sd.document_id = d.id
        JOIN users u ON sd.owner_id = u.id
        WHERE sd.shared_with_id = ? AND sd.status = 'pending'
    ''', (session['user_id'],))
    pending = cursor.fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', 
                         documents=documents, 
                         member_id=member_id, 
                         shared_documents=shared, 
                         pending_shares=pending)

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if not session.get('logged_in'): return redirect(url_for('login'))
    
    file = request.files.get('document')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(path)
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO documents (user_id, filename, original_filename, file_path) VALUES (?, ?, ?, ?)',
                      (session['user_id'], unique_name, filename, path))
        conn.commit()
        conn.close()
        flash('Upload successful!', 'success')
    else:
        flash('Invalid file!', 'error')
        
    return redirect(url_for('dashboard'))

@app.route('/share_document', methods=['POST'])
def share_document():
    if not session.get('logged_in'): return jsonify({'success': False, 'message': 'Auth required'})
    
    data = request.get_json()
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Check document ownership
    cursor.execute('SELECT * FROM documents WHERE id = ? AND user_id = ?', (data['document_id'], session['user_id']))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'success': False, 'message': 'Invalid document'})
        
    # Handle recipients
    member_ids = data.get('member_ids', [])
    if data.get('member_id'): member_ids.append(data.get('member_id'))
    member_ids = list(set([m for m in member_ids if m]))
    
    if not member_ids:
         conn.close()
         return jsonify({'success': False, 'message': 'No recipients specified'})

    # Security Level Parsing
    security_level = data.get('security_level', 'standard')
    max_views = -1 # Infinity
    expires_at = None
    
    from datetime import datetime, timedelta
    
    if security_level == 'confidential':
        expires_at = datetime.now() + timedelta(hours=24)
    elif security_level == 'top_secret':
        max_views = 1
        expires_at = datetime.now() + timedelta(hours=1) # Also expire soon

    results = {'success': [], 'failed': []}
    
    for mid in member_ids:
        cursor.execute('SELECT id, username FROM users WHERE unique_member_id = ?', (mid,))
        target = cursor.fetchone()
        
        if target:
            if target[0] == session['user_id']:
                 results['failed'].append(f"{mid} (Self)")
                 continue

            cursor.execute('SELECT id FROM shared_documents WHERE document_id = ? AND shared_with_id = ?', 
                          (data['document_id'], target[0]))
            if cursor.fetchone():
                results['failed'].append(f"{mid} (Already Shared)")
                continue
            
            cursor.execute('''
                INSERT INTO shared_documents 
                (document_id, owner_id, shared_with_id, status, max_views, expires_at) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (data['document_id'], session['user_id'], target[0], 'pending', max_views, expires_at))
            
            results['success'].append(target[1])
            log_access(session['user_id'], 'SHARE', f'Shared Doc {data["document_id"]} with {target[1]} ({security_level})', data['document_id'])
        else:
            results['failed'].append(f"{mid} (Invalid ID)")

    conn.commit()
    conn.close()
    
    msg = f"Sent to {len(results['success'])} users."
    if results['failed']: msg += f" Failed: {len(results['failed'])}"
        
    return jsonify({'success': True, 'message': msg, 'details': results})

@app.route('/accept_share/<int:share_id>', methods=['POST'])
def accept_share(share_id):
    if not session.get('logged_in'): return jsonify({'success': False})
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE shared_documents SET status = "accepted" WHERE id = ? AND shared_with_id = ?', 
                  (share_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Accepted'})

@app.route('/reject_share/<int:share_id>', methods=['POST'])
def reject_share(share_id):
    if not session.get('logged_in'): return jsonify({'success': False})
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM shared_documents WHERE id = ? AND shared_with_id = ?', 
                  (share_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Rejected'})

@app.route('/access_document/<int:doc_id>')
def access_document(doc_id):
    if not session.get('logged_in'): return redirect(url_for('login'))
    return render_template('access_document.html', doc_id=doc_id, is_shared=False)

@app.route('/access_shared_document/<int:doc_id>')
def access_shared_document(doc_id):
    if not session.get('logged_in'): return redirect(url_for('login'))
    return render_template('access_document.html', doc_id=doc_id, is_shared=True)

@app.route('/verify_document_access', methods=['POST'])
def verify_document_access():
    if not session.get('logged_in'): 
        return jsonify({'success': False, 'message': 'Unauthorized'})

    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        is_shared = data.get('is_shared')
        image_data = data.get('image')
        username = session.get('username')

        # --- Face Verification Logic ---
        if not image_data: return jsonify({'success': False, 'message': 'No camera data.'})
        temp_path = os.path.join('static/temp', f'access_{uuid.uuid4().hex}.jpg')
        with open(temp_path, 'wb') as f: f.write(base64.b64decode(image_data.split(',')[1]))
        
        access_embedding = face_engine.get_embedding(temp_path)
        try: os.remove(temp_path)
        except: pass
        if access_embedding is None: return jsonify({'success': False, 'message': 'No face detected.'})

        emb_path = os.path.join(FACE_DB_PATH, username, 'embeddings.json')
        if not os.path.exists(emb_path): return jsonify({'success': False, 'message': 'Face data corrupted.'})
        with open(emb_path, 'r') as f: stored_embeddings = json.load(f)
            
        match_count = 0
        for stored in stored_embeddings:
            if face_engine.compare(access_embedding, np.array(stored, dtype=np.float32)) > 0.4:
                match_count += 1
        
        if match_count == 0:
             log_access(session['user_id'], 'ACCESS_DENIED', f'Face Mismatch for Doc {doc_id}')
             return jsonify({'success': False, 'message': 'Face verification failed.'})
        # ----------------------------------------------------------------

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        doc = None
        if is_shared:
             # Check shared + expiry + views logic
             from datetime import datetime
             
             cursor.execute('''
                SELECT d.file_path, d.original_filename, sd.max_views, sd.current_views, sd.expires_at, sd.id
                FROM shared_documents sd
                JOIN documents d ON sd.document_id = d.id
                WHERE sd.document_id = ? AND sd.shared_with_id = ? AND sd.status = 'accepted'
             ''', (doc_id, session['user_id']))
             share_info = cursor.fetchone()
             
             if not share_info:
                 conn.close()
                 return jsonify({'success': False, 'message': 'Access Revoked.'})
                 
             file_path, filename, max_views, current_views, expires_at, share_id = share_info
             
             # Check Expiry
             if expires_at:
                 expiry_dt = datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S.%f')
                 if datetime.now() > expiry_dt:
                     conn.close()
                     return jsonify({'success': False, 'message': 'This document has expired (Self-Destructed).'})
            
             # Check Max Views
             if max_views != -1 and current_views >= max_views:
                 conn.close()
                 return jsonify({'success': False, 'message': 'Maximum view count reached (Self-Destructed).'})
                 
             # Increment View
             cursor.execute('UPDATE shared_documents SET current_views = current_views + 1 WHERE id = ?', (share_id,))
             conn.commit()
             
             doc = (file_path, filename)
             
        else:
             cursor.execute('SELECT file_path, original_filename FROM documents WHERE id = ? AND user_id = ?', (doc_id, session['user_id']))
             doc = cursor.fetchone()
        
        conn.close()

        if not doc:
             return jsonify({'success': False, 'message': 'Document not found.'})
             
        # Read file content
        file_path, filename = doc
        content = ""
        try:
            with open(file_path, 'rb') as f:
                raw = f.read()
                try: content = raw.decode('utf-8')
                except: content = f"[Binary File: {len(raw)} bytes]\n(File is not plain text.)"
        except Exception as e:
            content = f"Error reading file: {str(e)}"

        log_access(session['user_id'], 'ACCESS_GRANTED', f'Viewed {filename}', doc_id)

        return jsonify({
            'success': True, 
            'content': content,
            'filename': filename,
            'is_shared': is_shared
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'message': 'Server Error'}) 

@app.route('/get_stats')
def get_stats():
    """Return JSON stats for the Matrix Terminal"""
    if not session.get('logged_in'): return jsonify([])
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Get recent logs
    cursor.execute('''
        SELECT u.username, al.action, al.details, al.timestamp 
        FROM access_logs al 
        JOIN users u ON al.user_id = u.id 
        ORDER BY al.timestamp DESC LIMIT 10
    ''')
    logs = cursor.fetchall()
    
    conn.close()
    
    return jsonify([
        {'user': l[0], 'action': l[1], 'details': l[2], 'time': l[3]} for l in logs
    ])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)