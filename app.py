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
from deepface import DeepFace
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
FACE_DB_PATH = 'face_database'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACE_DB_PATH, exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

def generate_unique_member_id():
    """Generate a unique 6-digit member ID"""
    import random
    while True:
        # Generate 6-digit number
        member_id = f"{random.randint(100000, 999999)}"
        
        # Check if it already exists
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE unique_member_id = ?', (member_id,))
        exists = cursor.fetchone()
        conn.close()
        
        if not exists:
            return member_id

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            pin_hash TEXT NOT NULL,
            face_encoding_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if unique_member_id column exists, if not add it
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'unique_member_id' not in columns:
        print("Adding unique_member_id column to users table...")
        # Add column without UNIQUE constraint first
        cursor.execute('ALTER TABLE users ADD COLUMN unique_member_id TEXT')
        
        # Generate unique member IDs for existing users
        cursor.execute('SELECT id FROM users WHERE unique_member_id IS NULL')
        users_without_id = cursor.fetchall()
        
        for user in users_without_id:
            member_id = generate_unique_member_id()
            cursor.execute('UPDATE users SET unique_member_id = ? WHERE id = ?', 
                          (member_id, user[0]))
        
        # Now create a unique index
        try:
            cursor.execute('CREATE UNIQUE INDEX idx_unique_member_id ON users(unique_member_id)')
            print(f"Generated member IDs for {len(users_without_id)} existing users and created unique index")
        except sqlite3.OperationalError as e:
            if "already exists" not in str(e):
                print(f"Index creation warning: {e}")
            else:
                print(f"Generated member IDs for {len(users_without_id)} existing users")
    
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
    
    conn.commit()
    conn.close()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_pin(pin):
    """Hash PIN using bcrypt"""
    return bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt())

def verify_pin(pin, hashed):
    """Verify PIN against hash"""
    return bcrypt.checkpw(pin.encode('utf-8'), hashed)



def get_face_embedding(image_path):
    """Get face embedding using DeepFace with ArcFace backend - strict face detection"""
    try:
        # Use enforce_detection=True for training to ensure face is detected
        # This gives better quality embeddings
        embedding_obj = DeepFace.represent(
            img_path=image_path, 
            model_name="ArcFace", 
            enforce_detection=True,  # Strict detection for better accuracy
            detector_backend='opencv'  # Use opencv for better detection
        )
        return embedding_obj[0]["embedding"]
    except Exception as e:
        print(f"Face not detected or error in {image_path}: {e}")
        return None

def get_face_embedding_lenient(image_path):
    """Get face embedding with lenient detection for verification (backup)"""
    try:
        embedding_obj = DeepFace.represent(
            img_path=image_path, 
            model_name="ArcFace", 
            enforce_detection=False,
            detector_backend='opencv'
        )
        return embedding_obj[0]["embedding"]
    except Exception as e:
        print(f"Error getting embedding for {image_path}: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    try:
        a, b = np.array(vec1), np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except:
        return 0.0

def save_face_encoding(username, image_paths):
    """Save face encoding using DeepFace with multiple training images"""
    try:
        user_face_dir = os.path.join(FACE_DB_PATH, username)
        os.makedirs(user_face_dir, exist_ok=True)
        
        # Save all training images and get embeddings
        valid_embeddings = []
        valid_images = []
        
        for i, image_path in enumerate(image_paths):
            face_image_path = os.path.join(user_face_dir, f'training_{i}.jpg')
            
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    cv2.imwrite(face_image_path, img)
                    
                    # Get embedding (won't fail even if face not perfectly detected)
                    embedding = get_face_embedding(face_image_path)
                    if embedding is not None:
                        valid_embeddings.append(embedding)
                        valid_images.append(face_image_path)
                        print(f"Successfully processed training image {i}")
                    else:
                        print(f"Could not get embedding for training image {i}")
                        try:
                            os.remove(face_image_path)
                        except:
                            pass
        
        if len(valid_embeddings) < 3:  # Need at least 3 valid embeddings
            print(f"Only {len(valid_embeddings)} valid embeddings found, need at least 3")
            return None
        
        # Save embeddings to JSON file
        embeddings_file = os.path.join(user_face_dir, 'embeddings.json')
        with open(embeddings_file, 'w') as f:
            json.dump(valid_embeddings, f)
            
        # Save the first valid image as reference
        reference_path = os.path.join(user_face_dir, 'reference.jpg')
        if valid_images:
            shutil.copy2(valid_images[0], reference_path)
            
        print(f"Successfully saved {len(valid_embeddings)} embeddings for {username}")
        return reference_path
            
    except Exception as e:
        print(f"Error saving face encoding: {e}")
        return None

def save_face_training_from_base64(username, base64_images):
    """Save multiple face training images from base64 data with strict quality control"""
    try:
        user_face_dir = os.path.join(FACE_DB_PATH, username)
        os.makedirs(user_face_dir, exist_ok=True)
        
        valid_embeddings = []
        valid_images = []
        temp_paths = []
        
        print(f"Processing {len(base64_images)} images for {username}")
        
        for i, base64_image in enumerate(base64_images):
            try:
                # Decode base64 image
                image_data = base64_image.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                # Save temporary image
                temp_path = os.path.join('static/temp', f'training_{username}_{i}_{uuid.uuid4().hex}.jpg')
                with open(temp_path, 'wb') as f:
                    f.write(image_bytes)
                temp_paths.append(temp_path)
                
                # Get embedding with STRICT detection (face must be clearly visible)
                embedding = get_face_embedding(temp_path)
                if embedding is not None:
                    # Additional quality check: ensure embedding is not too similar to existing ones
                    # This prevents duplicate/poor quality images
                    is_unique = True
                    for existing_emb in valid_embeddings:
                        similarity = cosine_similarity(embedding, existing_emb)
                        if similarity > 0.95:  # Too similar to existing image
                            is_unique = False
                            print(f"Image {i+1} too similar to existing training image, skipping")
                            break
                    
                    if is_unique:
                        # Save to permanent location
                        permanent_path = os.path.join(user_face_dir, f'training_{len(valid_embeddings)}.jpg')
                        img = cv2.imread(temp_path)
                        cv2.imwrite(permanent_path, img)
                        
                        valid_embeddings.append(embedding)
                        valid_images.append(permanent_path)
                        print(f"Successfully processed image {i+1}/{len(base64_images)} (Valid: {len(valid_embeddings)})")
                else:
                    print(f"No clear face detected in image {i+1}, skipping...")
                    
            except Exception as e:
                print(f"Error processing image {i}: {e}, continuing...")
        
        # Clean up temporary files
        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
            except:
                pass
        
        # Require at least 5 high-quality embeddings for robust recognition
        if len(valid_embeddings) < 5:
            print(f"Only {len(valid_embeddings)} valid embeddings found, need at least 5")
            # Clean up saved images if not enough valid ones
            for img_path in valid_images:
                try:
                    os.remove(img_path)
                except:
                    pass
            return None
        
        # Save embeddings to JSON file
        embeddings_file = os.path.join(user_face_dir, 'embeddings.json')
        with open(embeddings_file, 'w') as f:
            json.dump(valid_embeddings, f)
            
        # Save the first valid image as reference
        reference_path = os.path.join(user_face_dir, 'reference.jpg')
        if valid_images:
            shutil.copy2(valid_images[0], reference_path)
            
        print(f"Successfully saved {len(valid_embeddings)} high-quality embeddings for {username}")
        return reference_path
            
    except Exception as e:
        print(f"Error saving face training: {e}")
        return None

def verify_face(username, captured_image_path):
    """Verify face using DeepFace with STRICT cosine similarity threshold"""
    try:
        user_face_dir = os.path.join(FACE_DB_PATH, username)
        embeddings_file = os.path.join(user_face_dir, 'embeddings.json')
        
        if not os.path.exists(embeddings_file):
            print(f"No embeddings file found for {username}")
            return False
        
        # Load stored embeddings
        with open(embeddings_file, 'r') as f:
            stored_embeddings = json.load(f)
        
        if not stored_embeddings:
            print(f"No stored embeddings for {username}")
            return False
        
        # Get embedding for captured image with STRICT detection
        test_embedding = get_face_embedding(captured_image_path)
        
        # If strict detection fails, try lenient (but will require higher threshold)
        use_strict = True
        if test_embedding is None:
            print("Strict detection failed, trying lenient detection...")
            test_embedding = get_face_embedding_lenient(captured_image_path)
            use_strict = False
            
        if test_embedding is None:
            print("Could not get embedding from captured image")
            return False
        
        # Calculate similarities with all stored embeddings
        similarities = []
        for stored_embedding in stored_embeddings:
            similarity = cosine_similarity(test_embedding, stored_embedding)
            similarities.append(similarity)
        
        # Get best match and average of top 3 matches
        similarities.sort(reverse=True)
        best_similarity = similarities[0] if similarities else 0.0
        
        # Use top 3 average for more robust verification
        top_3_avg = sum(similarities[:3]) / min(3, len(similarities)) if similarities else 0.0
        
        print(f"Verification for {username}:")
        print(f"  Best match: {best_similarity:.4f}")
        print(f"  Top 3 average: {top_3_avg:.4f}")
        print(f"  Detection mode: {'Strict' if use_strict else 'Lenient'}")
        
        # STRICT thresholds for accurate face recognition
        if use_strict:
            # Strict detection requires: best match > 0.50 AND top 3 avg > 0.45
            threshold_best = 0.50  # Much stricter than 0.35
            threshold_avg = 0.45
            verified = best_similarity > threshold_best and top_3_avg > threshold_avg
        else:
            # Lenient detection requires even higher thresholds
            threshold_best = 0.60
            threshold_avg = 0.55
            verified = best_similarity > threshold_best and top_3_avg > threshold_avg
        
        if verified:
            print(f"✅ Face VERIFIED for {username}")
        else:
            print(f"❌ Face REJECTED for {username} (thresholds not met)")
            
        return verified
        
    except Exception as e:
        print(f"Face verification error: {e}")
        return False

def get_user_by_username(username):
    """Get user by username"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_documents(user_id):
    """Get documents for a user"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE user_id = ?', (user_id,))
    documents = cursor.fetchall()
    conn.close()
    return documents

def get_user_by_member_id(member_id):
    """Get user by unique member ID"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE unique_member_id = ?', (member_id,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_shared_documents_for_user(user_id):
    """Get documents shared with a user"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sd.*, d.original_filename, d.upload_date, u.username as owner_username
        FROM shared_documents sd
        JOIN documents d ON sd.document_id = d.id
        JOIN users u ON sd.owner_id = u.id
        WHERE sd.shared_with_id = ? AND sd.status = 'accepted'
    ''', (user_id,))
    documents = cursor.fetchall()
    conn.close()
    return documents

def get_pending_shares(user_id):
    """Get pending document shares for a user"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sd.*, d.original_filename, u.username as owner_username
        FROM shared_documents sd
        JOIN documents d ON sd.document_id = d.id
        JOIN users u ON sd.owner_id = u.id
        WHERE sd.shared_with_id = ? AND sd.status = 'pending'
    ''', (user_id,))
    shares = cursor.fetchall()
    conn.close()
    return shares

def get_user_member_id(username):
    """Get user's unique member ID"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT unique_member_id FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')

@app.route('/signup_with_training', methods=['POST'])
def signup_with_training():
    """Handle signup with live face training"""
    try:
        data = request.get_json()
        username = data['username']
        email = data['email']
        pin = data['pin']
        face_images = data['face_images']
        
        # Validate input
        if not all([username, email, pin, face_images]):
            return jsonify({'success': False, 'message': 'Missing required fields'})
        
        if len(pin) != 4 or not pin.isdigit():
            return jsonify({'success': False, 'message': 'PIN must be 4 digits'})
        
        if len(face_images) < 5:
            return jsonify({'success': False, 'message': 'Need at least 5 face images for training'})
        
        # Check if user already exists
        if get_user_by_username(username):
            return jsonify({'success': False, 'message': 'Username already exists!'})
        
        # Save face training images with strict quality control
        face_path = save_face_training_from_base64(username, face_images)
        
        if face_path:
            # Hash PIN
            pin_hash = hash_pin(pin)
            
            # Generate unique member ID
            unique_member_id = generate_unique_member_id()
            
            # Save to database
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (username, email, pin_hash, face_encoding_path, unique_member_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, email, pin_hash, face_path, unique_member_id))
                conn.commit()
                return jsonify({
                    'success': True, 
                    'message': f'Registration successful! Your unique member ID is: {unique_member_id}. Please save it for document sharing.',
                    'member_id': unique_member_id
                })
            except sqlite3.IntegrityError:
                return jsonify({'success': False, 'message': 'Username or email already exists!'})
            finally:
                conn.close()
        else:
            return jsonify({'success': False, 'message': 'Face training failed. Not enough clear face images detected. Please ensure your face is well-lit and directly facing the camera. Try again with better lighting.'})
            
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/verify_face', methods=['POST'])
def verify_face_login():
    try:
        data = request.get_json()
        username = data['username']
        image_data = data['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Save temporary image
        temp_path = os.path.join('static/temp', f'login_{username}_{uuid.uuid4().hex}.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        # Verify face
        if verify_face(username, temp_path):
            session['face_verified'] = True
            session['username'] = username
            os.remove(temp_path)
            return jsonify({'success': True, 'message': 'Face verified successfully!'})
        else:
            os.remove(temp_path)
            return jsonify({'success': False, 'message': 'Face verification failed!'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/verify_pin', methods=['POST'])
def verify_pin_login():
    try:
        data = request.get_json()
        username = data['username']
        pin = data['pin']
        
        if not session.get('face_verified'):
            return jsonify({'success': False, 'message': 'Face verification required first!'})
        
        user = get_user_by_username(username)
        if user and verify_pin(pin, user[3]):  # user[3] is pin_hash
            session['logged_in'] = True
            session['user_id'] = user[0]
            session['username'] = user[1]
            return jsonify({'success': True, 'message': 'Login successful!'})
        else:
            return jsonify({'success': False, 'message': 'Invalid PIN!'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    documents = get_user_documents(user_id)
    shared_documents = get_shared_documents_for_user(user_id)
    pending_shares = get_pending_shares(user_id)
    
    # Get user's member ID using dedicated function
    member_id = get_user_member_id(session.get('username'))
    
    return render_template('dashboard.html', 
                         documents=documents, 
                         shared_documents=shared_documents,
                         pending_shares=pending_shares,
                         member_id=member_id)

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if 'document' not in request.files:
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    file = request.files['document']
    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(url_for('dashboard'))
    
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Save to database
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO documents (user_id, filename, original_filename, file_path)
            VALUES (?, ?, ?, ?)
        ''', (session['user_id'], unique_filename, original_filename, filepath))
        conn.commit()
        conn.close()
        
        flash('Document uploaded successfully!', 'success')
    else:
        flash('Invalid file type!', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/share_document', methods=['POST'])
def share_document():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Please login first'})
    
    try:
        data = request.get_json()
        document_id = data['document_id']
        member_id = data['member_id']
        
        # Verify the document belongs to the current user
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM documents WHERE id = ? AND user_id = ?', 
                      (document_id, session['user_id']))
        document = cursor.fetchone()
        
        if not document:
            conn.close()
            return jsonify({'success': False, 'message': 'Document not found or unauthorized'})
        
        # Find the user by member ID
        target_user = get_user_by_member_id(member_id)
        if not target_user:
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid member ID'})
        
        # Check if already shared
        cursor.execute('SELECT * FROM shared_documents WHERE document_id = ? AND shared_with_id = ?', 
                      (document_id, target_user[0]))
        existing_share = cursor.fetchone()
        
        if existing_share:
            conn.close()
            return jsonify({'success': False, 'message': 'Document already shared with this member'})
        
        # Create share request
        cursor.execute('''
            INSERT INTO shared_documents (document_id, owner_id, shared_with_id, status)
            VALUES (?, ?, ?, 'pending')
        ''', (document_id, session['user_id'], target_user[0]))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'Document shared with {target_user[1]} (ID: {member_id}). Waiting for acceptance.',
            'target_username': target_user[1]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/accept_share/<int:share_id>', methods=['POST'])
def accept_share(share_id):
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Please login first'})
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Verify the share belongs to the current user
        cursor.execute('SELECT * FROM shared_documents WHERE id = ? AND shared_with_id = ?', 
                      (share_id, session['user_id']))
        share = cursor.fetchone()
        
        if not share:
            conn.close()
            return jsonify({'success': False, 'message': 'Share not found or unauthorized'})
        
        # Update status to accepted
        cursor.execute('UPDATE shared_documents SET status = ? WHERE id = ?', 
                      ('accepted', share_id))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Document share accepted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/reject_share/<int:share_id>', methods=['POST'])
def reject_share(share_id):
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Please login first'})
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Verify the share belongs to the current user
        cursor.execute('SELECT * FROM shared_documents WHERE id = ? AND shared_with_id = ?', 
                      (share_id, session['user_id']))
        share = cursor.fetchone()
        
        if not share:
            conn.close()
            return jsonify({'success': False, 'message': 'Share not found or unauthorized'})
        
        # Delete the share
        cursor.execute('DELETE FROM shared_documents WHERE id = ?', (share_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Document share rejected'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/access_document/<int:doc_id>')
def access_document(doc_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template('access_document.html', doc_id=doc_id, is_shared=False)

@app.route('/access_shared_document/<int:doc_id>')
def access_shared_document(doc_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Verify user has access to this shared document
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sd.* FROM shared_documents sd
        WHERE sd.document_id = ? AND sd.shared_with_id = ? AND sd.status = 'accepted'
    ''', (doc_id, session['user_id']))
    share = cursor.fetchone()
    conn.close()
    
    if not share:
        flash('Document not found or access denied!', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('access_document.html', doc_id=doc_id, is_shared=True)

@app.route('/verify_document_access', methods=['POST'])
def verify_document_access():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Please login first'})
    
    try:
        data = request.get_json()
        image_data = data['image']
        doc_id = data['doc_id']
        is_shared = data.get('is_shared', False)
        
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        print(f"\n[DOCUMENT ACCESS] Verifying access for document {doc_id} by user {session.get('user_id')} (shared: {is_shared})")
        
        # Save temporary image for verification
        temp_path = os.path.join('static/temp', f'access_{session["username"]}_{uuid.uuid4().hex}.jpg')
        cv2.imwrite(temp_path, image)
        
        # Verify user's face
        verification_result = verify_face(session['username'], temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        if verification_result:
            # Get document info
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            if is_shared:
                # Check shared document access
                cursor.execute('''
                    SELECT d.filename FROM documents d
                    JOIN shared_documents sd ON d.id = sd.document_id
                    WHERE d.id = ? AND sd.shared_with_id = ? AND sd.status = 'accepted'
                ''', (doc_id, session['user_id']))
            else:
                # Check owned document access
                cursor.execute('SELECT filename FROM documents WHERE id = ? AND user_id = ?', 
                              (doc_id, session['user_id']))
            
            document = cursor.fetchone()
            conn.close()
            
            if document:
                filename = document[0]
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    
                    return jsonify({
                        'success': True, 
                        'content': content,
                        'filename': filename,
                        'is_shared': is_shared
                    })
                else:
                    return jsonify({'success': False, 'message': 'Document file not found'})
            else:
                return jsonify({'success': False, 'message': 'Document not found or access denied'})
        else:
            return jsonify({'success': False, 'message': 'Face verification failed. Access denied.'})
            
    except Exception as e:
        print(f"[DOCUMENT ACCESS] Error: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/download_document/<int:doc_id>')
def download_document(doc_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE id = ? AND user_id = ?', 
                   (doc_id, session['user_id']))
    document = cursor.fetchone()
    conn.close()
    
    if document:
        return redirect(f'/static/../{document[4]}')  # document[4] is file_path
    else:
        flash('Document not found!', 'error')
        return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)