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
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-dev-secret-key-change-me')

# ── Config ───────────────────────────────────────────────────────────────────
UPLOAD_FOLDER  = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE  = 16 * 1024 * 1024
FACE_DB_PATH   = 'face_database'
MODELS_DIR     = 'models'

# SMTP / Email config
SMTP_HOST     = os.getenv('SMTP_HOST',     'smtp.gmail.com')
SMTP_PORT     = int(os.getenv('SMTP_PORT', 587))
SMTP_USER     = os.getenv('SMTP_USER',     'securedocverify@gmail.com')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', 'ekpi xbff losj dvnk')
EMAIL_FROM    = os.getenv('EMAIL_FROM',    'securedocverify@gmail.com')

OTP_EXPIRY_MINUTES = 10  # OTP valid for 10 minutes

app.config['UPLOAD_FOLDER']       = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']  = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(FACE_DB_PATH,   exist_ok=True)
os.makedirs(MODELS_DIR,     exist_ok=True)
os.makedirs('static/temp',  exist_ok=True)

# ── Face Recognition Engine ──────────────────────────────────────────────────
class FaceEngine:
    def __init__(self):
        self.detector   = None
        self.recognizer = None
        self.base_dir   = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, MODELS_DIR)
        self.detector_path   = os.path.join(self.models_dir, "face_detection_yunet_2023mar.onnx")
        self.recognizer_path = os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx")
        os.makedirs(self.models_dir, exist_ok=True)
        self.download_models()
        self.init_models()

    def download_models(self):
        models = {
            self.detector_path:   "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?raw=true",
            self.recognizer_path: "https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?raw=true"
        }
        for path, url in models.items():
            min_size = 2 * 1024 * 1024 if "sface" in path else 200 * 1024
            needs_download = not os.path.exists(path) or os.path.getsize(path) < min_size
            if not needs_download:
                try:
                    with open(path, 'rb') as f:
                        header = f.read(512)
                    if b'<!DOCTYPE' in header or b'<html' in header or b'git-lfs' in header:
                        needs_download = True
                        try: os.remove(path)
                        except: pass
                except: needs_download = True
            if needs_download:
                print(f"[INFO] Downloading {os.path.basename(path)}...")
                try:
                    r = requests.get(url, allow_redirects=True)
                    if r.status_code == 200:
                        with open(path, 'wb') as f: f.write(r.content)
                        print(f"[OK] Downloaded {os.path.basename(path)}")
                    else: print(f"[ERROR] Download failed (status {r.status_code})")
                except Exception as e: print(f"[ERROR] Download error: {e}")

    def init_models(self):
        try:
            if not os.path.exists(self.detector_path) or not os.path.exists(self.recognizer_path):
                print(f"[ERROR] Model files missing"); return
            self.detector   = cv2.FaceDetectorYN.create(self.detector_path, "", (320, 320), 0.9, 0.3, 5000)
            self.recognizer = cv2.FaceRecognizerSF.create(self.recognizer_path, "")
            print("[OK] Face Recognition Models Loaded")
        except Exception as e:
            print(f"[WARNING] Model load error: {e}")

    def get_embedding(self, image_path):
        if self.detector is None or self.recognizer is None:
            self.init_models()
            if self.detector is None: return None
        try:
            img = cv2.imread(image_path)
            if img is None: return None
            h, w, _ = img.shape
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)
            if faces is None or len(faces) == 0: return None
            aligned = self.recognizer.alignCrop(img, faces[0])
            emb = self.recognizer.feature(aligned)
            return emb[0]
        except Exception as e:
            print(f"Embedding error: {e}"); return None

    def compare(self, emb1, emb2):
        if emb1 is None or emb2 is None: return 0.0
        return self.recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)

face_engine = FaceEngine()

# ── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn   = sqlite3.connect('users.db')
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
            expires_at TIMESTAMP,
            max_views INTEGER DEFAULT -1,
            current_views INTEGER DEFAULT 0,
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS otp_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            otp_code TEXT NOT NULL,
            purpose TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            used INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    migrate_db()

def migrate_db():
    try:
        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Ensure shared_documents columns exist
        cursor.execute('PRAGMA table_info(shared_documents)')
        cols = [r[1] for r in cursor.fetchall()]
        for col, defn in [('expires_at', 'TIMESTAMP'), ('max_views', 'INTEGER DEFAULT -1'), ('current_views', 'INTEGER DEFAULT 0')]:
            if col not in cols:
                cursor.execute(f'ALTER TABLE shared_documents ADD COLUMN {col} {defn}')

        # Ensure otp_tokens table exists (for existing DBs)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS otp_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp_code TEXT NOT NULL,
                purpose TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Migration error: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def log_access(user_id, action, details, doc_id=None):
    try:
        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO access_logs (user_id, document_id, action, details) VALUES (?, ?, ?, ?)',
                       (user_id, doc_id, action, details))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Log error: {e}")

def generate_unique_member_id():
    while True:
        mid = str(random.randint(100000, 999999))
        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE unique_member_id = ?', (mid,))
        exists = cursor.fetchone()
        conn.close()
        if not exists: return mid

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_pin(pin):
    return bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt())

def verify_pin(pin, hashed):
    return bcrypt.checkpw(pin.encode('utf-8'), hashed)

# ── OTP System ────────────────────────────────────────────────────────────────
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def save_otp(email, otp, purpose):
    """Store OTP in DB, invalidate any old ones for same email+purpose."""
    conn   = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Expire old OTPs
    cursor.execute("UPDATE otp_tokens SET used=1 WHERE email=? AND purpose=? AND used=0", (email, purpose))
    expires_at = datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    cursor.execute("INSERT INTO otp_tokens (email, otp_code, purpose, expires_at) VALUES (?, ?, ?, ?)",
                   (email, otp, purpose, expires_at))
    conn.commit()
    conn.close()

def verify_otp_db(email, otp, purpose):
    """Returns True and marks used if OTP is valid and not expired."""
    conn   = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, expires_at FROM otp_tokens WHERE email=? AND otp_code=? AND purpose=? AND used=0",
        (email, otp, purpose)
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, "Invalid OTP or already used."
    otp_id, expires_at_str = row
    try:
        expires_at = datetime.strptime(expires_at_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        expires_at = datetime.strptime(expires_at_str, '%Y-%m-%d %H:%M:%S')
    if datetime.now() > expires_at:
        conn.close()
        return False, f"OTP expired. Please request a new one."
    cursor.execute("UPDATE otp_tokens SET used=1 WHERE id=?", (otp_id,))
    conn.commit()
    conn.close()
    return True, "OTP verified."

def send_otp_email(to_email, otp, purpose):
    """Send a beautiful HTML OTP email."""
    purpose_labels = {
        'signup':   ('Account Verification', 'Complete your SecureDoc registration'),
        'login':    ('Login Verification',    'Complete your SecureDoc login'),
        'access':   ('Document Access',       'Verify your identity to access this document'),
        'share':    ('Document Share',        'Verify your identity to share this document'),
    }
    title, subtitle = purpose_labels.get(purpose, ('Verification', 'Your OTP code'))

    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:#0f0f13;font-family:Inter,sans-serif">
      <div style="max-width:520px;margin:40px auto;background:linear-gradient(135deg,#12131a,#1a1a24);
                  border:1px solid rgba(0,242,255,0.2);border-radius:20px;overflow:hidden">
        <div style="background:linear-gradient(90deg,#00f2ff22,#bd00ff22);padding:30px 40px;text-align:center;
                    border-bottom:1px solid rgba(255,255,255,0.07)">
          <span style="font-size:2rem">🛡️</span>
          <h1 style="color:#00f2ff;margin:10px 0 4px;font-size:1.6rem;letter-spacing:2px">SECUREDOC</h1>
          <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.85rem">{subtitle}</p>
        </div>
        <div style="padding:40px">
          <h2 style="color:#fff;text-align:center;margin:0 0 8px;font-size:1.2rem">{title}</h2>
          <p style="color:rgba(255,255,255,0.6);text-align:center;font-size:0.9rem;margin:0 0 30px">
            Enter this code to continue. Valid for {OTP_EXPIRY_MINUTES} minutes.
          </p>
          <div style="background:rgba(0,0,0,0.4);border:2px solid rgba(0,242,255,0.4);border-radius:16px;
                      padding:30px;text-align:center;margin-bottom:30px;
                      box-shadow:0 0 30px rgba(0,242,255,0.1)">
            <div style="font-size:3rem;font-weight:700;letter-spacing:12px;color:#00f2ff;
                        text-shadow:0 0 20px rgba(0,242,255,0.6)">{otp}</div>
          </div>
          <p style="color:rgba(255,255,255,0.35);font-size:0.8rem;text-align:center;margin:0">
            ⚠️ Never share this code. SecureDoc will never ask for it via chat or phone.<br>
            If you didn't request this, ignore this email.
          </p>
        </div>
        <div style="padding:20px;text-align:center;border-top:1px solid rgba(255,255,255,0.05)">
          <p style="color:rgba(255,255,255,0.2);font-size:0.75rem;margin:0">
            © SecureDoc Vault • Encrypted Document System
          </p>
        </div>
      </div>
    </body>
    </html>
    """
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[SecureDoc] Your {title} OTP: {otp}"
        msg['From']    = f"SecureDoc <{EMAIL_FROM}>"
        msg['To']      = to_email
        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        print(f"[INFO] OTP sent to {to_email} ({purpose})")
        return True
    except Exception as e:
        print(f"[ERROR] Email error: {e}")
        return False

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

# ── OTP Endpoints (general) ───────────────────────────────────────────────────
@app.route('/send_otp', methods=['POST'])
def send_otp():
    """Send OTP to an email for a given purpose. Works pre-login and post-login."""
    data    = request.get_json()
    email   = data.get('email', '').strip().lower()
    purpose = data.get('purpose', 'login')

    # For purposes other than signup, require login
    if purpose in ('access', 'share') and not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Not authenticated.'})

    # For login purpose, look up email from the username stored in session after face verify
    if purpose == 'login':
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'message': 'Face verification required first.'})
        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT email FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return jsonify({'success': False, 'message': 'User not found.'})
        email = row[0]  # Email fetched from DB — caller does not need to send it

    # For access/share, use the logged-in user's email from DB
    elif purpose in ('access', 'share'):
        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT email FROM users WHERE id = ?', (session['user_id'],))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return jsonify({'success': False, 'message': 'User not found.'})
        email = row[0]

    # For signup, the email must be provided by the client
    elif purpose == 'signup':
        if not email:
            return jsonify({'success': False, 'message': 'Email is required.'})

    otp = generate_otp()
    save_otp(email, otp, purpose)
    ok  = send_otp_email(email, otp, purpose)

    if ok:
        # Store which email we sent to in session (for verify step)
        session[f'otp_email_{purpose}'] = email
        # Mask email for display
        parts = email.split('@')
        masked = parts[0][:2] + '***@' + parts[1]
        return jsonify({'success': True, 'message': f'OTP sent to {masked}', 'masked_email': masked})
    else:
        return jsonify({'success': False, 'message': 'Failed to send OTP. Check server email config.'})

@app.route('/verify_otp', methods=['POST'])
def verify_otp_route():
    """Verify an OTP for a given purpose."""
    data    = request.get_json()
    otp     = data.get('otp', '').strip()
    purpose = data.get('purpose', 'login')

    email = session.get(f'otp_email_{purpose}')
    if not email:
        return jsonify({'success': False, 'message': 'No OTP session found. Please request again.'})

    ok, msg = verify_otp_db(email, otp, purpose)
    if ok:
        session[f'otp_verified_{purpose}'] = True
        session.pop(f'otp_email_{purpose}', None)
        return jsonify({'success': True, 'message': msg})
    else:
        return jsonify({'success': False, 'message': msg})

# ── Signup ────────────────────────────────────────────────────────────────────
@app.route('/signup_with_training', methods=['POST'])
def signup_with_training():
    try:
        data         = request.get_json()
        username     = data['username']
        email        = data['email'].strip().lower()
        pin          = data['pin']
        face_images  = data['face_images']

        # Require OTP verification for signup
        if not session.get('otp_verified_signup'):
            return jsonify({'success': False, 'message': 'Email OTP verification required before registration.'})

        user_face_dir    = os.path.join(FACE_DB_PATH, username)
        os.makedirs(user_face_dir, exist_ok=True)

        valid_embeddings = []
        valid_images     = []

        print(f"Processing {len(face_images)} training images...")

        for i, b64 in enumerate(face_images):
            try:
                image_bytes = base64.b64decode(b64.split(',')[1])
                temp_path   = os.path.join('static/temp', f'train_{i}_{uuid.uuid4().hex}.jpg')
                with open(temp_path, 'wb') as f: f.write(image_bytes)
                embedding = face_engine.get_embedding(temp_path)
                if embedding is not None:
                    perm_path = os.path.join(user_face_dir, f'face_{len(valid_embeddings)}.jpg')
                    shutil.move(temp_path, perm_path)
                    valid_embeddings.append(embedding.tolist())
                    valid_images.append(perm_path)
                else:
                    os.remove(temp_path)
            except Exception as e:
                print(f"Image {i} error: {e}")

        if len(valid_embeddings) < 3:
            return jsonify({'success': False, 'message': 'Not enough clear face photos. Try again with better lighting.'})

        with open(os.path.join(user_face_dir, 'embeddings.json'), 'w') as f:
            json.dump(valid_embeddings, f)

        ref_path = os.path.join(user_face_dir, 'reference.jpg')
        if valid_images: shutil.copy2(valid_images[0], ref_path)

        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            mid = generate_unique_member_id()
            cursor.execute(
                'INSERT INTO users (username, email, pin_hash, face_encoding_path, unique_member_id) VALUES (?, ?, ?, ?, ?)',
                (username, email, hash_pin(pin), ref_path, mid)
            )
            conn.commit()
            session.pop('otp_verified_signup', None)
            return jsonify({'success': True, 'message': 'Registration Successful! 🎉', 'member_id': mid})
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'message': 'Username or Email already exists.'})
        finally:
            conn.close()

    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'success': False, 'message': f'Server Error: {str(e)}'})

# ── Login ─────────────────────────────────────────────────────────────────────
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/verify_face', methods=['POST'])
def verify_face_route():
    try:
        data      = request.get_json()
        username  = data['username']
        image_data = data['image']

        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return jsonify({'success': False, 'message': 'User not found.'})

        temp_path = os.path.join('static/temp', f'login_{uuid.uuid4().hex}.jpg')
        with open(temp_path, 'wb') as f:
            f.write(base64.b64decode(image_data.split(',')[1]))

        login_embedding = face_engine.get_embedding(temp_path)
        os.remove(temp_path)

        if login_embedding is None:
            return jsonify({'success': False, 'message': 'No face detected. Look at the camera.'})

        emb_path = os.path.join(FACE_DB_PATH, username, 'embeddings.json')
        if not os.path.exists(emb_path):
            return jsonify({'success': False, 'message': 'Face data corrupted.'})

        with open(emb_path, 'r') as f:
            stored_embeddings = json.load(f)

        match_count = sum(
            1 for stored in stored_embeddings
            if face_engine.compare(login_embedding, np.array(stored, dtype=np.float32)) > 0.4
        )

        if match_count > 0:
            session['face_verified'] = True
            session['username']      = username
            return jsonify({'success': True, 'message': 'Face Verified!'})
        else:
            return jsonify({'success': False, 'message': 'Face verification failed.'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/verify_pin', methods=['POST'])
def verify_pin_route():
    try:
        data     = request.get_json()
        username = data['username']
        pin      = data['pin']

        if not session.get('face_verified') or session.get('username') != username:
            return jsonify({'success': False, 'message': 'Biometric verification required first.'})

        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        # Duress mode
        if pin == "0000" and user:
            session['logged_in']    = True
            session['user_id']      = user[0]
            session['is_panic_mode'] = True
            log_access(user[0], 'PANIC_LOGIN', 'Duress PIN Used')
            return jsonify({'success': True, 'message': 'Login Successful', 'require_otp': False})

        if user and verify_pin(pin, user[3]):
            # PIN correct — now require OTP
            session['pin_verified']     = True
            session['pending_user_id']  = user[0]
            return jsonify({'success': True, 'require_otp': True, 'message': 'PIN verified. OTP will be sent to your email.'})
        else:
            return jsonify({'success': False, 'message': 'Invalid PIN.'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/complete_login', methods=['POST'])
def complete_login():
    """Called after OTP is verified for login — finalises the session."""
    if not (session.get('pin_verified') and session.get('pending_user_id')):
        return jsonify({'success': False, 'message': 'Not authorized.'})
    if not session.get('otp_verified_login'):
        return jsonify({'success': False, 'message': 'OTP verification required.'})

    user_id = session['pending_user_id']
    session['logged_in']    = True
    session['user_id']      = user_id
    session['is_panic_mode'] = False
    session.pop('pin_verified', None)
    session.pop('pending_user_id', None)
    session.pop('otp_verified_login', None)
    log_access(user_id, 'LOGIN', 'Successful Login (Face + PIN + OTP)')
    return jsonify({'success': True, 'message': 'Login complete!'})

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'): return redirect(url_for('login'))

    if session.get('is_panic_mode'):
        return render_template('dashboard.html',
                               documents=[], member_id="000000",
                               shared_documents=[], pending_shares=[])

    conn   = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM documents WHERE user_id = ?', (session['user_id'],))
    documents = cursor.fetchall()

    cursor.execute('SELECT unique_member_id FROM users WHERE id = ?', (session['user_id'],))
    member_id = cursor.fetchone()[0]

    cursor.execute('''
        SELECT sd.*, d.original_filename, d.upload_date, u.username
        FROM shared_documents sd
        JOIN documents d ON sd.document_id = d.id
        JOIN users u ON sd.owner_id = u.id
        WHERE sd.shared_with_id = ? AND sd.status = 'accepted'
    ''', (session['user_id'],))
    shared = cursor.fetchall()

    cursor.execute('''
        SELECT sd.*, d.original_filename, d.upload_date, u.username
        FROM shared_documents sd
        JOIN documents d ON sd.document_id = d.id
        JOIN users u ON sd.owner_id = u.id
        WHERE sd.shared_with_id = ? AND sd.status = 'pending'
    ''', (session['user_id'],))
    pending = cursor.fetchall()

    conn.close()
    return render_template('dashboard.html', documents=documents, member_id=member_id,
                           shared_documents=shared, pending_shares=pending)

@app.route('/dashboard_stats')
def dashboard_stats():
    if not session.get('logged_in'): return jsonify({})
    conn   = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM documents WHERE user_id = ?', (session['user_id'],))
    doc_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM shared_documents WHERE owner_id = ? AND status = "accepted"', (session['user_id'],))
    shared_out = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM shared_documents WHERE shared_with_id = ? AND status = "accepted"', (session['user_id'],))
    shared_in = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM access_logs WHERE user_id = ?', (session['user_id'],))
    log_count = cursor.fetchone()[0]
    conn.close()
    return jsonify({'docs': doc_count, 'shared_out': shared_out, 'shared_in': shared_in, 'logs': log_count})

# ── Upload ────────────────────────────────────────────────────────────────────
@app.route('/upload_document', methods=['POST'])
def upload_document():
    if not session.get('logged_in'): return redirect(url_for('login'))
    file = request.files.get('document')
    if file and allowed_file(file.filename):
        filename    = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        path        = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(path)
        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO documents (user_id, filename, original_filename, file_path) VALUES (?, ?, ?, ?)',
                       (session['user_id'], unique_name, filename, path))
        conn.commit()
        conn.close()
        flash('Upload successful!', 'success')
    else:
        flash('Invalid file!', 'error')
    return redirect(url_for('dashboard'))

# ── Share Document (requires OTP) ─────────────────────────────────────────────
@app.route('/share_document', methods=['POST'])
def share_document():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'message': 'Auth required.'})

    # Require OTP verified for sharing
    if not session.pop('otp_verified_share', False):
        return jsonify({'success': False, 'message': 'OTP verification required to share documents.', 'require_otp': True})

    data   = request.get_json()
    conn   = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM documents WHERE id = ? AND user_id = ?', (data['document_id'], session['user_id']))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'success': False, 'message': 'Invalid document.'})

    member_ids = data.get('member_ids', [])
    if data.get('member_id'): member_ids.append(data['member_id'])
    member_ids = list(set(m for m in member_ids if m))

    if not member_ids:
        conn.close()
        return jsonify({'success': False, 'message': 'No recipients specified.'})

    security_level = data.get('security_level', 'standard')
    max_views  = -1
    expires_at = None
    if security_level == 'confidential':
        expires_at = datetime.now() + timedelta(hours=24)
    elif security_level == 'top_secret':
        max_views  = 1
        expires_at = datetime.now() + timedelta(hours=1)

    results = {'success': [], 'failed': []}
    for mid in member_ids:
        cursor.execute('SELECT id, username FROM users WHERE unique_member_id = ?', (mid,))
        target = cursor.fetchone()
        if target:
            if target[0] == session['user_id']:
                results['failed'].append(f"{mid} (Self)"); continue
            cursor.execute('SELECT id FROM shared_documents WHERE document_id = ? AND shared_with_id = ?',
                           (data['document_id'], target[0]))
            if cursor.fetchone():
                results['failed'].append(f"{mid} (Already Shared)"); continue
            cursor.execute(
                'INSERT INTO shared_documents (document_id, owner_id, shared_with_id, status, max_views, expires_at) VALUES (?, ?, ?, ?, ?, ?)',
                (data['document_id'], session['user_id'], target[0], 'pending', max_views, expires_at)
            )
            results['success'].append(target[1])
            log_access(session['user_id'], 'SHARE', f'Shared Doc {data["document_id"]} with {target[1]} ({security_level})', data['document_id'])
        else:
            results['failed'].append(f"{mid} (Invalid ID)")

    conn.commit()
    conn.close()

    msg = f"Sent to {len(results['success'])} users."
    if results['failed']: msg += f" Failed: {', '.join(results['failed'])}"
    return jsonify({'success': True, 'message': msg, 'details': results})

@app.route('/accept_share/<int:share_id>', methods=['POST'])
def accept_share(share_id):
    if not session.get('logged_in'): return jsonify({'success': False})
    conn   = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE shared_documents SET status = "accepted" WHERE id = ? AND shared_with_id = ?',
                   (share_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Accepted'})

@app.route('/reject_share/<int:share_id>', methods=['POST'])
def reject_share(share_id):
    if not session.get('logged_in'): return jsonify({'success': False})
    conn   = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM shared_documents WHERE id = ? AND shared_with_id = ?',
                   (share_id, session['user_id']))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Rejected'})

# ── Document Access ───────────────────────────────────────────────────────────
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
        return jsonify({'success': False, 'message': 'Unauthorized.'})
    try:
        data       = request.get_json()
        doc_id     = data.get('doc_id')
        is_shared  = data.get('is_shared')
        image_data = data.get('image')
        username   = session.get('username')

        # ── Face check ──
        if not image_data: return jsonify({'success': False, 'message': 'No camera data.'})
        temp_path = os.path.join('static/temp', f'access_{uuid.uuid4().hex}.jpg')
        with open(temp_path, 'wb') as f: f.write(base64.b64decode(image_data.split(',')[1]))

        emb = face_engine.get_embedding(temp_path)
        try: os.remove(temp_path)
        except: pass
        if emb is None: return jsonify({'success': False, 'message': 'No face detected.'})

        emb_path = os.path.join(FACE_DB_PATH, username, 'embeddings.json')
        if not os.path.exists(emb_path): return jsonify({'success': False, 'message': 'Face data corrupted.'})
        with open(emb_path, 'r') as f: stored = json.load(f)

        match = sum(1 for s in stored if face_engine.compare(emb, np.array(s, dtype=np.float32)) > 0.4)
        if match == 0:
            log_access(session['user_id'], 'ACCESS_DENIED', f'Face mismatch for doc {doc_id}')
            return jsonify({'success': False, 'message': 'Face verification failed.'})

        # ── OTP check ──
        if not session.pop('otp_verified_access', False):
            return jsonify({'success': False, 'message': 'OTP verification required.', 'require_otp': True})

        # ── Fetch document ──
        conn   = sqlite3.connect('users.db')
        cursor = conn.cursor()
        doc = None
        if is_shared:
            cursor.execute('''
                SELECT d.file_path, d.original_filename, sd.max_views, sd.current_views, sd.expires_at, sd.id
                FROM shared_documents sd
                JOIN documents d ON sd.document_id = d.id
                WHERE sd.document_id = ? AND sd.shared_with_id = ? AND sd.status = 'accepted'
            ''', (doc_id, session['user_id']))
            info = cursor.fetchone()
            if not info:
                conn.close()
                return jsonify({'success': False, 'message': 'Access Revoked.'})
            file_path, filename, max_views, cur_views, expires_at_str, share_id = info
            if expires_at_str:
                try: exp = datetime.strptime(expires_at_str, '%Y-%m-%d %H:%M:%S.%f')
                except: exp = datetime.strptime(expires_at_str, '%Y-%m-%d %H:%M:%S')
                if datetime.now() > exp:
                    conn.close()
                    return jsonify({'success': False, 'message': 'Document expired (Self-Destructed).'})
            if max_views != -1 and cur_views >= max_views:
                conn.close()
                return jsonify({'success': False, 'message': 'Max views reached (Self-Destructed).'})
            cursor.execute('UPDATE shared_documents SET current_views = current_views + 1 WHERE id = ?', (share_id,))
            conn.commit()
            doc = (file_path, filename)
        else:
            cursor.execute('SELECT file_path, original_filename FROM documents WHERE id = ? AND user_id = ?',
                           (doc_id, session['user_id']))
            doc = cursor.fetchone()
        conn.close()

        if not doc:
            return jsonify({'success': False, 'message': 'Document not found.'})

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
        return jsonify({'success': True, 'content': content, 'filename': filename, 'is_shared': is_shared})

    except Exception as e:
        print(f"Access error: {e}")
        return jsonify({'success': False, 'message': 'Server Error'})

# ── Stats / Logs ──────────────────────────────────────────────────────────────
@app.route('/get_stats')
def get_stats():
    if not session.get('logged_in'): return jsonify([])
    conn   = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.username, al.action, al.details, al.timestamp
        FROM access_logs al
        JOIN users u ON al.user_id = u.id
        ORDER BY al.timestamp DESC LIMIT 10
    ''')
    logs = cursor.fetchall()
    conn.close()
    return jsonify([{'user': l[0], 'action': l[1], 'details': l[2], 'time': l[3]} for l in logs])

# ── Logout ────────────────────────────────────────────────────────────────────
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)