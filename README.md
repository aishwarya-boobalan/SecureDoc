# Face Auth Document Locker

A secure document management system using facial recognition and PIN authentication.

## Features

- **Multi-layer Authentication**: Face recognition + PIN verification
- **Secure Document Storage**: Upload and store documents securely
- **Face Recognition**: Powered by DeepFace with ArcFace backend
- **Document Access Control**: Re-authentication required for document access
- **Modern Web Interface**: Clean, responsive design with Streamlit-like experience

## Tech Stack

- **Backend**: Flask
- **Face Recognition**: DeepFace
- **Database**: SQLite
- **Frontend**: HTML, CSS, JavaScript
- **Security**: bcrypt for PIN hashing

## Setup Instructions

### Windows
```bash
# Run the setup script
setup.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### macOS/Linux
```bash
# Run the setup script
chmod +x setup.sh
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Usage

1. **Sign Up**: Register with username, email, PIN, and face image
2. **Login**: 3-step authentication (username → face → PIN)
3. **Upload Documents**: Secure document storage after authentication
4. **Access Documents**: Re-authenticate with face + PIN to view documents

## Security Features

- PIN hashing with bcrypt
- Face encoding storage using DeepFace
- Session management
- File type and size validation
- Secure file storage

## Supported File Types

- Documents: PDF, DOCX, DOC, TXT
- Images: PNG, JPG, JPEG, GIF
- Maximum file size: 16MB

## Project Structure

```
face_auth_app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── users.db              # SQLite database (created automatically)
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── signup.html
│   ├── login.html
│   ├── dashboard.html
│   └── access_document.html
├── static/               # Static files
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── temp/            # Temporary files
├── uploads/             # Document storage
├── face_database/       # Face encodings storage
└── README.md
```

## API Endpoints

- `GET /` - Home page
- `GET/POST /signup` - User registration
- `GET /login` - Login page
- `POST /verify_face` - Face verification
- `POST /verify_pin` - PIN verification
- `GET /dashboard` - User dashboard
- `POST /upload_document` - Document upload
- `GET /access_document/<id>` - Document access page
- `POST /verify_document_access` - Document access verification
- `GET /download_document/<id>` - Document download
- `GET /logout` - User logout

## Security Considerations

- Change the Flask secret key in production
- Use HTTPS in production
- Consider additional rate limiting
- Implement proper error logging
- Add CSRF protection for production use

## License

This project is for educational purposes. Please ensure compliance with privacy laws when using facial recognition technology.