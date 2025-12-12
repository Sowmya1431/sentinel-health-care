# app.py
from flask import (
    Flask, render_template, Response, request, redirect, url_for,
    jsonify, send_from_directory, session, flash, abort
)
import os
from werkzeug.utils import secure_filename
from threading import Lock
from functools import wraps
import bcrypt
import datetime
import logging
import time
from uuid import uuid4

# Try to import pymongo but handle if it's not available / or DB is unreachable
try:
    from pymongo import MongoClient
    from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except Exception:
    PYMONGO_AVAILABLE = False
    # Create placeholders for exceptions used later
    class DuplicateKeyError(Exception):
        pass
    class ServerSelectionTimeoutError(Exception):
        pass

# import detector from main.py (existing)
from main import detector

#### CONFIG ####
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = {'mp4', 'avi', 'mov', 'mkv'}
EXPECTED_HOSPITAL_ID = "Hospital@45"

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "mydb"
USERS_COLL = "users"

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit uploads to 500 MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.secret_key = os.environ.get("FLASK_SECRET", "change_this_secret_for_prod")

# Ensure upload dir
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# In-memory fallback store (development only)
_in_memory_users = {}  # keyed by email -> user_doc
DB_CONNECTED = False
users = None

# Alerts collection handle and in-memory fallback (declare once so DB init can set alerts_coll)
alerts_coll = None
_in_memory_alerts = []

# Attempt to connect to MongoDB
if PYMONGO_AVAILABLE:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force server selection now (will raise if cannot connect)
        client.server_info()
        db = client[DB_NAME]
        users = db[USERS_COLL]
        # initialize alerts collection handle
        alerts_coll = db['alerts']
        try:
            alerts_coll.create_index("timestamp")
        except Exception as ex:
            logging.warning("Could not create index on alerts.timestamp: %s", ex)
        try:
            users.create_index("email", unique=True)
        except Exception as ex:
            logging.warning("Could not create index on users.email: %s", ex)
        DB_CONNECTED = True
        print("Connected to MongoDB at", MONGO_URI)
    except ServerSelectionTimeoutError as ex:
        logging.error("MongoDB not reachable: %s", ex)
        DB_CONNECTED = False
        users = None
        print("MongoDB not reachable — running with in-memory user store (dev only).")
    except Exception as ex:
        logging.error("Unexpected error connecting to MongoDB: %s", ex)
        DB_CONNECTED = False
        users = None
        print("MongoDB connection error — running with in-memory user store (dev only).")
else:
    print("pymongo not installed — running with in-memory user store (dev only).")

# Simple in-memory store for fall events (for analytics demo)
fall_events = []
events_lock = Lock()


def fall_callback(event_type, info):
    global alerts_coll, _in_memory_alerts, DB_CONNECTED
    if event_type == 'fall':
        with events_lock:
            fall_events.append(info)

    # persist alert (type, info, timestamp) to MongoDB or in-memory fallback
    alert_doc = {
        "type": event_type,
        "info": info,
        "timestamp": datetime.datetime.utcnow()
    }
    try:
        if DB_CONNECTED and alerts_coll is not None:
            alerts_coll.insert_one(alert_doc)
        else:
            _in_memory_alerts.append(alert_doc)
    except Exception:
        logging.exception("Failed to persist alert")


# ---- Helpers ----
def hash_text(plain_text: str) -> str:
    if not isinstance(plain_text, str):
        raise TypeError("Input must be string")
    return bcrypt.hashpw(plain_text.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_text(plain_text: str, hashed_text: str) -> bool:
    try:
        return bcrypt.checkpw(plain_text.encode("utf-8"), hashed_text.encode("utf-8"))
    except Exception:
        return False


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# DB helper wrappers (work for real DB or in-memory fallback)
def insert_user_doc(user_doc: dict):
    """Insert a user, raise DuplicateKeyError if email exists."""
    email = user_doc.get("email")
    if DB_CONNECTED and users is not None:
        # using MongoDB
        return users.insert_one(user_doc)
    else:
        # in-memory fallback
        if email in _in_memory_users:
            raise DuplicateKeyError("email exists")
        _in_memory_users[email] = user_doc
        class DummyResult:
            inserted_id = "inmemory:" + email
        return DummyResult()


def find_user_by_email(email: str):
    if DB_CONNECTED and users is not None:
        return users.find_one({"email": email})
    else:
        return _in_memory_users.get(email)


# ---- Auth routes (register / login / logout) ----

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    # If already logged in, go to upload
    if session.get("user_id"):
        return redirect(url_for("upload_page"))

    # keep non-sensitive form data so user doesn't retype on error
    form = {"username": "", "email": "", "role": "", "hospital_id": ""}
    error = None

    if request.method == 'POST':
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "").strip()
        hospital_id = request.form.get("hospital_id", "").strip()

        # preserve values (never preserve password)
        form["username"] = username
        form["email"] = email
        form["role"] = role
        form["hospital_id"] = hospital_id

        # Basic validation
        if not all([username, email, password, role, hospital_id]):
            error = "Missing required fields."
            return render_template('register.html', error=error, form=form)

        if hospital_id != EXPECTED_HOSPITAL_ID:
            error = "Invalid hospital id."
            return render_template('register.html', error=error, form=form)

        if role not in {"Nurse", "Doctor"}:
            error = "Invalid role."
            return render_template('register.html', error=error, form=form)

        # Very simple email format check
        if "@" not in email or "." not in email.split("@")[-1]:
            error = "Invalid email."
            return render_template('register.html', error=error, form=form)

        # Hash password and hospital_id and store user
        try:
            pw_hash = hash_text(password)
            hospital_hash = hash_text(hospital_id)
            user_doc = {
                "username": username,
                "email": email,
                "password_hashed": pw_hash,
                "hospital_id_hashed": hospital_hash,
                "role": role,
                "created_at": datetime.datetime.utcnow()
            }
            insert_user_doc(user_doc)
        except DuplicateKeyError:
            error = "Email already registered."
            return render_template('register.html', error=error, form=form)
        except Exception as e:
            logging.exception("DB insert error")
            # keep same behavior but render template with a friendly message
            error = "Database error. Please try again later."
            return render_template('register.html', error=error, form=form), 500

        # Successful register -> redirect to login
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login_page"))

    # GET: show register page (use your signup.html)
    return render_template('register.html', error=None, form=form)


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    # If already logged in, go to upload
    if session.get("user_id"):
        return redirect(url_for("upload_page"))

    form = {"email": ""}
    error = None

    if request.method == 'POST':
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # preserve email so user doesn't retype
        form["email"] = email

        if not email or not password:
            error = "Email and password required."
            return render_template('login.html', error=error, form=form)

        user = find_user_by_email(email)
        if not user:
            error = "Invalid credentials."
            return render_template('login.html', error=error, form=form)

        stored_hash = user.get("password_hashed")
        if not stored_hash or not check_text(password, stored_hash):
            error = "Invalid credentials."
            return render_template('login.html', error=error, form=form)

        # login success -> set session
        session["user_id"] = str(user.get("_id", "inmemory:" + email))
        session["email"] = user.get("email")
        session["username"] = user.get("username")
        session["role"] = user.get("role")

        # redirect to upload page after successful login
        return redirect(url_for("upload_page"))

    # GET: show login page
    return render_template('login.html', error=None, form=form)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))


# ---- Application routes (protected) ----

@app.route('/')
def index():
    # If not logged in, show public home page; if logged in, go to upload page.
    if not session.get("user_id"):
        return render_template('home.html')
    return redirect(url_for('upload_page'))


@app.route('/dashboard')
@login_required
def dashboard():
    # Render your dashboard.html template. Template can read session['username'] / session['email'] etc.
    return render_template('dashboard.html')


# ---- New: About and Admin routes ----
@app.route('/about')
def about():
    """Public about page (templates/about.html)."""
    return render_template('about.html')


@app.route('/admin')
@login_required
def admin_panel():
    """
    Admin panel (doctors only).
    If you want a separate admin role, adjust the role check below.
    """
    role = session.get("role", "")
    # simple role check — change to your own logic if needed
    if role != "Doctor":
        # unauthorized for non-doctors
        abort(403)
    return render_template('admin_panel.html')


# ---- Upload / play / streaming routes ----
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_page():
    """
    Handles both normal form submits (redirect -> play page) and AJAX uploads:
    - If request contains header X-Requested-With: XMLHttpRequest -> return JSON {success, filename}
    - Otherwise keep the existing redirect behavior for compatibility.
    """
    if request.method == 'POST':
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        if 'video' not in request.files:
            if is_ajax:
                return jsonify({"success": False, "message": "No file part"}), 400
            return 'No file part', 400

        file = request.files['video']
        if file.filename == '':
            if is_ajax:
                return jsonify({"success": False, "message": "No selected file"}), 400
            return 'No selected file', 400

        orig_name = secure_filename(file.filename)
        # validate extension
        if '.' in orig_name:
            ext = orig_name.rsplit('.', 1)[1].lower()
        else:
            ext = ''
        if ext not in ALLOWED_EXT:
            msg = f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXT))}"
            if is_ajax:
                return jsonify({"success": False, "message": msg}), 400
            return msg, 400

        # create unique filename
        unique_name = f"{int(time.time())}_{uuid4().hex}_{orig_name}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

        try:
            file.save(save_path)
        except Exception as e:
            logging.exception("Failed saving uploaded file")
            if is_ajax:
                return jsonify({"success": False, "message": "Failed to save file"}), 500
            return f"Failed to save file: {e}", 500

        # AJAX -> return JSON so frontend can start overlay stream without redirect
        if is_ajax:
            return jsonify({"success": True, "filename": unique_name})

        # Non-AJAX -> old behavior: redirect to play route
        return redirect(url_for('play_video', filename=unique_name))

    # GET: render upload page
    return render_template('upload_video.html')


@app.route('/play/<filename>')
@login_required
def play_video(filename):
    return render_template('play_video.html', filename=filename)


@app.route('/stream_video/<path:filename>')
@login_required
def stream_video(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        return 'Not found', 404
    return Response(detector.video_frame_generator(path, callback=fall_callback),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live')
@login_required
def live():
    return render_template('live.html')


@app.route('/stream_live')
@login_required
def stream_live():
    return Response(detector.camera_frame_generator(0, callback=fall_callback),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html')


@app.route('/api/analytics')
@login_required
def api_analytics():
    with events_lock:
        count = len(fall_events)
    return jsonify({'fall_count': count})


@app.route('/api/users')
@login_required
def api_users():
    """API endpoint to return list of users for admin panel."""
    # Only allow doctors to view users
    role = session.get("role", "")
    if role != "Doctor":
        return jsonify({"success": False, "message": "Forbidden"}), 403
    
    try:
        if DB_CONNECTED and users is not None:
            # Get users from MongoDB
            user_list = []
            for user_doc in users.find():
                user_list.append({
                    "_id": str(user_doc.get("_id", "")),
                    "username": user_doc.get("username", ""),
                    "email": user_doc.get("email", ""),
                    "role": user_doc.get("role", "")
                })
            return jsonify({"success": True, "users": user_list})
        else:
            # Get users from in-memory store
            user_list = []
            for email, user_doc in _in_memory_users.items():
                user_list.append({
                    "_id": email,
                    "username": user_doc.get("username", ""),
                    "email": user_doc.get("email", ""),
                    "role": user_doc.get("role", "")
                })
            return jsonify({"success": True, "users": user_list})
    except Exception as e:
        logging.exception("Failed to fetch users")
        return jsonify({"success": False, "message": "Error fetching users"}), 500


@app.route('/api/add_user', methods=['POST'])
@login_required
def api_add_user():
    """API endpoint to add a new user (admin only)."""
    # Only allow doctors to add users
    role = session.get("role", "")
    if role != "Doctor":
        return jsonify({"success": False, "message": "Forbidden"}), 403
    
    try:
        data = request.get_json()
        username = data.get("username", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        user_role = data.get("role", "").strip()
        hospital_id = data.get("hospital_id", "").strip()

        # Validation
        if not all([username, email, password, user_role, hospital_id]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400

        if hospital_id != EXPECTED_HOSPITAL_ID:
            return jsonify({"success": False, "message": "Invalid hospital ID"}), 400

        if user_role not in {"Nurse", "Doctor"}:
            return jsonify({"success": False, "message": "Invalid role"}), 400

        if "@" not in email or "." not in email.split("@")[-1]:
            return jsonify({"success": False, "message": "Invalid email format"}), 400

        # Hash password and hospital_id
        pw_hash = hash_text(password)
        hospital_hash = hash_text(hospital_id)
        
        user_doc = {
            "username": username,
            "email": email,
            "password_hashed": pw_hash,
            "hospital_id_hashed": hospital_hash,
            "role": user_role,
            "created_at": datetime.datetime.utcnow()
        }

        # Insert user
        insert_user_doc(user_doc)
        return jsonify({"success": True, "message": "User added successfully"})

    except DuplicateKeyError:
        return jsonify({"success": False, "message": "Email already registered"}), 400
    except Exception as e:
        logging.exception("Failed to add user")
        return jsonify({"success": False, "message": "Error adding user"}), 500


@app.route('/api/update_user', methods=['POST'])
@login_required
def api_update_user():
    """API endpoint to update a user (admin only)."""
    # Only allow doctors to update users
    role = session.get("role", "")
    if role != "Doctor":
        return jsonify({"success": False, "message": "Forbidden"}), 403
    
    try:
        data = request.get_json()
        user_id = data.get("user_id", "").strip()
        username = data.get("username", "").strip()
        email = data.get("email", "").strip().lower()
        user_role = data.get("role", "").strip()

        # Validation
        if not all([user_id, username, email, user_role]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400

        if user_role not in {"Nurse", "Doctor"}:
            return jsonify({"success": False, "message": "Invalid role"}), 400

        if "@" not in email or "." not in email.split("@")[-1]:
            return jsonify({"success": False, "message": "Invalid email format"}), 400

        # Update user
        if DB_CONNECTED and users is not None:
            from bson import ObjectId
            try:
                obj_id = ObjectId(user_id)
            except Exception:
                return jsonify({"success": False, "message": "Invalid user ID"}), 400
            
            result = users.update_one(
                {"_id": obj_id},
                {"$set": {
                    "username": username,
                    "email": email,
                    "role": user_role
                }}
            )
            if result.matched_count == 0:
                return jsonify({"success": False, "message": "User not found"}), 404
        else:
            # In-memory update
            if email in _in_memory_users:
                _in_memory_users[email]["username"] = username
                _in_memory_users[email]["email"] = email
                _in_memory_users[email]["role"] = user_role
            else:
                return jsonify({"success": False, "message": "User not found"}), 404

        return jsonify({"success": True, "message": "User updated successfully"})

    except Exception as e:
        logging.exception("Failed to update user")
        return jsonify({"success": False, "message": "Error updating user"}), 500


@app.route('/api/delete_user', methods=['POST'])
@login_required
def api_delete_user():
    """API endpoint to delete a user (admin only)."""
    # Only allow doctors to delete users
    role = session.get("role", "")
    if role != "Doctor":
        return jsonify({"success": False, "message": "Forbidden"}), 403
    
    try:
        data = request.get_json()
        user_id = data.get("user_id", "").strip()

        if not user_id:
            return jsonify({"success": False, "message": "Missing user ID"}), 400

        # Delete user
        if DB_CONNECTED and users is not None:
            from bson import ObjectId
            try:
                obj_id = ObjectId(user_id)
            except Exception:
                return jsonify({"success": False, "message": "Invalid user ID"}), 400
            
            result = users.delete_one({"_id": obj_id})
            if result.deleted_count == 0:
                return jsonify({"success": False, "message": "User not found"}), 404
        else:
            # In-memory delete
            found = False
            for email, user_doc in list(_in_memory_users.items()):
                if email == user_id:
                    del _in_memory_users[email]
                    found = True
                    break
            if not found:
                return jsonify({"success": False, "message": "User not found"}), 404

        return jsonify({"success": True, "message": "User deleted successfully"})

    except Exception as e:
        logging.exception("Failed to delete user")
        return jsonify({"success": False, "message": "Error deleting user"}), 500


@app.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    # Serve the original saved file
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ---- Error handlers ----
@app.errorhandler(403)
def forbidden(e):
    # render a friendly 403 page if you have templates/403.html; otherwise return text
    try:
        return render_template('403.html'), 403
    except Exception:
        return "Forbidden", 403


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
0