# auth_logic.py
"""
Register / Login logic for MongoDB (no Flask/web framework code).
- Register: requires username, email, password, role (Nurse/Doctor), hospital_id (must equal "Hospital@45")
  -> stores { username, email, password_hashed, role, hospital_id_hashed } in MongoDB (hospital_id saved only as a bcrypt hash)
- Login: requires email, password -> verifies credentials, returns user info
- Optional: helper to create JWT (requires PyJWT)
"""

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import bcrypt
import datetime
import jwt   # optional, install pyjwt if you want tokens

# CONFIG - change to your own values
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "mydb"
USERS_COLL = "users"
JWT_SECRET = "change_this_to_a_secure_secret"
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_MINUTES = 60 * 24  # 1 day

# Hospital ID that must be provided on register (but only stored as hashed)
EXPECTED_HOSPITAL_ID = "Hospital@45"

# --- Setup client & ensure indexes ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users = db[USERS_COLL]

# Ensure unique index on email
users.create_index("email", unique=True)


# --- Helpers ---
def hash_password(plain_text: str) -> str:
    """Return bcrypt hashed string (utf-8). Used for both passwords and hospital_id."""
    if not isinstance(plain_text, str):
        raise TypeError("Input must be a string")
    b = plain_text.encode("utf-8")
    hashed = bcrypt.hashpw(b, bcrypt.gensalt())
    return hashed.decode("utf-8")


def check_password(plain_text: str, hashed_text: str) -> bool:
    """Return True if plain_text matches hashed_text (bcrypt)."""
    try:
        return bcrypt.checkpw(plain_text.encode("utf-8"), hashed_text.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_jwt(payload: dict, expires_minutes: int = JWT_EXPIRES_MINUTES) -> str:
    """Create a JWT with standard exp claim. (Optional - requires pyjwt)"""
    exp = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_minutes)
    to_encode = payload.copy()
    to_encode.update({"exp": exp})
    token = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    # PyJWT >= 2.0 returns str, else bytes
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def decode_jwt(token: str) -> dict:
    """Decode and verify JWT; raises jwt exceptions on error."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# --- Core logic functions (framework-agnostic) ---
def register_user(data: dict) -> dict:
    """
    Register a new user.
    Expected data keys: username, email, password, role, hospital_id
    Returns a result dict: { success: bool, message: str, user: <user dict without sensitive fields> }
    """
    # Basic validation
    required = ["username", "email", "password", "role", "hospital_id"]
    for k in required:
        if k not in data or not data[k]:
            return {"success": False, "message": f"Missing required field: {k}"}

    username = str(data["username"]).strip()
    email = str(data["email"]).strip().lower()
    password = data["password"]
    role = str(data["role"]).strip()
    hospital_id = str(data["hospital_id"]).strip()

    # Validate hospital id (must match expected)
    if hospital_id != EXPECTED_HOSPITAL_ID:
        return {"success": False, "message": "Invalid hospital id."}

    # Validate role value
    allowed_roles = {"Nurse", "Doctor"}
    if role not in allowed_roles:
        return {"success": False, "message": f"Invalid role. Allowed: {', '.join(allowed_roles)}"}

    # Validate email format (simple)
    if "@" not in email or "." not in email.split("@")[-1]:
        return {"success": False, "message": "Invalid email address."}

    # Hash password and hospital_id
    try:
        hashed_pw = hash_password(password)
        hashed_hospital = hash_password(hospital_id)
    except Exception as e:
        return {"success": False, "message": f"Error hashing sensitive fields: {e}"}

    # Build user document (DO NOT include plaintext hospital_id or password)
    user_doc = {
        "username": username,
        "email": email,
        "password_hashed": hashed_pw,
        "hospital_id_hashed": hashed_hospital,
        "role": role,
        "created_at": datetime.datetime.utcnow()
    }

    # Insert into MongoDB
    try:
        result = users.insert_one(user_doc)
    except DuplicateKeyError:
        return {"success": False, "message": "Email already registered."}
    except Exception as e:
        return {"success": False, "message": f"Database error: {e}"}

    # Return created user info (excluding sensitive fields)
    user_out = {
        "_id": str(result.inserted_id),
        "username": username,
        "email": email,
        "role": role,
        "created_at": user_doc["created_at"].isoformat() + "Z"
    }
    return {"success": True, "message": "User registered successfully.", "user": user_out}


def login_user(data: dict, issue_jwt: bool = False) -> dict:
    """
    Login a user.
    Expected data keys: email, password
    If issue_jwt is True, returns token in the response on success.
    Returns: { success: bool, message: str, user: {...} , token: <optional> }
    """
    # Basic validation
    if "email" not in data or "password" not in data:
        return {"success": False, "message": "Email and password are required."}

    email = str(data["email"]).strip().lower()
    password = data["password"]

    # Find user
    user = users.find_one({"email": email})
    if not user:
        return {"success": False, "message": "Invalid credentials."}

    # Verify password
    stored_hash = user.get("password_hashed")
    if not stored_hash or not check_password(password, stored_hash):
        return {"success": False, "message": "Invalid credentials."}

    # Build user object to return (omit sensitive hashes)
    user_out = {
        "_id": str(user.get("_id")),
        "username": user.get("username"),
        "email": user.get("email"),
        "role": user.get("role"),
        "created_at": user.get("created_at").isoformat() + "Z" if user.get("created_at") else None
    }

    response = {"success": True, "message": "Login successful.", "user": user_out}

    if issue_jwt:
        # Minimal claims: sub/email/role
        payload = {"sub": str(user.get("_id")), "email": user.get("email"), "role": user.get("role")}
        token = create_jwt(payload)
        response["token"] = token

    return response


# --- Example usage (for testing / reference) ---
if __name__ == "__main__":
    # Example register (hospital_id will be validated then stored hashed)
    reg = register_user({
        "username": "Alice",
        "email": "alice@example.com",
        "password": "StrongP@ssw0rd",
        "role": "Nurse",
        "hospital_id": "Hospital@45"
    })
    print("Register:", reg)

    # Example login
    login = login_user({"email": "alice@example.com", "password": "StrongP@ssw0rd"}, issue_jwt=True)
    print("Login:", login)
