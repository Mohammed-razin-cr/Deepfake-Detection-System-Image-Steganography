"""Simple authentication system with email/password."""

import os
import json
import hashlib
import secrets
from pathlib import Path
from datetime import datetime

# User storage file
USERS_FILE = Path(__file__).parent / 'users.json'


def init_users_file():
    """Initialize users.json file if it doesn't exist."""
    if not USERS_FILE.exists():
        # Create default admin user
        default_users = {
            'users': [
                {
                    'email': 'admin@example.com',
                    'password_hash': hash_password('admin123'),
                    'created_at': datetime.now().isoformat(),
                    'is_admin': True
                }
            ]
        }
        save_users(default_users)
        return default_users
    return load_users()


def load_users():
    """Load users from JSON file."""
    if not USERS_FILE.exists():
        return init_users_file()
    
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # If file is corrupted, reinitialize
        return init_users_file()


def save_users(users_data):
    """Save users to JSON file."""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users_data, f, indent=2)
        return True
    except IOError:
        return False


def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password, password_hash):
    """Verify password against hash."""
    return hash_password(password) == password_hash


def register_user(email, password):
    """Register a new user."""
    users_data = load_users()
    
    # Check if user already exists
    for user in users_data.get('users', []):
        if user.get('email') == email.lower():
            return False, 'User already exists'
    
    # Create new user
    new_user = {
        'email': email.lower(),
        'password_hash': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'is_admin': False
    }
    
    users_data.setdefault('users', []).append(new_user)
    
    if save_users(users_data):
        return True, 'User registered successfully'
    return False, 'Failed to save user'


def authenticate_user(email, password):
    """Authenticate user and return user data if valid."""
    users_data = load_users()
    
    for user in users_data.get('users', []):
        if user.get('email') == email.lower():
            if verify_password(password, user.get('password_hash')):
                return True, {
                    'email': user.get('email'),
                    'is_admin': user.get('is_admin', False),
                    'created_at': user.get('created_at')
                }
            else:
                return False, 'Invalid password'
    
    return False, 'User not found'


def get_user_by_email(email):
    """Get user by email."""
    users_data = load_users()
    
    for user in users_data.get('users', []):
        if user.get('email') == email.lower():
            return {
                'email': user.get('email'),
                'is_admin': user.get('is_admin', False),
                'created_at': user.get('created_at')
            }
    return None

