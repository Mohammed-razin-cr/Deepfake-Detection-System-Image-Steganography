# Simple Authentication System

This project now uses a simple email/password authentication system. No external services required!

## Features

- ✅ Email/password login
- ✅ User registration
- ✅ Session-based authentication
- ✅ Protected API routes
- ✅ User management with JSON storage

## Default Credentials

When you first run the application, a default admin account is created:

- **Email**: `admin@example.com`
- **Password**: `admin123`

⚠️ **Important**: Change the default password after first login!

## How It Works

1. Users are stored in `web_app/users.json`
2. Passwords are hashed using SHA-256
3. Sessions are managed by Flask
4. All API routes are protected and require authentication

## Usage

### Login

1. Navigate to the application (you'll be redirected to login if not authenticated)
2. Enter your email and password
3. Click "Sign In"

### Register New Account

1. Click the "Register" tab on the login page
2. Enter your email and password (minimum 6 characters)
3. Confirm your password
4. Click "Create Account"
5. You'll be automatically logged in

### Logout

Click the "Logout" button in the header.

## User Storage

Users are stored in `web_app/users.json`. The file structure:

```json
{
  "users": [
    {
      "email": "admin@example.com",
      "password_hash": "...",
      "created_at": "2024-01-01T12:00:00",
      "is_admin": true
    }
  ]
}
```

## Security Notes

- Passwords are hashed (not stored in plain text)
- Sessions expire when browser is closed
- API routes require authentication
- Change the default admin password immediately

## Adding Users

### Method 1: Through the Web Interface
- Use the registration form on the login page

### Method 2: Manually (Advanced)
You can edit `web_app/users.json` directly, but you'll need to hash passwords. To hash a password:

```python
import hashlib
password = "your_password"
hashed = hashlib.sha256(password.encode()).hexdigest()
print(hashed)
```

Then add to users.json:
```json
{
  "email": "user@example.com",
  "password_hash": "<hashed_password>",
  "created_at": "2024-01-01T12:00:00",
  "is_admin": false
}
```

## API Endpoints

- `POST /api/auth/login` - Login with email/password
- `POST /api/auth/register` - Register new user
- `POST /api/auth/logout` - Logout current user
- `GET /api/auth/user` - Get current user info

All other API routes require authentication and will return 401 if not logged in.

## Troubleshooting

### Can't login
- Check email and password are correct
- Verify users.json exists and is readable
- Check Flask server logs for errors

### Users.json not found
- The file is created automatically on first run
- Make sure the `web_app/` directory is writable

### Session issues
- Clear browser cookies
- Check SECRET_KEY is set in Flask config
- Restart the Flask server

---

**That's it! Simple and secure authentication without external dependencies.**

