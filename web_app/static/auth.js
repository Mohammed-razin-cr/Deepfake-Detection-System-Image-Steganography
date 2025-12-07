// Simple Authentication JavaScript

// Switch between login and register tabs
function switchTab(tab) {
    // Update tab buttons
    document.getElementById('login-tab-btn').classList.toggle('active', tab === 'login');
    document.getElementById('register-tab-btn').classList.toggle('active', tab === 'register');
    
    // Update tab content
    document.getElementById('login-form').classList.toggle('active', tab === 'login');
    document.getElementById('register-form').classList.toggle('active', tab === 'register');
    
    // Clear messages
    showMessage('', 'error');
    showMessage('', 'success');
}

// Show error/success messages
function showMessage(message, type) {
    const errorEl = document.getElementById('error-message');
    const successEl = document.getElementById('success-message');
    
    if (type === 'error') {
        errorEl.textContent = message;
        errorEl.classList.toggle('show', message !== '');
        successEl.classList.remove('show');
    } else if (type === 'success') {
        successEl.textContent = message;
        successEl.classList.toggle('show', message !== '');
        errorEl.classList.remove('show');
    }
}

// Handle login form submission
async function handleLogin(event) {
    event.preventDefault();
    
    const email = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value;
    const submitBtn = document.getElementById('login-submit-btn');
    
    if (!email || !password) {
        showMessage('Please enter both email and password', 'error');
        return;
    }
    
    try {
        showMessage('', 'error');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Signing in...';
        
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email: email,
                password: password
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.status === 'success') {
            showMessage('Login successful! Redirecting...', 'success');
            setTimeout(() => {
                window.location.href = '/';
            }, 1000);
        } else {
            throw new Error(data.error || 'Login failed');
        }
    } catch (error) {
        console.error('Login error:', error);
        showMessage(error.message || 'Failed to login. Please check your credentials.', 'error');
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Sign In';
    }
}

// Handle register form submission
async function handleRegister(event) {
    event.preventDefault();
    
    const email = document.getElementById('register-email').value.trim();
    const password = document.getElementById('register-password').value;
    const passwordConfirm = document.getElementById('register-password-confirm').value;
    const submitBtn = document.getElementById('register-submit-btn');
    
    // Validate inputs
    if (!email || !password || !passwordConfirm) {
        showMessage('Please fill in all fields', 'error');
        return;
    }
    
    if (password.length < 6) {
        showMessage('Password must be at least 6 characters long', 'error');
        return;
    }
    
    if (password !== passwordConfirm) {
        showMessage('Passwords do not match', 'error');
        return;
    }
    
    try {
        showMessage('', 'error');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating account...';
        
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email: email,
                password: password
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.status === 'success') {
            showMessage('Account created successfully! Redirecting...', 'success');
            setTimeout(() => {
                window.location.href = '/';
            }, 1500);
        } else {
            throw new Error(data.error || 'Registration failed');
        }
    } catch (error) {
        console.error('Registration error:', error);
        showMessage(error.message || 'Failed to create account. Please try again.', 'error');
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-user-plus"></i> Create Account';
    }
}
