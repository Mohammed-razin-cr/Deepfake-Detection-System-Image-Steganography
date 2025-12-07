// Global Variables
let currentFile = null;
let currentFileType = null;
const API_BASE = '/api';

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    checkAuth();
    initializeEventListeners();
    loadHistory();
});

// Check authentication status
async function checkAuth() {
    try {
        const response = await fetch('/api/auth/user');
        const data = await response.json();
        
        if (data.status === 'success' && data.user) {
            // User is authenticated
            const userEmail = document.getElementById('user-email');
            if (userEmail) {
                userEmail.textContent = data.user.email || 'User';
            }
        } else {
            // User is not authenticated, redirect to login
            window.location.href = '/login';
        }
    } catch (error) {
        console.error('Auth check error:', error);
        window.location.href = '/login';
    }
}

// Logout function
async function logout() {
    try {
        const response = await fetch('/api/auth/logout', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            window.location.href = '/login';
        }
    } catch (error) {
        console.error('Logout error:', error);
        // Still redirect to login even if logout fails
        window.location.href = '/login';
    }
}

// Initialize all event listeners
function initializeEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            switchTab(this.dataset.tab);
        });
    });

    // Image upload
    const imageUploadArea = document.getElementById('image-upload-area');
    const imageInput = document.getElementById('image-input');
    
    imageUploadArea.addEventListener('click', () => imageInput.click());
    imageUploadArea.addEventListener('dragover', handleDragOver);
    imageUploadArea.addEventListener('dragleave', handleDragLeave);
    imageUploadArea.addEventListener('drop', (e) => handleDrop(e, 'image'));
    imageInput.addEventListener('change', (e) => handleFileSelect(e, 'image'));

    // Video upload
    const videoUploadArea = document.getElementById('video-upload-area');
    const videoInput = document.getElementById('video-input');
    
    videoUploadArea.addEventListener('click', () => videoInput.click());
    videoUploadArea.addEventListener('dragover', handleDragOver);
    videoUploadArea.addEventListener('dragleave', handleDragLeave);
    videoUploadArea.addEventListener('drop', (e) => handleDrop(e, 'video'));
    videoInput.addEventListener('change', (e) => handleFileSelect(e, 'video'));

    // Detection buttons
    document.getElementById('detect-image-btn').addEventListener('click', detectImage);
    document.getElementById('detect-video-btn').addEventListener('click', detectVideo);

    // Results actions
    document.getElementById('close-results').addEventListener('click', closeResults);
    document.getElementById('new-detection-btn').addEventListener('click', closeResults);
    document.getElementById('download-btn').addEventListener('click', downloadResults);

    // History
    document.getElementById('refresh-history').addEventListener('click', loadHistory);
    document.getElementById('clear-history').addEventListener('click', clearHistory);
    
    // Logout
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
    }

    // Encryption/Decryption upload areas
    const encryptUploadArea = document.getElementById('encrypt-upload-area');
    const encryptImageInput = document.getElementById('encrypt-image-input');
    const decryptUploadArea = document.getElementById('decrypt-upload-area');
    const decryptImageInput = document.getElementById('decrypt-image-input');

    if (encryptUploadArea && encryptImageInput) {
        encryptUploadArea.addEventListener('click', () => encryptImageInput.click());
        encryptUploadArea.addEventListener('dragover', handleDragOver);
        encryptUploadArea.addEventListener('dragleave', handleDragLeave);
        encryptUploadArea.addEventListener('drop', (e) => handleDrop(e, 'encrypt'));
        encryptImageInput.addEventListener('change', (e) => handleFileSelect(e, 'encrypt'));
    }

    if (decryptUploadArea && decryptImageInput) {
        decryptUploadArea.addEventListener('click', () => decryptImageInput.click());
        decryptUploadArea.addEventListener('dragover', handleDragOver);
        decryptUploadArea.addEventListener('dragleave', handleDragLeave);
        decryptUploadArea.addEventListener('drop', (e) => handleDrop(e, 'decrypt'));
        decryptImageInput.addEventListener('change', (e) => handleFileSelect(e, 'decrypt'));
    }

    // Encryption/Decryption buttons
    const encryptBtn = document.getElementById('encrypt-btn');
    const decryptBtn = document.getElementById('decrypt-btn');
    const checkEncryptedBtn = document.getElementById('check-encrypted-btn');
    const closeStegoResults = document.getElementById('close-stego-results');
    const newStegoBtn = document.getElementById('new-stego-btn');
    const downloadStegoBtn = document.getElementById('download-stego-btn');

    if (encryptBtn) encryptBtn.addEventListener('click', encryptImage);
    if (decryptBtn) decryptBtn.addEventListener('click', decryptImage);
    if (checkEncryptedBtn) checkEncryptedBtn.addEventListener('click', checkIfEncrypted);
    if (closeStegoResults) closeStegoResults.addEventListener('click', closeStegoResultsSection);
    if (newStegoBtn) newStegoBtn.addEventListener('click', closeStegoResultsSection);
    if (downloadStegoBtn) downloadStegoBtn.addEventListener('click', downloadEncryptedImage);
}

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');

    // Add active class to clicked button
    event.target.closest('.tab-btn').classList.add('active');
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.remove('dragover');
}

// Handle file drop
function handleDrop(e, fileType) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect({ target: { files: files } }, fileType);
    }
}

// Handle file selection
function handleFileSelect(e, fileType) {
    const files = e.target.files;
    if (files.length === 0) return;

    const file = files[0];
    currentFile = file;
    currentFileType = fileType;

    // Get upload area and button
    let uploadAreaId, detectBtn;
    if (fileType === 'image') {
        uploadAreaId = 'image-upload-area';
        detectBtn = document.getElementById('detect-image-btn');
    } else if (fileType === 'video') {
        uploadAreaId = 'video-upload-area';
        detectBtn = document.getElementById('detect-video-btn');
    } else if (fileType === 'encrypt') {
        uploadAreaId = 'encrypt-upload-area';
        detectBtn = document.getElementById('encrypt-btn');
    } else if (fileType === 'decrypt') {
        uploadAreaId = 'decrypt-upload-area';
        detectBtn = document.getElementById('decrypt-btn');
        const checkBtn = document.getElementById('check-encrypted-btn');
        if (checkBtn) checkBtn.disabled = false;
    }
    
    const uploadArea = document.getElementById(uploadAreaId);

    // Update upload area
    const fileSize = (file.size / (1024 * 1024)).toFixed(2);
    uploadArea.innerHTML = `
        <i class="fas fa-check-circle" style="color: var(--success-color);"></i>
        <h3>File Selected</h3>
        <p><strong>${file.name}</strong></p>
        <p>${fileSize} MB</p>
    `;

    // Enable detect/encrypt/decrypt button
    if (fileType === 'encrypt') {
        // For encrypt, check if message is also filled
        const message = document.getElementById('encrypt-message');
        if (message) {
            const checkEncryptReady = () => {
                detectBtn.disabled = !(currentFile && message.value.trim().length > 0);
            };
            message.addEventListener('input', checkEncryptReady);
            checkEncryptReady();
        }
    } else {
        detectBtn.disabled = false;
    }
}

// Detect image
async function detectImage() {
    if (!currentFile) {
        alert('Please select an image first');
        return;
    }

    await performDetection('image');
}

// Detect video
async function detectVideo() {
    if (!currentFile) {
        alert('Please select a video first');
        return;
    }

    await performDetection('video');
}

// Perform detection
async function performDetection(type) {
    const formData = new FormData();
    formData.append('file', currentFile);

    showProgress(true);

    try {
        const endpoint = type === 'image' ? '/api/detect/image' : '/api/detect/video';
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Detection failed');
        }

        const results = await response.json();
        displayResults(results);
        loadHistory();
    } catch (error) {
        console.error('Error:', error);
        alert(`Detection error: ${error.message}`);
    } finally {
        showProgress(false);
    }
}

// Show/hide progress
function showProgress(show) {
    document.getElementById('progress-container').classList.toggle('hidden', !show);
}

// Display results
function displayResults(results) {
    closeResults(); // Reset previous results
    
    const resultsSection = document.getElementById('results-section');
    const resultVerdict = document.getElementById('result-verdict');
    const fakeProb = parseFloat(results.fake_probability);
    const realProb = parseFloat(results.real_probability);
    const confidence = parseFloat(results.confidence);

    // Set verdict
    if (results.is_fake) {
        resultVerdict.className = 'result-verdict fake';
        resultVerdict.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${results.label}`;
    } else {
        resultVerdict.className = 'result-verdict authentic';
        resultVerdict.innerHTML = `<i class="fas fa-check-circle"></i> ${results.label}`;
    }

    // Update probabilities
    document.getElementById('fake-prob').textContent = (fakeProb * 100).toFixed(1) + '%';
    document.getElementById('real-prob').textContent = (realProb * 100).toFixed(1) + '%';
    document.getElementById('confidence').textContent = (confidence * 100).toFixed(1) + '%';

    // Update bars
    document.getElementById('fake-bar').style.width = (fakeProb * 100) + '%';
    document.getElementById('real-bar').style.width = (realProb * 100) + '%';
    document.getElementById('confidence-bar').style.width = (confidence * 100) + '%';

    // Update video info if present
    if (results.video_info) {
        document.getElementById('video-info').classList.remove('hidden');
        document.getElementById('video-duration').textContent = results.video_info.duration.toFixed(2) + ' s';
        document.getElementById('video-fps').textContent = results.video_info.fps.toFixed(1);
        document.getElementById('video-resolution').textContent = results.video_info.resolution;
        document.getElementById('video-frames').textContent = results.video_info.frame_count;
    } else {
        document.getElementById('video-info').classList.add('hidden');
    }

    // Display encrypted data detection (if present)
    displayEncryptedDataDetection(results);

    // Store current results for download
    window.currentResults = results;

    // Show results section
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Close results
function closeResults() {
    document.getElementById('results-section').classList.add('hidden');
    
    // Reset file selection
    currentFile = null;
    document.getElementById('image-upload-area').innerHTML = `
        <i class="fas fa-cloud-upload-alt"></i>
        <h3>Upload Image</h3>
        <p>Drag and drop or click to select image file</p>
        <p class="file-types">Supported: JPG, PNG, BMP, GIF</p>
    `;
    document.getElementById('video-upload-area').innerHTML = `
        <i class="fas fa-cloud-upload-alt"></i>
        <h3>Upload Video</h3>
        <p>Drag and drop or click to select video file</p>
        <p class="file-types">Supported: MP4, AVI, MOV, MKV, FLV (Max 500MB)</p>
    `;
    document.getElementById('detect-image-btn').disabled = true;
    document.getElementById('detect-video-btn').disabled = true;
}

// Download results
function downloadResults() {
    if (!window.currentResults) return;

    const data = JSON.stringify(window.currentResults, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `mohammed_razin_cr_detection_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// Load history
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        const historyList = document.getElementById('history-list');
        
        if (!data.history || data.history.length === 0) {
            historyList.innerHTML = '<p class="empty-message">No recent detections</p>';
            return;
        }

        historyList.innerHTML = data.history.map(item => `
            <div class="history-item">
                <div class="history-item-info">
                    <div class="history-item-name">${item.filename}</div>
                    <div class="history-item-details">
                        ${item.size_mb} MB • ${new Date(item.uploaded_at).toLocaleString()}
                    </div>
                </div>
                <a href="/downloads/${item.filename}" class="btn btn-small" download>
                    <i class="fas fa-download"></i> Download
                </a>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Clear history
async function clearHistory() {
    if (!confirm('Are you sure you want to clear all uploaded files?')) return;

    try {
        const response = await fetch('/api/clear-history', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'success') {
            alert('History cleared');
            loadHistory();
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        alert('Failed to clear history');
    }
}

// Update progress text
setInterval(() => {
    const progressText = document.getElementById('progress-text');
    if (!document.getElementById('progress-container').classList.contains('hidden')) {
        const messages = [
            'Processing your file...',
            'Analyzing with CNN...',
            'Running ResNext model...',
            'Checking with Vision Transformer...',
            'Computing ensemble prediction...',
            'Finalizing results...'
        ];
        
        const randomMessage = messages[Math.floor(Math.random() * messages.length)];
        progressText.textContent = randomMessage;
    }
}, 3000);

// Feedback handlers
document.getElementById('feedback-correct').addEventListener('click', function() {
    submitFeedback('correct');
});

document.getElementById('feedback-incorrect').addEventListener('click', function() {
    submitFeedback('incorrect');
});

// Retrain button handler
document.getElementById('retrain-btn').addEventListener('click', function() {
    triggerRetraining();
});

// Update RL stats when results are shown
function updateRLStats() {
    fetch('/api/rl/stats')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('feedback-count').textContent = data.feedback_count || 0;
                document.getElementById('rl-trainings').textContent = data.rl_stats.total_trainings || 0;
                document.getElementById('rl-improvements').textContent = data.rl_stats.improvements || 0;
                const avgLoss = data.rl_stats.avg_loss || 0;
                document.getElementById('rl-loss').textContent = avgLoss > 0 ? avgLoss.toFixed(4) : '-';
            }
        })
        .catch(error => console.error('Error updating RL stats:', error));
}

// Call this after showing results
function closeResults() {
    document.getElementById('results-section').classList.add('hidden');
    currentFile = null;
    currentFileType = null;
    updateRLStats();  // Update stats when results close
}

// Submit feedback
async function submitFeedback(feedbackType) {
    const feedbackMessage = document.getElementById('feedback-message');
    const buttons = document.querySelectorAll('.btn-feedback');
    
    // Get current prediction data
    const isFake = document.getElementById('result-verdict').textContent.toLowerCase().includes('deepfake');
    const fakeProb = parseFloat(document.getElementById('fake-prob').textContent) / 100;
    const realProb = parseFloat(document.getElementById('real-prob').textContent) / 100;
    const fileName = currentFile ? currentFile.name : 'unknown';
    
    try {
        // Disable buttons during submission
        buttons.forEach(btn => btn.disabled = true);
        
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                feedback_type: feedbackType,
                prediction: isFake ? 1 : 0,
                fake_probability: fakeProb,
                real_probability: realProb,
                file_name: fileName,
                file_type: currentFileType
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            feedbackMessage.classList.remove('hidden', 'error');
            feedbackMessage.classList.add('success');
            let message = `<i class="fas fa-check-circle"></i> ${data.message}`;
            if (data.rl_stats && data.rl_stats.total_trainings > 0) {
                message += ` (RL: ${data.rl_stats.total_trainings} trainings, ${data.rl_stats.improvements} improvements)`;
            }
            feedbackMessage.innerHTML = message;
            
            // Update stats
            updateRLStats();
        } else {
            throw new Error('Failed to submit feedback');
        }
    } catch (error) {
        console.error('Error submitting feedback:', error);
        feedbackMessage.classList.remove('hidden', 'success');
        feedbackMessage.classList.add('error');
        feedbackMessage.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error submitting feedback. Please try again.`;
    } finally {
        // Re-enable buttons
        buttons.forEach(btn => btn.disabled = false);
        
        // Hide message after 5 seconds
        setTimeout(() => {
            feedbackMessage.classList.add('hidden');
        }, 5000);
    }
}

// Trigger RL retraining
async function triggerRetraining() {
    const retrainMessage = document.getElementById('retrain-message');
    const retrainBtn = document.getElementById('retrain-btn');
    
    try {
        retrainBtn.disabled = true;
        retrainMessage.classList.remove('hidden', 'error', 'info');
        retrainMessage.classList.add('info');
        retrainMessage.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training model on collected feedback...';
        
        const response = await fetch('/api/rl/retrain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            retrainMessage.classList.remove('info', 'error');
            retrainMessage.classList.add('success');
            retrainMessage.innerHTML = `<i class="fas fa-check-circle"></i> ${data.message}<br>
                Trained: ${data.trained_samples} samples, Improvements: ${data.improvements}`;
            
            // Update stats
            updateRLStats();
        } else if (data.status === 'info') {
            retrainMessage.classList.remove('success', 'error');
            retrainMessage.classList.add('info');
            retrainMessage.innerHTML = `<i class="fas fa-info-circle"></i> ${data.message}`;
        } else {
            throw new Error(data.message || 'Retraining failed');
        }
    } catch (error) {
        console.error('Error triggering retraining:', error);
        retrainMessage.classList.remove('success', 'info');
        retrainMessage.classList.add('error');
        retrainMessage.innerHTML = `<i class="fas fa-exclamation-circle"></i> Retraining failed: ${error.message}`;
    } finally {
        retrainBtn.disabled = false;
        
        // Hide message after 6 seconds
        setTimeout(() => {
            retrainMessage.classList.add('hidden');
        }, 6000);
    }
}

// Load RL stats on page load
document.addEventListener('DOMContentLoaded', function() {
    updateRLStats();
});

// Update RL stats periodically
setInterval(updateRLStats, 10000);  // Update every 10 seconds

// ============================================
// STEGANOGRAPHY FUNCTIONS (Encrypt/Decrypt)
// ============================================

let encryptedImageFile = null;

// Encrypt Image
async function encryptImage() {
    if (!currentFile) {
        alert('Please select an image first');
        return;
    }

    const message = document.getElementById('encrypt-message').value.trim();
    if (!message) {
        alert('Please enter a message to hide in the image');
        return;
    }

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('message', message);

    showProgress(true);
    document.getElementById('progress-text').textContent = 'Encrypting image...';

    try {
        const response = await fetch('/api/encrypt/image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Encryption failed');
        }

        const results = await response.json();
        displayEncryptionResults(results, 'encrypt');
        encryptedImageFile = results.encrypted_file;
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Encryption error: ${error.message}`);
    } finally {
        showProgress(false);
    }
}

// Decrypt Image
async function decryptImage() {
    if (!currentFile) {
        alert('Please select an encrypted image first');
        return;
    }

    const formData = new FormData();
    formData.append('file', currentFile);

    showProgress(true);
    document.getElementById('progress-text').textContent = 'Decrypting image...';

    try {
        const response = await fetch('/api/decrypt/image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Decryption failed');
        }

        const results = await response.json();
        displayEncryptionResults(results, 'decrypt');
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Decryption error: ${error.message}`);
    } finally {
        showProgress(false);
    }
}

// Check if image is encrypted
async function checkIfEncrypted() {
    if (!currentFile) {
        alert('Please select an image first');
        return;
    }

    const formData = new FormData();
    formData.append('file', currentFile);

    showProgress(true);
    document.getElementById('progress-text').textContent = 'Checking image...';

    try {
        const response = await fetch('/api/check/encrypted', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Check failed');
        }

        const results = await response.json();
        displayEncryptionResults(results, 'check');
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Check error: ${error.message}`);
    } finally {
        showProgress(false);
    }
}

// Display encryption/decryption results
function displayEncryptionResults(results, type) {
    const resultsSection = document.getElementById('stego-results-section');
    const resultContent = document.getElementById('stego-result-content');
    const resultTitle = document.getElementById('stego-result-title');
    const downloadBtn = document.getElementById('download-stego-btn');

    if (type === 'encrypt' && results.status === 'success') {
        resultTitle.textContent = 'Image Encrypted Successfully';
        resultContent.innerHTML = `
            <div style="padding: 20px; background: #f0fdf4; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #10b981; margin-bottom: 15px;">
                    <i class="fas fa-check-circle"></i> Encryption Successful!
                </h3>
                <div style="margin: 15px 0;">
                    <p><strong>Encrypted File:</strong> ${results.encrypted_file}</p>
                    <p><strong>Original File:</strong> ${results.original_file}</p>
                    <p><strong>Message Length:</strong> ${results.message_length} characters</p>
                    <p><strong>Capacity Used:</strong> ${results.capacity_used}</p>
                </div>
                <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px;">
                    <p style="color: #64748b; font-size: 14px;">
                        <i class="fas fa-info-circle"></i> 
                        The encrypted image has been saved. You can download it using the button below.
                    </p>
                </div>
            </div>
        `;
        downloadBtn.style.display = 'block';
    } else if (type === 'decrypt' && results.status === 'success') {
        resultTitle.textContent = 'Data Extracted Successfully';
        const dataType = results.data_type || 'Text';
        resultContent.innerHTML = `
            <div style="padding: 20px; background: #eff6ff; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #3b82f6; margin-bottom: 15px;">
                    <i class="fas fa-unlock"></i> Decryption Successful!
                </h3>
                <div style="margin: 15px 0;">
                    <p><strong>Data Type:</strong> ${dataType}</p>
                    <p><strong>File:</strong> ${results.file_name}</p>
                </div>
                <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <p style="font-weight: 600; margin-bottom: 10px;">Extracted Data:</p>
                    <div style="background: #f8fafc; padding: 15px; border-radius: 6px; word-break: break-all; font-family: monospace; max-height: 300px; overflow-y: auto;">
                        ${results.extracted_data}
                    </div>
                </div>
                ${results.extracted_data.startsWith('http') ? `
                    <div style="margin-top: 15px;">
                        <a href="${results.extracted_data}" target="_blank" class="btn btn-primary">
                            <i class="fas fa-external-link-alt"></i> Open Link
                        </a>
                    </div>
                ` : ''}
            </div>
        `;
        downloadBtn.style.display = 'none';
    } else if (type === 'decrypt' && results.status === 'no_data') {
        resultTitle.textContent = 'No Encrypted Data Found';
        resultContent.innerHTML = `
            <div style="padding: 20px; background: #fef3c7; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #f59e0b; margin-bottom: 15px;">
                    <i class="fas fa-exclamation-triangle"></i> ${results.message}
                </h3>
                <p style="color: #64748b;">This image does not contain any hidden/encrypted data.</p>
            </div>
        `;
        downloadBtn.style.display = 'none';
    } else if (type === 'check') {
        resultTitle.textContent = 'Encryption Check Results';
        if (results.is_encrypted) {
            resultContent.innerHTML = `
                <div style="padding: 20px; background: #f0fdf4; border-radius: 10px; margin: 20px 0;">
                    <h3 style="color: #10b981; margin-bottom: 15px;">
                        <i class="fas fa-check-circle"></i> This image contains encrypted data!
                    </h3>
                    <p><strong>Capacity:</strong> ${results.capacity} characters</p>
                    <p style="margin-top: 15px;">
                        <button class="btn btn-primary" onclick="decryptImage()">
                            <i class="fas fa-unlock"></i> Decrypt Now
                        </button>
                    </p>
                </div>
            `;
        } else {
            resultContent.innerHTML = `
                <div style="padding: 20px; background: #fef3c7; border-radius: 10px; margin: 20px 0;">
                    <h3 style="color: #f59e0b; margin-bottom: 15px;">
                        <i class="fas fa-info-circle"></i> No encrypted data found
                    </h3>
                    <p><strong>Capacity:</strong> ${results.capacity} characters (available for encryption)</p>
                </div>
            `;
        }
        downloadBtn.style.display = 'none';
    }

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Close steganography results section
function closeStegoResultsSection() {
    document.getElementById('stego-results-section').classList.add('hidden');
    encryptedImageFile = null;
    
    // Reset encrypt form
    const encryptMessage = document.getElementById('encrypt-message');
    if (encryptMessage) encryptMessage.value = '';
    
    // Reset upload areas
    document.getElementById('encrypt-upload-area').innerHTML = `
        <i class="fas fa-cloud-upload-alt"></i>
        <h3>Upload Image to Encrypt</h3>
        <p>Drag and drop or click to select image file</p>
        <p class="file-types">Supported: PNG, JPG, JPEG</p>
    `;
    document.getElementById('decrypt-upload-area').innerHTML = `
        <i class="fas fa-cloud-upload-alt"></i>
        <h3>Upload Encrypted Image</h3>
        <p>Drag and drop or click to select encrypted image</p>
        <p class="file-types">Supported: PNG, JPG, JPEG</p>
    `;
    
    document.getElementById('encrypt-btn').disabled = true;
    document.getElementById('decrypt-btn').disabled = true;
    document.getElementById('check-encrypted-btn').disabled = true;
}

// Download encrypted image (force download to device)
function downloadEncryptedImage() {
    if (!encryptedImageFile) {
        alert('No encrypted image to download');
        return;
    }
    
    // Force download by creating a link element with download attribute
    const link = document.createElement('a');
    link.href = `/downloads/${encodeURIComponent(encryptedImageFile)}`;
    link.download = encryptedImageFile; // This forces download instead of opening
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    
    // Clean up after a delay
    setTimeout(() => {
        document.body.removeChild(link);
    }, 100);
}

// Display encrypted data detection in results
function displayEncryptedDataDetection(results) {
    const encryptedSection = document.getElementById('encrypted-data-section');
    const encryptedContent = document.getElementById('encrypted-data-content');
    
    if (!encryptedSection || !encryptedContent) return;
    
    // Check if encrypted data detection is available
    if (results.hasOwnProperty('is_encrypted')) {
        encryptedSection.style.display = 'block';
        
        if (results.is_encrypted && results.encrypted_data) {
            // Image contains encrypted data
            const dataType = results.encrypted_data_type || 'Text/Message';
            encryptedContent.innerHTML = `
                <div style="padding: 15px; background: white; border-radius: 8px; margin-top: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                        <i class="fas fa-check-circle" style="color: #10b981; font-size: 24px;"></i>
                        <div>
                            <h4 style="margin: 0; color: #10b981;">Encrypted Data Detected!</h4>
                            <p style="margin: 5px 0 0 0; color: #64748b; font-size: 14px;">Data Type: ${dataType}</p>
                        </div>
                    </div>
                    <div style="background: #f8fafc; padding: 15px; border-radius: 6px; margin-top: 15px;">
                        <p style="font-weight: 600; margin-bottom: 10px; color: #1e293b;">Extracted Hidden Data:</p>
                        <div style="background: white; padding: 12px; border-radius: 4px; word-break: break-all; font-family: monospace; max-height: 200px; overflow-y: auto; border: 1px solid #e2e8f0;">
                            ${results.encrypted_data}
                        </div>
                    </div>
                    ${results.encrypted_data.startsWith('http') ? `
                        <div style="margin-top: 15px;">
                            <a href="${results.encrypted_data}" target="_blank" class="btn btn-primary" style="display: inline-flex; align-items: center; gap: 8px;">
                                <i class="fas fa-external-link-alt"></i> Open Link
                            </a>
                        </div>
                    ` : ''}
                </div>
            `;
        } else {
            // No encrypted data found
            encryptedContent.innerHTML = `
                <div style="padding: 15px; background: white; border-radius: 8px; margin-top: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <i class="fas fa-info-circle" style="color: #64748b; font-size: 24px;"></i>
                        <div>
                            <h4 style="margin: 0; color: #64748b;">No Encrypted Data Found</h4>
                            <p style="margin: 5px 0 0 0; color: #64748b; font-size: 14px;">This image does not contain any hidden/encrypted data.</p>
                        </div>
                    </div>
                </div>
            `;
        }
    } else {
        // Encrypted data detection not available (for video or old API)
        encryptedSection.style.display = 'none';
    }
}

