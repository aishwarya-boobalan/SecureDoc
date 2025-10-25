// Global JavaScript functions for the application

// File upload validation
function validateFileUpload(input) {
    const file = input.files[0];
    const maxSize = 16 * 1024 * 1024; // 16MB
    const allowedTypes = ['application/pdf', 'application/msword', 
                         'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                         'text/plain', 'image/png', 'image/jpeg', 'image/gif'];
    
    if (file) {
        if (file.size > maxSize) {
            alert('File size must be less than 16MB');
            input.value = '';
            return false;
        }
        
        if (!allowedTypes.includes(file.type)) {
            alert('Invalid file type. Please upload PDF, DOCX, DOC, TXT, PNG, JPG, JPEG, or GIF files.');
            input.value = '';
            return false;
        }
    }
    
    return true;
}

// PIN input formatting
function formatPinInput(input) {
    input.addEventListener('input', function(e) {
        // Only allow digits
        this.value = this.value.replace(/\D/g, '');
        
        // Limit to 4 digits
        if (this.value.length > 4) {
            this.value = this.value.slice(0, 4);
        }
    });
}

// Initialize PIN inputs when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Format all PIN inputs
    const pinInputs = document.querySelectorAll('input[type="password"][pattern="\\d{4}"]');
    pinInputs.forEach(formatPinInput);
    
    // Add file upload validation
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            validateFileUpload(this);
        });
    });
});

// Show loading spinner
function showLoading(button) {
    const originalText = button.textContent;
    button.textContent = 'Loading...';
    button.disabled = true;
    
    return function() {
        button.textContent = originalText;
        button.disabled = false;
    };
}

// Copy text to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // You can add a toast notification here
        console.log('Copied to clipboard');
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Auto-hide flash messages
document.addEventListener('DOMContentLoaded', function() {
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(message => {
        setTimeout(() => {
            message.style.opacity = '0';
            setTimeout(() => {
                message.remove();
            }, 300);
        }, 5000);
    });
});