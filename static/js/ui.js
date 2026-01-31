/* UI Helper Script */

// Toast Notification System
const Toast = {
    container: null,

    init() {
        this.container = document.createElement('div');
        this.container.className = 'toast-container';
        document.body.appendChild(this.container);
    },

    show(message, type = 'info') {
        if (!this.container) this.init();

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span>${message}</span>`;
        
        this.container.appendChild(toast);

        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
};

// Initialize specific UI elements
document.addEventListener('DOMContentLoaded', () => {
    // File Upload Drag & Drop Enhancement
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        const dropZone = input.closest('.form-group');
        if (dropZone) {
            dropZone.style.border = '2px dashed var(--glass-border)';
            dropZone.style.padding = '20px';
            dropZone.style.borderRadius = '10px';
            dropZone.style.transition = '0.3s';

            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = 'var(--primary-color)';
                dropZone.style.background = 'rgba(0, 242, 255, 0.05)';
            });

            dropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = 'var(--glass-border)';
                dropZone.style.background = 'transparent';
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = 'var(--success)';
                dropZone.style.background = 'rgba(0, 255, 136, 0.05)';
                input.files = e.dataTransfer.files;
            });
        }
    });

    // Handle Flash Messages from Flask as Toasts
    const flashMessages = document.querySelectorAll('.flash-messages .alert');
    flashMessages.forEach(msg => {
        const type = msg.classList.contains('alert-error') ? 'error' : 'success';
        Toast.show(msg.textContent, type);
        msg.remove(); // Remove original element
    });
});
