// Registration wizard state
window.registrationWizard = {
    currentStep: 1,
    totalSteps: 4
};

// Update progress bar
function updateProgress() {
    const progress = (window.registrationWizard.currentStep / window.registrationWizard.totalSteps) * 100;
    document.getElementById('progressBar').style.width = progress + '%';
}

// Show specific step
function showStep(step) {
    document.querySelectorAll('.step-container').forEach(container => {
        container.classList.remove('active');
    });
    document.querySelector(`[data-step="${step}"]`).classList.add('active');
    updateProgress();
}

// Validate current step
function validateCurrentStep() {
    const container = document.querySelector(`[data-step="${window.registrationWizard.currentStep}"]`);
    const input = container.querySelector('input');
    const errorMsg = container.querySelector('.error-message');
    
    if (!input.value.trim()) {
        input.classList.add('error');
        errorMsg.style.display = 'block';
        return false;
    }
    
    if (window.registrationWizard.currentStep === 2) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(input.value)) {
            input.classList.add('error');
            errorMsg.style.display = 'block';
            return false;
        }
    }
    
    if (window.registrationWizard.currentStep === 3 && input.value.length < 8) {
        input.classList.add('error');
        errorMsg.style.display = 'block';
        return false;
    }
    
    input.classList.remove('error');
    errorMsg.style.display = 'none';
    return true;
}

// Navigate to next step
function nextStep() {
    if (validateCurrentStep() && window.registrationWizard.currentStep < window.registrationWizard.totalSteps) {
        window.registrationWizard.currentStep++;
        showStep(window.registrationWizard.currentStep);
    }
}

// Navigate to previous step
function previousStep() {
    if (window.registrationWizard.currentStep > 1) {
        window.registrationWizard.currentStep--;
        showStep(window.registrationWizard.currentStep);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Attach click handlers to all next buttons
    document.querySelectorAll('.btn-next').forEach(button => {
        button.addEventListener('click', nextStep);
    });
    
    // Attach click handlers to all back buttons
    document.querySelectorAll('.btn-back').forEach(button => {
        button.addEventListener('click', previousStep);
    });
    
    // Clear error on input
    document.querySelectorAll('input').forEach(input => {
        input.addEventListener('input', function() {
            this.classList.remove('error');
            this.parentElement.querySelector('.error-message').style.display = 'none';
        });
    });
    
    // Handle form submission
    document.getElementById('registrationForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!validateCurrentStep()) return;
        
        const submitBtn = document.getElementById('submitBtn');
        const messageDiv = document.getElementById('message');
        
        submitBtn.disabled = true;
        const loadingSpan = document.createElement('span');
        loadingSpan.className = 'loading';
        submitBtn.textContent = 'Creating Account';
        submitBtn.appendChild(loadingSpan);
        
        const formData = {
            name: document.getElementById('name').value.trim(),
            email: document.getElementById('email').value.trim().toLowerCase(),
            password: document.getElementById('password').value,
            secretKey: document.getElementById('secretKey').value,
            quotaGB: 100 // Fixed quota
        };
        
        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                messageDiv.className = 'message success';
                messageDiv.innerHTML = `
                    <svg class="icon" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                    </svg>
                    ${data.message || 'Account created! Redirecting to your photo gallery...'}
                `;
                
                setTimeout(() => {
                    window.location.href = '/';
                }, 2000);
            } else {
                messageDiv.className = 'message error';
                messageDiv.innerHTML = `
                    <svg class="icon" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                    </svg>
                    ${data.error || 'Registration failed. Please try again.'}
                `;
            }
        } catch (error) {
            messageDiv.className = 'message error';
            messageDiv.innerHTML = `
                <svg class="icon" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                </svg>
                Network error. Please check your connection.
            `;
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Create Account';
        }
    });
    
    // Enter key navigation
    document.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && window.registrationWizard.currentStep < window.registrationWizard.totalSteps) {
            e.preventDefault();
            nextStep();
        }
    });
});