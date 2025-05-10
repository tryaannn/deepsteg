/**
 * DeepSteg Enhanced - Main JavaScript (Part 1)
 * Menangani interaksi pengguna, validasi input, fitur keamanan, dan komunikasi dengan backend
 */

document.addEventListener('DOMContentLoaded', function() {
    // Global variables for notifications
    let notificationTimeout;
    
    /**
     * Function to show notification
     * @param {string} message - Pesan notifikasi
     * @param {string} type - Tipe notifikasi (success, info, warning, danger)
     * @param {number} duration - Durasi tampil notifikasi dalam ms (default: 5000)
     */
    function showNotification(message, type = 'info', duration = 5000) {
        // Create notification container if it doesn't exist
        let notificationContainer = document.getElementById('notification-container');
        if (!notificationContainer) {
            notificationContainer = document.createElement('div');
            notificationContainer.id = 'notification-container';
            notificationContainer.style.position = 'fixed';
            notificationContainer.style.top = '20px';
            notificationContainer.style.right = '20px';
            notificationContainer.style.zIndex = '9999';
            document.body.appendChild(notificationContainer);
        }
        
        // Clear any existing notification
        clearTimeout(notificationTimeout);
        notificationContainer.innerHTML = '';
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show`;
        notification.style.minWidth = '300px';
        notification.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Add to container
        notificationContainer.appendChild(notification);
        
        // Set timeout to remove notification
        notificationTimeout = setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notificationContainer.removeChild(notification);
            }, 300);
        }, duration);
    }
    
    /**
     * Validasi file gambar
     * @param {File} file - File gambar yang akan divalidasi
     * @returns {boolean} - Hasil validasi (true jika valid)
     */
    function validateImageFile(file) {
        if (!file) {
            return false;
        }
        
        // Check file size (max 16MB)
        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            showNotification('Ukuran file terlalu besar (maksimum 16MB).', 'warning');
            return false;
        }
        
        // Check file type
        const acceptedTypes = ['image/jpeg', 'image/png', 'image/bmp'];
        if (!acceptedTypes.includes(file.type)) {
            showNotification('Format file tidak didukung. Gunakan JPG, PNG, atau BMP.', 'warning');
            return false;
        }
        
        return true;
    }
    
    /**
     * Estimasi kapasitas gambar untuk pesan
     * @param {HTMLImageElement} img - Elemen gambar
     * @returns {number} - Estimasi kapasitas dalam karakter
     */
    function estimateMessageCapacity(img) {
        if (!img || !img.naturalWidth || !img.naturalHeight) {
            return 0;
        }
        
        // Setiap pixel dapat menyimpan 3 bit (1 per channel RGB)
        // Setiap karakter membutuhkan 8 bit
        // Kurangi 32 bit untuk header dan 8 bit untuk terminator
        const totalBits = img.naturalWidth * img.naturalHeight * 3 - 40;
        const capacity = Math.floor(totalBits / 8);
        
        // Berikan margin keamanan 10%
        return Math.floor(capacity * 0.9);
    }
    
    /**
     * Handle image preview on upload
     * @param {string} inputId - ID elemen input file
     * @param {string} previewId - ID elemen preview gambar
     * @param {function} callback - Callback setelah gambar dimuat (opsional)
     */
    function handleImagePreview(inputId, previewId, callback) {
        const input = document.getElementById(inputId);
        const preview = document.getElementById(previewId);
        
        if (input && preview) {
            input.addEventListener('change', function() {
                const file = this.files[0];
                if (file && validateImageFile(file)) {
                    const reader = new FileReader();
                    
                    reader.onload = function(e) {
                        const previewImg = preview.querySelector('.preview-img');
                        const placeholder = preview.querySelector('.placeholder');
                        
                        // Reset image sebelum menambahkan sumber baru
                        previewImg.onload = function() {
                            // Update visual
                            previewImg.style.display = 'block';
                            if (placeholder) placeholder.style.display = 'none';
                            
                            // Panggil callback jika ada
                            if (typeof callback === 'function') {
                                callback(previewImg);
                            }
                        };
                        
                        previewImg.src = e.target.result;
                    };
                    
                    reader.onerror = function() {
                        showNotification('Gagal membaca file gambar.', 'danger');
                    };
                    
                    reader.readAsDataURL(file);
                } else if (file) {
                    // Reset file input
                    input.value = '';
                }
            });
            
            // Allow clicking on preview area to trigger file input
            preview.addEventListener('click', function() {
                input.click();
            });
        }
    }
    
    /**
     * Update character count and visual feedback
     */
    function updateCharCount() {
        const messageInput = document.getElementById('message');
        const charCount = document.getElementById('charCount');
        
        if (!messageInput || !charCount) return;
        
        const length = messageInput.value.length;
        charCount.textContent = `${length} karakter`;
        
        // Get capacity if available
        const capacity = parseInt(charCount.dataset.capacity || '0');
        
        if (capacity > 0) {
            const percentUsed = (length / capacity) * 100;
            
            // Update visual feedback
            if (percentUsed > 90) {
                charCount.className = 'text-danger';
                messageInput.classList.add('border-danger');
            } else if (percentUsed > 70) {
                charCount.className = 'text-warning';
                messageInput.classList.remove('border-danger');
            } else {
                charCount.className = 'text-muted';
                messageInput.classList.remove('border-danger');
            }
        }
    }

    /**
     * Update password strength indicator
     * @param {string} password - Password to check strength
     */
    function updatePasswordStrength(password) {
        const strengthBar = document.getElementById('passwordStrengthBar');
        const strengthText = document.getElementById('passwordStrengthText');
        const feedbackContainer = document.getElementById('passwordFeedback');
        const passwordStrength = document.getElementById('passwordStrength');
        
        if (!strengthBar || !strengthText || !feedbackContainer || !passwordStrength) return;
        
        // Show strength indicator when there's a password
        if (password) {
            passwordStrength.style.display = 'block';
        } else {
            passwordStrength.style.display = 'none';
            return;
        }
        
        // Call the API to check password strength
        fetch('/api/check-password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ password: password }),
        })
        .then(response => response.json())
        .then(data => {
            // Update strength bar
            const score = data.score;
            const percentage = (score / 5) * 100; // 5 is max score
            strengthBar.style.width = `${percentage}%`;
            
            // Update bar color based on strength
            strengthBar.className = 'progress-bar';
            if (score < 2) {
                strengthBar.classList.add('bg-danger');
            } else if (score < 4) {
                strengthBar.classList.add('bg-warning');
            } else {
                strengthBar.classList.add('bg-success');
            }
            
            // Update strength text
            strengthText.textContent = `Kekuatan Password: ${data.strength}`;
            
            // Update feedback
            if (data.feedback && data.feedback.length > 0) {
                feedbackContainer.innerHTML = '';
                data.feedback.forEach(item => {
                    const feedback = document.createElement('small');
                    feedback.className = 'text-muted d-block';
                    feedback.innerHTML = `<i class="fas fa-info-circle me-1"></i>${item}`;
                    feedbackContainer.appendChild(feedback);
                });
            } else {
                feedbackContainer.innerHTML = '';
            }
        })
        .catch(error => {
            console.error('Error checking password strength:', error);
            passwordStrength.style.display = 'none';
        });
    }
    
    /**
     * Update the security-related UI elements
     */
    function setupSecurityControls() {
        // Get elements
        const useEncryption = document.getElementById('useEncryption');
        const passwordSection = document.getElementById('passwordSection');
        const password = document.getElementById('password');
        const togglePassword = document.getElementById('togglePassword');
        
        const useCompression = document.getElementById('useCompression');
        const compressionSection = document.getElementById('compressionSection');
        const compressionLevel = document.getElementById('compressionLevel');
        const compressionLevelValue = document.getElementById('compressionLevelValue');
        
        // Toggle password visibility
        if (togglePassword && password) {
            togglePassword.addEventListener('click', function() {
                const type = password.getAttribute('type') === 'password' ? 'text' : 'password';
                password.setAttribute('type', type);
                togglePassword.innerHTML = type === 'password' ? '<i class="fas fa-eye"></i>' : '<i class="fas fa-eye-slash"></i>';
            });
        }
        
        // Toggle encryption section
        if (useEncryption && passwordSection) {
            useEncryption.addEventListener('change', function() {
                passwordSection.style.display = this.checked ? 'block' : 'none';
                if (!this.checked) {
                    // Clear password when encryption is disabled
                    if (password) password.value = '';
                }
            });
        }
        
        // Update password strength when typing
        if (password) {
            password.addEventListener('input', function() {
                updatePasswordStrength(this.value);
            });
        }
        
        // Toggle compression section
        if (useCompression && compressionSection) {
            useCompression.addEventListener('change', function() {
                compressionSection.style.display = this.checked ? 'block' : 'none';
            });
        }
        
        // Update compression level display
        if (compressionLevel && compressionLevelValue) {
            compressionLevel.addEventListener('input', function() {
                compressionLevelValue.textContent = this.value;
            });
        }
    }
    
    /**
     * Setup decode-specific password controls
     */
    function setupDecodePasswordControls() {
        const decodePassword = document.getElementById('decodePassword');
        const toggleDecodePassword = document.getElementById('toggleDecodePassword');
        const passwordInput = document.getElementById('passwordInput');
        const togglePasswordInput = document.getElementById('togglePasswordInput');
        
        // Toggle main decode password visibility
        if (toggleDecodePassword && decodePassword) {
            toggleDecodePassword.addEventListener('click', function() {
                const type = decodePassword.getAttribute('type') === 'password' ? 'text' : 'password';
                decodePassword.setAttribute('type', type);
                toggleDecodePassword.innerHTML = type === 'password' ? '<i class="fas fa-eye"></i>' : '<i class="fas fa-eye-slash"></i>';
            });
        }
        
        // Toggle password input visibility in the password required popup
        if (togglePasswordInput && passwordInput) {
            togglePasswordInput.addEventListener('click', function() {
                const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
                passwordInput.setAttribute('type', type);
                togglePasswordInput.innerHTML = type === 'password' ? '<i class="fas fa-eye"></i>' : '<i class="fas fa-eye-slash"></i>';
            });
        }
        
        // Handle password form submission
        const passwordForm = document.getElementById('passwordForm');
        if (passwordForm && passwordInput) {
            passwordForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Get the password and try to decode again
                const password = passwordInput.value;
                
                if (!password) {
                    showNotification('Silakan masukkan password.', 'warning');
                    return;
                }
                
                // Get file from the main form
                const fileInput = document.getElementById('stegoImageUpload');
                if (!fileInput || !fileInput.files[0]) {
                    showNotification('Tidak ada file yang dipilih. Silakan upload ulang gambar.', 'warning');
                    return;
                }
                
                // Process with the new password
                decodeWithPassword(fileInput.files[0], password);
            });
        }
    }
    
    /**
     * Decode image with provided password
     * @param {File} file - Stego image file
     * @param {string} password - Password for decryption
     */
    function decodeWithPassword(file, password) {
        // Show loading
        document.getElementById('passwordRequiredContainer').style.display = 'none';
        document.getElementById('decodeLoadingContainer').style.display = 'block';
        
        // Create form data
        const formData = new FormData();
        formData.append('image', file);
        formData.append('password', password);
        
        // Send request to server
        fetch('/api/decode', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading
            document.getElementById('decodeLoadingContainer').style.display = 'none';
            
            if (data.error) {
                showNotification('Error: ' + data.error, 'danger');
                // Show the password form again
                document.getElementById('passwordRequiredContainer').style.display = 'block';
                return;
            }
            
            // Handle different status types
            if (data.status === 'password_required') {
                // Still requires password (wrong password)
                showNotification('Password salah. Silakan coba lagi.', 'warning');
                document.getElementById('passwordRequiredContainer').style.display = 'block';
                
                // Clear the password input
                const passwordInput = document.getElementById('passwordInput');
                if (passwordInput) passwordInput.value = '';
                
                return;
            }
            
            // Success or other status - display the result
            displayDecodeResult(data);
        })
        .catch(error => {
            document.getElementById('decodeLoadingContainer').style.display = 'none';
            document.getElementById('passwordRequiredContainer').style.display = 'block';
            showNotification('Error: ' + error.message, 'danger');
        });
    }

    /**
     * Display decode result in the UI
     * @param {Object} data - Response data from server
     */
    function displayDecodeResult(data) {
        // Hide placeholders
        document.getElementById('decodePlaceholderContainer').style.display = 'none';
        document.getElementById('passwordRequiredContainer').style.display = 'none';
        
        // Show result
        document.getElementById('messageResultContainer').style.display = 'block';
        const extractedMessageEl = document.getElementById('extractedMessage');
        const messageMetadata = document.getElementById('messageMetadata');
        
        // Display extracted message
        if (data.message && data.message.trim()) {
            extractedMessageEl.textContent = data.message;
            extractedMessageEl.className = 'message-box';
            
            if (data.status === 'success') {
                extractedMessageEl.classList.add('message-success');
                showNotification('Pesan berhasil diekstrak dari gambar!', 'success');
            } else if (data.status === 'warning') {
                extractedMessageEl.classList.add('message-warning');
                showNotification('Peringatan: ' + data.message, 'warning');
            }
            
            // Display metadata if available
            if (messageMetadata && data.metadata) {
                let metadataText = '';
                
                // Add method info
                if (data.metadata.method === 'deep_learning') {
                    metadataText += 'Ekstraksi: Model Deep Learning';
                } else if (data.metadata.method === 'lsb_secure') {
                    metadataText += 'Ekstraksi: LSB Secure';
                }
                
                // Add encryption info
                if (data.metadata.is_encrypted) {
                    metadataText += ' | Enkripsi: Ya';
                }
                
                // Add compression info
                if (data.metadata.is_compressed) {
                    metadataText += ' | Kompresi: Ya';
                }
                
                // Add execution time if available
                if (data.metadata.execution_time) {
                    metadataText += ` | Waktu: ${data.metadata.execution_time.toFixed(2)}s`;
                }
                
                messageMetadata.textContent = metadataText;
            }
        } else {
            // No message found
            extractedMessageEl.textContent = "Tidak ada pesan yang ditemukan dalam gambar ini.";
            extractedMessageEl.className = 'message-box text-muted';
            if (messageMetadata) messageMetadata.textContent = '';
            showNotification('Tidak ada pesan yang dapat diekstrak dari gambar ini.', 'info');
        }
    }
    
    /**
     * Update metrics display
     * @param {Object} metrics - Objek berisi metrik kualitas gambar
     */
    function updateMetrics(metrics) {
        if (!metrics) return;
        
        // Reset all classes first
        document.querySelectorAll('.progress-bar').forEach(bar => {
            bar.classList.remove('bg-success', 'bg-info', 'bg-warning', 'bg-danger');
        });
        
        // PSNR (higher is better, typical range: 30-50 dB for good quality)
        const psnrValue = document.getElementById('psnrValue');
        const psnrBar = document.getElementById('psnrBar');
        
        if (psnrValue && psnrBar) {
            psnrValue.textContent = metrics.psnr.toFixed(2) + ' dB';
            // Scale PSNR to percentage (assuming 50+ dB is excellent)
            const psnrPercent = Math.min(100, (metrics.psnr / 50) * 100);
            psnrBar.style.width = psnrPercent + '%';
            psnrBar.setAttribute('aria-valuenow', psnrPercent);
            
            // Color coding
            if (metrics.psnr > 40) {
                psnrBar.classList.add('bg-success');
            } else if (metrics.psnr > 30) {
                psnrBar.classList.add('bg-info');
            } else if (metrics.psnr > 20) {
                psnrBar.classList.add('bg-warning');
            } else {
                psnrBar.classList.add('bg-danger');
            }
        }
        
        // SSIM (higher is better, range: 0-1)
        const ssimValue = document.getElementById('ssimValue');
        const ssimBar = document.getElementById('ssimBar');
        
        if (ssimValue && ssimBar) {
            ssimValue.textContent = metrics.ssim.toFixed(4);
            const ssimPercent = metrics.ssim * 100;
            ssimBar.style.width = ssimPercent + '%';
            ssimBar.setAttribute('aria-valuenow', ssimPercent);
            
            // Color coding
            if (metrics.ssim > 0.95) {
                ssimBar.classList.add('bg-success');
            } else if (metrics.ssim > 0.9) {
                ssimBar.classList.add('bg-info');
            } else if (metrics.ssim > 0.8) {
                ssimBar.classList.add('bg-warning');
            } else {
                ssimBar.classList.add('bg-danger');
            }
        }
        
        // MSE (lower is better)
        const mseValue = document.getElementById('mseValue');
        const mseBar = document.getElementById('mseBar');
        
        if (mseValue && mseBar) {
            mseValue.textContent = metrics.mse.toFixed(2);
            // Inverse scale (0 is perfect, 100+ is poor)
            const msePercent = Math.max(0, 100 - Math.min(100, metrics.mse));
            mseBar.style.width = msePercent + '%';
            mseBar.setAttribute('aria-valuenow', msePercent);
            
            // Color coding
            if (metrics.mse < 1) {
                mseBar.classList.add('bg-success');
            } else if (metrics.mse < 5) {
                mseBar.classList.add('bg-info');
            } else if (metrics.mse < 10) {
                mseBar.classList.add('bg-warning');
            } else {
                mseBar.classList.add('bg-danger');
            }
        }
        
        // Histogram Similarity (higher is better, range: 0-1)
        const histValue = document.getElementById('histValue');
        const histBar = document.getElementById('histBar');
        
        if (histValue && histBar) {
            histValue.textContent = metrics.hist_similarity.toFixed(4);
            const histPercent = metrics.hist_similarity * 100;
            histBar.style.width = histPercent + '%';
            histBar.setAttribute('aria-valuenow', histPercent);
            
            // Color coding
            if (metrics.hist_similarity > 0.95) {
                histBar.classList.add('bg-success');
            } else if (metrics.hist_similarity > 0.9) {
                histBar.classList.add('bg-info');
            } else if (metrics.hist_similarity > 0.8) {
                histBar.classList.add('bg-warning');
            } else {
                histBar.classList.add('bg-danger');
            }
        }
        
        // Capacity Used (if available)
        const capValue = document.getElementById('capacityValue');
        const capBar = document.getElementById('capacityBar');
        
        if (capValue && capBar && metrics.capacity_used !== undefined) {
            const capPercent = Math.min(100, metrics.capacity_used);
            capValue.textContent = capPercent.toFixed(1) + '%';
            capBar.style.width = capPercent + '%';
            capBar.setAttribute('aria-valuenow', capPercent);
            
            // Color coding
            if (capPercent < 50) {
                capBar.classList.add('bg-success');
            } else if (capPercent < 75) {
                capBar.classList.add('bg-info');
            } else if (capPercent < 90) {
                capBar.classList.add('bg-warning');
            } else {
                capBar.classList.add('bg-danger');
            }
        }
        
        // File Size (if available)
        const sizeElement = document.getElementById('fileSize');
        if (sizeElement && metrics.file_size !== undefined) {
            const sizeKB = (metrics.file_size / 1024).toFixed(1);
            sizeElement.textContent = sizeKB + ' KB';
        }
        
        // Update security badges
        const encryptionBadge = document.getElementById('encryptionBadge');
        const encryptionStatus = document.getElementById('encryptionStatus');
        const compressionBadge = document.getElementById('compressionBadge');
        const compressionStatus = document.getElementById('compressionStatus');
        const passwordReminder = document.getElementById('passwordReminder');
        
        if (encryptionBadge && encryptionStatus && metrics.encrypted !== undefined) {
            if (metrics.encrypted) {
                encryptionBadge.className = 'security-badge p-2 rounded bg-success text-white';
                encryptionStatus.textContent = 'Aktif';
                
                // Show password reminder
                if (passwordReminder) passwordReminder.style.display = 'block';
            } else {
                encryptionBadge.className = 'security-badge p-2 rounded bg-light';
                encryptionStatus.textContent = 'Tidak Aktif';
                
                // Hide password reminder
                if (passwordReminder) passwordReminder.style.display = 'none';
            }
        }
        
        if (compressionBadge && compressionStatus && metrics.compressed !== undefined) {
            if (metrics.compressed) {
                compressionBadge.className = 'security-badge p-2 rounded bg-info text-white';
                compressionStatus.textContent = 'Aktif';
            } else {
                compressionBadge.className = 'security-badge p-2 rounded bg-light';
                compressionStatus.textContent = 'Tidak Aktif';
            }
        }
    }

    // Initialize image previews
    const messageInput = document.getElementById('message');
    const charCount = document.getElementById('charCount');
    
    // Setup encode image preview with capacity calculation callback
    handleImagePreview('imageUpload', 'imagePreview', function(img) {
        // Update message capacity if we have a message input
        if (messageInput && charCount) {
            const capacity = estimateMessageCapacity(img);
            charCount.dataset.capacity = capacity;
            
            // Update character count
            updateCharCount();
            
            // Show capacity info
            const capacityInfo = document.getElementById('capacityInfo');
            if (capacityInfo) {
                capacityInfo.textContent = `Kapasitas maksimum: ~${capacity} karakter`;
                capacityInfo.style.display = 'block';
            }
        }
    });
    
    // Setup decode image preview
    handleImagePreview('stegoImageUpload', 'stegoImagePreview');
    
    // Character counter for message input
    if (messageInput && charCount) {
        // Initial count
        charCount.textContent = `0 karakter`;
        
        // Update on input
        messageInput.addEventListener('input', updateCharCount);
    }
    
    // Setup security controls (password, compression, etc.)
    setupSecurityControls();
    
    // Setup decode password controls
    setupDecodePasswordControls();
    
    // Copy message button
    const copyMessageBtn = document.getElementById('copyMessageBtn');
    
    if (copyMessageBtn) {
        copyMessageBtn.addEventListener('click', function() {
            const message = document.getElementById('extractedMessage').textContent;
            
            if (!message || message.trim() === '' || message.includes('Tidak ada pesan')) {
                showNotification('Tidak ada pesan untuk disalin.', 'warning');
                return;
            }
            
            // Use modern clipboard API
            navigator.clipboard.writeText(message)
                .then(() => {
                    const originalText = this.innerHTML;
                    this.innerHTML = '<i class="fas fa-check me-1"></i>Disalin!';
                    showNotification('Pesan berhasil disalin ke clipboard!', 'success');
                    
                    setTimeout(() => {
                        this.innerHTML = originalText;
                    }, 2000);
                })
                .catch(err => {
                    // Fallback for older browsers
                    try {
                        const textArea = document.createElement('textarea');
                        textArea.value = message;
                        textArea.style.position = 'fixed';
                        textArea.style.opacity = '0';
                        document.body.appendChild(textArea);
                        textArea.select();
                        const successful = document.execCommand('copy');
                        document.body.removeChild(textArea);
                        
                        if (successful) {
                            const originalText = this.innerHTML;
                            this.innerHTML = '<i class="fas fa-check me-1"></i>Disalin!';
                            showNotification('Pesan berhasil disalin ke clipboard!', 'success');
                            
                            setTimeout(() => {
                                this.innerHTML = originalText;
                            }, 2000);
                        } else {
                            throw new Error('Copy command failed');
                        }
                    } catch (fallbackError) {
                        showNotification('Gagal menyalin teks: ' + err, 'danger');
                    }
                });
        });
    }
    
    // Encode form submission
    const encodeForm = document.getElementById('encodeForm');
    
    if (encodeForm) {
        encodeForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const imageFile = document.getElementById('imageUpload').files[0];
            const message = document.getElementById('message').value.trim();
            
            if (!imageFile) {
                showNotification('Silakan pilih gambar terlebih dahulu.', 'warning');
                return;
            }
            
            if (!validateImageFile(imageFile)) {
                return;
            }
            
            if (!message) {
                showNotification('Silakan masukkan pesan yang ingin disembunyikan.', 'warning');
                return;
            }
            
            // Check capacity if available
            const capacity = parseInt(charCount.dataset.capacity || '0');
            if (capacity > 0 && message.length > capacity) {
                if (!confirm(`Pesan melebihi kapasitas yang direkomendasikan (${message.length}/${capacity} karakter). Pesan mungkin akan terpotong. Lanjutkan?`)) {
                    return;
                }
            }
            
            // Get encryption settings
            const useEncryption = document.getElementById('useEncryption').checked;
            let password = '';
            if (useEncryption) {
                password = document.getElementById('password').value;
                if (!password) {
                    showNotification('Anda mengaktifkan enkripsi. Silakan masukkan password.', 'warning');
                    return;
                }
                
                // Confirm password is strong enough
                const passwordStrengthBar = document.getElementById('passwordStrengthBar');
                if (passwordStrengthBar && passwordStrengthBar.classList.contains('bg-danger')) {
                    if (!confirm('Password Anda lemah. Disarankan untuk menggunakan password yang lebih kuat. Tetap lanjutkan?')) {
                        return;
                    }
                }
            }
            
            // Get compression settings
            const useCompression = document.getElementById('useCompression').checked;
            let compressionLevel = 6; // Default
            if (useCompression) {
                compressionLevel = parseInt(document.getElementById('compressionLevel').value);
            }
            
            // Disable submit button to prevent multiple submissions
            const submitBtn = encodeForm.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Memproses...';
            
            // Show loading state
            document.getElementById('placeholderContainer').style.display = 'none';
            document.getElementById('resultContainer').style.display = 'none';
            document.getElementById('loadingContainer').style.display = 'block';
            
            // Create form data
            const formData = new FormData();
            formData.append('image', imageFile);
            formData.append('message', message);
            formData.append('use_encryption', useEncryption.toString());
            if (useEncryption) {
                formData.append('password', password);
            }
            formData.append('use_compression', useCompression.toString());
            if (useCompression) {
                formData.append('compression_level', compressionLevel.toString());
            }

            // Send request to server
            fetch('/api/encode', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    // Handle HTTP errors
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Hide loading state
                document.getElementById('loadingContainer').style.display = 'none';
                
                if (data.error) {
                    showNotification('Error: ' + data.error, 'danger');
                    document.getElementById('placeholderContainer').style.display = 'block';
                    return;
                }
                
                if (data.status === 'success') {
                    // Show success notification
                    showNotification('Pesan berhasil disembunyikan dalam gambar!', 'success');
                    
                    // Show result
                    document.getElementById('resultContainer').style.display = 'block';
                    
                    // Display stego and original images for comparison
                    const resultImage = document.getElementById('resultImage');
                    const originalImage = document.getElementById('originalImage');
                    
                    resultImage.src = 'data:image/png;base64,' + data.image;
                    originalImage.src = 'data:image/png;base64,' + data.original_image;
                    
                    // Handle image load error
                    resultImage.onerror = function() {
                        showNotification('Gagal menampilkan gambar hasil. Coba download gambar langsung.', 'warning');
                        resultImage.style.display = 'none';
                    };
                    
                    // Set download link with correct filename
                    const downloadBtn = document.getElementById('downloadBtn');
                    if (downloadBtn) {
                        downloadBtn.href = '/download/' + data.filename;
                        downloadBtn.download = 'stego_image.png';  // Suggest filename to browser
                    }
                    
                    // Update metrics
                    updateMetrics(data.metrics);
                    
                    // Show download help tooltip
                    const downloadHelp = document.getElementById('downloadHelp');
                    if (downloadHelp) {
                        downloadHelp.style.display = 'block';
                    }
                } else {
                    showNotification('Terjadi kesalahan yang tidak diketahui.', 'warning');
                    document.getElementById('placeholderContainer').style.display = 'block';
                }
            })
            .catch(error => {
                document.getElementById('loadingContainer').style.display = 'none';
                document.getElementById('placeholderContainer').style.display = 'block';
                showNotification('Error: ' + error.message, 'danger');
            })
            .finally(() => {
                // Re-enable submit button
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
            });
        });
    }
    
    // Decode form submission
    const decodeForm = document.getElementById('decodeForm');
    
    if (decodeForm) {
        decodeForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const stegoFile = document.getElementById('stegoImageUpload').files[0];
            
            if (!stegoFile) {
                showNotification('Silakan pilih gambar stego terlebih dahulu.', 'warning');
                return;
            }
            
            if (!validateImageFile(stegoFile)) {
                return;
            }
            
            // Get password if provided
            const password = document.getElementById('decodePassword').value.trim();
            
            // Disable submit button to prevent multiple submissions
            const submitBtn = decodeForm.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Memproses...';
            
            // Show loading state
            document.getElementById('decodePlaceholderContainer').style.display = 'none';
            document.getElementById('messageResultContainer').style.display = 'none';
            document.getElementById('passwordRequiredContainer').style.display = 'none';
            document.getElementById('decodeLoadingContainer').style.display = 'block';
            
            // Create form data
            const formData = new FormData();
            formData.append('image', stegoFile);
            if (password) {
                formData.append('password', password);
            }
            
            // Send request to server
            fetch('/api/decode', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    // Handle HTTP errors
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Hide loading state
                document.getElementById('decodeLoadingContainer').style.display = 'none';
                
                if (data.error) {
                    showNotification('Error: ' + data.error, 'danger');
                    document.getElementById('decodePlaceholderContainer').style.display = 'block';
                    return;
                }
                
                // Handle password required case
                if (data.status === 'password_required') {
                    showNotification('Pesan ini terenkripsi. Masukkan password untuk mendekripsi.', 'warning');
                    document.getElementById('passwordRequiredContainer').style.display = 'block';
                    
                    return;
                }
                
                // Display the result for success and warning cases
                displayDecodeResult(data);
            })
            .catch(error => {
                document.getElementById('decodeLoadingContainer').style.display = 'none';
                document.getElementById('decodePlaceholderContainer').style.display = 'block';
                showNotification('Error: ' + error.message, 'danger');
            })
            .finally(() => {
                // Re-enable submit button
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
            });
        });
    }
    
    // Add error handling for image loading
    document.querySelectorAll('.preview-img').forEach(img => {
        img.addEventListener('error', function() {
            this.style.display = 'none';
            const placeholder = this.parentElement.querySelector('.placeholder');
            if (placeholder) {
                placeholder.style.display = 'block';
            }
            showNotification('Gagal memuat gambar. Format file mungkin tidak didukung.', 'warning');
        });
    });
});