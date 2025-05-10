/**
 * DeepSteg - JavaScript untuk Frontend
 * Menangani interaksi pengguna, validasi input, dan komunikasi dengan backend
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
    
    /**
     * Update character count and visual feedback
     */
    function updateCharCount() {
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
    
    // Character counter for message input
    if (messageInput && charCount) {
        // Initial count
        charCount.textContent = `0 karakter`;
        
        // Update on input
        messageInput.addEventListener('input', updateCharCount);
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
                    
                    // Display stego image
                    const resultImage = document.getElementById('resultImage');
                    resultImage.src = 'data:image/png;base64,' + data.image;
                    
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
                        
                        // Add event listener to track download errors
                        downloadBtn.addEventListener('click', function(e) {
                            // Set a flag to check if the download started
                            let downloadStarted = false;
                            
                            // Use setTimeout to check if download started
                            setTimeout(function() {
                                if (!downloadStarted) {
                                    showNotification('Terjadi masalah saat download. Coba refresh halaman dan coba lagi.', 'warning');
                                }
                            }, 1000);
                            
                            // Listen for browser navigation event which happens on successful download
                            window.addEventListener('blur', function() {
                                downloadStarted = true;
                            }, { once: true });
                        });
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
            
            // Disable submit button to prevent multiple submissions
            const submitBtn = decodeForm.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Memproses...';
            
            // Show loading state
            document.getElementById('decodePlaceholderContainer').style.display = 'none';
            document.getElementById('messageResultContainer').style.display = 'none';
            document.getElementById('decodeLoadingContainer').style.display = 'block';
            
            // Create form data
            const formData = new FormData();
            formData.append('image', stegoFile);
            
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
                
                // Show result
                document.getElementById('messageResultContainer').style.display = 'block';
                const extractedMessageEl = document.getElementById('extractedMessage');
                
                // Display extracted message
                if (data.message && data.message.trim()) {
                    extractedMessageEl.textContent = data.message;
                    
                    if (data.status === 'success') {
                        extractedMessageEl.classList.add('message-success');
                        showNotification('Pesan berhasil diekstrak dari gambar!', 'success');
                    } else if (data.status === 'warning') {
                        extractedMessageEl.classList.add('message-warning');
                        showNotification('Peringatan: ' + data.message, 'warning');
                    }
                } else {
                    // No message found
                    extractedMessageEl.textContent = "Tidak ada pesan yang ditemukan dalam gambar ini.";
                    extractedMessageEl.classList.add('text-muted');
                    showNotification('Tidak ada pesan yang dapat diekstrak dari gambar ini.', 'info');
                }
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
        
        // Additional metrics can be added here
    }
    
    // Add smooth scrolling for anchors
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#') return; // Skip empty anchors
            
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
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
    
    // Fix for download button on mobile devices
    const downloadButton = document.getElementById('downloadBtn');
    if (downloadButton) {
        downloadButton.addEventListener('touchend', function(e) {
            e.preventDefault();
            window.location.href = this.getAttribute('href');
        });
    }
});