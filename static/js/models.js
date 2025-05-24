/**
 * models.js
 * JavaScript untuk halaman manajemen model dan detector
 */

document.addEventListener('DOMContentLoaded', function() {
    // Global variables
    let modelUsageChart = null;
    let performanceChart = null;
    let downloadProgressModal = null;
    let modelInfoModal = null;
    
    // Initialize models page
    initModelsPage();
    
    function initModelsPage() {
        // Initialize modals
        initModals();
        
        // Load initial data
        loadModelStatistics();
        loadModels();
        loadDetectors();
        
        // Setup event listeners
        setupEventListeners();
        
        // Initialize charts
        initCharts();
    }
    
    function initModals() {
        // Initialize Bootstrap modals
        const downloadModalElement = document.getElementById('downloadProgressModal');
        const modelInfoModalElement = document.getElementById('modelInfoModal');
        
        if (downloadModalElement) {
            downloadProgressModal = new bootstrap.Modal(downloadModalElement);
        }
        
        if (modelInfoModalElement) {
            modelInfoModal = new bootstrap.Modal(modelInfoModalElement);
        }
    }
    
    function setupEventListeners() {
        // Refresh buttons
        const refreshModelsBtn = document.getElementById('refreshModels');
        const refreshDetectorsBtn = document.getElementById('refreshDetectors');
        
        if (refreshModelsBtn) {
            refreshModelsBtn.addEventListener('click', function() {
                loadModels();
                showNotification('Model list refreshed', 'info');
            });
        }
        
        if (refreshDetectorsBtn) {
            refreshDetectorsBtn.addEventListener('click', function() {
                loadDetectors();
                showNotification('Detector list refreshed', 'info');
            });
        }
        
        // Download all buttons
        const downloadAllModelsBtn = document.getElementById('downloadAllModels');
        const downloadAllDetectorsBtn = document.getElementById('downloadAllDetectors');
        
        if (downloadAllModelsBtn) {
            downloadAllModelsBtn.addEventListener('click', downloadAllModels);
        }
        
        if (downloadAllDetectorsBtn) {
            downloadAllDetectorsBtn.addEventListener('click', downloadAllDetectors);
        }
        
        // Cancel download button
        const cancelDownloadBtn = document.getElementById('cancelDownload');
        if (cancelDownloadBtn) {
            cancelDownloadBtn.addEventListener('click', cancelDownload);
        }
    }
    
    function loadModelStatistics() {
        fetch('/api/advanced-stats')
            .then(response => response.json())
            .then(data => {
                updateStatistics(data);
            })
            .catch(error => {
                console.error('Error loading statistics:', error);
            });
    }
    
    function updateStatistics(data) {
        // Update stat cards
        const totalModelsEl = document.getElementById('totalModels');
        const totalDetectorsEl = document.getElementById('totalDetectors');
        const downloadedModelsEl = document.getElementById('downloadedModels');
        const storageUsedEl = document.getElementById('storageUsed');
        
        if (totalModelsEl) {
            totalModelsEl.textContent = data.available_models || 0;
        }
        
        if (totalDetectorsEl) {
            totalDetectorsEl.textContent = data.available_detectors || 0;
        }
        
        if (downloadedModelsEl) {
            const downloaded = (data.local_models || 0) + (data.local_detectors || 0);
            downloadedModelsEl.textContent = downloaded;
        }
        
        if (storageUsedEl) {
            // Calculate approximate storage (this would come from server in real implementation)
            const approxStorage = ((data.local_models || 0) * 15) + ((data.local_detectors || 0) * 8);
            storageUsedEl.textContent = `${approxStorage} MB`;
        }
    }
    
    function loadModels() {
        const grid = document.getElementById('modelsGrid');
        if (!grid) return;
        
        // Show loading
        grid.innerHTML = `
            <div class="col-12 text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading models...</span>
                </div>
                <p class="mt-2">Loading available models...</p>
            </div>
        `;
        
        fetch('/api/models')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    displayModels(data.models);
                } else {
                    throw new Error(data.error || 'Failed to load models');
                }
            })
            .catch(error => {
                console.error('Error loading models:', error);
                grid.innerHTML = `
                    <div class="col-12 text-center py-5">
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Failed to load models: ${error.message}
                        </div>
                    </div>
                `;
            });
    }
    
    function displayModels(models) {
        const grid = document.getElementById('modelsGrid');
        if (!grid) return;
        
        if (Object.keys(models).length === 0) {
            grid.innerHTML = `
                <div class="col-12 text-center py-5">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        No models available. Check your internet connection or try refreshing.
                    </div>
                </div>
            `;
            return;
        }
        
        let html = '';
        
        Object.entries(models).forEach(([modelName, modelInfo]) => {
            const isLocal = modelInfo.is_local || false;
            const statusClass = isLocal ? 'downloaded' : 'available';
            const statusText = isLocal ? 'Downloaded' : 'Available';
            
            html += `
                <div class="col-lg-6 col-xl-4 mb-4">
                    <div class="model-card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h6 class="mb-0">${modelInfo.name || modelName}</h6>
                            <span class="model-status ${statusClass}">${statusText}</span>
                        </div>
                        <div class="card-body">
                            <p class="card-text">${modelInfo.description || 'No description available'}</p>
                            
                            <div class="model-info">
                                <div class="info-item">
                                    <span class="info-label">Type:</span>
                                    <span class="info-value">${modelInfo.type || 'Steganography'}</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">Size:</span>
                                    <span class="info-value">${modelInfo.size_mb || 'Unknown'} MB</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">Message Length:</span>
                                    <span class="info-value">${modelInfo.message_length || 'Variable'} bits</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">Priority:</span>
                                    <span class="info-value">${modelInfo.priority || 'Normal'}</span>
                                </div>
                            </div>
                            
                            <div class="model-actions">
                                ${isLocal ? 
                                    `<button class="btn btn-outline-danger btn-sm" onclick="deleteModel('${modelName}')">
                                        <i class="fas fa-trash me-1"></i>Delete
                                    </button>
                                    <button class="btn btn-primary btn-sm" onclick="testModel('${modelName}')">
                                        <i class="fas fa-play me-1"></i>Test
                                    </button>` :
                                    `<button class="btn btn-primary btn-sm" onclick="downloadModel('${modelName}')">
                                        <i class="fas fa-download me-1"></i>Download
                                    </button>`
                                }
                                <button class="btn btn-outline-secondary btn-sm" onclick="showModelInfo('${modelName}')">
                                    <i class="fas fa-info me-1"></i>Info
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        grid.innerHTML = html;
    }
    
    function loadDetectors() {
        const grid = document.getElementById('detectorsGrid');
        if (!grid) return;
        
        // Show loading
        grid.innerHTML = `
            <div class="col-12 text-center py-5">
                <div class="spinner-border text-success" role="status">
                    <span class="visually-hidden">Loading detectors...</span>
                </div>
                <p class="mt-2">Loading available detectors...</p>
            </div>
        `;
        
        fetch('/api/detectors')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    displayDetectors(data.detectors);
                } else {
                    throw new Error(data.error || 'Failed to load detectors');
                }
            })
            .catch(error => {
                console.error('Error loading detectors:', error);
                grid.innerHTML = `
                    <div class="col-12 text-center py-5">
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Failed to load detectors: ${error.message}
                        </div>
                    </div>
                `;
            });
    }
    
    function displayDetectors(detectors) {
        const grid = document.getElementById('detectorsGrid');
        if (!grid) return;
        
        if (Object.keys(detectors).length === 0) {
            grid.innerHTML = `
                <div class="col-12 text-center py-5">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        No detectors available. Check your internet connection or try refreshing.
                    </div>
                </div>
            `;
            return;
        }
        
        let html = '';
        
        Object.entries(detectors).forEach(([detectorName, detectorInfo]) => {
            const isLocal = detectorInfo.is_local || false;
            const statusClass = isLocal ? 'downloaded' : 'available';
            const statusText = isLocal ? 'Downloaded' : 'Available';
            
            html += `
                <div class="col-lg-6 col-xl-4 mb-4">
                    <div class="detector-card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h6 class="mb-0">${detectorInfo.name || detectorName}</h6>
                            <span class="model-status ${statusClass}">${statusText}</span>
                        </div>
                        <div class="card-body">
                            <p class="card-text">${detectorInfo.description || 'No description available'}</p>
                            
                            <div class="model-info">
                                <div class="info-item">
                                    <span class="info-label">Type:</span>
                                    <span class="info-value">${detectorInfo.type || 'Steganalysis'}</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">Size:</span>
                                    <span class="info-value">${detectorInfo.size_mb || 'Unknown'} MB</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">Accuracy:</span>
                                    <span class="info-value">${detectorInfo.accuracy || 'Unknown'}%</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">Priority:</span>
                                    <span class="info-value">${detectorInfo.priority || 'Normal'}</span>
                                </div>
                            </div>
                            
                            <div class="model-actions">
                                ${isLocal ? 
                                    `<button class="btn btn-outline-danger btn-sm" onclick="deleteDetector('${detectorName}')">
                                        <i class="fas fa-trash me-1"></i>Delete
                                    </button>
                                    <button class="btn btn-success btn-sm" onclick="testDetector('${detectorName}')">
                                        <i class="fas fa-play me-1"></i>Test
                                    </button>` :
                                    `<button class="btn btn-success btn-sm" onclick="downloadDetector('${detectorName}')">
                                        <i class="fas fa-download me-1"></i>Download
                                    </button>`
                                }
                                <button class="btn btn-outline-secondary btn-sm" onclick="showDetectorInfo('${detectorName}')">
                                    <i class="fas fa-info me-1"></i>Info
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        grid.innerHTML = html;
    }
    
    function initCharts() {
        initModelUsageChart();
        initPerformanceChart();
        loadModelComparison();
    }
    
    function initModelUsageChart() {
        const canvas = document.getElementById('modelUsageChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Sample data - in real implementation, this would come from API
        const sampleData = {
            labels: ['GAN Basic', 'GAN High Capacity', 'CNN Basic', 'Deep Detector'],
            datasets: [{
                label: 'Usage Count',
                data: [25, 15, 30, 20],
                backgroundColor: [
                    'rgba(52, 152, 219, 0.8)',
                    'rgba(46, 204, 113, 0.8)',
                    'rgba(231, 76, 60, 0.8)',
                    'rgba(243, 156, 18, 0.8)'
                ],
                borderColor: [
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(231, 76, 60, 1)',
                    'rgba(243, 156, 18, 1)'
                ],
                borderWidth: 2
            }]
        };
        
        modelUsageChart = new Chart(ctx, {
            type: 'doughnut',
            data: sampleData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    function initPerformanceChart() {
        const canvas = document.getElementById('performanceChart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Sample data - in real implementation, this would come from API
        const sampleData = {
            labels: ['GAN Basic', 'GAN High Cap', 'CNN Basic', 'Deep Detector'],
            datasets: [{
                label: 'PSNR (dB)',
                data: [42.5, 38.2, null, null],
                backgroundColor: 'rgba(52, 152, 219, 0.5)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 2,
                yAxisID: 'y'
            }, {
                label: 'SSIM',
                data: [0.95, 0.92, null, null],
                backgroundColor: 'rgba(46, 204, 113, 0.5)',
                borderColor: 'rgba(46, 204, 113, 1)',
                borderWidth: 2,
                yAxisID: 'y1'
            }, {
                label: 'Accuracy (%)',
                data: [null, null, 94.2, 96.8],
                backgroundColor: 'rgba(231, 76, 60, 0.5)',
                borderColor: 'rgba(231, 76, 60, 1)',
                borderWidth: 2,
                yAxisID: 'y2'
            }]
        };
        
        performanceChart = new Chart(ctx, {
            type: 'bar',
            data: sampleData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'PSNR (dB)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'SSIM'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    },
                    y2: {
                        type: 'linear',
                        display: false,
                        min: 0,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }
    
    function loadModelComparison() {
        const tbody = document.querySelector('#modelComparisonTable tbody');
        if (!tbody) return;
        
        // Sample data - in real implementation, this would come from API
        const sampleData = [
            {
                name: 'GAN Basic',
                type: 'Steganography',
                accuracy: '42.5 dB PSNR',
                speed: '1200',
                size: '12.5',
                usage: '25',
                status: 'downloaded'
            },
            {
                name: 'GAN High Capacity',
                type: 'Steganography',
                accuracy: '38.2 dB PSNR',
                speed: '2800',
                size: '25.0',
                usage: '15',
                status: 'available'
            },
            {
                name: 'CNN Basic',
                type: 'Steganalysis',
                accuracy: '94.2%',
                speed: '450',
                size: '5.5',
                usage: '30',
                status: 'downloaded'
            },
            {
                name: 'Deep Detector',
                type: 'Steganalysis',
                accuracy: '96.8%',
                speed: '1100',
                size: '18.0',
                usage: '20',
                status: 'available'
            }
        ];
        
        let html = '';
        sampleData.forEach(model => {
            const statusBadge = model.status === 'downloaded' ? 
                '<span class="badge bg-success">Downloaded</span>' : 
                '<span class="badge bg-secondary">Available</span>';
            
            html += `
                <tr>
                    <td><strong>${model.name}</strong></td>
                    <td>${model.type}</td>
                    <td>${model.accuracy}</td>
                    <td>${model.speed} ms</td>
                    <td>${model.size} MB</td>
                    <td>${model.usage}</td>
                    <td>${statusBadge}</td>
                </tr>
            `;
        });
        
        tbody.innerHTML = html;
    }
    
    // Global functions for button clicks
    window.downloadModel = function(modelName) {
        showDownloadProgress(`Downloading model: ${modelName}`, 'model');
        
        fetch('/api/download-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model_name: modelName })
        })
        .then(response => response.json())
        .then(data => {
            hideDownloadProgress();
            if (data.status === 'success') {
                showNotification(`Model ${modelName} downloaded successfully!`, 'success');
                loadModels(); // Refresh the models list
                loadModelStatistics(); // Refresh statistics
            } else {
                showNotification(`Failed to download model: ${data.error}`, 'danger');
            }
        })
        .catch(error => {
            hideDownloadProgress();
            showNotification(`Error downloading model: ${error.message}`, 'danger');
        });
    };
    
    window.downloadDetector = function(detectorName) {
        showDownloadProgress(`Downloading detector: ${detectorName}`, 'detector');
        
        fetch('/api/download-detector', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ detector_name: detectorName })
        })
        .then(response => response.json())
        .then(data => {
            hideDownloadProgress();
            if (data.status === 'success') {
                showNotification(`Detector ${detectorName} downloaded successfully!`, 'success');
                loadDetectors(); // Refresh the detectors list
                loadModelStatistics(); // Refresh statistics
            } else {
                showNotification(`Failed to download detector: ${data.error}`, 'danger');
            }
        })
        .catch(error => {
            hideDownloadProgress();
            showNotification(`Error downloading detector: ${error.message}`, 'danger');
        });
    };
    
    window.deleteModel = function(modelName) {
        if (!confirm(`Are you sure you want to delete model "${modelName}"?`)) {
            return;
        }
        
        // In real implementation, this would call an API endpoint
        showNotification(`Model ${modelName} deleted (simulated)`, 'info');
        loadModels(); // Refresh the models list
        loadModelStatistics(); // Refresh statistics
    };
    
    window.deleteDetector = function(detectorName) {
        if (!confirm(`Are you sure you want to delete detector "${detectorName}"?`)) {
            return;
        }
        
        // In real implementation, this would call an API endpoint
        showNotification(`Detector ${detectorName} deleted (simulated)`, 'info');
        loadDetectors(); // Refresh the detectors list
        loadModelStatistics(); // Refresh statistics
    };
    
    window.testModel = function(modelName) {
        showNotification(`Testing model ${modelName}... (simulated)`, 'info');
        // In real implementation, this would redirect to test page or show test dialog
    };
    
    window.testDetector = function(detectorName) {
        showNotification(`Testing detector ${detectorName}... (simulated)`, 'info');
        // In real implementation, this would redirect to analysis page with detector selected
    };
    
    window.showModelInfo = function(modelName) {
        // Fetch detailed model info and show in modal
        fetch(`/api/model-info/${modelName}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    displayModelInfoModal(data.model_info);
                } else {
                    showNotification(`Failed to load model info: ${data.error}`, 'danger');
                }
            })
            .catch(error => {
                showNotification(`Error loading model info: ${error.message}`, 'danger');
            });
    };
    
    window.showDetectorInfo = function(detectorName) {
        // Fetch detailed detector info and show in modal
        fetch(`/api/detector-info/${detectorName}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    displayDetectorInfoModal(data.detector_info);
                } else {
                    showNotification(`Failed to load detector info: ${data.error}`, 'danger');
                }
            })
            .catch(error => {
                showNotification(`Error loading detector info: ${error.message}`, 'danger');
            });
    };
    
    function downloadAllModels() {
        if (!confirm('This will download all available models. This may take a while and use significant bandwidth. Continue?')) {
            return;
        }
        
        showDownloadProgress('Downloading all models...', 'batch');
        
        // In real implementation, this would call an API endpoint
        setTimeout(() => {
            hideDownloadProgress();
            showNotification('All models downloaded successfully! (simulated)', 'success');
            loadModels();
            loadModelStatistics();
        }, 5000);
    }
    
    function downloadAllDetectors() {
        if (!confirm('This will download all available detectors. This may take a while and use significant bandwidth. Continue?')) {
            return;
        }
        
        showDownloadProgress('Downloading all detectors...', 'batch');
        
        // In real implementation, this would call an API endpoint
        setTimeout(() => {
            hideDownloadProgress();
            showNotification('All detectors downloaded successfully! (simulated)', 'success');
            loadDetectors();
            loadModelStatistics();
        }, 5000);
    }
    
    function showDownloadProgress(title, type) {
        const modalElement = document.getElementById('downloadProgressModal');
        const titleElement = document.getElementById('downloadProgressModalLabel');
        const itemElement = document.getElementById('downloadingItem');
        const progressBar = document.getElementById('downloadProgressBar');
        const progressText = document.getElementById('downloadProgressText');
        
        if (titleElement) titleElement.textContent = title;
        if (itemElement) itemElement.textContent = title;
        
        // Reset progress
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', 0);
        }
        if (progressText) progressText.textContent = '0%';
        
        // Show modal
        if (downloadProgressModal) {
            downloadProgressModal.show();
        }
        
        // Simulate download progress
        simulateDownloadProgress();
    }
    
    function simulateDownloadProgress() {
        const progressBar = document.getElementById('downloadProgressBar');
        const progressText = document.getElementById('downloadProgressText');
        const downloadedSize = document.getElementById('downloadedSize');
        const totalSize = document.getElementById('totalSize');
        const downloadSpeed = document.getElementById('downloadSpeed');
        
        let progress = 0;
        const totalSizeMB = 15 + Math.random() * 20; // Random size between 15-35 MB
        
        if (totalSize) totalSize.textContent = `${totalSizeMB.toFixed(1)} MB`;
        
        const interval = setInterval(() => {
            progress += Math.random() * 10 + 2; // Random progress between 2-12%
            if (progress > 100) progress = 100;
            
            const currentSize = (progress / 100) * totalSizeMB;
            const speed = 200 + Math.random() * 800; // Random speed between 200-1000 KB/s
            
            if (progressBar) {
                progressBar.style.width = `${progress}%`;
                progressBar.setAttribute('aria-valuenow', progress);
            }
            if (progressText) progressText.textContent = `${Math.round(progress)}%`;
            if (downloadedSize) downloadedSize.textContent = `${currentSize.toFixed(1)} MB`;
            if (downloadSpeed) downloadSpeed.textContent = `${Math.round(speed)} KB/s`;
            
            if (progress >= 100) {
                clearInterval(interval);
                setTimeout(() => {
                    hideDownloadProgress();
                }, 1000);
            }
        }, 300);
    }
    
    function hideDownloadProgress() {
        if (downloadProgressModal) {
            downloadProgressModal.hide();
        }
    }
    
    function cancelDownload() {
        // In real implementation, this would cancel the actual download
        hideDownloadProgress();
        showNotification('Download cancelled', 'info');
    }
    
    function displayModelInfoModal(modelInfo) {
        // Update modal content
        const elements = {
            'modalModelName': modelInfo.name || 'Unknown',
            'modalModelType': modelInfo.type || 'Steganography',
            'modalModelVersion': modelInfo.version || '1.0',
            'modalModelSize': `${modelInfo.size_mb || 'Unknown'} MB`,
            'modalMessageLength': `${modelInfo.message_length || 'Variable'} bits`,
            'modalImageSize': modelInfo.image_size || 'Variable',
            'modalAccuracy': modelInfo.accuracy || 'Unknown',
            'modalSpeed': `${modelInfo.speed || 'Unknown'} ms`,
            'modalModelDescription': modelInfo.description || 'No description available',
            'modalUsageCount': modelInfo.usage_count || 0,
            'modalSuccessRate': `${modelInfo.success_rate || 0}%`,
            'modalAvgTime': `${modelInfo.avg_time || 0}ms`
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });
        
        // Show modal
        if (modelInfoModal) {
            modelInfoModal.show();
        }
    }
    
    function displayDetectorInfoModal(detectorInfo) {
        // Similar to model info but for detectors
        displayModelInfoModal(detectorInfo); // Reuse the same modal structure
    }
    
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
        
        // Create notification
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show`;
        notification.style.minWidth = '300px';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Add to container
        notificationContainer.appendChild(notification);
        
        // Auto-remove after duration
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, duration);
    }
});