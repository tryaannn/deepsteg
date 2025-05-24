/**
 * analyze.js
 * JavaScript untuk halaman analysis steganalysis
 */

document.addEventListener('DOMContentLoaded', function() {
    // Global variables
    let analysisChart = null;
    let confidenceChart = null;
    let noisePatternChart = null;
    
    // Initialize analysis functionality
    initAnalysisPage();
    
    function initAnalysisPage() {
        // Setup image preview
        setupImagePreview();
        
        // Setup form submission
        setupAnalysisForm();
        
        // Initialize charts
        initializeCharts();
        
        // Setup analysis options
        setupAnalysisOptions();
    }
    
    function setupImagePreview() {
        const input = document.getElementById('analyzeImageUpload');
        const preview = document.getElementById('analyzeImagePreview');
        
        if (!input || !preview) return;
        
        input.addEventListener('change', function() {
            const file = this.files[0];
            if (file && validateImageFile(file)) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    const previewImg = preview.querySelector('.preview-img');
                    const placeholder = preview.querySelector('.placeholder');
                    
                    previewImg.onload = function() {
                        previewImg.style.display = 'block';
                        if (placeholder) placeholder.style.display = 'none';
                    };
                    
                    previewImg.src = e.target.result;
                };
                
                reader.onerror = function() {
                    showNotification('Gagal membaca file gambar.', 'danger');
                };
                
                reader.readAsDataURL(file);
            } else if (file) {
                input.value = '';
            }
        });
        
        // Allow clicking on preview area to trigger file input
        preview.addEventListener('click', function() {
            input.click();
        });
    }
    
    function setupAnalysisForm() {
        const form = document.getElementById('analyzeForm');
        if (!form) return;
        
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const imageFile = document.getElementById('analyzeImageUpload').files[0];
            const detector = document.getElementById('detectorSelect').value;
            
            if (!imageFile) {
                showNotification('Silakan pilih gambar untuk dianalisis.', 'warning');
                return;
            }
            
            if (!validateImageFile(imageFile)) {
                return;
            }
            
            performAnalysis(imageFile, detector);
        });
    }
    
    function performAnalysis(imageFile, detector) {
        // Show loading state
        showLoadingState();
        
        // Disable submit button
        const submitBtn = document.getElementById('analyzeBtn');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Menganalisis...';
        
        // Create form data
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('detector', detector);
        
        // Get analysis options
        const lsbAnalysis = document.getElementById('enableLSBAnalysis').checked;
        const histogramAnalysis = document.getElementById('enableHistogramAnalysis').checked;
        const frequencyAnalysis = document.getElementById('enableFrequencyAnalysis').checked;
        
        formData.append('lsb_analysis', lsbAnalysis.toString());
        formData.append('histogram_analysis', histogramAnalysis.toString());
        formData.append('frequency_analysis', frequencyAnalysis.toString());
        
        // Simulate analysis progress
        simulateAnalysisProgress();
        
        // Send request to server
        fetch('/api/analyze-image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            hideLoadingState();
            
            if (data.error) {
                showNotification('Error: ' + data.error, 'danger');
                return;
            }
            
            if (data.status === 'success') {
                displayAnalysisResult(data.analysis);
                showNotification('Analisis berhasil diselesaikan!', 'success');
            } else {
                showNotification('Terjadi kesalahan yang tidak diketahui.', 'warning');
            }
        })
        .catch(error => {
            hideLoadingState();
            showNotification('Error: ' + error.message, 'danger');
        })
        .finally(() => {
            // Re-enable submit button
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
        });
    }
    
    function showLoadingState() {
        document.getElementById('analyzePlaceholderContainer').style.display = 'none';
        document.getElementById('analysisResultContainer').style.display = 'none';
        document.getElementById('analyzeLoadingContainer').style.display = 'block';
    }
    
    function hideLoadingState() {
        document.getElementById('analyzeLoadingContainer').style.display = 'none';
    }
    
    function simulateAnalysisProgress() {
        const progressBar = document.getElementById('analysisProgress');
        if (!progressBar) return;
        
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) progress = 95;
            
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
            
            if (progress >= 95) {
                clearInterval(interval);
            }
        }, 300);
        
        // Complete progress when analysis is done
        setTimeout(() => {
            clearInterval(interval);
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
        }, 5000);
    }
    
    function displayAnalysisResult(analysisData) {
        // Show result container
        document.getElementById('analysisResultContainer').style.display = 'block';
        
        // Update detection summary
        updateDetectionSummary(analysisData.detection_results);
        
        // Update detailed results
        updateDetectionTab(analysisData.detection_results);
        updateStatisticsTab(analysisData.statistical_analysis);
        updateVisualizationTab(analysisData.visualizations);
        
        // Update charts
        updateConfidenceChart(analysisData.detection_results.confidence);
        if (analysisData.statistical_analysis && analysisData.statistical_analysis.histogram) {
            updateHistogramChart(analysisData.statistical_analysis.histogram);
        }
        
        // Update noise pattern chart
        if (analysisData.visualizations && analysisData.visualizations.noise_pattern) {
            updateNoisePatternChart(analysisData.visualizations.noise_pattern);
        }
    }
    
    function updateDetectionSummary(detection) {
        const statusElement = document.getElementById('detectionStatus');
        const iconElement = document.getElementById('statusIcon');
        const resultElement = document.getElementById('detectionResult');
        const textElement = document.getElementById('detectionText');
        
        // Remove existing classes
        statusElement.className = 'detection-status p-3 rounded';
        iconElement.className = 'status-icon mb-2';
        
        if (detection.is_stego) {
            statusElement.classList.add('stego');
            iconElement.classList.add('stego');
            iconElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
            resultElement.textContent = 'STEGANOGRAFI TERDETEKSI';
            textElement.textContent = 'Gambar mengandung pesan tersembunyi';
        } else if (detection.confidence < 0.7) {
            statusElement.classList.add('suspicious');
            iconElement.classList.add('suspicious');
            iconElement.innerHTML = '<i class="fas fa-question-circle"></i>';
            resultElement.textContent = 'TIDAK PASTI';
            textElement.textContent = 'Kemungkinan mengandung steganografi';
        } else {
            statusElement.classList.add('clean');
            iconElement.classList.add('clean');
            iconElement.innerHTML = '<i class="fas fa-check-circle"></i>';
            resultElement.textContent = 'GAMBAR BERSIH';
            textElement.textContent = 'Tidak ada steganografi terdeteksi';
        }
    }
    
    function updateDetectionTab(detection) {
        // Update detection score
        const detectionScore = document.getElementById('detectionScore');
        const detectionScoreBar = document.getElementById('detectionScoreBar');
        
        if (detectionScore && detectionScoreBar) {
            const score = (detection.detection_score * 100).toFixed(1);
            detectionScore.textContent = `${score}%`;
            detectionScoreBar.style.width = `${score}%`;
            detectionScoreBar.setAttribute('aria-valuenow', score);
            
            // Color code the progress bar
            detectionScoreBar.className = 'progress-bar';
            if (detection.detection_score > 0.8) {
                detectionScoreBar.classList.add('bg-danger');
            } else if (detection.detection_score > 0.5) {
                detectionScoreBar.classList.add('bg-warning');
            } else {
                detectionScoreBar.classList.add('bg-success');
            }
        }
        
        // Update estimated payload
        const estimatedPayload = document.getElementById('estimatedPayload');
        if (estimatedPayload) {
            if (detection.estimated_payload > 0) {
                estimatedPayload.textContent = `${detection.estimated_payload} bytes`;
            } else {
                estimatedPayload.textContent = 'Tidak terdeteksi';
            }
        }
        
        // Update detector info
        const usedDetector = document.getElementById('usedDetector');
        const executionTime = document.getElementById('executionTime');
        
        if (usedDetector) {
            usedDetector.textContent = detection.detector_used || 'Unknown';
        }
        
        if (executionTime) {
            const time = detection.execution_time || 0;
            executionTime.textContent = `${time.toFixed(2)}s`;
        }
    }
    
    function updateStatisticsTab(statistics) {
        if (!statistics) return;
        
        // Update statistical tests
        const chiSquareValue = document.getElementById('chiSquareValue');
        const chiSquarePValue = document.getElementById('chiSquarePValue');
        const chiSquareResult = document.getElementById('chiSquareResult');
        
        if (statistics.chi_square_test) {
            const test = statistics.chi_square_test;
            if (chiSquareValue) chiSquareValue.textContent = test.value.toFixed(4);
            if (chiSquarePValue) chiSquarePValue.textContent = test.p_value.toFixed(4);
            if (chiSquareResult) {
                chiSquareResult.textContent = test.result;
                chiSquareResult.className = `test-result ${test.result.toLowerCase()}`;
            }
        }
        
        const lsbPairValue = document.getElementById('lsbPairValue');
        const lsbPairPValue = document.getElementById('lsbPairPValue');
        const lsbPairResult = document.getElementById('lsbPairResult');
        
        if (statistics.lsb_pair_analysis) {
            const test = statistics.lsb_pair_analysis;
            if (lsbPairValue) lsbPairValue.textContent = test.value.toFixed(4);
            if (lsbPairPValue) lsbPairPValue.textContent = test.p_value.toFixed(4);
            if (lsbPairResult) {
                lsbPairResult.textContent = test.result;
                lsbPairResult.className = `test-result ${test.result.toLowerCase()}`;
            }
        }
    }
    
    function updateVisualizationTab(visualizations) {
        if (!visualizations) return;
        
        // Update LSB visualization
        const lsbViualization = document.getElementById('lsbVisualization');
        if (lsbViualization && visualizations.lsb_plane) {
            lsbViualization.src = 'data:image/png;base64,' + visualizations.lsb_plane;
            lsbViualization.style.display = 'block';
        }
        
        // Update analysis summary
        const analysisSummary = document.getElementById('analysisSummary');
        if (analysisSummary && visualizations.summary) {
            analysisSummary.innerHTML = visualizations.summary;
        }
    }
    
    function updateConfidenceChart(confidence) {
        const canvas = document.getElementById('confidenceChart');
        const valueElement = document.getElementById('confidenceValue');
        const descriptionElement = document.getElementById('confidenceDescription');
        
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const percentage = Math.round(confidence * 100);
        
        // Update value display
        if (valueElement) valueElement.textContent = `${percentage}%`;
        
        // Update description
        if (descriptionElement) {
            if (percentage > 80) {
                descriptionElement.textContent = 'Sangat yakin';
                descriptionElement.className = 'text-success';
            } else if (percentage > 60) {
                descriptionElement.textContent = 'Cukup yakin';
                descriptionElement.className = 'text-info';
            } else if (percentage > 40) {
                descriptionElement.textContent = 'Tidak pasti';
                descriptionElement.className = 'text-warning';
            } else {
                descriptionElement.textContent = 'Kurang yakin';
                descriptionElement.className = 'text-danger';
            }
        }
        
        // Create donut chart
        if (confidenceChart) {
            confidenceChart.destroy();
        }
        
        confidenceChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [percentage, 100 - percentage],
                    backgroundColor: [
                        percentage > 60 ? '#e74c3c' : '#2ecc71',
                        '#ecf0f1'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                cutout: '70%'
            }
        });
    }
    
    function updateHistogramChart(histogramData) {
        const canvas = document.getElementById('histogramChart');
        if (!canvas || !histogramData) return;
        
        const ctx = canvas.getContext('2d');
        
        if (analysisChart) {
            analysisChart.destroy();
        }
        
        // Prepare data for chart
        const labels = Array.from({length: 256}, (_, i) => i);
        
        analysisChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Red Channel',
                    data: histogramData.red || [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 1,
                    fill: true,
                    pointRadius: 0
                }, {
                    label: 'Green Channel',
                    data: histogramData.green || [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 1,
                    fill: true,
                    pointRadius: 0
                }, {
                    label: 'Blue Channel',
                    data: histogramData.blue || [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 1,
                    fill: true,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Pixel Value'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    function updateNoisePatternChart(noiseData) {
        const canvas = document.getElementById('noisePatternChart');
        if (!canvas || !noiseData) return;
        
        const ctx = canvas.getContext('2d');
        
        if (noisePatternChart) {
            noisePatternChart.destroy();
        }
        
        noisePatternChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Noise Pattern',
                    data: noiseData,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    pointRadius: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'X Position'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Y Position'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    function initializeCharts() {
        // Initialize empty confidence chart
        const confidenceCanvas = document.getElementById('confidenceChart');
        if (confidenceCanvas) {
            updateConfidenceChart(0);
        }
    }
    
    function setupAnalysisOptions() {
        const options = ['enableLSBAnalysis', 'enableHistogramAnalysis', 'enableFrequencyAnalysis'];
        
        options.forEach(optionId => {
            const checkbox = document.getElementById(optionId);
            if (checkbox) {
                checkbox.addEventListener('change', function() {
                    // You can add logic here to show/hide related sections
                    console.log(`${optionId} changed to:`, this.checked);
                });
            }
        });
    }
    
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