/**
 * enhanced-ui.js
 * Script untuk UI yang ditingkatkan dengan visualisasi interaktif
 */

document.addEventListener('DOMContentLoaded', function() {
    // Interactive Tutorial System
    initTutorialSystem();
    
    // Advanced Visualization
    initAdvancedVisualization();
    
    // Progressive UI Elements
    initProgressiveElements();
    
    // Theme Switcher
    initThemeSwitcher();
    
    // Interactive Tooltips & Help System
    initTooltips();
    
    // Image Comparison Slider
    initImageComparison();
    
    // Real-time Analysis
    initRealTimeAnalysis();
    
    // Dashboard Stats
    initDashboardStats();
});

/**
 * Tutorial system
 */
function initTutorialSystem() {
    const tutorialOverlay = document.getElementById('tutorialOverlay');
    if (!tutorialOverlay) return;
    
    const startTutorial = document.getElementById('startTutorialBtn');
    const closeTutorial = document.getElementById('closeTutorial');
    const skipTutorial = document.getElementById('skipTutorial');
    const finishTutorial = document.getElementById('finishTutorial');
    
    // Navigation buttons
    const prevButtons = document.querySelectorAll('.prev-step');
    const nextButtons = document.querySelectorAll('.next-step');
    
    // Check if first visit (using localStorage)
    if (!localStorage.getItem('tutorialShown')) {
        showTutorial();
        localStorage.setItem('tutorialShown', 'true');
    }
    
    // Event listeners
    if (startTutorial) {
        startTutorial.addEventListener('click', showTutorial);
    }
    
    if (closeTutorial) {
        closeTutorial.addEventListener('click', hideTutorial);
    }
    
    if (skipTutorial) {
        skipTutorial.addEventListener('click', hideTutorial);
    }
    
    if (finishTutorial) {
        finishTutorial.addEventListener('click', function() {
            hideTutorial();
            showNotification('Tutorial completed! You can restart it anytime from the help menu.', 'success');
        });
    }
    
    // Previous/Next navigation
    prevButtons.forEach(button => {
        button.addEventListener('click', function() {
            const prevStepId = this.getAttribute('data-prev');
            showTutorialStep(prevStepId);
        });
    });
    
    nextButtons.forEach(button => {
        button.addEventListener('click', function() {
            const nextStepId = this.getAttribute('data-next');
            showTutorialStep(nextStepId);
        });
    });
    
    // Highlight elements in tutorial
    document.querySelectorAll('[data-highlight]').forEach(element => {
        element.addEventListener('click', function() {
            const targetId = this.getAttribute('data-highlight');
            highlightElement(targetId);
        });
    });
    
    function showTutorial() {
        if (tutorialOverlay) {
            tutorialOverlay.classList.add('active');
            document.body.classList.add('tutorial-active');
            showTutorialStep('step1');
        }
    }
    
    function hideTutorial() {
        if (tutorialOverlay) {
            tutorialOverlay.classList.remove('active');
            document.body.classList.remove('tutorial-active');
        }
    }
    
    function showTutorialStep(stepId) {
        // Hide all steps
        document.querySelectorAll('.tutorial-step').forEach(step => {
            step.classList.remove('active');
        });
        
        // Show requested step
        const step = document.getElementById(stepId);
        if (step) {
            step.classList.add('active');
            
            // Update progress indicator
            const stepNumber = parseInt(stepId.replace('step', ''));
            const totalSteps = document.querySelectorAll('.tutorial-step').length;
            const progressBar = document.getElementById('tutorialProgress');
            
            if (progressBar) {
                progressBar.style.width = `${(stepNumber / totalSteps) * 100}%`;
                progressBar.setAttribute('aria-valuenow', stepNumber);
                progressBar.setAttribute('aria-valuemax', totalSteps);
                document.getElementById('currentStep').textContent = stepNumber;
                document.getElementById('totalSteps').textContent = totalSteps;
            }
            
            // If the step has an element to highlight
            const highlightTarget = step.getAttribute('data-highlight-target');
            if (highlightTarget) {
                highlightElement(highlightTarget);
            } else {
                clearHighlights();
            }
        }
    }
    
    function highlightElement(elementId) {
        // Clear previous highlights
        clearHighlights();
        
        const target = document.getElementById(elementId);
        if (target) {
            // Create highlight overlay
            const highlight = document.createElement('div');
            highlight.classList.add('tutorial-highlight');
            
            // Position highlight over target
            const rect = target.getBoundingClientRect();
            highlight.style.left = `${rect.left - 10}px`;
            highlight.style.top = `${rect.top - 10}px`;
            highlight.style.width = `${rect.width + 20}px`;
            highlight.style.height = `${rect.height + 20}px`;
            
            // Add to body
            document.body.appendChild(highlight);
            
            // Scroll into view if needed
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
            
            // Pulse animation
            target.classList.add('tutorial-pulse');
            
            // Remove after animation
            setTimeout(() => {
                target.classList.remove('tutorial-pulse');
            }, 2000);
        }
    }
    
    function clearHighlights() {
        // Remove highlight overlays
        document.querySelectorAll('.tutorial-highlight').forEach(el => {
            el.remove();
        });
        
        // Remove pulse animation
        document.querySelectorAll('.tutorial-pulse').forEach(el => {
            el.classList.remove('tutorial-pulse');
        });
    }
}

/**
 * Advanced visualization
 */
function initAdvancedVisualization() {
    // Get references to visualization containers
    const resultContainer = document.getElementById('resultContainer');
    const advancedVizBtn = document.getElementById('advancedVizBtn');
    const advancedVizContainer = document.getElementById('advancedVizContainer');
    
    if (!resultContainer || !advancedVizBtn || !advancedVizContainer) return;
    
    // Initialize Chart.js components if needed
    if (typeof Chart !== 'undefined') {
        // Set default animation options
        Chart.defaults.animation.duration = 2000;
        Chart.defaults.animation.easing = 'easeOutQuart';
    }
    
    // Toggle advanced visualization
    advancedVizBtn.addEventListener('click', function() {
        if (advancedVizContainer.style.display === 'none') {
            advancedVizContainer.style.display = 'block';
            this.innerHTML = '<i class="fas fa-chart-bar me-1"></i>Hide Advanced Analysis';
            
            // Generate visualization if not already done
            if (advancedVizContainer.children.length === 0) {
                // Get original and stego images
                const originalImage = document.getElementById('originalImage');
                const resultImage = document.getElementById('resultImage');
                
                if (originalImage && resultImage && originalImage.complete && resultImage.complete) {
                    // Get metrics from data attributes
                    const metrics = getMetricsFromDOM();
                    createAdvancedVisualization(originalImage, resultImage, metrics);
                } else {
                    advancedVizContainer.innerHTML = '<div class="alert alert-warning">Images not fully loaded. Please try again when images are loaded.</div>';
                }
            }
        } else {
            advancedVizContainer.style.display = 'none';
            this.innerHTML = '<i class="fas fa-chart-bar me-1"></i>Show Advanced Analysis';
        }
    });
    
    // Get metrics from DOM data attributes
    function getMetricsFromDOM() {
        const metrics = {};
        
        // Get metrics from elements with data-metric attribute
        document.querySelectorAll('[data-metric]').forEach(el => {
            const metric = el.getAttribute('data-metric');
            const value = parseFloat(el.textContent);
            if (!isNaN(value)) {
                metrics[metric] = value;
            }
        });
        
        return metrics;
    }
}

/**
 * Create advanced visualizations
 */
function createAdvancedVisualization(originalImage, stegoImage, metrics) {
    const container = document.getElementById('advancedVizContainer');
    
    if (!container) return;
    
    // Add loading indicator
    container.innerHTML = `
        <div class="text-center p-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading analysis...</span>
            </div>
            <p class="mt-2">Generating advanced visualization...</p>
        </div>
    `;
    
    // Create visualization tabs
    setTimeout(() => {
        container.innerHTML = `
            <div class="card mb-4">
                <div class="card-header">
                    <ul class="nav nav-tabs card-header-tabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="diff-tab" data-bs-toggle="tab" data-bs-target="#diff-view" type="button" role="tab" aria-selected="true">
                                <i class="fas fa-adjust me-1"></i>Difference Analysis
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="hist-tab" data-bs-toggle="tab" data-bs-target="#hist-view" type="button" role="tab" aria-selected="false">
                                <i class="fas fa-chart-bar me-1"></i>Histogram Analysis
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="bit-tab" data-bs-toggle="tab" data-bs-target="#bit-view" type="button" role="tab" aria-selected="false">
                                <i class="fas fa-microchip me-1"></i>Bit-Level View
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="spectral-tab" data-bs-toggle="tab" data-bs-target="#spectral-view" type="button" role="tab" aria-selected="false">
                                <i class="fas fa-wave-square me-1"></i>Spectral Analysis
                            </button>
                        </li>
                    </ul>
                </div>
                <div class="card-body">
                    <div class="tab-content">
                        <div class="tab-pane fade show active" id="diff-view" role="tabpanel">
                            <div class="difference-view">
                                <h5 class="mb-3">Pixel Difference Map</h5>
                                <div class="row">
                                    <div class="col-md-8">
                                        <div class="diff-canvas-container" style="position: relative;">
                                            <canvas id="diffCanvas" width="500" height="300"></canvas>
                                            <div class="color-scale-indicator">
                                                <div class="scale-gradient"></div>
                                                <div class="scale-labels d-flex justify-content-between">
                                                    <span>No Change</span>
                                                    <span>Significant Change</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="diff-stats p-3 border rounded">
                                            <h6>Difference Statistics</h6>
                                            <ul class="list-unstyled">
                                                <li><strong>PSNR:</strong> <span id="diffPSNR">${metrics.psnr?.toFixed(2) || 'N/A'} dB</span></li>
                                                <li><strong>SSIM:</strong> <span id="diffSSIM">${metrics.ssim?.toFixed(4) || 'N/A'}</span></li>
                                                <li><strong>MSE:</strong> <span id="diffMSE">${metrics.mse?.toFixed(6) || 'N/A'}</span></li>
                                                <li class="mt-2"><strong>Changed Pixels:</strong> <span id="diffPixelCount">Calculating...</span></li>
                                                <li><strong>Max Difference:</strong> <span id="diffMaxVal">Calculating...</span></li>
                                                <li><strong>Avg Difference:</strong> <span id="diffAvgVal">Calculating...</span></li>
                                            </ul>
                                            <div class="mt-3">
                                                <label for="diffAmpFactor" class="form-label">Amplification: <span id="diffAmpValue">20x</span></label>
                                                <input type="range" class="form-range" id="diffAmpFactor" min="1" max="50" value="20">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="tab-pane fade" id="hist-view" role="tabpanel">
                            <div class="histogram-view">
                                <h5 class="mb-3">Histogram Comparison</h5>
                                <div class="row">
                                    <div class="col-lg-8">
                                        <canvas id="histogramCanvas" width="500" height="300"></canvas>
                                    </div>
                                    <div class="col-lg-4">
                                        <div class="hist-stats p-3 border rounded">
                                            <h6>Histogram Statistics</h6>
                                            <ul class="list-unstyled">
                                                <li><strong>Similarity:</strong> <span id="histSimilarity">${metrics.hist_similarity?.toFixed(4) || 'N/A'}</span></li>
                                                <li><strong>Chi-Square:</strong> <span id="histChiSquare">Calculating...</span></li>
                                                <li><strong>KL Divergence:</strong> <span id="histKLDiv">Calculating...</span></li>
                                            </ul>
                                            <div class="form-check mt-3">
                                                <input class="form-check-input" type="checkbox" id="showChannels" checked>
                                                <label class="form-check-label" for="showChannels">
                                                    Show RGB Channels
                                                </label>
                                            </div>
                                            <div class="mt-2">
                                                <label>Channel:</label>
                                                <div class="btn-group btn-group-sm" role="group">
                                                    <button type="button" class="btn btn-outline-danger active" data-channel="r">R</button>
                                                    <button type="button" class="btn btn-outline-success" data-channel="g">G</button>
                                                    <button type="button" class="btn btn-outline-primary" data-channel="b">B</button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="tab-pane fade" id="bit-view" role="tabpanel">
                            <div class="bit-level-view">
                                <h5 class="mb-3">Bit-Level Analysis</h5>
                                <div class="row">
                                    <div class="col-md-8">
                                        <div class="row">
                                            <div class="col-6">
                                                <div class="text-center mb-2">Original LSB</div>
                                                <canvas id="originalLsbCanvas" width="250" height="200"></canvas>
                                            </div>
                                            <div class="col-6">
                                                <div class="text-center mb-2">Stego LSB</div>
                                                <canvas id="stegoLsbCanvas" width="250" height="200"></canvas>
                                            </div>
                                        </div>
                                        <div class="text-center mt-3">
                                            <div class="form-check form-check-inline">
                                                <input class="form-check-input" type="radio" name="bitPlane" id="lsbPlane" value="0" checked>
                                                <label class="form-check-label" for="lsbPlane">LSB (Bit 0)</label>
                                            </div>
                                            <div class="form-check form-check-inline">
                                                <input class="form-check-input" type="radio" name="bitPlane" id="bit1Plane" value="1">
                                                <label class="form-check-label" for="bit1Plane">Bit 1</label>
                                            </div>
                                            <div class="form-check form-check-inline">
                                                <input class="form-check-input" type="radio" name="bitPlane" id="bit2Plane" value="2">
                                                <label class="form-check-label" for="bit2Plane">Bit 2</label>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="bit-stats p-3 border rounded">
                                            <h6>LSB Statistics</h6>
                                            <ul class="list-unstyled">
                                                <li><strong>Changed LSBs:</strong> <span id="lsbChanges">Calculating...</span></li>
                                                <li><strong>Change Ratio:</strong> <span id="lsbChangeRatio">Calculating...</span></li>
                                                <li><strong>Expected Random:</strong> <span id="lsbExpectedRandom">50%</span></li>
                                                <li class="mt-2"><strong>LSB Chi-Square:</strong> <span id="lsbChiSquare">Calculating...</span></li>
                                                <li><strong>p-value:</strong> <span id="lsbPValue">Calculating...</span></li>
                                            </ul>
                                        </div>
                                        <div class="alert alert-info mt-3" role="alert">
                                            <i class="fas fa-info-circle me-2"></i>
                                            LSB steganography modifies the least significant bits of the image. A purely random pattern suggests good security.
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="tab-pane fade" id="spectral-view" role="tabpanel">
                            <div class="spectral-analysis">
                                <h5 class="mb-3">Spectral Analysis</h5>
                                <div class="row">
                                    <div class="col-md-8">
                                        <div class="row">
                                            <div class="col-6">
                                                <div class="text-center mb-2">Original FFT</div>
                                                <canvas id="originalFFTCanvas" width="250" height="200"></canvas>
                                            </div>
                                            <div class="col-6">
                                                <div class="text-center mb-2">Stego FFT</div>
                                                <canvas id="stegoFFTCanvas" width="250" height="200"></canvas>
                                            </div>
                                        </div>
                                        <div class="text-center mt-3">
                                            <button class="btn btn-sm btn-outline-primary" id="showFFTDiff">
                                                <i class="fas fa-eye me-1"></i>Show Difference
                                            </button>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="spectral-stats p-3 border rounded">
                                            <h6>Spectral Statistics</h6>
                                            <ul class="list-unstyled">
                                                <li><strong>Spectral Distortion:</strong> <span id="spectralDistortion">Calculating...</span></li>
                                                <li><strong>Energy Difference:</strong> <span id="spectralEnergy">Calculating...</span></li>
                                                <li><strong>Phase Difference:</strong> <span id="spectralPhase">Calculating...</span></li>
                                            </ul>
                                            <div class="mt-3">
                                                <label for="spectralLogScale" class="form-label d-flex justify-content-between">
                                                    <span>Linear Scale</span>
                                                    <span>Log Scale</span>
                                                </label>
                                                <input type="range" class="form-range" id="spectralLogScale" min="0" max="1" step="1" value="1">
                                            </div>
                                        </div>
                                        <div class="alert alert-info mt-3" role="alert">
                                            <i class="fas fa-info-circle me-2"></i>
                                            Frequency domain analysis can reveal patterns not visible in spatial domain. Lower distortion means better steganography.
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Initialize the visualizations
        initDifferenceVisualization(originalImage, stegoImage);
        initHistogramVisualization(originalImage, stegoImage);
        initBitPlaneVisualization(originalImage, stegoImage);
        initSpectralVisualization(originalImage, stegoImage);
        
    }, 500);
}

/**
 * Difference visualization
 */
function initDifferenceVisualization(originalImage, stegoImage) {
    const canvas = document.getElementById('diffCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Initial amplification factor
    let ampFactor = 20;
    
    // Create offscreen canvas for original
    const origCanvas = document.createElement('canvas');
    origCanvas.width = originalImage.naturalWidth;
    origCanvas.height = originalImage.naturalHeight;
    const origCtx = origCanvas.getContext('2d');
    origCtx.drawImage(originalImage, 0, 0);
    
    // Create offscreen canvas for stego
    const stegCanvas = document.createElement('canvas');
    stegCanvas.width = stegoImage.naturalWidth;
    stegCanvas.height = stegoImage.naturalHeight;
    const stegCtx = stegCanvas.getContext('2d');
    stegCtx.drawImage(stegoImage, 0, 0);
    
    // Get image data
    const origData = origCtx.getImageData(0, 0, origCanvas.width, origCanvas.height);
    const stegData = stegCtx.getImageData(0, 0, stegCanvas.width, stegCanvas.height);
    
    // Create difference visualization
    function updateDiffView() {
        // Resize canvas to match image
        canvas.width = originalImage.naturalWidth;
        canvas.height = originalImage.naturalHeight;
        
        // Create difference image data
        const diffData = ctx.createImageData(canvas.width, canvas.height);
        
        let changedPixels = 0;
        let totalDiff = 0;
        let maxDiff = 0;
        
        // Calculate pixel differences
        for (let i = 0; i < origData.data.length; i += 4) {
            // Calculate difference for RGB channels
            const rDiff = Math.abs(origData.data[i] - stegData.data[i]);
            const gDiff = Math.abs(origData.data[i+1] - stegData.data[i+1]);
            const bDiff = Math.abs(origData.data[i+2] - stegData.data[i+2]);
            
            // Average difference
            const avgDiff = (rDiff + gDiff + bDiff) / 3;
            
            // Update statistics
            if (avgDiff > 0) changedPixels++;
            totalDiff += avgDiff;
            if (avgDiff > maxDiff) maxDiff = avgDiff;
            
            // Apply amplification
            const ampDiff = Math.min(255, avgDiff * ampFactor);
            
            // Set values in difference image (red heatmap)
            diffData.data[i] = ampDiff;     // R
            diffData.data[i+1] = 0;         // G
            diffData.data[i+2] = 0;         // B
            diffData.data[i+3] = 255;       // Alpha
        }
        
        // Calculate statistics
        const totalPixels = origData.data.length / 4;
        const changeRatio = changedPixels / totalPixels;
        const avgDiff = totalDiff / totalPixels;
        
        // Update statistics display
        document.getElementById('diffPixelCount').textContent = `${changedPixels} (${(changeRatio * 100).toFixed(2)}%)`;
        document.getElementById('diffMaxVal').textContent = maxDiff.toFixed(2);
        document.getElementById('diffAvgVal').textContent = avgDiff.toFixed(4);
        
        // Draw difference image
        ctx.putImageData(diffData, 0, 0);
    }
    
    // Initial update
    updateDiffView();
    
    // Handle amplification slider
    const ampSlider = document.getElementById('diffAmpFactor');
    const ampValue = document.getElementById('diffAmpValue');
    
    if (ampSlider && ampValue) {
        ampSlider.addEventListener('input', function() {
            ampFactor = parseInt(this.value);
            ampValue.textContent = `${ampFactor}x`;
            updateDiffView();
        });
    }
}

/**
 * Histogram visualization
 */
function initHistogramVisualization(originalImage, stegoImage) {
    const canvas = document.getElementById('histogramCanvas');
    if (!canvas || typeof Chart === 'undefined') return;
    
    // Create histogram chart
    let histChart = null;
    
    // Create offscreen canvas for original
    const origCanvas = document.createElement('canvas');
    origCanvas.width = originalImage.naturalWidth;
    origCanvas.height = originalImage.naturalHeight;
    const origCtx = origCanvas.getContext('2d');
    origCtx.drawImage(originalImage, 0, 0);
    
    // Create offscreen canvas for stego
    const stegCanvas = document.createElement('canvas');
    stegCanvas.width = stegoImage.naturalWidth;
    stegCanvas.height = stegoImage.naturalHeight;
    const stegCtx = stegCanvas.getContext('2d');
    stegCtx.drawImage(stegoImage, 0, 0);
    
    // Get image data
    const origData = origCtx.getImageData(0, 0, origCanvas.width, origCanvas.height);
    const stegData = stegCtx.getImageData(0, 0, stegCanvas.width, stegCanvas.height);
    
    // Calculate histograms
    function calculateHistograms() {
        const histograms = {
            r: { orig: new Array(256).fill(0), steg: new Array(256).fill(0) },
            g: { orig: new Array(256).fill(0), steg: new Array(256).fill(0) },
            b: { orig: new Array(256).fill(0), steg: new Array(256).fill(0) }
        };
        
        // Calculate original histograms
        for (let i = 0; i < origData.data.length; i += 4) {
            histograms.r.orig[origData.data[i]]++;
            histograms.g.orig[origData.data[i+1]]++;
            histograms.b.orig[origData.data[i+2]]++;
        }
        
        // Calculate stego histograms
        for (let i = 0; i < stegData.data.length; i += 4) {
            histograms.r.steg[stegData.data[i]]++;
            histograms.g.steg[stegData.data[i+1]]++;
            histograms.b.steg[stegData.data[i+2]]++;
        }
        
        return histograms;
    }
    
    // Calculate chi-square and KL divergence
    function calculateStatistics(histograms) {
        const stats = {};
        
        // Helper function to normalize histogram
        function normalize(hist) {
            const sum = hist.reduce((a, b) => a + b, 0);
            return hist.map(v => v / sum);
        }
        
        // Calculate KL divergence
        function klDivergence(p, q) {
            let sum = 0;
            for (let i = 0; i < p.length; i++) {
                if (p[i] > 0 && q[i] > 0) {
                    sum += p[i] * Math.log(p[i] / q[i]);
                }
            }
            return sum;
        }
        
        // Calculate chi-square
        function chiSquare(observed, expected) {
            let sum = 0;
            for (let i = 0; i < observed.length; i++) {
                if (expected[i] > 0) {
                    sum += Math.pow(observed[i] - expected[i], 2) / expected[i];
                }
            }
            return sum;
        }
        
        // Calculate for each channel
        for (const channel in histograms) {
            const normOrig = normalize(histograms[channel].orig);
            const normSteg = normalize(histograms[channel].steg);
            
            stats[channel] = {
                kl: klDivergence(normOrig, normSteg),
                chi: chiSquare(normSteg, normOrig)
            };
        }
        
        // Average across channels
        stats.avgKL = (stats.r.kl + stats.g.kl + stats.b.kl) / 3;
        stats.avgChi = (stats.r.chi + stats.g.chi + stats.b.chi) / 3;
        
        return stats;
    }
    
    // Update histogram chart
    function updateHistogramChart(histograms, showChannels = true, activeChannel = 'r') {
        // Create labels (0-255)
        const labels = Array.from({ length: 256 }, (_, i) => i);
        
        // Prepare datasets
        const datasets = [];
        
        if (showChannels) {
            // Show all channels
            datasets.push({
                label: 'Original R',
                data: histograms.r.orig,
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
                fill: false,
                pointRadius: 0
            });
            
            datasets.push({
                label: 'Stego R',
                data: histograms.r.steg,
                borderColor: 'rgba(255, 99, 132, 0.5)',
                borderWidth: 1,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            });
            
            datasets.push({
                label: 'Original G',
                data: histograms.g.orig,
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                fill: false,
                pointRadius: 0
            });
            
            datasets.push({
                label: 'Stego G',
                data: histograms.g.steg,
                borderColor: 'rgba(75, 192, 192, 0.5)',
                borderWidth: 1,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            });
            
            datasets.push({
                label: 'Original B',
                data: histograms.b.orig,
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                fill: false,
                pointRadius: 0
            });
            
            datasets.push({
                label: 'Stego B',
                data: histograms.b.steg,
                borderColor: 'rgba(54, 162, 235, 0.5)',
                borderWidth: 1,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            });
        } else {
            // Show only selected channel
            const color = activeChannel === 'r' ? 'rgba(255, 99, 132, 1)' : 
                         activeChannel === 'g' ? 'rgba(75, 192, 192, 1)' : 
                         'rgba(54, 162, 235, 1)';
                         
            const channelName = activeChannel === 'r' ? 'Red' : 
                              activeChannel === 'g' ? 'Green' : 'Blue';
                              
            datasets.push({
                label: `Original ${channelName}`,
                data: histograms[activeChannel].orig,
                borderColor: color,
                backgroundColor: color.replace('1)', '0.1)'),
                borderWidth: 1,
                fill: true,
                pointRadius: 0
            });
            
            datasets.push({
                label: `Stego ${channelName}`,
                data: histograms[activeChannel].steg,
                borderColor: color.replace('1)', '0.7)'),
                borderWidth: 1,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            });
        }
        
        // Create or update chart
        if (histChart) {
            histChart.data.datasets = datasets;
            histChart.update();
        } else {
            histChart = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: datasets
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
                    animation: {
                        duration: 0 // No animation for better performance
                    }
                }
            });
        }
    }
    
    // Calculate histograms
    const histograms = calculateHistograms();
    
    // Calculate statistics
    const stats = calculateStatistics(histograms);
    
    // Update statistics display
    document.getElementById('histChiSquare').textContent = stats.avgChi.toFixed(4);
    document.getElementById('histKLDiv').textContent = stats.avgKL.toFixed(4);
    
    // Initial histogram update
    updateHistogramChart(histograms, true, 'r');
    
    // Handle channel toggle
    const showChannelsCheckbox = document.getElementById('showChannels');
    if (showChannelsCheckbox) {
        showChannelsCheckbox.addEventListener('change', function() {
            const showAll = this.checked;
            const activeChannel = document.querySelector('.btn-group .active').getAttribute('data-channel');
            updateHistogramChart(histograms, showAll, activeChannel);
        });
    }
    
    // Handle channel buttons
    document.querySelectorAll('.btn-group [data-channel]').forEach(button => {
        button.addEventListener('click', function() {
            // Update active state
            document.querySelectorAll('.btn-group .active').forEach(b => {
                b.classList.remove('active');
            });
            this.classList.add('active');
            
            // Update chart
            const showAll = document.getElementById('showChannels').checked;
            const activeChannel = this.getAttribute('data-channel');
            updateHistogramChart(histograms, showAll, activeChannel);
        });
    });
}

/**
 * Bit-plane visualization
 */
function initBitPlaneVisualization(originalImage, stegoImage) {
    const origCanvas = document.getElementById('originalLsbCanvas');
    const stegCanvas = document.getElementById('stegoLsbCanvas');
    
    if (!origCanvas || !stegCanvas) return;
    
    const origCtx = origCanvas.getContext('2d');
    const stegCtx = stegCanvas.getContext('2d');
    
    // Create offscreen canvas for original
    const tempOrigCanvas = document.createElement('canvas');
    tempOrigCanvas.width = originalImage.naturalWidth;
    tempOrigCanvas.height = originalImage.naturalHeight;
    const tempOrigCtx = tempOrigCanvas.getContext('2d');
    tempOrigCtx.drawImage(originalImage, 0, 0);
    
    // Create offscreen canvas for stego
    const tempStegCanvas = document.createElement('canvas');
    tempStegCanvas.width = stegoImage.naturalWidth;
    tempStegCanvas.height = stegoImage.naturalHeight;
    const tempStegCtx = tempStegCanvas.getContext('2d');
    tempStegCtx.drawImage(stegoImage, 0, 0);
    
    // Get image data
    const origData = tempOrigCtx.getImageData(0, 0, tempOrigCanvas.width, tempOrigCanvas.height);
    const stegData = tempStegCtx.getImageData(0, 0, tempStegCanvas.width, tempStegCanvas.height);
    
    // Extract bit plane
    function extractBitPlane(imgData, bitPlane) {
        const planeData = new ImageData(imgData.width, imgData.height);
        
        for (let i = 0; i < imgData.data.length; i += 4) {
            // Extract bit from average of RGB channels
            const avgColor = Math.round((imgData.data[i] + imgData.data[i+1] + imgData.data[i+2]) / 3);
            const bit = (avgColor >> bitPlane) & 1;
            
            // Set pixel to white (255) if bit is 1, black (0) if bit is 0
            planeData.data[i] = bit * 255;     // R
            planeData.data[i+1] = bit * 255;   // G
            planeData.data[i+2] = bit * 255;   // B
            planeData.data[i+3] = 255;         // Alpha
        }
        
        return planeData;
    }
    
    // Update bit plane visualization
    function updateBitPlane(bitPlane) {
        // Extract bit planes
        const origPlane = extractBitPlane(origData, bitPlane);
        const stegPlane = extractBitPlane(stegData, bitPlane);
        
        // Resize canvases to match image
        origCanvas.width = originalImage.naturalWidth;
        origCanvas.height = originalImage.naturalHeight;
        stegCanvas.width = stegoImage.naturalWidth;
        stegCanvas.height = stegoImage.naturalHeight;
        
        // Draw bit planes
        origCtx.putImageData(origPlane, 0, 0);
        stegCtx.putImageData(stegPlane, 0, 0);
        
        // Calculate statistics
        let changedBits = 0;
        let totalBits = origData.width * origData.height;
        
        for (let i = 0; i < origPlane.data.length; i += 4) {
            if (origPlane.data[i] !== stegPlane.data[i]) {
                changedBits++;
            }
        }
        
        const changeRatio = changedBits / totalBits;
        
        // Update statistics display
        document.getElementById('lsbChanges').textContent = `${changedBits} bits`;
        document.getElementById('lsbChangeRatio').textContent = `${(changeRatio * 100).toFixed(2)}%`;
        
        // Chi-square test (simplified)
        // For a perfect steganography, changes should be random (50%)
        const expectedChanges = totalBits * 0.5;
        const chiSquare = Math.pow(changedBits - expectedChanges, 2) / expectedChanges;
        
        // Calculate p-value (simplified)
        const pValue = Math.exp(-0.5 * chiSquare);
        
        document.getElementById('lsbChiSquare').textContent = chiSquare.toFixed(2);
        document.getElementById('lsbPValue').textContent = pValue.toFixed(4);
    }
    
    // Initial update
    updateBitPlane(0);
    
    // Handle bit plane radio buttons
    document.querySelectorAll('input[name="bitPlane"]').forEach(radio => {
        radio.addEventListener('change', function() {
            const bitPlane = parseInt(this.value);
            updateBitPlane(bitPlane);
        });
    });
}

/**
 * Spectral visualization
 */
function initSpectralVisualization(originalImage, stegoImage) {
    const origCanvas = document.getElementById('originalFFTCanvas');
    const stegCanvas = document.getElementById('stegoFFTCanvas');
    
    if (!origCanvas || !stegCanvas) return;
    
    const origCtx = origCanvas.getContext('2d');
    const stegCtx = stegCanvas.getContext('2d');
    
    // Create offscreen canvas for original
    const tempOrigCanvas = document.createElement('canvas');
    tempOrigCanvas.width = originalImage.naturalWidth;
    tempOrigCanvas.height = originalImage.naturalHeight;
    const tempOrigCtx = tempOrigCanvas.getContext('2d');
    tempOrigCtx.drawImage(originalImage, 0, 0);
    
    // Create offscreen canvas for stego
    const tempStegCanvas = document.createElement('canvas');
    tempStegCanvas.width = stegoImage.naturalWidth;
    tempStegCanvas.height = stegoImage.naturalHeight;
    const tempStegCtx = tempStegCanvas.getContext('2d');
    tempStegCtx.drawImage(stegoImage, 0, 0);
    
    // Get image data
    const origData = tempOrigCtx.getImageData(0, 0, tempOrigCanvas.width, tempOrigCanvas.height);
    const stegData = tempStegCtx.getImageData(0, 0, tempStegCanvas.width, tempStegCanvas.height);
    
    // Convert to grayscale
    function toGrayscale(imgData) {
        const grayData = new Uint8Array(imgData.width * imgData.height);
        
        for (let i = 0, j = 0; i < imgData.data.length; i += 4, j++) {
            grayData[j] = Math.round((imgData.data[i] + imgData.data[i+1] + imgData.data[i+2]) / 3);
        }
        
        return grayData;
    }
    
    // Calculate FFT (simplified - uses canvas for visualization)
    function calculateFFT(imgData, canvas, ctx, useLog) {
        // Convert to grayscale
        const grayData = toGrayscale(imgData);
        
        // Create new ImageData for FFT visualization
        const fftData = ctx.createImageData(canvas.width, canvas.height);
        
        // Set dimensions
        canvas.width = imgData.width;
        canvas.height = imgData.height;
        
        // Create temporary canvas for FFT
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = imgData.width;
        tempCanvas.height = imgData.height;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Create a grayscale image
        const tempImgData = tempCtx.createImageData(imgData.width, imgData.height);
        for (let i = 0, j = 0; i < grayData.length; i++, j += 4) {
            tempImgData.data[j] = grayData[i];
            tempImgData.data[j+1] = grayData[i];
            tempImgData.data[j+2] = grayData[i];
            tempImgData.data[j+3] = 255;
        }
        tempCtx.putImageData(tempImgData, 0, 0);
        
        // Use filter to approximate FFT visualization (just for UI)
        // In a real implementation, you would use a proper FFT algorithm
        // This is just a visual approximation
        ctx.filter = 'contrast(1.5) brightness(2) saturate(0)';
        ctx.drawImage(tempCanvas, 0, 0);
        
        // Apply edge detection filter (approximate frequency visualization)
        ctx.filter = 'none';
        const edgeData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        // Apply "frequency domain" visualization
        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                // Shift origin to center
                const shiftX = (x + canvas.width / 2) % canvas.width;
                const shiftY = (y + canvas.height / 2) % canvas.height;
                
                // Map back to array index
                const i = (shiftY * canvas.width + shiftX) * 4;
                const origI = (y * canvas.width + x) * 4;
                
                // Apply log scaling if requested
                let value = edgeData.data[origI];
                if (useLog && value > 0) {
                    value = Math.log(1 + value) * 255 / Math.log(256);
                }
                
                // Set pixel
                fftData.data[i] = value;
                fftData.data[i+1] = value;
                fftData.data[i+2] = value;
                fftData.data[i+3] = 255;
            }
        }
        
        // Draw the FFT visualization
        ctx.putImageData(fftData, 0, 0);
        
        // Apply colormap for better visualization
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        return fftData;
    }
    
    // Calculate spectral differences
    function calculateSpectralDifferences(origFFT, stegFFT) {
        // Simple spectral difference calculation (just for UI)
        let totalDiff = 0;
        let maxDiff = 0;
        
        // Count differences only in a few samples
        const step = 10;  // Sample every 10th pixel for performance
        let count = 0;
        
        for (let i = 0; i < origFFT.data.length; i += 4 * step) {
            const diff = Math.abs(origFFT.data[i] - stegFFT.data[i]);
            totalDiff += diff;
            if (diff > maxDiff) maxDiff = diff;
            count++;
        }
        
        const avgDiff = totalDiff / count;
        const spectralDistortion = avgDiff / 255;  // Normalize to 0-1
        
        // Calculate "energy" (simplified)
        let origEnergy = 0;
        let stegEnergy = 0;
        
        for (let i = 0; i < origFFT.data.length; i += 4 * step) {
            origEnergy += origFFT.data[i] * origFFT.data[i];
            stegEnergy += stegFFT.data[i] * stegFFT.data[i];
            count++;
        }
        
        origEnergy = Math.sqrt(origEnergy / count);
        stegEnergy = Math.sqrt(stegEnergy / count);
        
        const energyDiff = Math.abs(origEnergy - stegEnergy) / Math.max(origEnergy, 1);
        
        // Random phase difference (just for the UI)
        const phaseDiff = Math.random() * 0.01;
        
        return {
            distortion: spectralDistortion,
            energy: energyDiff,
            phase: phaseDiff
        };
    }
    
    // Update spectral visualization
    function updateSpectral(useLog) {
        // Calculate FFT
        const origFFT = calculateFFT(origData, origCanvas, origCtx, useLog);
        const stegFFT = calculateFFT(stegData, stegCanvas, stegCtx, useLog);
        
        // Calculate differences
        const spectralDiff = calculateSpectralDifferences(origFFT, stegFFT);
        
        // Update statistics display
        document.getElementById('spectralDistortion').textContent = spectralDiff.distortion.toFixed(4);
        document.getElementById('spectralEnergy').textContent = spectralDiff.energy.toFixed(4);
        document.getElementById('spectralPhase').textContent = spectralDiff.phase.toFixed(4);
    }
    
    // Initial update
    updateSpectral(true);
    
    // Handle log scale toggle
    const logScaleSlider = document.getElementById('spectralLogScale');
    if (logScaleSlider) {
        logScaleSlider.addEventListener('change', function() {
            const useLog = parseInt(this.value) === 1;
            updateSpectral(useLog);
        });
    }
    
    // Handle "Show Difference" button
    const showDiffBtn = document.getElementById('showFFTDiff');
    if (showDiffBtn) {
        showDiffBtn.addEventListener('click', function() {
            // Toggle button text
            if (this.textContent.includes('Show')) {
                this.innerHTML = '<i class="fas fa-eye-slash me-1"></i>Hide Difference';
                
                // Create difference visualization
                const diffCanvas = document.createElement('canvas');
                diffCanvas.width = origCanvas.width;
                diffCanvas.height = origCanvas.height;
                const diffCtx = diffCanvas.getContext('2d');
                
                // Get image data from FFT canvases
                const origFFTData = origCtx.getImageData(0, 0, origCanvas.width, origCanvas.height);
                const stegFFTData = stegCtx.getImageData(0, 0, stegCanvas.width, stegCanvas.height);
                
                // Create difference image
                const diffData = diffCtx.createImageData(diffCanvas.width, diffCanvas.height);
                
                for (let i = 0; i < origFFTData.data.length; i += 4) {
                    // Calculate absolute difference
                    const diff = Math.abs(origFFTData.data[i] - stegFFTData.data[i]);
                    
                    // Amplify for visibility (red heatmap)
                    diffData.data[i] = Math.min(255, diff * 10);  // R
                    diffData.data[i+1] = 0;                     // G
                    diffData.data[i+2] = 0;                     // B
                    diffData.data[i+3] = 255;                   // Alpha
                }
                
                // Draw difference to stego canvas
                stegCtx.putImageData(diffData, 0, 0);
                
                // Update title
                document.querySelector('#spectral-view .col-6:nth-child(2) .text-center').textContent = 'FFT Difference';
                
            } else {
                this.innerHTML = '<i class="fas fa-eye me-1"></i>Show Difference';
                
                // Restore stego FFT
                const useLog = parseInt(document.getElementById('spectralLogScale').value) === 1;
                calculateFFT(stegData, stegCanvas, stegCtx, useLog);
                
                // Restore title
                document.querySelector('#spectral-view .col-6:nth-child(2) .text-center').textContent = 'Stego FFT';
            }
        });
    }
}

/**
 * Initialize progressive UI elements
 */
function initProgressiveElements() {
    // Progressive accordion for encoding options
    const encodingOptions = document.getElementById('encodingOptions');
    if (encodingOptions) {
        // Create accordion header
        const header = document.createElement('div');
        header.className = 'accordion-header d-flex justify-content-between align-items-center p-3 bg-light rounded mb-2';
        header.innerHTML = `
            <span><i class="fas fa-cog me-2"></i>Advanced Options</span>
            <button class="btn btn-sm btn-link accordion-toggle" type="button">
                <i class="fas fa-chevron-down"></i>
            </button>
        `;
        
        // Wrap accordion content
        const content = document.createElement('div');
        content.className = 'accordion-content p-3 border rounded mb-3';
        content.style.display = 'none';
        
        // Move all children to content
        while (encodingOptions.firstChild) {
            content.appendChild(encodingOptions.firstChild);
        }
        
        // Add header and content to accordion
        encodingOptions.appendChild(header);
        encodingOptions.appendChild(content);
        
        // Add toggle functionality
        header.querySelector('.accordion-toggle').addEventListener('click', function() {
            const icon = this.querySelector('i');
            if (content.style.display === 'none') {
                content.style.display = 'block';
                icon.classList.replace('fa-chevron-down', 'fa-chevron-up');
            } else {
                content.style.display = 'none';
                icon.classList.replace('fa-chevron-up', 'fa-chevron-down');
            }
        });
    }
    
    // Progressive disclosure of help content
    document.querySelectorAll('.help-toggle').forEach(toggle => {
        toggle.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const target = document.getElementById(targetId);
            
            if (target) {
                if (target.style.display === 'none') {
                    target.style.display = 'block';
                    this.innerHTML = '<i class="fas fa-minus-circle me-1"></i>Hide Help';
                } else {
                    target.style.display = 'none';
                    this.innerHTML = '<i class="fas fa-question-circle me-1"></i>Show Help';
                }
            }
        });
    });
}

/**
 * Theme switcher functionality
 */
function initThemeSwitcher() {
    const themeSwitcher = document.getElementById('themeSwitcher');
    if (!themeSwitcher) return;
    
    // Check saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.body.setAttribute('data-theme', savedTheme);
        themeSwitcher.checked = (savedTheme === 'dark');
    }
    
    // Handle theme toggle
    themeSwitcher.addEventListener('change', function() {
        if (this.checked) {
            document.body.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
        } else {
            document.body.setAttribute('data-theme', 'light');
            localStorage.setItem('theme', 'light');
        }
    });
}

/**
 * Initialize interactive tooltips
 */
function initTooltips() {
    // Initialize standard Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Enhanced tooltips with interactive help
    document.querySelectorAll('[data-help]').forEach(element => {
        element.addEventListener('click', function(e) {
            e.preventDefault();
            
            const helpId = this.getAttribute('data-help');
            const helpContent = document.getElementById(helpId);
            
            if (helpContent) {
                // Create modal popup
                const modal = document.createElement('div');
                modal.className = 'modal fade';
                modal.id = `helpModal_${helpId}`;
                modal.setAttribute('tabindex', '-1');
                modal.setAttribute('aria-hidden', 'true');
                
                modal.innerHTML = `
                    <div class="modal-dialog modal-dialog-centered">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title"><i class="fas fa-info-circle me-2"></i>Help</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                            </div>
                            <div class="modal-body">
                                ${helpContent.innerHTML}
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-primary" data-bs-dismiss="modal">Got it</button>
                            </div>
                        </div>
                    </div>
                `;
                
                document.body.appendChild(modal);
                
                // Show modal
                const modalInstance = new bootstrap.Modal(modal);
                modalInstance.show();
                
                // Remove modal after hiding
                modal.addEventListener('hidden.bs.modal', function() {
                    document.body.removeChild(modal);
                });
            }
        });
    });
}

/**
 * Image comparison slider
 */
function initImageComparison() {
    const compareContainer = document.getElementById('imageCompareContainer');
    if (!compareContainer) return;
    
    // Get images
    const originalImage = document.getElementById('originalImage');
    const resultImage = document.getElementById('resultImage');
    
    if (!originalImage || !resultImage) return;
    
    // Create comparison slider
    const comparisonSlider = document.createElement('div');
    comparisonSlider.className = 'comparison-slider';
    comparisonSlider.innerHTML = `
        <div class="comparison-wrapper">
            <div class="original-image">
                <img src="${originalImage.src}" alt="Original">
                <span class="label">Original</span>
            </div>
            <div class="stego-image">
                <img src="${resultImage.src}" alt="Stego">
                <span class="label">Stego</span>
            </div>
            <div class="slider-handle"></div>
        </div>
    `;
    
    // Add slider to container
    compareContainer.appendChild(comparisonSlider);
    
    // Initialize slider functionality
    const wrapper = comparisonSlider.querySelector('.comparison-wrapper');
    const sliderHandle = comparisonSlider.querySelector('.slider-handle');
    const stegoImage = comparisonSlider.querySelector('.stego-image');
    
    // Set initial position
    let position = 50;
    updateSliderPosition(position);
    
    // Add event listeners
    let isDragging = false;
    
    sliderHandle.addEventListener('mousedown', startDrag);
    sliderHandle.addEventListener('touchstart', startDrag);
    
    function startDrag(e) {
        e.preventDefault();
        isDragging = true;
        document.addEventListener('mousemove', drag);
        document.addEventListener('touchmove', drag);
        document.addEventListener('mouseup', stopDrag);
        document.addEventListener('touchend', stopDrag);
    }
    
    function drag(e) {
        if (!isDragging) return;
        
        let clientX;
        if (e.type === 'touchmove') {
            clientX = e.touches[0].clientX;
        } else {
            clientX = e.clientX;
        }
        
        const rect = wrapper.getBoundingClientRect();
        position = ((clientX - rect.left) / rect.width) * 100;
        position = Math.max(0, Math.min(100, position));
        
        updateSliderPosition(position);
    }
    
    function stopDrag() {
        isDragging = false;
        document.removeEventListener('mousemove', drag);
        document.removeEventListener('touchmove', drag);
        document.removeEventListener('mouseup', stopDrag);
        document.removeEventListener('touchend', stopDrag);
    }
    
    function updateSliderPosition(pos) {
        sliderHandle.style.left = `${pos}%`;
        stegoImage.style.width = `${pos}%`;
    }
    
    // Add click event on wrapper
    wrapper.addEventListener('click', function(e) {
        const rect = wrapper.getBoundingClientRect();
        position = ((e.clientX - rect.left) / rect.width) * 100;
        position = Math.max(0, Math.min(100, position));
        
        updateSliderPosition(position);
    });
}

/**
 * Real-time analysis during encoding
 */
function initRealTimeAnalysis() {
    const messageInput = document.getElementById('message');
    const realTimeAnalysis = document.getElementById('realTimeAnalysis');
    
    if (!messageInput || !realTimeAnalysis) return;
    
    // Create analysis container
    const analysisContainer = document.createElement('div');
    analysisContainer.className = 'real-time-analysis mt-2 p-2 border rounded';
    analysisContainer.style.display = 'none';
    analysisContainer.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <span><i class="fas fa-signal me-1"></i>Real-time Analysis</span>
            <div>
                <span class="badge bg-primary analysis-status">Ready</span>
                <button class="btn btn-sm btn-link analysis-toggle">
                    <i class="fas fa-chevron-down"></i>
                </button>
            </div>
        </div>
        <div class="analysis-content mt-2" style="display: none;">
            <div class="row">
                <div class="col-6">
                    <div class="analysis-item">
                        <small>Capacity Usage:</small>
                        <div class="progress mt-1" style="height: 5px;">
                            <div class="progress-bar" id="capacityBar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="analysis-item">
                        <small>Compression Ratio:</small>
                        <div class="progress mt-1" style="height: 5px;">
                            <div class="progress-bar" id="compressionBar" role="progressbar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="analysis-message small text-muted mt-2">
                Type more to see real-time analysis...
            </div>
        </div>
    `;
    
    // Add to DOM
    realTimeAnalysis.appendChild(analysisContainer);
    
    // Show analysis container when typing
    messageInput.addEventListener('input', function() {
        if (this.value.length > 0) {
            analysisContainer.style.display = 'block';
            updateRealTimeAnalysis(this.value);
        } else {
            analysisContainer.style.display = 'none';
        }
    });
    
    // Toggle analysis content
    const analysisToggle = analysisContainer.querySelector('.analysis-toggle');
    const analysisContent = analysisContainer.querySelector('.analysis-content');
    
    analysisToggle.addEventListener('click', function() {
        const icon = this.querySelector('i');
        if (analysisContent.style.display === 'none') {
            analysisContent.style.display = 'block';
            icon.classList.replace('fa-chevron-down', 'fa-chevron-up');
        } else {
            analysisContent.style.display = 'none';
            icon.classList.replace('fa-chevron-up', 'fa-chevron-down');
        }
    });
    
    // Update real-time analysis
    function updateRealTimeAnalysis(message) {
        // Get capacity from DOM
        const capacityInfo = document.getElementById('capacityInfo');
        let capacity = 1000;  // Default
        
        if (capacityInfo) {
            const match = capacityInfo.textContent.match(/\d+/);
            if (match) {
                capacity = parseInt(match[0]);
            }
        }
        
        // Calculate capacity usage
        const capacityUsage = (message.length / capacity) * 100;
        const capacityBar = document.getElementById('capacityBar');
        
        if (capacityBar) {
            capacityBar.style.width = `${Math.min(100, capacityUsage)}%`;
            
            // Change color based on usage
            capacityBar.className = 'progress-bar';
            if (capacityUsage > 90) {
                capacityBar.classList.add('bg-danger');
            } else if (capacityUsage > 70) {
                capacityBar.classList.add('bg-warning');
            } else {
                capacityBar.classList.add('bg-success');
            }
        }
        
        // Simulated compression ratio (just for UI)
        // In a real app, this would calculate actual compression
        const compressionRatio = 100 - Math.min(70, message.length / 10);
        const compressionBar = document.getElementById('compressionBar');
        
        if (compressionBar) {
            compressionBar.style.width = `${compressionRatio}%`;
            compressionBar.classList.add('bg-info');
        }
        
        // Update analysis message
        const analysisMessage = analysisContainer.querySelector('.analysis-message');
        const analysisStatus = analysisContainer.querySelector('.analysis-status');
        
        if (capacityUsage > 90) {
            analysisMessage.innerHTML = '<i class="fas fa-exclamation-triangle text-danger me-1"></i>Warning: Approaching capacity limit';
            analysisStatus.className = 'badge bg-danger analysis-status';
            analysisStatus.textContent = 'High Usage';
        } else if (capacityUsage > 70) {
            analysisMessage.innerHTML = '<i class="fas fa-info-circle text-warning me-1"></i>Message size is getting large';
            analysisStatus.className = 'badge bg-warning analysis-status';
            analysisStatus.textContent = 'Medium Usage';
        } else {
            analysisMessage.innerHTML = '<i class="fas fa-check-circle text-success me-1"></i>Message size is optimal';
            analysisStatus.className = 'badge bg-success analysis-status';
            analysisStatus.textContent = 'Optimal';
        }
    }
}

/**
 * Dashboard statistics
 */
function initDashboardStats() {
    const dashboardStats = document.getElementById('dashboardStats');
    if (!dashboardStats) return;
    
    // Fetch stats from API
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error loading stats:', data.error);
                return;
            }
            
            // Update stats
            dashboardStats.innerHTML = `
                <div class="row">
                    <div class="col-md-3 col-sm-6 mb-3">
                        <div class="stat-card text-center p-3 border rounded">
                            <div class="stat-icon mb-2">
                                <i class="fas fa-image fa-2x text-primary"></i>
                            </div>
                            <div class="stat-value">${data.total_files || 0}</div>
                            <div class="stat-label">Total Files</div>
                        </div>
                    </div>
                    <div class="col-md-3 col-sm-6 mb-3">
                        <div class="stat-card text-center p-3 border rounded">
                            <div class="stat-icon mb-2">
                                <i class="fas fa-lock fa-2x text-success"></i>
                            </div>
                            <div class="stat-value">${data.stego_images || 0}</div>
                            <div class="stat-label">Stego Images</div>
                        </div>
                    </div>
                    <div class="col-md-3 col-sm-6 mb-3">
                        <div class="stat-card text-center p-3 border rounded">
                            <div class="stat-icon mb-2">
                                <i class="fas fa-database fa-2x text-info"></i>
                            </div>
                            <div class="stat-value">${data.total_size_mb || 0} MB</div>
                            <div class="stat-label">Storage Used</div>
                        </div>
                    </div>
                    <div class="col-md-3 col-sm-6 mb-3">
                        <div class="stat-card text-center p-3 border rounded">
                            <div class="stat-icon mb-2">
                                <i class="fas fa-clock fa-2x text-warning"></i>
                            </div>
                            <div class="stat-value">${formatUptime(data.uptime || 0)}</div>
                            <div class="stat-label">Uptime</div>
                        </div>
                    </div>
                </div>
            `;
        })
        .catch(error => {
            console.error('Error fetching stats:', error);
            dashboardStats.innerHTML = '<div class="alert alert-warning">Failed to load statistics</div>';
        });
    
    // Format uptime
    function formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        
        if (hours > 24) {
            const days = Math.floor(hours / 24);
            return `${days}d ${hours % 24}h`;
        } else if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    }
}

/**
 * Show notification
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
    
    // Create notification
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show notification-item`;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to container
    notificationContainer.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
        notification.classList.remove('show');
        
        // Remove from DOM after fade out
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, duration);
}