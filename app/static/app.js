/**
 * Deepfake Detector Frontend Application
 * 
 * Handles file upload, WebSocket communication, and real-time
 * UI updates for deepfake analysis results.
 */

// State
let currentJobId = null;
let websocket = null;
let uploadedFile = null;

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileBtn = document.getElementById('file-btn');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const previewImage = document.getElementById('preview-image');
const fileName = document.getElementById('file-name');
const fileDetails = document.getElementById('file-details');
const fileFaces = document.getElementById('file-faces');
const analyzeBtn = document.getElementById('analyze-btn');
const verdictContent = document.getElementById('verdict-content');
const currentJobIdEl = document.getElementById('current-job-id');
const agentsGrid = document.getElementById('agents-grid');
const eventLog = document.getElementById('event-log');
const clearLogBtn = document.getElementById('clear-log-btn');
const gpuStatus = document.getElementById('gpu-status');
const agentCount = document.getElementById('agent-count');
const jobsList = document.getElementById('jobs-list');

// Agent display names, colors, and Mandrake roles
// Mandrake Architecture: Detection Agents (primary analysis), Verification Agents (cross-validation)
const AGENTS = {
    'CNNClassifierAgent': {
        name: 'CNN Classifier',
        color: '#6366f1',
        icon: '🧠',
        role: 'Detection Agent',  // Mandrake: Primary detection
        pattern: 'Checkpoint'     // Byzantine: Immediate result persistence
    },
    'ViTClassifierAgent': {
        name: 'ViT Classifier',
        color: '#8b5cf6',
        icon: '👁️',
        role: 'Detection Agent',
        pattern: 'Checkpoint'
    },
    'FrequencyAgent': {
        name: 'Frequency Analysis',
        color: '#06b6d4',
        icon: '📊',
        role: 'Detection Agent',
        pattern: 'Checkpoint'
    },
    'EmbeddingAnomalyAgent': {
        name: 'Embedding Anomaly',
        color: '#10b981',
        icon: '🔍',
        role: 'Verification Agent',  // Mandrake: Cross-validates identity
        pattern: 'Remind'            // Byzantine: Retry on timeout
    },
    'FaceXrayLikeAgent': {
        name: 'Face X-Ray',
        color: '#f59e0b',
        icon: '🔬',
        role: 'Verification Agent',
        pattern: 'Remind'
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkHealth();
    loadRecentJobs();
});

function initializeEventListeners() {
    // File input - direct click handler
    fileBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });

    // File input change - handle file selection
    fileInput.addEventListener('change', (e) => {
        e.stopPropagation();
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    // Drop zone events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // Analyze button
    analyzeBtn.addEventListener('click', () => {
        if (uploadedFile) {
            startAnalysis();
        }
    });

    // Clear log
    clearLogBtn.addEventListener('click', clearEventLog);
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        logEvent('error', 'Please select an image file');
        return;
    }

    uploadedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;

        // Get image dimensions
        const img = new Image();
        img.onload = () => {
            const sizeKB = (file.size / 1024).toFixed(1);
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            const sizeStr = file.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;

            fileName.textContent = file.name;
            fileDetails.textContent = `${img.width}×${img.height} • ${sizeStr}`;
            fileFaces.textContent = '';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Show file info, hide drop zone content
    dropZone.style.display = 'none';
    fileInfo.style.display = 'flex';

    logEvent('info', `Selected: ${file.name}`);
}

async function startAnalysis() {
    if (!uploadedFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    // Reset UI
    resetAgentCards();
    showLoadingVerdict();

    try {
        // Upload file
        const formData = new FormData();
        formData.append('file', uploadedFile);

        logEvent('info', 'Uploading image...');

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        currentJobId = data.job_id;
        currentJobIdEl.textContent = `#${currentJobId}`;

        logEvent('success', `Job started: ${currentJobId}`);

        // Connect WebSocket for real-time updates
        connectWebSocket(currentJobId);

        // Initialize agent cards
        initializeAgentCards();

    } catch (error) {
        logEvent('error', `Analysis failed: ${error.message}`);
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze';
    }
}

function connectWebSocket(jobId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${jobId}`;

    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
        logEvent('info', 'Connected to analysis stream');
    };

    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketEvent(data);
    };

    websocket.onerror = (error) => {
        logEvent('error', 'WebSocket error');
        console.error('WebSocket error:', error);
    };

    websocket.onclose = () => {
        logEvent('info', 'Analysis stream closed');
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze';
    };
}

function handleWebSocketEvent(event) {
    const { event_type, data } = event;

    switch (event_type) {
        case 'connected':
            logEvent('info', 'Waiting for agent results...');
            break;

        case 'job_started':
            logEvent('info', `Analysis started with ${data.agents?.length || 5} agents`);
            break;

        case 'agent_started':
            updateAgentCard(data.agent_name, 'running');
            logEvent('info', `${AGENTS[data.agent_name]?.name || data.agent_name}: Started`);
            break;

        case 'agent_completed':
            updateAgentCard(data.agent_name, 'completed', data);
            logEvent('success', `${AGENTS[data.agent_name]?.name || data.agent_name}: Score ${(data.score * 100).toFixed(0)}%`);
            break;

        case 'agent_error':
            updateAgentCard(data.agent_name, 'error', data);
            logEvent('error', `${AGENTS[data.agent_name]?.name || data.agent_name}: ${data.error}`);
            break;

        case 'agent_retry':
            logEvent('warn', `${AGENTS[data.agent_name]?.name || data.agent_name}: Retrying (${data.retry_count})`);
            break;

        case 'job_completed':
            showVerdict(data);
            logEvent('success', `Analysis complete: ${data.verdict} (${(data.confidence * 100).toFixed(0)}% confidence)`);
            addToRecentJobs(currentJobId, data);
            // Fetch complete results from API to update any agents that finished after quorum
            fetchCompleteResults(currentJobId);
            break;

        case 'job_failed':
            showErrorVerdict(data.error || 'Analysis failed');
            logEvent('error', `Analysis failed: ${data.error}`);
            break;

        case 'ping':
            // Keepalive, ignore
            break;

        default:
            console.log('Unknown event:', event_type, data);
    }
}

function initializeAgentCards() {
    agentsGrid.innerHTML = '';

    for (const [agentId, agent] of Object.entries(AGENTS)) {
        const card = document.createElement('div');
        card.className = 'agent-card';
        card.id = `agent-${agentId}`;

        // Determine role class for styling
        const roleClass = agent.role === 'Detection Agent' ? 'role-detection' : 'role-verification';

        card.innerHTML = `
            <div class="agent-header">
                <span class="agent-icon">${agent.icon}</span>
                <span class="agent-name">${agent.name}</span>
                <span class="agent-status status-queued">Queued</span>
            </div>
            <div class="agent-role-bar ${roleClass}">
                <span class="role-label">${agent.role}</span>
                <span class="pattern-badge" title="Byzantine Fault-Tolerance Pattern">${agent.pattern}</span>
            </div>
            <div class="agent-body">
                <div class="agent-score-container">
                    <div class="agent-score">--</div>
                    <div class="agent-score-label">Score</div>
                </div>
                <div class="agent-explanation">Waiting for analysis...</div>
            </div>
            <div class="agent-details" style="display: none;"></div>
        `;
        agentsGrid.appendChild(card);
    }
}

function updateAgentCard(agentName, status, data = {}) {
    const card = document.getElementById(`agent-${agentName}`);
    if (!card) return;

    const statusEl = card.querySelector('.agent-status');
    const scoreEl = card.querySelector('.agent-score');
    const explanationEl = card.querySelector('.agent-explanation');
    const detailsEl = card.querySelector('.agent-details');

    // Update status
    statusEl.className = `agent-status status-${status}`;
    statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);

    if (status === 'running') {
        card.classList.add('running');
        scoreEl.innerHTML = '<span class="loading-spinner"></span>';
        explanationEl.textContent = 'Analyzing...';
    } else if (status === 'completed') {
        card.classList.remove('running');
        card.classList.add('completed');

        const score = (data.score * 100).toFixed(0);
        scoreEl.textContent = `${score}%`;
        scoreEl.style.color = getScoreColor(data.score);
        explanationEl.textContent = data.explanation || 'Analysis complete';

        // Show ArcFace metric if available (for EmbeddingAnomalyAgent)
        if (agentName === 'EmbeddingAnomalyAgent' && data.arcface_similarity !== undefined) {
            detailsEl.innerHTML = `
                <div class="detail-item">
                    <span class="detail-label">ArcFace Similarity:</span>
                    <span class="detail-value">${(data.arcface_similarity * 100).toFixed(1)}%</span>
                </div>
            `;
            detailsEl.style.display = 'block';
        }

        // Show latency if available
        if (data.latency_ms) {
            const latencyHtml = `
                <div class="detail-item">
                    <span class="detail-label">Latency:</span>
                    <span class="detail-value">${data.latency_ms.toFixed(0)}ms</span>
                </div>
            `;
            detailsEl.innerHTML += latencyHtml;
            detailsEl.style.display = 'block';
        }
    } else if (status === 'error') {
        card.classList.remove('running');
        card.classList.add('error');
        scoreEl.textContent = '!';
        explanationEl.textContent = data.error || 'Error occurred';
    }
}

function resetAgentCards() {
    for (const card of document.querySelectorAll('.agent-card')) {
        card.classList.remove('running', 'completed', 'error');
    }
}

function showLoadingVerdict() {
    verdictContent.innerHTML = `
        <div class="verdict-loading">
            <div class="loading-spinner large"></div>
            <p>Analyzing image with multiple agents...</p>
        </div>
    `;
}

function showVerdict(data) {
    const verdictClass = data.verdict === 'FAKE' ? 'fake' : 'real';
    const confidencePercent = (data.confidence * 100).toFixed(0);
    const scorePercent = (data.final_score * 100).toFixed(0);
    const agentsCompleted = data.agents_completed || 5;
    const agentsTotal = data.agents_total || 5;

    verdictContent.innerHTML = `
        <div class="verdict-result ${verdictClass}">
            <div class="aggregator-header">
                <span class="aggregator-label">🎯 Aggregator Agent</span>
                <span class="pattern-badge" title="Byzantine Fault-Tolerance: Finalize when quorum reached">Continue</span>
            </div>
            <div class="verdict-badge">${data.verdict}</div>
            <div class="verdict-stats">
                <div class="stat">
                    <span class="stat-value">${confidencePercent}%</span>
                    <span class="stat-label">Confidence</span>
                </div>
                <div class="stat">
                    <span class="stat-value">${scorePercent}%</span>
                    <span class="stat-label">Fake Score</span>
                </div>
                <div class="stat">
                    <span class="stat-value">${agentsCompleted}/${agentsTotal}</span>
                    <span class="stat-label">Agents</span>
                </div>
            </div>
            <p class="verdict-explanation">${data.explanation}</p>
            ${data.quorum_reached ? '<span class="quorum-badge">✓ Byzantine Consensus (Quorum Reached)</span>' : ''}
        </div>
    `;
}

function showErrorVerdict(error) {
    verdictContent.innerHTML = `
        <div class="verdict-error">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            <p>${error}</p>
        </div>
    `;
}

function getScoreColor(score) {
    if (score >= 0.7) return '#ef4444';  // Red for high fake probability
    if (score >= 0.5) return '#f59e0b';  // Orange for moderate
    if (score >= 0.3) return '#eab308';  // Yellow for low
    return '#22c55e';  // Green for likely real
}

// Event Log Functions
function logEvent(type, message) {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `
        <span class="log-time">[${time}]</span>
        <span class="log-message">${message}</span>
    `;
    eventLog.appendChild(entry);
    eventLog.scrollTop = eventLog.scrollHeight;
}

function clearEventLog() {
    eventLog.innerHTML = `
        <div class="log-entry info">
            <span class="log-time">[${new Date().toLocaleTimeString()}]</span>
            <span class="log-message">Log cleared</span>
        </div>
    `;
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        const gpuDot = gpuStatus.querySelector('.status-dot');
        const gpuText = gpuStatus.querySelector('span:last-child');

        if (data.gpu_available) {
            gpuDot.classList.add('active');
            gpuText.textContent = `GPU: ${data.gpu_name || 'Available'}`;
        } else {
            gpuDot.classList.remove('active');
            gpuText.textContent = 'GPU: CPU Mode';
        }

        if (data.agents) {
            agentCount.textContent = data.agents.length;
        }

    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Recent Jobs
async function loadRecentJobs() {
    try {
        const response = await fetch('/api/jobs?limit=5');
        const data = await response.json();

        if (data.jobs && data.jobs.length > 0) {
            jobsList.innerHTML = '';
            for (const job of data.jobs) {
                addJobToList(job);
            }
        }
    } catch (error) {
        console.error('Failed to load recent jobs:', error);
    }
}

function addJobToList(job) {
    const li = document.createElement('li');
    li.className = 'job-item';

    const verdictClass = job.verdict === 'FAKE' ? 'fake' : job.verdict === 'REAL' ? 'real' : '';
    const confidence = job.confidence ? `${(job.confidence * 100).toFixed(0)}%` : '--';

    li.innerHTML = `
        <span class="job-id">#${job.job_id}</span>
        <span class="job-verdict ${verdictClass}">${job.verdict || 'Processing'}</span>
        <span class="job-confidence">${confidence}</span>
    `;

    li.addEventListener('click', () => loadJobResult(job.job_id));
    jobsList.appendChild(li);
}

function addToRecentJobs(jobId, data) {
    // Remove placeholder if present
    const placeholder = jobsList.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    // Add to top of list
    const li = document.createElement('li');
    li.className = 'job-item';

    const verdictClass = data.verdict === 'FAKE' ? 'fake' : 'real';
    const confidence = `${(data.confidence * 100).toFixed(0)}%`;

    li.innerHTML = `
        <span class="job-id">#${jobId}</span>
        <span class="job-verdict ${verdictClass}">${data.verdict}</span>
        <span class="job-confidence">${confidence}</span>
    `;

    li.addEventListener('click', () => loadJobResult(jobId));
    jobsList.insertBefore(li, jobsList.firstChild);

    // Keep only 5 recent
    while (jobsList.children.length > 5) {
        jobsList.removeChild(jobsList.lastChild);
    }
}

async function loadJobResult(jobId) {
    try {
        const response = await fetch(`/api/result/${jobId}`);
        const data = await response.json();

        currentJobId = jobId;
        currentJobIdEl.textContent = `#${jobId}`;

        // Update agent cards
        initializeAgentCards();
        for (const [agentName, state] of Object.entries(data.agent_states)) {
            if (state.result) {
                // Include arcface_similarity from details if available
                const resultData = {
                    ...state.result,
                    arcface_similarity: state.result.details?.arcface_similarity
                };
                updateAgentCard(agentName, 'completed', resultData);
            } else if (state.error) {
                updateAgentCard(agentName, 'error', state.error);
            }
        }

        // Update verdict
        if (data.aggregated_result) {
            showVerdict(data.aggregated_result);
        }

        logEvent('info', `Loaded job ${jobId}`);

    } catch (error) {
        logEvent('error', `Failed to load job ${jobId}`);
    }
}

/**
 * Fetch complete results from API after job completion.
 * This ensures all agent results are displayed, even those
 * that finished after quorum was reached.
 */
async function fetchCompleteResults(jobId) {
    try {
        // Small delay to allow any final results to be persisted
        await new Promise(resolve => setTimeout(resolve, 500));

        const response = await fetch(`/api/result/${jobId}`);
        const data = await response.json();

        // Update each agent card with complete data
        for (const [agentName, state] of Object.entries(data.agent_states)) {
            if (state.result) {
                // Include arcface_similarity from details if available
                const resultData = {
                    ...state.result,
                    arcface_similarity: state.result.details?.arcface_similarity
                };
                updateAgentCard(agentName, 'completed', resultData);
            } else if (state.error) {
                updateAgentCard(agentName, 'error', state.error);
            }
        }

        // Update verdict with complete agent count
        if (data.aggregated_result) {
            showVerdict(data.aggregated_result);
        }

    } catch (error) {
        console.error('Failed to fetch complete results:', error);
    }
}

