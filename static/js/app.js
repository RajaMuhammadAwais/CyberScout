// OSINT Reconnaissance Tool - Frontend JavaScript

class OSINTApp {
    constructor() {
        this.currentTaskId = null;
        this.taskCheckInterval = null;
        this.activeTasks = new Map();
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadActiveTasks();
        this.setupAutoRefresh();
    }
    
    bindEvents() {
        // Form submission
        document.getElementById('reconnaissance-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startReconnaissance();
        });
        
        // Target validation
        document.getElementById('validate-target').addEventListener('click', () => {
            this.validateTarget();
        });
        
        // Target input change
        document.getElementById('target').addEventListener('input', () => {
            this.clearTargetValidation();
        });
        
        // Select all modules
        document.getElementById('select-all-modules').addEventListener('change', (e) => {
            this.toggleAllModules(e.target.checked);
        });
        
        // Refresh tasks
        document.getElementById('refresh-tasks').addEventListener('click', () => {
            this.loadActiveTasks();
        });
        
        // Auto-validate target on blur
        document.getElementById('target').addEventListener('blur', () => {
            const target = document.getElementById('target').value.trim();
            if (target) {
                this.validateTarget();
            }
        });
    }
    
    setupAutoRefresh() {
        // Refresh active tasks every 5 seconds
        setInterval(() => {
            this.loadActiveTasks();
        }, 5000);
    }
    
    async startReconnaissance() {
        const form = document.getElementById('reconnaissance-form');
        const submitBtn = document.getElementById('start-reconnaissance');
        
        try {
            // Validate form
            const formData = this.getFormData();
            if (!this.validateForm(formData)) {
                return;
            }
            
            // Disable form
            this.setLoadingState(true);
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Starting...';
            
            // Send request
            const response = await fetch('/api/start_reconnaissance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.currentTaskId = result.task_id;
                this.showAlert('success', `Reconnaissance started successfully! Task ID: ${result.task_id.substring(0, 8)}`);
                this.loadActiveTasks();
                this.scrollToTasks();
            } else {
                this.showAlert('danger', result.error || 'Failed to start reconnaissance');
            }
            
        } catch (error) {
            console.error('Error starting reconnaissance:', error);
            this.showAlert('danger', 'Network error occurred');
        } finally {
            this.setLoadingState(false);
            submitBtn.innerHTML = '<i class="fas fa-play me-2"></i>Start Reconnaissance';
        }
    }
    
    getFormData() {
        const target = document.getElementById('target').value.trim();
        const modules = [];
        
        // Get selected modules
        document.querySelectorAll('input[type="checkbox"][value]').forEach(checkbox => {
            if (checkbox.checked && checkbox.value !== '') {
                modules.push(checkbox.value);
            }
        });
        
        return {
            target: target,
            modules: modules,
            rate_limit: parseFloat(document.getElementById('rate-limit').value),
            timeout: parseInt(document.getElementById('timeout').value),
            max_concurrent: parseInt(document.getElementById('max-concurrent').value),
            verbose: document.getElementById('verbose').checked
        };
    }
    
    validateForm(formData) {
        if (!formData.target) {
            this.showAlert('warning', 'Please enter a target');
            return false;
        }
        
        if (formData.modules.length === 0) {
            this.showAlert('warning', 'Please select at least one reconnaissance module');
            return false;
        }
        
        return true;
    }
    
    async validateTarget() {
        const targetInput = document.getElementById('target');
        const target = targetInput.value.trim();
        const validationDiv = document.getElementById('target-validation');
        const targetTypeDiv = document.getElementById('target-type');
        
        if (!target) {
            this.clearTargetValidation();
            return;
        }
        
        try {
            const response = await fetch('/api/validate_target', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ target: target })
            });
            
            const result = await response.json();
            
            if (result.valid) {
                targetInput.classList.remove('target-invalid');
                targetInput.classList.add('target-valid');
                
                validationDiv.innerHTML = `
                    <div class="alert alert-success py-2">
                        <i class="fas fa-check me-2"></i>Valid target format
                    </div>
                `;
                
                // Update target type badge
                const badgeClass = `target-type-${result.target_type}`;
                targetTypeDiv.innerHTML = `
                    <span class="badge ${badgeClass}">${result.target_type.toUpperCase()}</span>
                `;
                
            } else {
                targetInput.classList.remove('target-valid');
                targetInput.classList.add('target-invalid');
                
                validationDiv.innerHTML = `
                    <div class="alert alert-danger py-2">
                        <i class="fas fa-times me-2"></i>${result.error || 'Invalid target format'}
                    </div>
                `;
                
                targetTypeDiv.innerHTML = '<span class="badge bg-secondary">Invalid</span>';
            }
            
        } catch (error) {
            console.error('Target validation error:', error);
            this.showAlert('warning', 'Could not validate target');
        }
    }
    
    clearTargetValidation() {
        const targetInput = document.getElementById('target');
        const validationDiv = document.getElementById('target-validation');
        const targetTypeDiv = document.getElementById('target-type');
        
        targetInput.classList.remove('target-valid', 'target-invalid');
        validationDiv.innerHTML = '';
        targetTypeDiv.innerHTML = '<span class="badge bg-secondary">Unknown</span>';
    }
    
    toggleAllModules(checked) {
        document.querySelectorAll('input[type="checkbox"][value]').forEach(checkbox => {
            if (checkbox.value !== '') {
                checkbox.checked = checked;
            }
        });
    }
    
    async loadActiveTasks() {
        try {
            const response = await fetch('/api/active_tasks');
            const result = await response.json();
            
            if (response.ok) {
                this.displayTasks(result.tasks);
            } else {
                console.error('Failed to load tasks:', result.error);
            }
            
        } catch (error) {
            console.error('Error loading tasks:', error);
        }
    }
    
    displayTasks(tasks) {
        const container = document.getElementById('tasks-container');
        
        if (!tasks || tasks.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="fas fa-inbox fa-2x mb-2"></i>
                    <p>No active tasks</p>
                </div>
            `;
            return;
        }
        
        const tasksHtml = tasks.map(task => this.createTaskCard(task)).join('');
        container.innerHTML = tasksHtml;
        
        // Store tasks for reference
        tasks.forEach(task => {
            this.activeTasks.set(task.task_id, task);
        });
    }
    
    createTaskCard(task) {
        const statusColor = this.getStatusColor(task.status);
        const statusIcon = this.getStatusIcon(task.status);
        const progressWidth = task.progress || 0;
        
        return `
            <div class="card task-card task-status-${task.status} fade-in">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <div>
                            <h6 class="card-title mb-1">
                                <i class="${statusIcon} me-2 text-${statusColor}"></i>
                                ${task.target}
                            </h6>
                            <small class="text-muted">
                                Task ID: ${task.task_id.substring(0, 8)}
                            </small>
                        </div>
                        <div class="text-end">
                            <span class="badge bg-${statusColor}">${task.status.toUpperCase()}</span>
                            ${task.status === 'running' ? 
                                `<button class="btn btn-sm btn-outline-danger ms-2" onclick="app.cancelTask('${task.task_id}')">
                                    <i class="fas fa-stop"></i>
                                </button>` : ''}
                        </div>
                    </div>
                    
                    <div class="mb-2">
                        <small class="text-muted">Modules: ${task.modules.join(', ')}</small>
                    </div>
                    
                    ${task.status === 'running' ? `
                        <div class="progress mb-2">
                            <div class="progress-bar bg-${statusColor}" 
                                 style="width: ${progressWidth}%" 
                                 role="progressbar">
                                ${progressWidth}%
                            </div>
                        </div>
                    ` : ''}
                    
                    <div class="d-flex justify-content-between align-items-center">
                        <small class="text-muted">
                            ${task.start_time ? 
                                `Started: ${new Date(task.start_time).toLocaleString()}` : 
                                'Not started'}
                        </small>
                        
                        ${task.status === 'completed' ? `
                            <div class="btn-group btn-group-sm">
                                <button class="btn btn-outline-primary" onclick="app.viewResults('${task.task_id}')">
                                    <i class="fas fa-eye"></i> View
                                </button>
                                <button class="btn btn-outline-success" onclick="app.downloadResults('${task.task_id}', 'json')">
                                    <i class="fas fa-download"></i> JSON
                                </button>
                                <button class="btn btn-outline-info" onclick="app.downloadResults('${task.task_id}', 'csv')">
                                    <i class="fas fa-download"></i> CSV
                                </button>
                            </div>
                        ` : ''}
                        
                        ${task.status === 'failed' && task.error ? `
                            <small class="text-danger">Error: ${task.error}</small>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    getStatusColor(status) {
        const colors = {
            'pending': 'secondary',
            'running': 'info',
            'completed': 'success',
            'failed': 'danger',
            'cancelled': 'warning'
        };
        return colors[status] || 'secondary';
    }
    
    getStatusIcon(status) {
        const icons = {
            'pending': 'fas fa-clock',
            'running': 'fas fa-spinner fa-spin',
            'completed': 'fas fa-check',
            'failed': 'fas fa-times',
            'cancelled': 'fas fa-ban'
        };
        return icons[status] || 'fas fa-question';
    }
    
    async viewResults(taskId) {
        try {
            const response = await fetch(`/api/task_results/${taskId}`);
            const results = await response.json();
            
            if (response.ok) {
                this.displayResults(results);
                this.scrollToResults();
            } else {
                this.showAlert('danger', results.error || 'Failed to load results');
            }
            
        } catch (error) {
            console.error('Error loading results:', error);
            this.showAlert('danger', 'Failed to load results');
        }
    }
    
    displayResults(results) {
        const container = document.getElementById('results-container');
        
        if (!results || !results.results) {
            container.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                    <p>No results available</p>
                </div>
            `;
            return;
        }
        
        const summary = results.summary || {};
        const modules = results.results || {};
        
        let html = `
            <div class="results-summary fade-in">
                <h4><i class="fas fa-chart-bar me-2"></i>Results Summary</h4>
                <div class="row mt-3">
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3>${summary.total_findings || 0}</h3>
                            <small>Total Findings</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3>${summary.modules_successful || 0}</h3>
                            <small>Successful Modules</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3>${results.duration_seconds ? Math.round(results.duration_seconds) : 0}s</h3>
                            <small>Duration</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3>${results.target || 'N/A'}</h3>
                            <small>Target</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Display module results
        Object.entries(modules).forEach(([moduleName, moduleData]) => {
            html += this.createModuleResults(moduleName, moduleData);
        });
        
        container.innerHTML = html;
    }
    
    createModuleResults(moduleName, moduleData) {
        if (moduleData.error) {
            return `
                <div class="results-module fade-in">
                    <div class="results-module-header bg-danger text-white">
                        <h5 class="mb-0">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            ${moduleName.toUpperCase()} - Error
                        </h5>
                    </div>
                    <div class="results-module-content">
                        <div class="alert alert-danger">
                            ${moduleData.error}
                        </div>
                    </div>
                </div>
            `;
        }
        
        let content = '';
        
        switch (moduleName) {
            case 'dns':
                content = this.createDNSResults(moduleData);
                break;
            case 'dorks':
                content = this.createDorksResults(moduleData);
                break;
            case 'breach':
                content = this.createBreachResults(moduleData);
                break;
            case 'social':
                content = this.createSocialResults(moduleData);
                break;
            case 'emails':
                content = this.createEmailResults(moduleData);
                break;
            default:
                content = this.createGenericResults(moduleData);
        }
        
        return `
            <div class="results-module fade-in">
                <div class="results-module-header" data-bs-toggle="collapse" 
                     data-bs-target="#module-${moduleName}">
                    <h5 class="mb-0">
                        <i class="fas fa-chevron-right me-2"></i>
                        ${this.getModuleIcon(moduleName)} ${moduleName.toUpperCase()}
                        <span class="float-end">
                            <small class="text-muted">Click to expand</small>
                        </span>
                    </h5>
                </div>
                <div class="collapse show" id="module-${moduleName}">
                    <div class="results-module-content">
                        ${content}
                    </div>
                </div>
            </div>
        `;
    }
    
    getModuleIcon(moduleName) {
        const icons = {
            'dns': '<i class="fas fa-server text-primary me-2"></i>',
            'dorks': '<i class="fab fa-google text-danger me-2"></i>',
            'breach': '<i class="fas fa-exclamation-triangle text-warning me-2"></i>',
            'social': '<i class="fas fa-users text-info me-2"></i>',
            'emails': '<i class="fas fa-envelope text-success me-2"></i>'
        };
        return icons[moduleName] || '<i class="fas fa-puzzle-piece me-2"></i>';
    }
    
    createDNSResults(data) {
        let html = '';
        
        // DNS Records
        const records = data.records || {};
        if (Object.keys(records).length > 0) {
            html += '<h6>DNS Records:</h6>';
            Object.entries(records).forEach(([recordType, values]) => {
                html += `
                    <div class="mb-2">
                        <strong class="dns-record-type">${recordType}:</strong>
                        ${values.map(value => `<div class="dns-record">${value}</div>`).join('')}
                    </div>
                `;
            });
        }
        
        // Subdomains
        const subdomains = data.subdomains || [];
        if (subdomains.length > 0) {
            html += `<h6 class="mt-3">Subdomains (${subdomains.length}):</h6>`;
            subdomains.slice(0, 10).forEach(subdomain => {
                html += `
                    <div class="result-item">
                        <strong>${subdomain.full_domain}</strong>
                        ${subdomain.records ? `<br><small class="text-muted">${JSON.stringify(subdomain.records)}</small>` : ''}
                    </div>
                `;
            });
            if (subdomains.length > 10) {
                html += `<small class="text-muted">... and ${subdomains.length - 10} more</small>`;
            }
        }
        
        return html || '<p class="text-muted">No DNS data found</p>';
    }
    
    createDorksResults(data) {
        const results = data.results || [];
        const totalResults = data.total_results || 0;
        
        if (totalResults === 0) {
            return '<p class="text-muted">No Google dork results found</p>';
        }
        
        let html = `<h6>Search Results (${totalResults}):</h6>`;
        
        results.slice(0, 10).forEach((result, index) => {
            html += `
                <div class="result-item">
                    <div class="d-flex justify-content-between">
                        <h6 class="mb-1">${index + 1}. ${result.title || 'Untitled'}</h6>
                        <small class="text-muted">Score: ${(result.relevance_score || 0).toFixed(1)}</small>
                    </div>
                    <a href="${result.url}" target="_blank" class="text-decoration-none">
                        ${result.url}
                    </a>
                    <br>
                    <small class="text-primary">${result.domain}</small>
                    ${result.snippet ? `<p class="mt-2 mb-0 text-muted small">${result.snippet}</p>` : ''}
                </div>
            `;
        });
        
        if (results.length > 10) {
            html += `<small class="text-muted">... and ${results.length - 10} more results</small>`;
        }
        
        return html;
    }
    
    createBreachResults(data) {
        let html = '';
        
        if (data.total_breaches !== undefined) {
            // Single email check results
            html += `
                <div class="row mb-3">
                    <div class="col-md-6">
                        <div class="text-center">
                            <h4 class="text-warning">${data.total_breaches}</h4>
                            <small>Breaches Found</small>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="text-center">
                            <h4 class="text-info">${data.paste_count || 0}</h4>
                            <small>Paste Appearances</small>
                        </div>
                    </div>
                </div>
            `;
            
            const breaches = data.breaches || [];
            if (breaches.length > 0) {
                html += '<h6>Breach Details:</h6>';
                breaches.forEach(breach => {
                    html += `
                        <div class="breach-item ${breach.IsVerified === false ? 'breach-critical' : ''}">
                            <h6>${breach.Name}</h6>
                            <p><strong>Date:</strong> ${breach.BreachDate}</p>
                            <p><strong>Accounts Affected:</strong> ${(breach.PwnCount || 0).toLocaleString()}</p>
                            ${breach.Description ? `<p class="small">${breach.Description}</p>` : ''}
                        </div>
                    `;
                });
            }
        } else if (data.breaches_found !== undefined) {
            // Multiple email check results
            const breachesFound = data.breaches_found || [];
            html += `<h6>Compromised Accounts (${breachesFound.length}):</h6>`;
            
            breachesFound.forEach(breachData => {
                html += `
                    <div class="breach-item">
                        <strong>${breachData.email}</strong>
                        <span class="badge bg-warning ms-2">${breachData.total_breaches} breaches</span>
                    </div>
                `;
            });
        }
        
        return html || '<p class="text-muted">No breach data found</p>';
    }
    
    createSocialResults(data) {
        const profiles = data.profiles_found || [];
        const mentions = data.mentions_found || [];
        
        let html = '';
        
        if (profiles.length > 0) {
            html += `<h6>Profiles Found (${profiles.length}):</h6>`;
            profiles.forEach(profile => {
                const platformClass = `platform-${profile.platform}`;
                html += `
                    <div class="social-profile">
                        <div class="social-platform-icon ${platformClass}">
                            <i class="fab fa-${profile.platform}"></i>
                        </div>
                        <div class="flex-grow-1">
                            <h6 class="mb-1">${profile.username}</h6>
                            <small class="text-muted">${profile.platform}</small>
                            ${profile.url ? `<br><a href="${profile.url}" target="_blank" class="small">${profile.url}</a>` : ''}
                        </div>
                        <div class="text-end">
                            <small class="text-muted">Score: ${(profile.relevance_score || 0).toFixed(1)}</small>
                        </div>
                    </div>
                `;
            });
        }
        
        if (mentions.length > 0) {
            html += `<h6 class="mt-3">Mentions (${mentions.length}):</h6>`;
            mentions.slice(0, 5).forEach(mention => {
                html += `
                    <div class="result-item">
                        <strong>${mention.platform}</strong>
                        ${mention.url ? `<br><a href="${mention.url}" target="_blank" class="small">${mention.url}</a>` : ''}
                        <br><small class="text-muted">Relevance: ${(mention.relevance_score || 0).toFixed(1)}</small>
                    </div>
                `;
            });
        }
        
        return html || '<p class="text-muted">No social media data found</p>';
    }
    
    createEmailResults(data) {
        const emailsFound = data.emails_found || [];
        const commonPatterns = data.common_patterns || [];
        
        let html = '';
        
        if (emailsFound.length > 0) {
            html += `<h6>Email Addresses (${emailsFound.length}):</h6>`;
            emailsFound.forEach(email => {
                html += `<div class="result-item"><code>${email}</code></div>`;
            });
        }
        
        if (commonPatterns.length > 0) {
            html += `<h6 class="mt-3">Common Patterns:</h6>`;
            commonPatterns.forEach(pattern => {
                html += `<div class="result-item"><code>${pattern}</code></div>`;
            });
        }
        
        return html || '<p class="text-muted">No email data found</p>';
    }
    
    createGenericResults(data) {
        return `<pre class="small">${JSON.stringify(data, null, 2)}</pre>`;
    }
    
    async downloadResults(taskId, format) {
        try {
            const response = await fetch(`/api/download_results/${taskId}/${format}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `osint_results_${taskId.substring(0, 8)}.${format}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showAlert('success', `Results downloaded as ${format.toUpperCase()}`);
            } else {
                const error = await response.json();
                this.showAlert('danger', error.error || 'Download failed');
            }
            
        } catch (error) {
            console.error('Download error:', error);
            this.showAlert('danger', 'Download failed');
        }
    }
    
    async cancelTask(taskId) {
        if (!confirm('Are you sure you want to cancel this task?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/cancel_task/${taskId}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showAlert('info', 'Task cancelled successfully');
                this.loadActiveTasks();
            } else {
                this.showAlert('danger', result.error || 'Failed to cancel task');
            }
            
        } catch (error) {
            console.error('Cancel task error:', error);
            this.showAlert('danger', 'Failed to cancel task');
        }
    }
    
    setLoadingState(loading) {
        const form = document.getElementById('reconnaissance-form');
        const inputs = form.querySelectorAll('input, button, select');
        
        inputs.forEach(input => {
            input.disabled = loading;
        });
        
        if (loading) {
            form.classList.add('loading');
        } else {
            form.classList.remove('loading');
        }
    }
    
    showAlert(type, message) {
        const container = document.getElementById('alert-container');
        const alertId = 'alert-' + Date.now();
        
        const alert = document.createElement('div');
        alert.id = alertId;
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            <i class="fas fa-${this.getAlertIcon(type)} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        container.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement) {
                alertElement.remove();
            }
        }, 5000);
    }
    
    getAlertIcon(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-circle',
            'warning': 'exclamation-triangle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
    
    scrollToTasks() {
        document.getElementById('active-tasks').scrollIntoView({ 
            behavior: 'smooth' 
        });
    }
    
    scrollToResults() {
        document.getElementById('results').scrollIntoView({ 
            behavior: 'smooth' 
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OSINTApp();
});

// Utility functions for external access
window.osintUtils = {
    formatDate: (dateString) => {
        return new Date(dateString).toLocaleString();
    },
    
    copyToClipboard: (text) => {
        navigator.clipboard.writeText(text).then(() => {
            window.app.showAlert('success', 'Copied to clipboard');
        }).catch(() => {
            window.app.showAlert('warning', 'Failed to copy to clipboard');
        });
    },
    
    exportResults: (data, filename) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
};
