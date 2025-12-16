/**
 * Chat Application JavaScript
 * Handles real-time messaging, chart display, and UI interactions
 */

class ChatApp {
    constructor() {
        // Get template data
        const templateData = JSON.parse(document.getElementById('template-data').textContent);
        templateData.messages = JSON.parse(document.getElementById('messages-data').textContent);
        this.chatId = templateData.chatId;
        this.isProcessing = templateData.isProcessing;

        // DOM elements
        this.messagesContainer = document.getElementById('messages-container');
        this.messageForm = document.getElementById('message-form');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.statusIndicator = document.getElementById('status-indicator');
        this.chartSection = document.getElementById('chart-section');
        this.chartContainer = document.getElementById('chart-container');

        // State
        this.pendingMessages = new Set();
        this.chartInstances = {};


        this.pollInProgress = false;

        // Initialize
        this.init();
    }

    init() {
        // Initialize existing charts
        const initialMessages = JSON.parse(document.getElementById('messages-data').textContent);
        this.initializeCharts(initialMessages);

        // Setup event listeners
        this.setupEventListeners();

        // Auto-focus and scroll
        this.messageInput.focus();
        this.scrollToBottom();
    }

    setupEventListeners() {
        // Form submission
        this.messageForm.addEventListener('submit', (e) => this.handleSubmit(e));

        // Enter key handling
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.messageForm.requestSubmit();
            }
        });
    }

    initializeCharts(messages) {
        messages.forEach(message => {
            if (message.charts && message.charts.length > 0) {
                message.charts.forEach(chart => {
                    this.addChartToGallery(chart);
                });
                this.showChartSection();
            }
        });
    }

    showChartSection() {
        this.chartSection.classList.add('active');
    }

    hideChartSection() {
        if (Object.keys(this.chartInstances).length === 0) {
            this.chartSection.classList.remove('active');
        }
    }

    addChartToGallery(chartData) {
        // Create chart container
        const chartItem = document.createElement('div');
        chartItem.className = 'chart-item';
        chartItem.innerHTML = `
            <div class="chart-title">${this.escapeHtml(chartData.title)}</div>
            <div class="chart-wrapper">
                <canvas class="chart-canvas" id="chart-${chartData.id}"></canvas>
            </div>
        `;

        this.chartContainer.appendChild(chartItem);

        // Get canvas context
        const ctx = document.getElementById(`chart-${chartData.id}`).getContext('2d');

        // Destroy existing chart instance with same ID (avoid duplicates)
        if (this.chartInstances[chartData.id]) {
            this.chartInstances[chartData.id].destroy();
        }

        console.log(chartData)

        // Create Chart.js chart
        this.chartInstances[chartData.id] = new Chart(ctx, chartData);
    }

    addMessage(type, content, messageId = null, status = 'completed') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        if (messageId) messageDiv.setAttribute('data-message-id', messageId);
        if (status) messageDiv.setAttribute('data-status', status);

        let messageContent;
        if (status === 'pending') {
            messageContent = `
                <div class="typing-indicator">
                    Thinking
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            `;
            messageDiv.classList.add('pending');
        } else {
            messageContent = this.formatMessageContent(content);
        }

        messageDiv.innerHTML = `
            <div class="message-content">${messageContent}</div>
            <div class="message-meta">${this.formatTimestamp(new Date())}</div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        return messageDiv;
    }

    updateMessage(messageId, content, charts = [], status) {
        const messageDiv = document.querySelector(`[data-message-id="${messageId}"]`);
        if (messageDiv && (status !== 'pending')) {
            messageDiv.classList.remove('pending');
            messageDiv.setAttribute('data-status', 'completed');

            const contentDiv = messageDiv.querySelector('.message-content');
            contentDiv.innerHTML = this.formatMessageContent(content);

            // Add charts if any
            if (charts && charts.length > 0) {
                charts.forEach(chart => {
                    this.addChartToGallery(chart);
                });
                this.showChartSection();
            }
        }
        if (messageDiv && (status === 'pending')) {
            let messageContent = `
                <div class="typing-indicator">
                    ${content}
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            `;
            messageDiv.classList.add('pending');

            messageDiv.innerHTML = `
                <div class="message-content">${messageContent}</div>
                <div class="message-meta">${this.formatTimestamp(new Date())}</div>
            `;
        }
    }

    updateProcessingState(processing) {
        this.isProcessing = processing;
        this.messageInput.disabled = processing;
        this.sendButton.disabled = processing;

        this.statusIndicator.className = `status-indicator ${processing ? 'processing' : 'ready'}`;
        this.statusIndicator.textContent = processing ? 'Processing...' : 'Ready';
    }

    pollMessageStatus(messageId) {
        const pollInterval = setInterval(() => {
            if (this.pollInProgress) return;
            this.pollInProgress = true;

            const xhr = new XMLHttpRequest();
            xhr.open('GET', `/earthrise/api/chats/${this.chatId}/messages/${messageId}/status/`);
            xhr.onload = () => {
                this.pollInProgress = false;
                if (xhr.status >= 200 && xhr.status < 300) {
                    const data = JSON.parse(xhr.responseText);
                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(pollInterval);
                        this.pendingMessages.delete(messageId);
                        this.updateMessage(messageId, data.content || data.error, data.charts || [], 'completed');
                        this.updateProcessingState(false);
                    }
                    if (data.status === 'pending') {
                        const node = data.current_node || 'Waiting...';
                        this.updateMessage(messageId, `Processing node: ${node}`, [], 'pending');
                    }
                } else {
                    clearInterval(pollInterval);
                    this.pollInProgress = false;
                    this.pendingMessages.delete(messageId);
                    this.updateProcessingState(false);
                    this.showError('Connection error. Please refresh the page.');
                }
            };
            xhr.onerror = () => {
                this.pollInProgress = false;
                clearInterval(pollInterval);
                this.pendingMessages.delete(messageId);
                this.updateProcessingState(false);
                this.showError('Connection error. Please refresh the page.');
            };
            xhr.send();
        }, 1000);
    }

    handleSubmit(e) {
        e.preventDefault();
        e.stopPropagation();

        if (this.isProcessing || !this.messageInput.value.trim()) {
            return;
        }

        const content = this.messageInput.value.trim();
        this.messageInput.value = '';

        // Add user message
        this.addMessage('user', content);

        $.ajax({
            url: `/earthrise/api/chats/${this.chatId}/messages/`,
            method: 'POST',
            contentType: 'application/json',
            dataType: 'json',
            data: JSON.stringify({ content }),
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
            success: (data) => {
                if (data.success) {
                    this.addMessage('assistant', '', data.assistant_message_id, 'pending');
                    this.pendingMessages.add(data.assistant_message_id);
                    this.updateProcessingState(true);
                    this.pollMessageStatus(data.assistant_message_id);
                } else {
                    console.error('Failed to send message:', data.error);
                    this.showError('Failed to send message: ' + data.error);
                }
            },
            error: (xhr, status, error) => {
                console.error('Error sending message:', error);
                this.showError('Error sending message. Please try again.');
            }
        });
    }

    /* handleSubmit(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log("Sending test request...");
        const xhr = new XMLHttpRequest();
        xhr.open('GET', '/api/test/', true);
        xhr.onload = () => console.log("DONE", xhr.status);
        xhr.onerror = () => console.error("ERR");
        xhr.send();
    } */

    formatMessageContent(content) {
        return this.escapeHtml(content).replace(/\n/g, '<br>');
    }

    formatTimestamp(date) {
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    showError(message) {
        // Create a temporary error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #f8d7da;
            color: #721c24;
            padding: 12px 16px;
            border-radius: 4px;
            border: 1px solid #f5c6cb;
            z-index: 1000;
            max-width: 300px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        `;
        errorDiv.textContent = message;

        document.body.appendChild(errorDiv);

        // Remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
}

// Utility functions for Chart.js configuration
const ChartUtils = {
    getDefaultOptions: (chartType) => {
        const baseOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            }
        };

        switch (chartType) {
            case 'line':
                return {
                    ...baseOptions,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                };
            case 'bar':
                return {
                    ...baseOptions,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                };
            case 'pie':
            case 'doughnut':
                return {
                    ...baseOptions,
                    plugins: {
                        ...baseOptions.plugins,
                        legend: {
                            position: 'right',
                        }
                    }
                };
            default:
                return baseOptions;
        }
    },

    generateColors: (count) => {
        const colors = [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 205, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)',
            'rgba(199, 199, 199, 0.8)',
            'rgba(83, 102, 255, 0.8)'
        ];

        const result = [];
        for (let i = 0; i < count; i++) {
            result.push(colors[i % colors.length]);
        }
        return result;
    }
};

// Initialize the chat application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.chatApp) {
        // Refresh status when page becomes visible
        window.chatApp.messageInput.focus();
    }
});

// Handle beforeunload to warn about pending messages
window.addEventListener('beforeunload', (e) => {
    if (window.chatApp && window.chatApp.pendingMessages.size > 0) {
        e.preventDefault();
        e.returnValue = 'You have messages being processed. Are you sure you want to leave?';
        return e.returnValue;
    }
});