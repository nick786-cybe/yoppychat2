// =================================================================
// 1. GLOBALLY ACCESSIBLE FUNCTIONS
// =================================================================

/**
 * Displays a custom confirmation dialog.
 * @param {string} title - The dialog title.
 * @param {string} message - The confirmation message.
 * @param {string} confirmText - Text for the confirmation button.
 * @param {function} onConfirm - Callback function executed on confirmation.
 */
window.showCustomConfirm = function(title, message, confirmText, onConfirm) {
    const dialog = document.createElement('div');
    dialog.className = 'confirm-dialog';
    // This line is the fix. It correctly hides the dialog initially.
    dialog.style.display = 'none';
    dialog.innerHTML = `
        <div class="confirm-content">
            <h3>${title}</h3>
            <p>${message}</p>
            <div class="confirm-actions">
                <button class="cancel-btn">Cancel</button>
                <button class="confirm-btn">${confirmText}</button>
            </div>
        </div>
    `;

    document.body.appendChild(dialog);

    // We now show the dialog by changing its display property.
    dialog.style.display = 'flex';

    const confirmBtn = dialog.querySelector('.confirm-btn');
    const cancelBtn = dialog.querySelector('.cancel-btn');
    const closeDialog = () => document.body.removeChild(dialog);

    cancelBtn.onclick = closeDialog;
    confirmBtn.onclick = () => {
        onConfirm();
        closeDialog();
    };
}

/**
 * Initiates clearing the chat history for a specific channel.
 * @param {string} channel - The name of the channel ('general' or a specific name).
 */
window.clearChat = function(channel) {
    window.showCustomConfirm(
        'Clear Chat History',
        'Are you sure you want to clear this chat? This action cannot be undone.',
        'Clear',
        () => {
            const formData = new FormData();
            formData.append('channel_name', channel);
            fetch('/clear_chat', { method: 'POST', body: formData })
                .then(res => {
                    if (res.ok) {
                        window.location.reload();
                    } else {
                        return Promise.reject('Failed to clear chat.');
                    }
                })
                .catch(err => {
                    if (window.showNotification) {
                        window.showNotification(err.toString(), 'error');
                    }
                });
        }
    );
}

/**
 * Sets a channel as the default for the community.
 * @param {string} channelId - The ID of the channel to set as default.
 * @param {HTMLElement} buttonElement - The button that was clicked.
 */
window.setDefaultChannel = function(channelId, buttonElement) {
    if (!buttonElement || buttonElement.disabled) return;

    const originalContent = buttonElement.innerHTML;
    buttonElement.disabled = true;
    buttonElement.innerHTML = `<div class="button-spinner" style="width:16px;height:16px;border-width:2px;"></div><span>Setting...</span>`;

    fetch(`/set-default-channel/${channelId}`, { method: 'POST' })
        .then(res => {
            if (!res.ok) return res.json().then(err => Promise.reject(err));
            return res.json();
        })
        .then(data => {
            if (window.showNotification) {
                window.showNotification(data.message, 'success');
            }
            setTimeout(() => window.location.reload(), 1500);
        })
        .catch(err => {
            if (window.showNotification) {
                window.showNotification(err.message || 'An error occurred.', 'error');
            }
            buttonElement.disabled = false;
            buttonElement.innerHTML = originalContent;
        });
}


/**
 * Toggles a channel's privacy between personal and shared for community admins.
 * @param {string} channelId - The ID of the channel to toggle.
 * @param {boolean} isShared - The new desired state (true for shared, false for personal).
 */
window.toggleChannelPrivacy = function(channelId, isShared) {
    const toggle = document.getElementById('shareToggle');
    if(toggle) toggle.disabled = true;

    fetch(`/api/toggle_channel_privacy/${channelId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    })
    .then(res => {
        if (!res.ok) {
            return res.json().then(err => Promise.reject(err));
        }
        return res.json();
    })
    .then(data => {
        if (window.showNotification) {
            window.showNotification(data.message, 'success');
        }
        // A page reload is the simplest way to reflect the change everywhere (sidebar, etc.)
        setTimeout(() => window.location.reload(), 1500);
    })
    .catch(err => {
        if (window.showNotification) {
            window.showNotification(err.message || 'An error occurred.', 'error');
        }
        // Revert the toggle state on failure
        if(toggle) toggle.checked = !isShared;
    })
    .finally(() => {
        // Keep it disabled on success because the page will reload
        if(toggle && !document.querySelector('.notification.success')) {
            toggle.disabled = false;
        }
    });
}

/**
 * Toggles the visibility of the sources list associated with a button.
 * @param {HTMLElement} btn - The button element that was clicked.
 */
window.toggleSources = function(btn) {
    const sourcesSection = btn.closest('.sources-section');
    if (!sourcesSection) return;

    const list = sourcesSection.querySelector('.sources-list');
    if (!list) return;

    const indicator = btn.querySelector('.toggle-indicator');
    const isExpanded = list.style.display === 'block';

    list.style.display = isExpanded ? 'none' : 'block';
    if (indicator) {
        indicator.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(180deg)';
    }
}

/**
 * Initiates a background task to refresh a channel's content.
 * @param {string} channelId - The ID of the channel to refresh.
 * @param {HTMLElement} buttonElement - The button that triggered the refresh.
 */
window.refreshChannel = function(channelId, buttonElement) {
    if (!buttonElement || buttonElement.disabled) return;

    const originalContent = buttonElement.innerHTML;
    buttonElement.disabled = true;
    const spinnerSize = buttonElement.classList.contains('action-btn') ? '20px' : '16px';
    buttonElement.innerHTML = `<div class="button-spinner" style="width:${spinnerSize};height:${spinnerSize};border-width:2px;"></div><span>Refreshing...</span>`;

    const dropdownMenuMobile = document.getElementById('dropdown-menu-mobile');
    if (dropdownMenuMobile?.classList.contains('show')) {
        dropdownMenuMobile.classList.remove('show');
    }

    fetch(`/refresh_channel/${channelId}`, { method: 'POST' })
        .then(response => response.ok ? response.json() : response.json().then(err => Promise.reject(err)))
        .then(data => {
            if (data.status === 'success' && data.task_id) {
                checkTaskStatus(data.task_id, buttonElement, originalContent);
            } else {
                throw new Error(data.message || 'Failed to start refresh task.');
            }
        })
        .catch(error => {
            if (window.showNotification) {
                window.showNotification(error.message || 'Refresh failed to start.', 'error');
            }
            restoreButton(buttonElement, originalContent, 'state-failure');
        });
}

/**
 * Polls the backend to check the status of a background task.
 * @param {string} taskId - The ID of the task to check.
 * @param {HTMLElement} buttonElement - The button associated with the task.
 * @param {string} originalContent - The original HTML content of the button.
 */
function checkTaskStatus(taskId, buttonElement, originalContent) {
    const interval = setInterval(() => {
        fetch(`/task_result/${taskId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'complete' || data.status === 'failed') {
                    clearInterval(interval);
                    if (window.showNotification && data.message) {
                        const messageType = data.status === 'complete' ? 'success' : 'error';
                        window.showNotification(data.message, messageType);
                    }
                    const finalStateClass = data.status === 'complete' ? 'state-success' : 'state-failure';
                    restoreButton(buttonElement, originalContent, finalStateClass);
                    if (data.status === 'complete' && data.message?.toLowerCase().includes('success')) {
                        setTimeout(() => window.location.reload(), 1500);
                    }
                }
            })
            .catch(error => {
                clearInterval(interval);
                if (window.showNotification) window.showNotification('Error checking task status.', 'error');
                restoreButton(buttonElement, originalContent, 'state-failure');
            });
    }, 3000);
}

/**
 * Restores a button to its original state after an async operation.
 * @param {HTMLElement} button - The button to restore.
 * @param {string} originalHTML - The original inner HTML of the button.
 * @param {string} finalStateClass - A class to add temporarily (e.g., 'state-success').
 */
function restoreButton(button, originalHTML, finalStateClass) {
    if (!button) return;
    button.innerHTML = originalHTML;
    button.classList.remove('state-loading');
    if (finalStateClass) button.classList.add(finalStateClass);

    setTimeout(() => {
        button.disabled = false;
        if (finalStateClass) button.classList.remove(finalStateClass);
    }, 3000);
}


// =================================================================
// 2. SCRIPT INITIALIZATION
// =================================================================
document.addEventListener('DOMContentLoaded', function() {

    // --- A. Element Selectors & Page Data ---
    const toggleRightSidebarBtn = document.getElementById('toggle-right-sidebar-btn');
    const desktopChatLayout = document.querySelector('.desktop-chat-layout');
    const questionForm = document.getElementById('questionForm');
    const questionText = document.getElementById('questionText');
    const submitBtn = document.getElementById('submitBtn');
    const conversationHistory = document.getElementById('conversation-history');
    const charCount = document.querySelector('.char-count');

    // --- B. Helper Functions ---
    const adjustTextareaHeight = () => {
        if (!questionText) return;
        questionText.style.height = 'auto';
        const newHeight = Math.min(questionText.scrollHeight, 120);
        questionText.style.height = `${newHeight}px`;
    };

    const updateCharCount = () => {
        if (!questionText || !charCount) return;
        const count = questionText.value.length;
        charCount.textContent = `${count}/1000`;
        charCount.classList.toggle('near-limit', count > 800);
    };

    const escapeHtml = (text) => {
        if (typeof text !== 'string') return '';
        return text.replace(/[&<>'"]/g, tag => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
        }[tag]));
    };

    const addMessageToChat = (role, text) => {
        const pageData = document.getElementById('chat-page-data').dataset;
        const channelName = pageData.channelName;
        const channelThumbnail = pageData.channelThumbnail;
        const qnaPair = document.createElement('div');
        qnaPair.className = 'qna-pair';
        let html = '';

        if (role === 'user') {
            html = `
                <div class="question-box">
                    <div class="question-content">${escapeHtml(text)}</div>
                </div>
            `;
        } else {
            const avatarHtml = channelThumbnail
                ? `<div class="answer-avatar-container avatar-container">
                        <img src="${channelThumbnail}" alt="Avatar" class="answer-avatar">
                        <span class="ai-badge">AI</span>
                    </div>`
                : `<div class="answer-avatar-container avatar-container">
                        <div class="answer-avatar-placeholder">🤖</div>
                        <span class="ai-badge">AI</span>
                    </div>`;
            const label = channelName ? `${escapeHtml(channelName)}` : 'Answer';
            html = `
                <div class="answer-box">
                    <div class="answer-header">${avatarHtml}<span class="answer-label">${label}</span></div>
                    <div class="answer-content"></div>
                    <div class="typing-container">
                        <div class="typing-indicator">
                            <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
                        </div>
                    </div>
                    <div class="sources-section"></div>
                </div>
            `;
        }

        qnaPair.innerHTML = html;
        conversationHistory.appendChild(qnaPair);
        conversationHistory.scrollTop = conversationHistory.scrollHeight;

        return role === 'ai' ? qnaPair.querySelector('.answer-box') : null;
    };

    const renderSources = (sources, answerBoxElement) => {
        if (!answerBoxElement) return;

        let sourcesSection = answerBoxElement.querySelector('.sources-section');
        sourcesSection.innerHTML = '';

        let buttonsHTML = '';
        let listHTML = '';

        if (sources && sources.length > 0) {
            buttonsHTML += `
                <button class="toggle-sources-btn" onclick="toggleSources(this)">
                    <svg class="sources-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><path fill="currentColor" d="M4,6H2V20a2,2 0 0,0 2,2H18V18H4V6M20,2H8A2,2 0 0,0 6,4V16a2,2 0 0,0 2,2H20a2,2 0 0,0 2-2V4a2,2 0 0,0-2-2Z"></path></svg>
                    Sources (${sources.length})
                    <span class="toggle-indicator">▼</span>
                </button>
            `;
            const sourceLinks = sources.map(s => `<div class="source-item"><a href="${escapeHtml(s.url)}" target="_blank" class="source-link">${escapeHtml(s.title)}</a></div>`).join('');
            listHTML = `<div class="sources-list" style="display: none;">${sourceLinks}</div>`;
        }

        buttonsHTML += `
            <button class="toggle-sources-btn regenerate-btn-js" onclick="regenerateAnswer(this)">
                <svg class="sources-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
                Regenerate
            </button>`;

        sourcesSection.innerHTML = `<div class="source-buttons">${buttonsHTML}</div>${listHTML}`;
    };

    // --- C. Core Logic: Form Submission & Streaming ---
    const handleFormSubmit = (event) => {
        const pageData = document.getElementById('chat-page-data').dataset;
        const channelName = pageData.channelName;
        event.preventDefault();

        document.querySelectorAll('.regenerate-btn-js').forEach(btn => btn.remove());

        // --- Get the selected tone ---
        const selectedToneButton = document.querySelector('#tone-selector .tone-option.active');
        const selectedTone = selectedToneButton ? selectedToneButton.dataset.tone : 'Casual';

        const question = questionText.value.trim();
        if (!question) return;

        submitBtn.disabled = true;
        submitBtn.classList.remove('active');
        questionText.value = '';
        adjustTextareaHeight();
        updateCharCount();

        addMessageToChat('user', question);
        const aiAnswerBox = addMessageToChat('ai', '');
        const answerContent = aiAnswerBox.querySelector('.answer-content');
        const typingContainer = aiAnswerBox.querySelector('.typing-container');
        typingContainer.classList.add('active');

        const formData = new FormData();
        formData.append('question', question);
        formData.append('channel_name', channelName);
        formData.append('tone', selectedTone); // Send the tone to the backend

        fetch('/stream_answer', { method: 'POST', body: formData })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => Promise.reject(err));
                }
                return response.body.getReader();
            })
            .then(reader => {
                if (reader) processStream(reader, answerContent, typingContainer, aiAnswerBox);
            })
            .catch(err => {
                console.error('Fetch error:', err);
                if (err && err.status === 'limit_reached') {
                    if (window.showUpgradePopup) {
                        window.showUpgradePopup(err.message);
                    } else {
                        answerContent.innerHTML = `<p class="error-message">${escapeHtml(err.message)}</p>`;
                    }
                } else {
                    const message = err.message || 'Failed to send question. Please try again.';
                    answerContent.innerHTML = `<p class="error-message">${escapeHtml(message)}</p>`;
                }
                typingContainer.classList.remove('active');
            });
    };

    const processStream = (reader, answerContent, typingContainer, aiAnswerBox) => {
        const decoder = new TextDecoder();
        let fullAnswerText = '';
        let foundSources = [];

        const push = () => {
            reader.read().then(({ done, value }) => {
                if (done) { return; }
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n\n');
                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        if (data.trim() === '[DONE]') return;
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.error) {
                                throw new Error(parsed.message || 'An unknown error occurred.');
                            }
                            if (parsed.answer) {
                                fullAnswerText += parsed.answer;
                                answerContent.innerHTML = marked.parse(fullAnswerText);
                            }
                            if (parsed.sources) {
                                foundSources = parsed.sources;
                            }
                            if (parsed.updated_query_string) {
                                const planQueriesElement = document.querySelector('.plan-queries');
                                if (planQueriesElement) {
                                    planQueriesElement.innerHTML = parsed.updated_query_string;
                                }
                            }
                        } catch (e) {
                            console.error('Error parsing stream data:', e);
                            throw e;
                        }
                    }
                });
                conversationHistory.scrollTop = conversationHistory.scrollHeight;
                push();
            }).catch(error => {
                console.error("Stream reading error:", error);
                answerContent.innerHTML = `<p class="error-message">An error occurred during generation.</p>`;
            }).finally(() => {
                if (typingContainer) typingContainer.classList.remove('active');
                renderSources(foundSources, aiAnswerBox);
            });
        };
        push();
    };

    window.regenerateAnswer = function(btn) {
        if (btn.disabled) return;
        btn.disabled = true;
        btn.innerHTML = 'Regenerating...';

        document.querySelectorAll('.regenerate-btn-js').forEach(b => {
            if (b !== btn) b.remove();
        });

        const lastQnaPair = btn.closest('.qna-pair');
        if (!lastQnaPair) {
            btn.innerHTML = 'Regenerate';
            btn.disabled = false;
            return;
        }

        const questionContent = lastQnaPair.previousElementSibling?.querySelector('.question-content');
        const answerBox = btn.closest('.answer-box');

        if (!questionContent || !answerBox) return;

        const lastQuestion = questionContent.textContent;
        const answerContent = answerBox.querySelector('.answer-content');
        const sourcesSection = answerBox.querySelector('.sources-section');
        const typingContainer = answerBox.querySelector('.typing-container');

        answerContent.innerHTML = '';
        if (sourcesSection) sourcesSection.innerHTML = '';
        if (typingContainer) typingContainer.classList.add('active');

        const selectedToneButton = document.querySelector('#tone-selector .tone-option.active');
        const selectedTone = selectedToneButton ? selectedToneButton.dataset.tone : 'Casual';

        const formData = new FormData();
        formData.append('question', lastQuestion);
        formData.append('channel_name', document.getElementById('chat-page-data').dataset.channelName);
        formData.append('tone', selectedTone); // Also send tone on regeneration

        fetch('/stream_answer', { method: 'POST', body: formData })
            .then(response => {
                if (!response.ok) return response.json().then(err => Promise.reject(err));
                return response.body.getReader();
            })
            .then(reader => {
                if (reader) processStream(reader, answerContent, typingContainer, answerBox);
            })
            .catch(err => {
                console.error('Regeneration fetch error:', err);
                answerContent.innerHTML = `<p class="error-message">Failed to regenerate answer. Please try again.</p>`;
                if (typingContainer) typingContainer.classList.remove('active');
                renderSources([], answerBox);
            });
    }

    // --- D. Event Listeners ---
    if (questionForm) {
        questionForm.addEventListener('submit', handleFormSubmit);
    }

    if (questionText) {
        questionText.addEventListener('input', () => {
            const hasText = questionText.value.trim().length > 0;
            submitBtn.disabled = !hasText;
            submitBtn.classList.toggle('active', hasText);
            adjustTextareaHeight();
            updateCharCount();
        });
        questionText.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                if (!submitBtn.disabled) {
                    questionForm.requestSubmit();
                }
            }
        });
    }

    // Tone selector click handler
    const toneSelector = document.getElementById('tone-selector');
    if (toneSelector) {
        toneSelector.addEventListener('click', (event) => {
            if (event.target.classList.contains('tone-option')) {
                toneSelector.querySelectorAll('.tone-option').forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');
            }
        });
    }

    const setupKebabMenu = (btn, menu) => {
        if (btn && menu) {
            btn.addEventListener('click', (event) => {
                event.stopPropagation();
                menu.classList.toggle('show');
            });
        }
    };
    setupKebabMenu(document.getElementById('kebab-menu-btn-mobile'), document.getElementById('dropdown-menu-mobile'));

    window.addEventListener('click', () => {
        document.getElementById('dropdown-menu-mobile')?.classList.remove('show');
    });

    const hamburgerBtnMobile = document.getElementById('hamburgerBtnMobile');
    const mainHamburger = document.getElementById('hamburgerBtn');
    if (hamburgerBtnMobile && mainHamburger) {
        hamburgerBtnMobile.addEventListener('click', () => mainHamburger.click());
    }

    // --- E. Sidebar Toggle Logic ---
    if (toggleRightSidebarBtn && desktopChatLayout) {
        const sidebarStateKey = 'rightSidebarCollapsed';

        const applySidebarState = () => {
            if (localStorage.getItem(sidebarStateKey) === 'true') {
                desktopChatLayout.classList.add('right-sidebar-collapsed');
            } else {
                desktopChatLayout.classList.remove('right-sidebar-collapsed');
            }
        };

        applySidebarState(); // Apply state on initial load

        toggleRightSidebarBtn.addEventListener('click', () => {
            const isCollapsed = desktopChatLayout.classList.toggle('right-sidebar-collapsed');
            localStorage.setItem(sidebarStateKey, isCollapsed);
        });
    }

    // --- F. Initial Page Setup ---
    const init = () => {
        if (questionText) {
            questionText.focus();
            adjustTextareaHeight();
            updateCharCount();
        }
        if (conversationHistory) {
            conversationHistory.scrollTop = conversationHistory.scrollHeight;
        }
    };
    init();
});

function askExample(element) {
    const questionText = document.getElementById('questionText');
    const questionForm = document.getElementById('questionForm');
    if (questionText && questionForm) {
        questionText.value = element.textContent;
        // Correctly dispatch an input event first to enable the button
        questionText.dispatchEvent(new Event('input', { bubbles: true }));
        // Then dispatch the submit event
        questionForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    }
}
