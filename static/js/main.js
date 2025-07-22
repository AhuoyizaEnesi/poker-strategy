const API_BASE = 'http://localhost:5000';

let currentSection = 'dashboard';
let sessionData = null;
let bankrollData = null;
let handAnalysisData = null;
let recentAnalysisHistory = [];

document.addEventListener('DOMContentLoaded', function() {
    console.log('PokerPro Analytics - Initializing...');
    
    setupNavigation();
    initializeDashboard();
    
    const sessionDateInput = document.getElementById('session-date');
    if (sessionDateInput) {
        sessionDateInput.value = new Date().toISOString().split('T')[0];
    }
    
    displayRecentAnalysis();
    setInterval(updateAnalysisTimestamps, 60000);
    
    const opponentsInput = document.getElementById('opponents');
    if (opponentsInput) {
        opponentsInput.setAttribute('max', '9');
        opponentsInput.addEventListener('input', function() {
            if (parseInt(this.value) > 9) {
                this.value = 9;
                showWarning('Maximum Opponents', 'Maximum 9 opponents allowed (10 total seats at poker table).', 'warning');
            }
            if (parseInt(this.value) < 1) {
                this.value = 1;
            }
        });
    }
    
    console.log('Dashboard initialized');
});

function showWarning(title, message, type = 'warning') {
    const warningOverlay = document.createElement('div');
    warningOverlay.className = 'warning-overlay';
    warningOverlay.innerHTML = `
        <div class="warning-content ${type}">
            <div class="warning-icon">
                <i class="fas ${type === 'error' ? 'fa-exclamation-triangle' : 'fa-exclamation-circle'}"></i>
            </div>
            <h2>${title}</h2>
            <p>${message}</p>
            <button class="warning-close-btn" onclick="closeWarning()">
                <i class="fas fa-times"></i> Close
            </button>
        </div>
    `;
    
    if (!document.querySelector('.warning-styles')) {
        const style = document.createElement('style');
        style.className = 'warning-styles';
        style.textContent = `
            .warning-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                animation: warningFadeIn 0.3s ease-out;
            }
            
            @keyframes warningFadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            .warning-content {
                background: white;
                border-radius: 15px;
                padding: 40px;
                max-width: 500px;
                text-align: center;
                animation: warningSlideIn 0.4s ease-out;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            }
            
            @keyframes warningSlideIn {
                from { transform: translateY(-50px) scale(0.9); opacity: 0; }
                to { transform: translateY(0) scale(1); opacity: 1; }
            }
            
            .warning-content.error {
                border-left: 5px solid #ef4444;
            }
            
            .warning-content.warning {
                border-left: 5px solid #f59e0b;
            }
            
            .warning-icon {
                font-size: 4rem;
                margin-bottom: 20px;
            }
            
            .warning-content.error .warning-icon {
                color: #ef4444;
            }
            
            .warning-content.warning .warning-icon {
                color: #f59e0b;
            }
            
            .warning-content h2 {
                color: #333;
                margin-bottom: 15px;
            }
            
            .warning-content p {
                color: #666;
                margin-bottom: 30px;
                line-height: 1.6;
            }
            
            .warning-close-btn {
                background: #4f46e5;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .warning-close-btn:hover {
                background: #4338ca;
                transform: translateY(-2px);
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(warningOverlay);
}

function closeWarning() {
    const overlay = document.querySelector('.warning-overlay');
    if (overlay) {
        overlay.remove();
    }
}

function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetSection = this.getAttribute('data-section');
            showSection(targetSection);
            
            navLinks.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

function showSection(sectionName) {
    const sections = document.querySelectorAll('.dashboard-section, .content-section');
    sections.forEach(section => {
        section.classList.remove('active');
        section.style.display = 'none';
    });
    
    const targetSection = document.getElementById(sectionName);
    if (targetSection) {
        targetSection.classList.add('active');
        targetSection.style.display = 'block';
    }
    
    currentSection = sectionName;
    
    if (sectionName === 'dashboard') {
        loadDashboardData();
    }
}

function initializeDashboard() {
    loadDashboardData();
    setupDashboardCharts();
}

async function loadDashboardData() {
    try {
        const response = await fetch(`${API_BASE}/api/session_data`);
        sessionData = await response.json();
        updateDashboardMetrics();
    } catch (error) {
        console.log('Using demo dashboard data');
        sessionData = generateDemoSessionData();
        updateDashboardMetrics();
    }
}

function generateDemoSessionData() {
    return {
        totalProfit: 2847.50,
        winRate: 68.4,
        hourlyRate: 23.85,
        totalHours: 127.3,
        sessionsPlayed: 23,
        avgSessionProfit: 123.80
    };
}

function updateDashboardMetrics() {
    if (!sessionData) return;
    
    document.getElementById('total-profit').textContent = `${sessionData.totalProfit}`;
    document.getElementById('win-rate').textContent = `${sessionData.winRate}%`;
    document.getElementById('hourly-rate').textContent = `${sessionData.hourlyRate}/h`;
    document.getElementById('sessions-played').textContent = sessionData.sessionsPlayed;
    
    const profitElement = document.getElementById('total-profit');
    if (sessionData.totalProfit >= 0) {
        profitElement.className = 'metric-value profit-positive';
    } else {
        profitElement.className = 'metric-value profit-negative';
    }
    
    const hourlyElement = document.getElementById('hourly-rate');
    if (sessionData.hourlyRate >= 0) {
        hourlyElement.className = 'metric-value profit-positive';
    } else {
        hourlyElement.className = 'metric-value profit-negative';
    }
}

function setupDashboardCharts() {
    setupEquityChart();
    setupProfitChart();
}

function setupEquityChart() {
    const ctx = document.getElementById('equity-chart');
    if (!ctx) return;
    
    window.equityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Win', 'Tie', 'Lose'],
            datasets: [{
                data: [65, 5, 30],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderWidth: 2,
                borderColor: '#1f2937'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function setupProfitChart() {
    const ctx = document.getElementById('profit-chart');
    if (!ctx) return;
    
    const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    const profitData = [1200, 1850, 1650, 2100, 1950, 2400];
    
    window.profitChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Monthly Profit',
                data: profitData,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#e5e7eb'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#e5e7eb'
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

function refreshDashboard() {
    const refreshBtn = document.querySelector('.refresh-btn');
    refreshBtn.disabled = true;
    refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
    
    setTimeout(() => {
        loadDashboardData();
        
        if (window.equityChart) {
            const newData = [
                Math.floor(Math.random() * 40) + 50,
                Math.floor(Math.random() * 10) + 3,
                Math.floor(Math.random() * 40) + 20
            ];
            window.equityChart.data.datasets[0].data = newData;
            window.equityChart.update();
        }
        
        refreshBtn.disabled = false;
        refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Data';
    }, 1500);
}

function analyzeHand() {
    const holeCard1 = document.getElementById('hole-card-1').value;
    const holeCard2 = document.getElementById('hole-card-2').value;
    const opponents = parseInt(document.getElementById('opponents').value);
    
    if (!holeCard1 || !holeCard2) {
        showWarning('Missing Input', 'Please enter both hole cards to analyze the hand.', 'warning');
        return;
    }
    
    if (opponents < 1 || opponents > 9) {
        showWarning('Invalid Input', 'Number of opponents must be between 1 and 9. In poker, there are maximum 10 seats at a table.', 'error');
        return;
    }
    
    console.log('Analyzing hand...');
    
    document.getElementById('loading-overlay').classList.add('active');
    document.getElementById('loading-message').textContent = 'Analyzing hand equity...';
    
    analyzeHandBackend().then(success => {
        document.getElementById('loading-overlay').classList.remove('active');
        document.getElementById('hand-results').style.display = 'block';
    }).catch(error => {
        console.log('Backend failed, using demo analysis:', error.message);
        document.getElementById('loading-overlay').classList.remove('active');
        document.getElementById('hand-results').style.display = 'block';
        
        showWarning('Backend Unavailable', 'Cannot connect to analysis server. Using demo calculations.', 'warning');
        analyzeHandDemo();
    });
}

async function analyzeHandBackend() {
    try {
        const holeCard1 = document.getElementById('hole-card-1').value.trim().toUpperCase();
        const holeCard2 = document.getElementById('hole-card-2').value.trim().toUpperCase();
        const position = document.getElementById('position').value;
        const opponents = parseInt(document.getElementById('opponents').value);
        
        if (!holeCard1 || !holeCard2) {
            throw new Error('Please enter both hole cards');
        }
        
        if (holeCard1.length !== 2 || holeCard2.length !== 2) {
            throw new Error('Cards must be 2 characters (e.g., As, Kh)');
        }
        
        if (holeCard1 === holeCard2) {
            throw new Error('Cannot have duplicate cards');
        }
        
        if (opponents < 1 || opponents > 9) {
            throw new Error('Number of opponents must be between 1 and 9');
        }
        
        const validRanks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
        const validSuits = ['H', 'D', 'C', 'S'];
        
        if (!validRanks.includes(holeCard1[0]) || !validSuits.includes(holeCard1[1]) ||
            !validRanks.includes(holeCard2[0]) || !validSuits.includes(holeCard2[1])) {
            throw new Error('Invalid card format. Use rank (2-9,T,J,Q,K,A) + suit (h,d,c,s)');
        }
        
        const holeCards = holeCard1 + holeCard2;
        
        const positionMap = {
            'UTG': 'early_position',
            'UTG+1': 'early_position',
            'MP': 'middle_position',
            'MP+1': 'middle_position',
            'CO': 'late_position',
            'BTN': 'button',
            'SB': 'early_position',
            'BB': 'early_position'
        };
        
        const requestData = {
            holeCards: holeCards,
            position: positionMap[position] || 'middle_position',
            opponentCount: opponents
        };
        
        const response = await fetch(`${API_BASE}/api/analyze_hand`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`Analysis failed: ${response.statusText}`);
        }
        
        const analysisData = await response.json();
        
        updateEquityChart(analysisData.equity, 100 - analysisData.equity);
        updateRecentAnalysis(holeCard1, holeCard2, position, analysisData.equity, analysisData.recommendation);
        
        document.getElementById('hand-analysis-content').innerHTML = `
            <div class="result-summary">
                <h4>Hand: ${holeCard1} ${holeCard2} in ${position}</h4>
                <p>Against ${opponents} opponents</p>
                <div class="equity-result">
                    <strong>Equity: ${analysisData.equity}% (${analysisData.handCategory})</strong>
                </div>
                <div class="recommendation ${analysisData.recommendation.toLowerCase()}">
                    <strong>Recommendation: ${analysisData.recommendation}</strong>
                </div>
                <div class="confidence">
                    Confidence: ${analysisData.confidence.toFixed(1)}%
                </div>
                <div class="gto-strategy">
                    <h5>GTO Frequencies:</h5>
                    ${Object.entries(analysisData.gtoStrategy).map(([action, freq]) => 
                        `<span class="freq-item">${action}: ${(freq * 100).toFixed(1)}%</span>`
                    ).join(' | ')}
                </div>
            </div>
        `;
        
        return true;
    } catch (error) {
        console.error('Backend hand analysis failed:', error);
        throw error;
    }
}

function analyzeHandDemo() {
    const holeCard1 = document.getElementById('hole-card-1').value;
    const holeCard2 = document.getElementById('hole-card-2').value;
    const position = document.getElementById('position').value;
    const opponents = document.getElementById('opponents').value;
    
    let demoEquity = 50;
    if (holeCard1[0] === 'A' || holeCard2[0] === 'A') demoEquity += 15;
    if (holeCard1[0] === holeCard2[0]) demoEquity += 10;
    demoEquity -= (parseInt(opponents) - 1) * 3;
    demoEquity = Math.max(15, Math.min(85, demoEquity));
    
    const recommendation = demoEquity > 55 ? 'Raise' : demoEquity > 35 ? 'Call' : 'Fold';
    
    updateEquityChart(demoEquity, 100 - demoEquity);
    updateRecentAnalysis(holeCard1.toUpperCase(), holeCard2.toUpperCase(), position, demoEquity.toFixed(1), recommendation);
    
    document.getElementById('hand-analysis-content').innerHTML = `
        <div class="result-summary">
            <h4>Hand: ${holeCard1} ${holeCard2} in ${position}</h4>
            <p>Against ${opponents} opponents</p>
            <div class="equity-result">
                <strong>Equity: ${demoEquity.toFixed(1)}% ± 2.1% (Demo Mode)</strong>
            </div>
            <div class="recommendation">
                <strong>Recommendation: ${recommendation}</strong>
            </div>
            <div class="confidence">
                Confidence: 85%
            </div>
        </div>
    `;
}

function updateEquityChart(winPercent, losePercent) {
    if (window.equityChart) {
        const tiePercent = Math.max(0, 100 - winPercent - losePercent);
        window.equityChart.data.datasets[0].data = [winPercent, tiePercent, losePercent];
        window.equityChart.update();
    }
}

function updateRecentAnalysis(card1, card2, position, equity, recommendation) {
    const now = new Date();
    const timeAgo = "Just now";
    
    const newAnalysis = {
        cards: `${card1}${card2}`,
        position: position,
        equity: equity,
        recommendation: recommendation,
        timestamp: now,
        timeAgo: timeAgo
    };
    
    recentAnalysisHistory.unshift(newAnalysis);
    recentAnalysisHistory = recentAnalysisHistory.slice(0, 5);
    
    displayRecentAnalysis();
    updateAnalysisTimestamps();
}

function displayRecentAnalysis() {
    const analysisContainer = document.querySelector('.analysis-list');
    if (!analysisContainer) return;
    
    if (recentAnalysisHistory.length === 0) {
        analysisContainer.innerHTML = `
            <div class="analysis-item">
                <div class="analysis-hand">
                    <span class="cards">No analyses yet</span>
                </div>
                <div class="analysis-result">
                    <span class="equity">Use Hand Analyzer to see results here</span>
                </div>
            </div>
        `;
        return;
    }
    
    analysisContainer.innerHTML = recentAnalysisHistory.map(analysis => `
        <div class="analysis-item">
            <div class="analysis-hand">
                <span class="cards">${analysis.cards}</span>
                <span class="position">${analysis.position}</span>
            </div>
            <div class="analysis-result">
                <span class="equity">${analysis.equity}% equity</span>
                <span class="action ${analysis.recommendation.toLowerCase()}">${analysis.recommendation}</span>
            </div>
            <div class="analysis-time">${analysis.timeAgo}</div>
        </div>
    `).join('');
}

function updateAnalysisTimestamps() {
    const now = new Date();
    
    recentAnalysisHistory.forEach(analysis => {
        const diffMs = now - analysis.timestamp;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        
        if (diffMins < 1) {
            analysis.timeAgo = "Just now";
        } else if (diffMins < 60) {
            analysis.timeAgo = `${diffMins} min ago`;
        } else if (diffHours < 24) {
            analysis.timeAgo = `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        } else {
            analysis.timeAgo = "Earlier";
        }
    });
    
    displayRecentAnalysis();
}

function analyzeBankroll() {
    const bankroll = document.getElementById('bankrollAmount').value;
    const buyIn = document.getElementById('buyInAmount').value;
    const variance = document.getElementById('varianceLevel').value;
    
    if (!bankroll || !buyIn) {
        showWarning('Missing Input', 'Please enter both bankroll amount and buy-in amount.', 'warning');
        return;
    }
    
    console.log('Analyzing bankroll...');
    
    document.getElementById('loading-overlay').classList.add('active');
    document.getElementById('loading-message').textContent = 'Analyzing bankroll risk...';
    
    analyzeBankrollBackend().then(success => {
        document.getElementById('loading-overlay').classList.remove('active');
        document.getElementById('bankroll-results').style.display = 'block';
    }).catch(error => {
        console.log('Backend failed, using demo bankroll analysis:', error.message);
        document.getElementById('loading-overlay').classList.remove('active');
        document.getElementById('bankroll-results').style.display = 'block';
        
        showWarning('Backend Unavailable', 'Cannot connect to analysis server. Using demo calculations.', 'warning');
        analyzeBankrollDemo();
    });
}

async function analyzeBankrollBackend() {
    try {
        const bankroll = parseFloat(document.getElementById('bankrollAmount').value);
        const buyIn = parseFloat(document.getElementById('buyInAmount').value);
        const variance = parseFloat(document.getElementById('varianceLevel').value);
        
        const requestData = {
            bankroll: bankroll,
            buyIn: buyIn,
            variance: variance
        };
        
        const response = await fetch(`${API_BASE}/api/bankroll_analysis`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`Bankroll analysis failed: ${response.statusText}`);
        }
        
        const analysisData = await response.json();
        
        document.getElementById('bankrollResults').innerHTML = `
            <div class="result-item">
                <span class="result-label">Risk Level:</span>
                <span class="result-value risk-${analysisData.riskLevel.toLowerCase()}">${analysisData.riskLevel}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Buy-in Ratio:</span>
                <span class="result-value">${analysisData.buyInRatio}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Risk of Ruin:</span>
                <span class="result-value">${analysisData.riskOfRuin}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Max Recommended Buy-in:</span>
                <span class="result-value">${analysisData.maxBuyIn}</span>
            </div>
        `;
        
        return true;
    } catch (error) {
        console.error('Backend bankroll analysis failed:', error);
        throw error;
    }
}

function analyzeBankrollDemo() {
    const bankroll = parseFloat(document.getElementById('bankrollAmount').value);
    const buyIn = parseFloat(document.getElementById('buyInAmount').value);
    const variance = parseFloat(document.getElementById('varianceLevel').value);
    
    if (bankroll <= 0) {
        showWarning('Invalid Input', 'Bankroll must be greater than 0', 'error');
        return;
    }
    
    if (buyIn > bankroll) {
        showWarning('Invalid Input', 'Buy-in cannot be larger than bankroll!', 'error');
        return;
    }
    
    const buyInRatio = (buyIn / bankroll * 100);
    
    let riskOfRuin = 0;
    if (buyInRatio > 0) {
        const baseRisk = Math.min(buyInRatio * variance * 2, 50);
        riskOfRuin = Math.max(0.1, baseRisk);
    }
    
    const maxBuyIn = bankroll * 0.02;
    
    let riskLevel = 'Low';
    if (buyInRatio > 2) riskLevel = 'Medium';
    if (buyInRatio > 5) riskLevel = 'High';
    
    if (riskLevel === 'High') {
        showWarning('High Risk Detected!', 
            `Your current setup has a ${riskOfRuin.toFixed(1)}% risk of ruin. Consider reducing your buy-in size.`, 
            'error');
    }
    
    document.getElementById('bankrollResults').innerHTML = `
        <div class="result-item">
            <span class="result-label">Risk Level:</span>
            <span class="result-value risk-${riskLevel.toLowerCase()}">${riskLevel} (Demo)</span>
        </div>
        <div class="result-item">
            <span class="result-label">Buy-in Ratio:</span>
            <span class="result-value">${buyInRatio.toFixed(2)}%</span>
        </div>
        <div class="result-item">
            <span class="result-label">Risk of Ruin:</span>
            <span class="result-value">${riskOfRuin.toFixed(1)}%</span>
        </div>
        <div class="result-item">
            <span class="result-label">Max Recommended Buy-in:</span>
            <span class="result-value">${maxBuyIn.toFixed(0)}</span>
        </div>
    `;
}

function analyzeGTO() {
    console.log('Analyzing GTO strategy...');
    
    document.getElementById('loading-overlay').classList.add('active');
    document.getElementById('loading-message').textContent = 'Calculating optimal strategy...';
    
    analyzeGTOBackend().then(success => {
        document.getElementById('loading-overlay').classList.remove('active');
        document.getElementById('gto-results').style.display = 'block';
    }).catch(error => {
        console.log('Backend failed, using demo GTO analysis:', error.message);
        document.getElementById('loading-overlay').classList.remove('active');
        document.getElementById('gto-results').style.display = 'block';
        
        showWarning('Backend Unavailable', 'Cannot connect to analysis server. Using demo calculations.', 'warning');
        analyzeGTODemo();
    });
}

async function analyzeGTOBackend() {
    try {
        const scenario = document.getElementById('gto-scenario').value;
        const position = document.getElementById('gto-position').value;
        const boardTexture = document.getElementById('board-texture').value;
        const potSize = parseFloat(document.getElementById('pot-size').value) || 100;
        const stackSize = parseFloat(document.getElementById('stack-size').value) || 10000;
        
        const requestData = {
            scenarioType: scenario,
            position: position,
            boardTexture: boardTexture,
            potSize: potSize,
            stackSize: stackSize
        };
        
        const response = await fetch(`${API_BASE}/api/gto_solver`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`GTO analysis failed: ${response.statusText}`);
        }
        
        const analysisData = await response.json();
        
        displayGTOResults(analysisData);
        
        return true;
    } catch (error) {
        console.error('Backend GTO analysis failed:', error);
        throw error;
    }
}

function analyzeGTODemo() {
    const position = document.getElementById('gto-position').value;
    const scenario = document.getElementById('gto-scenario').value;
    
    let frequencies = {};
    
    if (scenario === 'preflop') {
        if (position === 'early_position') {
            frequencies = { fold: 75.0, call: 8.0, raise: 17.0 };
        } else if (position === 'button') {
            frequencies = { fold: 45.0, call: 25.0, raise: 30.0 };
        } else {
            frequencies = { fold: 60.0, call: 18.0, raise: 22.0 };
        }
    } else {
        frequencies = { check: 55.0, bet: 45.0 };
    }
    
    const demoData = {
        optimalStrategy: frequencies,
        recommendation: Object.keys(frequencies).reduce((a, b) => frequencies[a] > frequencies[b] ? a : b),
        exploitability: 1.2,
        stackToPotRatio: 5.5
    };
    
    displayGTOResults(demoData);
}

function displayGTOResults(data) {
    const frequenciesHtml = Object.entries(data.optimalStrategy).map(([action, freq]) => `
        <div class="frequency-item">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span class="action-name">${action.charAt(0).toUpperCase() + action.slice(1)}</span>
                <span class="frequency-value">${freq}%</span>
            </div>
            <div class="frequency-bar">
                <div class="frequency-fill" style="width: ${freq}%;"></div>
            </div>
        </div>
    `).join('');
    
    document.getElementById('gto-analysis-content').innerHTML = `
        <div class="result-summary">
            <h4>Optimal Strategy Analysis</h4>
            <div class="recommendation">
                <strong>Primary Action: ${data.recommendation.charAt(0).toUpperCase() + data.recommendation.slice(1)}</strong>
            </div>
            <div class="gto-frequencies">
                ${frequenciesHtml}
            </div>
            <div class="exploitability">
                Exploitability: ${data.exploitability}%
                ${data.stackToPotRatio ? `| SPR: ${data.stackToPotRatio}` : ''}
            </div>
        </div>
    `;
}

function analyzeTournament() {
    console.log('Analyzing tournament situation...');
    
    document.getElementById('loading-overlay').classList.add('active');
    document.getElementById('loading-message').textContent = 'Calculating tournament strategy...';
    
    analyzeTournamentBackend().then(success => {
        document.getElementById('loading-overlay').classList.remove('active');
        document.getElementById('tournament-results').style.display = 'block';
    }).catch(error => {
        console.log('Backend failed, using demo tournament analysis:', error.message);
        document.getElementById('loading-overlay').classList.remove('active');
        document.getElementById('tournament-results').style.display = 'block';
        
        showWarning('Backend Unavailable', 'Cannot connect to analysis server. Using demo calculations.', 'warning');
        analyzeTournamentDemo();
    });
}

async function analyzeTournamentBackend() {
    try {
        const stackSize = parseInt(document.getElementById('stack-size-tournament').value) || 20000;
        const blindLevel = parseInt(document.getElementById('blind-level').value) || 3;
        const playersLeft = parseInt(document.getElementById('players-left').value) || 45;
        const totalPlayers = parseInt(document.getElementById('total-players').value) || 100;
        
        const requestData = {
            stackSize: stackSize,
            blindLevel: blindLevel,
            playersLeft: playersLeft,
            totalPlayers: totalPlayers
        };
        
        const response = await fetch(`${API_BASE}/api/tournament_analysis`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`Tournament analysis failed: ${response.statusText}`);
        }
        
        const analysisData = await response.json();
        
        displayTournamentResults(analysisData);
        
        return true;
    } catch (error) {
        console.error('Backend tournament analysis failed:', error);
        throw error;
    }
}

function analyzeTournamentDemo() {
    const stackSize = parseInt(document.getElementById('stack-size-tournament').value) || 20000;
    const blindLevel = parseInt(document.getElementById('blind-level').value) || 3;
    const playersLeft = parseInt(document.getElementById('players-left').value) || 45;
    
    const blindLevels = [
        { small: 25, big: 50, ante: 0 },
        { small: 50, big: 100, ante: 0 },
        { small: 75, big: 150, ante: 0 },
        { small: 100, big: 200, ante: 25 },
        { small: 150, big: 300, ante: 50 }
    ];
    
    const currentBlinds = blindLevels[Math.min(blindLevel - 1, blindLevels.length - 1)];
    const costPerRound = currentBlinds.small + currentBlinds.big + (currentBlinds.ante * 9);
    const mRatio = stackSize / costPerRound;
    
    let phase, strategy, icmPressure;
    
    if (mRatio > 20) {
        phase = "Early/Accumulation";
        strategy = "Play tight-aggressive, build stack slowly";
        icmPressure = "Low";
    } else if (mRatio > 10) {
        phase = "Middle";
        strategy = "Increase aggression, steal blinds";
        icmPressure = "Medium";
    } else if (mRatio > 5) {
        phase = "Late/Push-Fold";
        strategy = "Push-fold strategy, look for spots";
        icmPressure = "High";
    } else {
        phase = "Critical";
        strategy = "Desperate mode, any two cards in good spots";
        icmPressure = "Critical";
    }
    
    const demoData = {
        mRatio: mRatio.toFixed(1),
        phase: phase,
        strategy: strategy,
        icmPressure: icmPressure,
        playersLeft: playersLeft,
        currentBlinds: currentBlinds
    };
    
    displayTournamentResults(demoData);
}

function displayTournamentResults(data) {
    const mRatioClass = data.mRatio > 20 ? 'm-ratio-healthy' : 
                       data.mRatio > 10 ? 'm-ratio-medium' : 
                       data.mRatio > 5 ? 'm-ratio-warning' : 'm-ratio-critical';
    
    const icmClass = data.icmPressure === 'Low' ? 'icm-low' : 
                     data.icmPressure === 'Medium' ? 'icm-medium' : 'icm-high';
    
    document.getElementById('tournament-analysis-content').innerHTML = `
        <div class="result-summary">
            <h4>Tournament Analysis</h4>
            <div class="tournament-stats">
                <div class="stat-item">
                    <span class="stat-label">M-Ratio:</span>
                    <span class="stat-value ${mRatioClass}">${data.mRatio}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Phase:</span>
                    <span class="stat-value">${data.phase}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">ICM Pressure:</span>
                    <span class="stat-value ${icmClass}">${data.icmPressure}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Players Left:</span>
                    <span class="stat-value">${data.playersLeft}</span>
                </div>
            </div>
            <div class="strategy-recommendation">
                <strong>Strategy: ${data.strategy}</strong>
            </div>
        </div>
    `;
    
    if (data.mRatio < 5) {
        setTimeout(() => {
            showWarning('Critical Stack Size!', 
                'Your M-ratio is critically low. Consider push-fold strategy with any reasonable hand.', 
                'error');
        }, 500);
    }
}

function addSession() {
    const date = document.getElementById('session-date').value;
    const duration = parseFloat(document.getElementById('session-duration').value);
    const profit = parseFloat(document.getElementById('session-profit').value);
    const gameType = document.getElementById('game-type').value;
    
    if (!date) {
        showWarning('Missing Input', 'Please enter a session date.', 'warning');
        return;
    }
    
    if (!duration || duration <= 0) {
        showWarning('Invalid Input', 'Session duration must be greater than 0.', 'error');
        return;
    }
    
    if (isNaN(profit)) {
        showWarning('Invalid Input', 'Please enter a valid profit/loss amount.', 'error');
        return;
    }
    
    const hourlyRate = profit / duration;
    const sessionRow = document.createElement('tr');
    sessionRow.innerHTML = `
        <td>${date}</td>
        <td>${duration.toFixed(1)}h</td>
        <td class="${profit >= 0 ? 'profit-positive' : 'profit-negative'}">${profit.toFixed(2)}</td>
        <td>${gameType}</td>
        <td class="${hourlyRate >= 0 ? 'profit-positive' : 'profit-negative'}">${hourlyRate.toFixed(2)}/h</td>
    `;
    
    const tbody = document.querySelector('#sessions-table tbody');
    if (tbody) {
        tbody.insertBefore(sessionRow, tbody.firstChild);
    }
    
    showWarning('Session Added', 
        `Successfully added ${gameType} session with ${profit >= 0 ? 'profit' : 'loss'} of ${Math.abs(profit).toFixed(2)}.`, 
        'success');
    
    document.getElementById('session-form').reset();
    document.getElementById('session-date').value = new Date().toISOString().split('T')[0];
    
    if (sessionData) {
        sessionData.totalProfit = (sessionData.totalProfit || 0) + profit;
        sessionData.totalHours = (sessionData.totalHours || 0) + duration;
        sessionData.sessionsPlayed = (sessionData.sessionsPlayed || 0) + 1;
        sessionData.hourlyRate = sessionData.totalProfit / sessionData.totalHours;
        
        if (currentSection === 'dashboard') {
            updateDashboardMetrics();
        }
    }
}

function toggleEquityChart(mode) {
    const buttons = document.querySelectorAll('.chart-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    const activeButton = document.querySelector(`[onclick="toggleEquityChart('${mode}')"]`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
    
    if (!window.equityChart) return;
    
    if (mode === 'strength') {
        window.equityChart.data.labels = ['Premium', 'Strong', 'Playable', 'Marginal', 'Fold'];
        window.equityChart.data.datasets[0].data = [15, 25, 30, 20, 10];
        window.equityChart.data.datasets[0].backgroundColor = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#6b7280'];
    } else {
        window.equityChart.data.labels = ['Win', 'Tie', 'Lose'];
        window.equityChart.data.datasets[0].data = [65, 5, 30];
        window.equityChart.data.datasets[0].backgroundColor = ['#10b981', '#f59e0b', '#ef4444'];
    }
    
    window.equityChart.update();
}

window.showSection = showSection;
window.refreshDashboard = refreshDashboard;
window.analyzeHand = analyzeHand;
window.analyzeBankroll = analyzeBankroll;
window.analyzeGTO = analyzeGTO;
window.analyzeTournament = analyzeTournament;
window.addSession = addSession;
window.toggleEquityChart = toggleEquityChart;
window.closeWarning = closeWarning;