# 🃏 PokerPro Analytics

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/yourusername/pokerpro-analytics?style=for-the-badge&color=orange)
![GitHub forks](https://img.shields.io/github/forks/yourusername/pokerpro-analytics?style=for-the-badge&color=blue)
![License](https://img.shields.io/github/license/yourusername/pokerpro-analytics?style=for-the-badge&color=green)
![Python](https://img.shields.io/badge/python-3.7+-blue?style=for-the-badge&logo=python&logoColor=white)
![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen?style=for-the-badge&logo=firefox&logoColor=white)

**A Web-Based Platform for Real-Time Poker Analysis & Strategic Optimization**

*Transform poker from intuition-based gambling into precision science*

**🌐 [Live Demo](https://enesi.pythonanywhere.com/) • [🚀 Quick Start](#-quick-start) • [📚 Features](#-core-features) • [🎯 API Docs](#-api-reference) • [🤝 Contributing](#-contributing)**

</div>

---

## 🎯 What is PokerPro Analytics?

PokerPro Analytics empowers poker players of all skill levels to make **data-driven decisions** at the table. By combining real-time mathematical analysis, game theory optimal (GTO) strategies, and comprehensive bankroll management, it transforms how you approach the game.

<div align="center">

```
┌─────────────────────────────────────────────────────────────┐
│                   🃏 PokerPro Dashboard                     │
├─────────────────────────────────────────────────────────────┤
│  Performance Dashboard                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Current     │  │ Expected    │  │ Total       │       │
│  │ Equity      │  │ Value       │  │ Profit      │       │
│  │ 67.3% ±2.1% │  │ +2.4 BB/100 │  │ $0.00       │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                           │
│  Recent Analysis:                                         │
│  🃏 A♠ K♥ (BTN) → 74.2% equity → Raise                   │
│  🃏 Q♦ Q♣ (CO)  → 82.1% equity → Raise                   │
│  🃏 7♠ 2♦ (UTG) → 31.4% equity → Fold                    │
└─────────────────────────────────────────────────────────────┘
```

**🔗 Try it live: [enesi.pythonanywhere.com](https://enesi.pythonanywhere.com/)**

</div>

### ✨ Why Choose PokerPro?

| 🎲 **Traditional Poker** | 🧠 **PokerPro Analytics** |
|--------------------------|----------------------------|
| Gut feeling decisions    | Monte Carlo simulations (±2.1% accuracy) |
| Emotional bankroll management | Kelly Criterion risk assessment |
| Guessing opponent ranges | Real-time equity calculations |
| Trial and error learning | GTO-based strategy recommendations |

---

## 🚀 Core Features

### 🎯 **Performance Dashboard**
Your poker command center with real-time analytics:

- **📊 Live Equity Tracking**: Current session equity with confidence intervals
- **💰 Expected Value Calculator**: BB/100 and hourly rate projections  
- **📈 Win Rate Analysis**: Session-based performance metrics
- **🃏 Recent Hand History**: Latest analyzed hands with recommendations

### 🔍 **Hand Analyzer** 
Advanced Monte Carlo simulation engine:

```
Input Format: As, Kh, Qd, Jc, Ts
┌─────────────────────────────────┐
│ A♠ K♥ vs 3 opponents            │
│ Position: Button                │
│ ────────────────────────────    │
│ Equity: 74.2% (±1.8%)          │
│ Recommendation: RAISE 2.5x      │
│ EV: +2.4 BB                     │
└─────────────────────────────────┘
```

### 🧮 **GTO Solver**
Game Theory Optimal strategy analysis:
- Preflop & postflop decision trees
- Nash equilibrium calculations
- Exploitability analysis
- Position-aware frequencies

### 💰 **Bankroll Management**
Scientific risk assessment tools:
- Kelly Criterion optimization
- Risk-of-ruin modeling
- Buy-in recommendations
- Variance analysis

### 🏆 **Tournament Strategy**
ICM and tournament-specific tools:
- M-ratio calculations
- Push/fold charts
- Bubble factor analysis
- Prize pool considerations

### 📊 **Session Review**
Comprehensive performance tracking:
- Session history with profit/loss
- Hourly rate calculations
- Game type categorization
- Leak identification

---

## 🛠️ Technology Stack

<div align="center">

| **Backend** | **Frontend** | **Analytics** |
|-------------|--------------|---------------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black) | ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) |
| ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) | ![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chart.js&logoColor=white) |
| ![PythonAnywhere](https://img.shields.io/badge/PythonAnywhere-1D4F8C?style=for-the-badge) | ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) | ![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-4CAF50?style=for-the-badge) |

</div>

---

## ⚡ Quick Start

### 🌐 Option 1: Use Live Demo
Visit **[enesi.pythonanywhere.com](https://enesi.pythonanywhere.com/)** - no installation required!

### 🔧 Option 2: Local Installation

#### 📋 Prerequisites
```bash
Python >= 3.7
Modern web browser (Chrome, Firefox, Safari, Edge)
```

#### 🚀 Installation Steps
```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/pokerpro-analytics.git
cd pokerpro-analytics

# 2️⃣ Install dependencies
pip install flask flask-cors numpy

# 3️⃣ Launch the application
python app.py
```

#### 🌐 Access Locally
Open your browser and navigate to **http://localhost:5000**

---

## 📁 Project Structure

```
pokerpro-analytics/
│
├── 🐍 app.py              # Flask backend & API routes
├── 📁 static/
│   ├── 🎨 css/style.css   # Dashboard styling
│   └── ⚡ js/main.js      # Monte Carlo engine & UI
├── 📁 templates/
│   └── 🌐 index.html      # Main dashboard interface
├── 📄 README.md           # You are here!
└── 📋 requirements.txt    # Python dependencies
```

---

## 🎯 Usage Examples

### 🃏 Hand Analysis Workflow

<table>
<tr>
<td width="50%">

**Step 1: Input Your Hand**
```
Cards: As Kh
Position: Button  
Opponents: 3
```

**Step 2: Get Analysis**
- Click "Analyze Hand"
- Monte Carlo simulation runs
- Results appear instantly

</td>
<td width="50%">

**Real Output from Dashboard:**
```
┌─────────────────────────────┐
│ A♠ K♥ (BTN)                 │
│ Equity: 74.2% (±1.8%)       │
│ Recommendation: Raise        │
│ Expected Value: +2.4 BB      │
│ Action: Raise 2.5x           │
└─────────────────────────────┘
```

*This shows actual output from the live demo*

</td>
</tr>
</table>

### 📊 Performance Tracking

The dashboard automatically tracks your session data:

```
Recent Analysis History:
┌─────────────────────────────────────────────────┐
│ A♠ K♥ (BTN) → 74.2% equity → Raise    2 min ago │
│ Q♦ Q♣ (CO)  → 82.1% equity → Raise    5 min ago │  
│ 7♠ 2♦ (UTG) → 31.4% equity → Fold     8 min ago │
└─────────────────────────────────────────────────┘

Performance Metrics:
• Current Equity: 67.3% ±2.1%
• Expected Value: +2.4 BB/100  
• Total Profit: $0.00 (new session)
• Win Rate: 0% (0 sessions logged)
```

---

## 📡 API Reference

<details>
<summary><strong>🃏 Hand Analysis Endpoint</strong></summary>

```http
POST /api/analyze_hand
Content-Type: application/json
```

**Request:**
```json
{
  "holeCards": "AhKs",
  "position": "button", 
  "opponentCount": 3,
  "boardCards": []
}
```

**Response:**
```json
{
  "equity": 74.2,
  "confidence_interval": 1.8,
  "recommendation": "raise",
  "action": "Raise 2.5x",
  "expectedValue": 2.4,
  "timestamp": "2 min ago"
}
```

</details>

<details>
<summary><strong>💰 Bankroll Analysis Endpoint</strong></summary>

```http
POST /api/bankroll_analysis  
Content-Type: application/json
```

**Request:**
```json
{
  "bankroll": 10000,
  "buyIn": 100,
  "gameType": "cash",
  "riskTolerance": "moderate"
}
```

**Response:**
```json
{
  "riskLevel": "Low Risk",
  "recommendedStakes": "0.50/1.00",
  "riskOfRuin": 0.8,
  "kellyCriterion": 0.12,
  "maxBuyins": 100
}
```

</details>

<details>
<summary><strong>📊 Session Tracking Endpoint</strong></summary>

```http
POST /api/add_session
Content-Type: application/json
```

**Request:**
```json
{
  "duration": 180,
  "profit": 250.50,
  "gameType": "NL Hold'em",
  "stakes": "1/2",
  "location": "Online"
}
```

</details>

---

## 🎓 Who Is This For?

<div align="center">

| 🌱 **Beginners** | 📈 **Intermediate** | 🏆 **Advanced** |
|-------------------|---------------------|------------------|
| Learn which hands to play | Identify and fix leaks | Fine-tune GTO strategies |
| Understand position importance | Master bankroll management | Exploit population tendencies |
| Build solid fundamentals | Tournament strategy | Multi-table optimization |
| **Live Example:** 7♠ 2♦ UTG → Fold | **Live Example:** A♠ K♥ BTN → Raise | **Live Example:** Q♦ Q♣ CO → Raise |

</div>

---

## 🔥 Live Demo Highlights

**🌐 Try these features at [enesi.pythonanywhere.com](https://enesi.pythonanywhere.com/):**

✅ **Hand Analyzer** - Enter "As, Kh" and see real-time equity  
✅ **GTO Solver** - Get optimal strategy recommendations  
✅ **Performance Dashboard** - View live session statistics  
✅ **Bankroll Manager** - Calculate optimal buy-in sizes  
✅ **Session Tracker** - Log and analyze your poker sessions  

*No registration required - start analyzing hands immediately!*

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 🔧 Development Setup

1. **Fork** the repository
2. **Clone** your fork locally  
3. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
4. **Test** with the live demo for reference
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### 📝 Code Style Guidelines

- **Python**: Follow PEP 8 standards
- **JavaScript**: Use ES6+ features  
- **UI**: Match the existing dashboard design
- **Testing**: Verify calculations against live demo

### 🐛 Found a Bug?

[Open an issue](https://github.com/yourusername/pokerpro-analytics/issues) with:
- Link to the live demo showing the issue
- Clear description of expected vs actual behavior
- Steps to reproduce
- Browser and device information

---

## 🔍 Troubleshooting

<details>
<summary><strong>❌ Local server not working?</strong></summary>

- ✅ Check if Flask server is running on port 5000
- ✅ Compare with live demo at enesi.pythonanywhere.com
- ✅ Verify all dependencies are installed
- ✅ Check console for error messages

</details>

<details>
<summary><strong>📊 Calculations seem wrong?</strong></summary>

- ✅ Test the same input on the live demo
- ✅ Verify card format: use "As, Kh, Qd" format
- ✅ Check opponent count (1-9 players)
- ✅ Ensure position is valid (UTG, MP, CO, BTN, SB, BB)

</details>

<details>
<summary><strong>🎯 Getting different results than expected?</strong></summary>

- ✅ Monte Carlo simulations have variance (±2.1% typical)
- ✅ Compare multiple runs for consistency
- ✅ Check if board cards are specified correctly
- ✅ Verify position impacts on equity calculations

</details>

---

## 📊 Project Stats

<div align="center">

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/yourusername/pokerpro-analytics?style=flat-square&color=orange)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/pokerpro-analytics?style=flat-square&color=blue)
![GitHub issues](https://img.shields.io/github/issues/yourusername/pokerpro-analytics?style=flat-square&color=red)
![Uptime](https://img.shields.io/uptimerobot/status/m793494864-dcbe0a8c7bfc4e5c9d5b1b1e?style=flat-square&label=Live%20Demo)

</div>

---

## 🎉 What's Next?

**Upcoming Features:**
- 🤖 AI opponent modeling
- 📱 Mobile app version
- 🎪 Multi-table tournament tracking
- 🔄 Real-time hand importing
- 👥 Community strategy sharing

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

**🌐 [Try the Live Demo](https://enesi.pythonanywhere.com/) | 🐛 [Report Issues](https://github.com/yourusername/pokerpro-analytics/issues) | 💬 [Join Discussion](https://github.com/yourusername/pokerpro-analytics/discussions)**

Made with ❤️ and ♠️♥️♦️♣️ by the PokerPro team

[🔝 Back to top](#-pokerpro-analytics)

</div>
