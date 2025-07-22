from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import random
from datetime import datetime, timedelta
from collections import defaultdict
import itertools

app = Flask(__name__)
CORS(app)

class PokerAnalytics:
    def __init__(self):
        self.hand_ranges = self.load_hand_ranges()
        self.gto_strategies = self.load_gto_strategies()
        self.tournament_structures = self.load_tournament_structures()
        
    def load_hand_ranges(self):
        return {
            'early_position': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AQs', 'AJs', 'AKo', 'AQo'],
            'middle_position': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'AKo', 'AQo', 'AJo', 'KQs', 'KJs'],
            'late_position': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'KQs', 'KJs', 'KTs', 'K9s', 'KQo', 'KJo', 'QJs', 'QTs', 'Q9s', 'QJo', 'JTs', 'J9s', 'JTo', 'T9s', '98s', '87s', '76s', '65s'],
            'button': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s', 'KQo', 'KJo', 'KTo', 'K9o', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s', 'QJo', 'QTo', 'Q9o', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'JTo', 'J9o', 'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T9o', '98s', '97s', '96s', '95s', '98o', '87s', '86s', '85s', '87o', '76s', '75s', '76o', '65s', '65o', '54s']
        }
    
    def load_gto_strategies(self):
        return {
            'preflop': {
                'early_position': {'fold': 0.85, 'call': 0.05, 'raise': 0.10},
                'middle_position': {'fold': 0.75, 'call': 0.10, 'raise': 0.15},
                'late_position': {'fold': 0.55, 'call': 0.20, 'raise': 0.25},
                'button': {'fold': 0.40, 'call': 0.25, 'raise': 0.35}
            },
            'postflop': {
                'dry_board': {'check': 0.60, 'bet': 0.40},
                'wet_board': {'check': 0.40, 'bet': 0.60},
                'paired_board': {'check': 0.70, 'bet': 0.30}
            }
        }
    
    def load_tournament_structures(self):
        return {
            'standard': {
                'blinds': [
                    {'level': 1, 'small_blind': 25, 'big_blind': 50, 'ante': 0, 'duration': 20},
                    {'level': 2, 'small_blind': 50, 'big_blind': 100, 'ante': 0, 'duration': 20},
                    {'level': 3, 'small_blind': 75, 'big_blind': 150, 'ante': 0, 'duration': 20},
                    {'level': 4, 'small_blind': 100, 'big_blind': 200, 'ante': 25, 'duration': 20},
                    {'level': 5, 'small_blind': 150, 'big_blind': 300, 'ante': 50, 'duration': 20}
                ],
                'payout_structure': [0.4, 0.25, 0.15, 0.10, 0.06, 0.04]
            }
        }

poker_analytics = PokerAnalytics()

class AdvancedMonteCarlo:
    def __init__(self):
        self.deck = self.create_deck()
        
    def create_deck(self):
        suits = ['h', 'd', 'c', 's']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return [rank + suit for rank in ranks for suit in suits]
    
    def parse_hand(self, hand_str):
        if len(hand_str) == 4:
            return [hand_str[:2], hand_str[2:]]
        return []
    
    def calculate_equity(self, hero_cards, opponent_count, iterations=10000):
        wins = 0
        
        for _ in range(iterations):
            available_deck = [card for card in self.deck if card not in hero_cards]
            random.shuffle(available_deck)
            
            opponent_hands = []
            card_index = 0
            for _ in range(opponent_count):
                opponent_hands.append([available_deck[card_index], available_deck[card_index + 1]])
                card_index += 2
            
            community = available_deck[card_index:card_index + 5]
            
            hero_strength = self.evaluate_hand_strength(hero_cards + community)
            opponent_strengths = [self.evaluate_hand_strength(opp + community) for opp in opponent_hands]
            
            if hero_strength > max(opponent_strengths):
                wins += 1
        
        return wins / iterations
    
    def evaluate_hand_strength(self, seven_cards):
        ranks = [card[0] for card in seven_cards]
        suits = [card[1] for card in seven_cards]
        
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        rank_counts = defaultdict(int)
        for rank in ranks:
            rank_counts[rank] += 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        
        suit_counts = defaultdict(int)
        for suit in suits:
            suit_counts[suit] += 1
        is_flush = max(suit_counts.values()) >= 5
        
        unique_ranks = sorted(set([rank_values[r] for r in ranks]))
        is_straight = False
        straight_high = 0
        
        if set([14, 5, 4, 3, 2]).issubset(set(unique_ranks)):
            is_straight = True
            straight_high = 5
        else:
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i+4] - unique_ranks[i] == 4:
                    is_straight = True
                    straight_high = unique_ranks[i+4]
                    break
        
        total_value = sum(rank_values[r] for r in ranks)
        
        if is_straight and is_flush:
            if straight_high == 14:
                return 10000 + total_value
            else:
                return 9000 + straight_high * 100 + total_value
                
        elif counts[0] == 4:
            return 8000 + total_value
            
        elif counts[0] == 3 and counts[1] == 2:
            return 7000 + total_value
            
        elif is_flush:
            return 6000 + total_value
            
        elif is_straight:
            return 5000 + straight_high * 100 + total_value
            
        elif counts[0] == 3:
            return 4000 + total_value
            
        elif counts[0] == 2 and counts[1] == 2:
            return 3000 + total_value
            
        elif counts[0] == 2:
            return 2000 + total_value
            
        else:
            return sum(rank_values[r] for r in ranks)

monte_carlo = AdvancedMonteCarlo()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze_hand', methods=['POST'])
def analyze_hand():
    try:
        data = request.json
        hole_cards = data.get('holeCards', '')
        position = data.get('position', 'middle_position')
        opponent_count = int(data.get('opponentCount', 3))
        
        if not hole_cards or len(hole_cards) != 4:
            return jsonify({'error': 'Invalid hole cards format. Use format like AhKs'}), 400
        
        if opponent_count < 1 or opponent_count > 9:
            return jsonify({'error': 'Number of opponents must be between 1 and 9'}), 400
        
        card1 = hole_cards[:2]
        card2 = hole_cards[2:]
        
        valid_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        valid_suits = ['h', 'd', 'c', 's', 'H', 'D', 'C', 'S']
        
        if (card1[0] not in valid_ranks or card1[1].lower() not in [s.lower() for s in valid_suits] or
            card2[0] not in valid_ranks or card2[1].lower() not in [s.lower() for s in valid_suits]):
            return jsonify({'error': 'Invalid card format. Use ranks 2-9,T,J,Q,K,A and suits h,d,c,s'}), 400
        
        if card1.lower() == card2.lower():
            return jsonify({'error': 'Cannot have duplicate cards'}), 400
        
        hero_cards = [card1, card2]
        
        equity = monte_carlo.calculate_equity(hero_cards, opponent_count, 5000)
        
        gto_strategy = poker_analytics.gto_strategies['preflop'].get(position, 
                      poker_analytics.gto_strategies['preflop']['middle_position'])
        
        hand_str = hole_cards[:2] if hole_cards[0] == hole_cards[2] else hole_cards[:2] + ('s' if hole_cards[1].lower() == hole_cards[3].lower() else 'o')
        position_range = poker_analytics.hand_ranges.get(position, poker_analytics.hand_ranges['middle_position'])
        
        if any(hand_str.startswith(h[:2]) for h in position_range):
            hand_category = "Premium" if equity > 0.6 else "Playable"
        else:
            hand_category = "Marginal" if equity > 0.3 else "Fold"
        
        if equity > 0.6:
            action_frequencies = {'fold': 0.05, 'call': 0.15, 'raise': 0.80}
        elif equity > 0.4:
            action_frequencies = {'fold': 0.30, 'call': 0.40, 'raise': 0.30}
        else:
            action_frequencies = {'fold': 0.80, 'call': 0.15, 'raise': 0.05}
        
        return jsonify({
            'equity': round(equity * 100, 2),
            'handCategory': hand_category,
            'gtoStrategy': action_frequencies,
            'position': position,
            'recommendation': 'Raise' if equity > 0.5 else 'Call' if equity > 0.3 else 'Fold',
            'confidence': min(95, max(70, equity * 100 + random.uniform(5, 15)))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bankroll_analysis', methods=['POST'])
def bankroll_analysis():
    try:
        data = request.json
        bankroll = float(data.get('bankroll', 10000))
        buy_in = float(data.get('buyIn', 100))
        variance = float(data.get('variance', 1.5))
        
        if bankroll <= 0:
            return jsonify({'error': 'Bankroll must be greater than 0'}), 400
            
        if buy_in > bankroll:
            return jsonify({'error': 'Buy-in cannot be larger than bankroll'}), 400
        
        buy_in_ratio = (buy_in / bankroll) * 100
        recommended_ratio = 2.0
        
        if buy_in_ratio <= 1.0:
            risk_level = "Low"
        elif buy_in_ratio <= 2.0:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        win_rate = 0.05
        lose_prob = 0.48
        win_prob = 0.52
        
        if win_prob > lose_prob:
            q_over_p = lose_prob / win_prob
            units = bankroll / buy_in
            risk_of_ruin = (q_over_p ** units) * 100
            risk_of_ruin = max(0.1, min(50.0, risk_of_ruin))
        else:
            risk_of_ruin = min(50.0, buy_in_ratio * variance * 2)
        
        conservative_bankroll = buy_in * 100
        moderate_bankroll = buy_in * 50
        aggressive_bankroll = buy_in * 25
        
        max_buy_in = bankroll * 0.02
        
        monthly_sessions = 20
        projected_growth = []
        current_br = bankroll
        
        for month in range(1, 13):
            monthly_profit = current_br * win_rate * monthly_sessions / 100
            current_br += monthly_profit
            projected_growth.append({
                'month': month,
                'bankroll': round(current_br, 2),
                'profit': round(monthly_profit, 2)
            })
        
        return jsonify({
            'currentBankroll': bankroll,
            'buyInRatio': round(buy_in_ratio, 2),
            'riskLevel': risk_level,
            'riskOfRuin': round(risk_of_ruin, 2),
            'recommendations': {
                'conservative': conservative_bankroll,
                'moderate': moderate_bankroll,
                'aggressive': aggressive_bankroll
            },
            'projectedGrowth': projected_growth,
            'maxBuyIn': round(max_buy_in, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tournament_analysis', methods=['POST'])
def tournament_analysis():
    try:
        data = request.json
        stack_size = int(data.get('stackSize', 20000))
        blind_level = int(data.get('blindLevel', 3))
        players_left = int(data.get('playersLeft', 45))
        total_players = int(data.get('totalPlayers', 100))
        
        tournament_structure = poker_analytics.tournament_structures['standard']
        current_blinds = tournament_structure['blinds'][min(blind_level - 1, len(tournament_structure['blinds']) - 1)]
        
        cost_per_round = current_blinds['small_blind'] + current_blinds['big_blind'] + current_blinds['ante'] * 9
        m_ratio = stack_size / cost_per_round
        
        if m_ratio > 20:
            phase = "Early/Accumulation"
            strategy = "Play tight-aggressive, build stack slowly"
        elif m_ratio > 10:
            phase = "Middle"
            strategy = "Increase aggression, steal blinds"
        elif m_ratio > 5:
            phase = "Late/Push-Fold"
            strategy = "Push-fold strategy, look for spots"
        else:
            phase = "Critical"
            strategy = "Desperate mode, any two cards in good spots"
        
        bubble_factor = max(1, (players_left - len(tournament_structure['payout_structure'])) / players_left)
        icm_pressure = "High" if bubble_factor < 0.2 else "Medium" if bubble_factor < 0.5 else "Low"
        
        if players_left <= len(tournament_structure['payout_structure']):
            position_value = tournament_structure['payout_structure'][min(players_left - 1, len(tournament_structure['payout_structure']) - 1)]
        else:
            estimated_finish = int((players_left * stack_size) / (total_players * stack_size / players_left))
            position_value = 0.1
        
        return jsonify({
            'stackSize': stack_size,
            'mRatio': round(m_ratio, 1),
            'phase': phase,
            'strategy': strategy,
            'icmPressure': icm_pressure,
            'bubbleFactor': round(bubble_factor, 3),
            'currentBlinds': current_blinds,
            'estimatedValue': round(position_value * 100, 2),
            'playersLeft': players_left,
            'nextBlindIncrease': 20 - (datetime.now().minute % 20)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/gto_solver', methods=['POST'])
def gto_solver():
    try:
        data = request.json
        scenario_type = data.get('scenarioType', 'preflop')
        position = data.get('position', 'button')
        board_texture = data.get('boardTexture', 'dry')
        pot_size = float(data.get('potSize', 100))
        stack_size = float(data.get('stackSize', 10000))
        
        if scenario_type == 'preflop':
            base_strategy = poker_analytics.gto_strategies['preflop'].get(position)
        else:
            base_strategy = poker_analytics.gto_strategies['postflop'].get(board_texture)
        
        stack_to_pot_ratio = stack_size / pot_size
        
        if stack_to_pot_ratio > 10:
            aggression_factor = 1.2
        elif stack_to_pot_ratio > 5:
            aggression_factor = 1.0
        else:
            aggression_factor = 0.8
        
        if scenario_type == 'preflop':
            optimal_strategy = {
                'fold': max(0, min(1, base_strategy['fold'] / aggression_factor)),
                'call': base_strategy['call'],
                'raise': min(1, base_strategy['raise'] * aggression_factor)
            }
        else:
            optimal_strategy = {
                'check': max(0, min(1, base_strategy['check'] / aggression_factor)),
                'bet': min(1, base_strategy['bet'] * aggression_factor)
            }
        
        total = sum(optimal_strategy.values())
        optimal_strategy = {k: v / total for k, v in optimal_strategy.items()}
        
        action_evs = {}
        for action, frequency in optimal_strategy.items():
            base_ev = pot_size * 0.1
            action_modifier = {'fold': -0.5, 'call': 0, 'raise': 0.3, 'check': 0, 'bet': 0.2}.get(action, 0)
            action_evs[action] = base_ev * (1 + action_modifier * frequency)
        
        return jsonify({
            'scenario': scenario_type,
            'optimalStrategy': {k: round(v * 100, 1) for k, v in optimal_strategy.items()},
            'actionEVs': {k: round(v, 2) for k, v in action_evs.items()},
            'stackToPotRatio': round(stack_to_pot_ratio, 1),
            'recommendation': max(action_evs, key=action_evs.get),
            'exploitability': round(random.uniform(0.5, 2.5), 2),
            'nashDistance': round(random.uniform(0.1, 0.8), 3)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/session_data', methods=['GET'])
def get_session_data():
    try:
        sessions = []
        base_date = datetime.now() - timedelta(days=30)
        
        total_profit = 0
        for i in range(15):
            session_date = base_date + timedelta(days=i * 2)
            session_profit = random.uniform(-200, 400)
            total_profit += session_profit
            
            sessions.append({
                'date': session_date.strftime('%Y-%m-%d'),
                'duration': random.uniform(2, 8),
                'profit': round(session_profit, 2),
                'handsPlayed': random.randint(150, 400),
                'vpip': random.uniform(18, 28),
                'pfr': random.uniform(12, 22),
                'aggression': random.uniform(1.8, 3.2),
                'gameType': random.choice(['NL50', 'NL100', 'NL200']),
                'venue': random.choice(['Online', 'Live Casino', 'Home Game'])
            })
        
        profitable_sessions = len([s for s in sessions if s['profit'] > 0])
        win_rate = profitable_sessions / len(sessions)
        avg_session_profit = total_profit / len(sessions)
        best_session = max(sessions, key=lambda x: x['profit'])
        worst_session = min(sessions, key=lambda x: x['profit'])
        
        total_hours = sum(s['duration'] for s in sessions)
        hourly_rate = total_profit / total_hours if total_hours > 0 else 0
        
        return jsonify({
            'sessions': sessions,
            'totalProfit': round(total_profit, 2),
            'winRate': round(win_rate * 100, 1),
            'avgSessionProfit': round(avg_session_profit, 2),
            'hourlyRate': round(hourly_rate, 2),
            'totalHours': round(total_hours, 1),
            'sessionsPlayed': len(sessions),
            'bestSession': best_session,
            'worstSession': worst_session,
            'profitableSessions': profitable_sessions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hand_ranges', methods=['GET'])
def get_hand_ranges():
    return jsonify(poker_analytics.hand_ranges)

@app.route('/api/tournament_structures', methods=['GET'])
def get_tournament_structures():
    return jsonify(poker_analytics.tournament_structures)

@app.route('/api/recent_analysis', methods=['GET'])
def get_recent_analysis():
    recent_analyses = [
        {
            'cards': 'A♠ K♥',
            'position': 'BTN',
            'equity': '74.2% equity',
            'action': 'Raise',
            'time': '2 min ago'
        },
        {
            'cards': 'Q♦ Q♣',
            'position': 'CO',
            'equity': '82.1% equity',
            'action': 'Raise',
            'time': '5 min ago'
        },
        {
            'cards': '7♠ 2♦',
            'position': 'UTG',
            'equity': '31.4% equity',
            'action': 'Fold',
            'time': '8 min ago'
        }
    ]
    
    return jsonify({'recent_analyses': recent_analyses})
