import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from scipy import stats

@dataclass
class BankrollConfig:
    """Bankroll management configuration"""
    conservative_buyins: int = 40
    standard_buyins: int = 25
    aggressive_buyins: int = 15
    stop_loss_buyins: int = 5
    move_up_buyins: int = 30
    move_down_buyins: int = 15

class BankrollManager:
    """Advanced bankroll management system"""
    
    def __init__(self):
        self.config = BankrollConfig()
        self.risk_tolerance_multipliers = {
            'conservative': 1.5,
            'standard': 1.0,
            'aggressive': 0.7,
            'ultra_aggressive': 0.5
        }
    
    def analyze_scenario(self, game_state: Dict, simulation_results: Dict) -> Dict:
        """Analyze bankroll impact of current scenario"""
        stack_size = game_state.get('stack_size', 100)
        game_type = game_state.get('game_type', 'cash')
        big_blind = game_state.get('big_blind', 1)
        
        # Calculate effective buy-in
        buyin_amount = stack_size * big_blind
        
        # Get simulation risk metrics
        risk_metrics = self.calculate_risk_metrics(simulation_results, buyin_amount)
        
        # Assess bankroll requirements
        bankroll_requirements = self.calculate_bankroll_requirements(buyin_amount, risk_metrics)
        
        # Generate recommendations
        recommendations = self.generate_bankroll_recommendations(risk_metrics, bankroll_requirements)
        
        return {
            'buyin_amount': buyin_amount,
            'risk_metrics': risk_metrics,
            'bankroll_requirements': bankroll_requirements,
            'recommendations': recommendations,
            'kelly_criterion': self.calculate_kelly_criterion(simulation_results),
            'risk_of_ruin': self.calculate_risk_of_ruin(risk_metrics, bankroll_requirements)
        }
    
    def calculate_risk_metrics(self, simulation_results: Dict, buyin_amount: float) -> Dict:
        """Calculate risk metrics from simulation results"""
        # Extract key metrics from simulation
        expected_value = simulation_results.get('expected_value', 0)
        win_rate = simulation_results.get('win_rate', 0.5)
        variance = simulation_results.get('variance', 100)
        
        # Calculate standard deviation in buyins
        std_dev_buyins = math.sqrt(variance) / buyin_amount if buyin_amount > 0 else 0
        
        # Calculate downswing metrics
        max_drawdown = self.estimate_max_drawdown(win_rate, std_dev_buyins)
        percentile_drawdowns = self.calculate_drawdown_percentiles(std_dev_buyins)
        
        return {
            'expected_value_bb': expected_value,
            'win_rate': win_rate,
            'std_dev_buyins': std_dev_buyins,
            'max_drawdown_buyins': max_drawdown,
            'drawdown_percentiles': percentile_drawdowns,
            'sharpe_ratio': self.calculate_sharpe_ratio(expected_value, std_dev_buyins),
            'volatility_category': self.categorize_volatility(std_dev_buyins)
        }
    
    def estimate_max_drawdown(self, win_rate: float, std_dev: float) -> float:
        """Estimate maximum expected drawdown"""
        # Empirical formula for poker downswings
        if win_rate <= 0:
            return float('inf')
        
        # Convert win rate to edge
        edge = (win_rate - 0.5) * 2
        
        # Estimate using Kelly criterion and volatility
        if edge > 0:
            kelly_fraction = edge / (std_dev ** 2) if std_dev > 0 else 0
            max_drawdown = (std_dev ** 2) / (2 * edge) * math.log(1 / 0.01)  # 1% risk level
            return min(max_drawdown, 100)  # Cap at 100 buyins
        else:
            return 50  # Default for break-even players
    
    def calculate_drawdown_percentiles(self, std_dev: float) -> Dict:
        """Calculate drawdown at different percentiles"""
        # Using normal distribution approximation
        percentiles = [50, 75, 90, 95, 99]
        drawdowns = {}
        
        for p in percentiles:
            z_score = stats.norm.ppf(p / 100)
            drawdown = z_score * std_dev * 0.5  # Empirical adjustment
            drawdowns[f'{p}th_percentile'] = max(0, drawdown)
        
        return drawdowns
    
    def calculate_sharpe_ratio(self, expected_value: float, std_dev: float) -> float:
        """Calculate Sharpe ratio for the strategy"""
        if std_dev == 0:
            return 0
        
        # Convert to per-session metrics
        return expected_value / std_dev if std_dev > 0 else 0
    
    def categorize_volatility(self, std_dev: float) -> str:
        """Categorize volatility level"""
        if std_dev < 1.0:
            return 'low'
        elif std_dev < 2.0:
            return 'medium'
        elif std_dev < 3.0:
            return 'high'
        else:
            return 'very_high'
    
    def calculate_bankroll_requirements(self, buyin_amount: float, risk_metrics: Dict) -> Dict:
        """Calculate bankroll requirements for different risk tolerances"""
        max_drawdown = risk_metrics['max_drawdown_buyins']
        volatility = risk_metrics['volatility_category']
        
        # Base requirements
        base_buyins = {
            'conservative': self.config.conservative_buyins,
            'standard': self.config.standard_buyins,
            'aggressive': self.config.aggressive_buyins
        }
        
        # Adjust for volatility
        volatility_multipliers = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3,
            'very_high': 1.6
        }
        
        multiplier = volatility_multipliers.get(volatility, 1.0)
        
        requirements = {}
        for risk_level, base in base_buyins.items():
            adjusted_buyins = base * multiplier
            # Ensure we have enough for estimated drawdowns
            min_buyins = max_drawdown * 2 + 10  # 2x drawdown plus buffer
            final_buyins = max(adjusted_buyins, min_buyins)
            
            requirements[risk_level] = {
                'buyins_required': int(final_buyins),
                'total_amount': final_buyins * buyin_amount,
                'safety_margin': final_buyins - max_drawdown
            }
        
        return requirements
    
    def generate_bankroll_recommendations(self, risk_metrics: Dict, requirements: Dict) -> List[str]:
        """Generate bankroll management recommendations"""
        recommendations = []
        
        volatility = risk_metrics['volatility_category']
        max_drawdown = risk_metrics['max_drawdown_buyins']
        sharpe_ratio = risk_metrics['sharpe_ratio']
        
        # Volatility-based recommendations
        if volatility == 'very_high':
            recommendations.extend([
                "Very high volatility detected - consider moving down stakes",
                "Increase bankroll requirements by 50-60%",
                "Implement strict stop-loss limits"
            ])
        elif volatility == 'high':
            recommendations.extend([
                "High volatility - maintain larger bankroll buffer",
                "Consider more conservative play style"
            ])
        elif volatility == 'low':
            recommendations.append("Low volatility allows for more aggressive bankroll management")
        
        # Drawdown-based recommendations
        if max_drawdown > 30:
            recommendations.extend([
                f"Estimated maximum drawdown: {max_drawdown:.1f} buyins",
                "Consider bankroll of at least 2x estimated drawdown",
                "Implement mental game preparation for downswings"
            ])
        
        # Performance-based recommendations
        if sharpe_ratio > 1.0:
            recommendations.append("Excellent risk-adjusted returns - consider moving up stakes")
        elif sharpe_ratio < 0.5:
            recommendations.append("Poor risk-adjusted returns - review strategy")
        
        # General recommendations
        recommendations.extend([
            "Track all sessions for accurate bankroll analysis",
            "Review bankroll strategy monthly",
            "Never play above your bankroll limits"
        ])
        
        return recommendations
    
    def calculate_kelly_criterion(self, simulation_results: Dict) -> Dict:
        """Calculate Kelly Criterion for optimal bet sizing"""
        win_rate = simulation_results.get('win_rate', 0.5)
        avg_win = simulation_results.get('avg_win_amount', 1)
        avg_loss = simulation_results.get('avg_loss_amount', 1)
        
        if avg_loss == 0:
            return {'kelly_fraction': 0, 'recommendation': 'insufficient_data'}
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received (avg_win/avg_loss), p = win probability, q = loss probability
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b if b > 0 else 0
        
        # Adjust for poker (fractional Kelly)
        recommended_fraction = kelly_fraction * 0.25  # 1/4 Kelly for safety
        
        return {
            'full_kelly': kelly_fraction,
            'recommended_fraction': max(0, recommended_fraction),
            'interpretation': self.interpret_kelly(kelly_fraction),
            'bankroll_percentage': min(recommended_fraction * 100, 5)  # Cap at 5%
        }
    
    def interpret_kelly(self, kelly_fraction: float) -> str:
        """Interpret Kelly Criterion result"""
        if kelly_fraction <= 0:
            return "No edge detected - do not play this game"
        elif kelly_fraction < 0.05:
            return "Very small edge - extremely conservative betting"
        elif kelly_fraction < 0.15:
            return "Small edge - conservative betting recommended"
        elif kelly_fraction < 0.30:
            return "Good edge - moderate betting appropriate"
        else:
            return "Strong edge - aggressive betting possible (but use fractional Kelly)"
    
    def calculate_risk_of_ruin(self, risk_metrics: Dict, requirements: Dict) -> Dict:
        """Calculate risk of ruin for different bankroll sizes"""
        win_rate = risk_metrics['win_rate']
        std_dev = risk_metrics['std_dev_buyins']
        
        risk_calculations = {}
        
        for risk_level, req in requirements.items():
            bankroll_buyins = req['buyins_required']
            
            # Simplified risk of ruin calculation
            if win_rate <= 0.5:
                risk_of_ruin = 1.0  # Certain ruin for losing players
            else:
                # Using approximation for risk of ruin
                edge = (win_rate - 0.5) * 2
                variance_factor = std_dev ** 2
                
                # Risk of ruin formula approximation
                if edge > 0 and variance_factor > 0:
                    alpha = math.sqrt(variance_factor / edge)
                    risk_of_ruin = math.exp(-2 * bankroll_buyins / alpha)
                else:
                    risk_of_ruin = 1.0
            
            risk_calculations[risk_level] = {
                'risk_percentage': min(risk_of_ruin * 100, 100),
                'recommendation': self.interpret_risk_of_ruin(risk_of_ruin)
            }
        
        return risk_calculations
    
    def interpret_risk_of_ruin(self, risk: float) -> str:
        """Interpret risk of ruin percentage"""
        risk_pct = risk * 100
        
        if risk_pct < 1:
            return "Very safe - excellent bankroll management"
        elif risk_pct < 5:
            return "Safe - good bankroll management"
        elif risk_pct < 10:
            return "Moderate risk - acceptable for experienced players"
        elif risk_pct < 25:
            return "High risk - consider larger bankroll"
        else:
            return "Very high risk - inadequate bankroll"
    
    def run_simulation(self, bankroll_params: Dict) -> Dict:
        """Run comprehensive bankroll simulation"""
        starting_bankroll = bankroll_params.get('starting_bankroll', 10000)
        stake_level = bankroll_params.get('stake_level', 100)
        win_rate_bb100 = bankroll_params.get('win_rate', 5)  # bb/100 hands
        std_deviation = bankroll_params.get('std_deviation', 80)  # bb/100 hands
        sessions = bankroll_params.get('sessions', 1000)
        strategy = bankroll_params.get('strategy', 'conservative')
        
        # Convert parameters
        session_hands = 100  # Assume 100 hands per session
        session_mean = win_rate_bb100 * session_hands / 100 * (stake_level / 100)
        session_std = std_deviation * session_hands / 100 * (stake_level / 100)
        
        # Run Monte Carlo simulation
        simulation_results = self.monte_carlo_bankroll_simulation(
            starting_bankroll, session_mean, session_std, sessions, stake_level, strategy
        )
        
        # Calculate additional metrics
        final_analysis = self.analyze_simulation_results(simulation_results, bankroll_params)
        
        return {
            'simulation_parameters': bankroll_params,
            'raw_results': simulation_results,
            'analysis': final_analysis,
            'recommendations': self.generate_simulation_recommendations(final_analysis)
        }
    
    def monte_carlo_bankroll_simulation(self, starting_bankroll: float, session_mean: float, 
                                      session_std: float, sessions: int, stake_level: float, 
                                      strategy: str) -> Dict:
        """Run Monte Carlo simulation of bankroll progression"""
        num_simulations = 1000
        results = []
        
        # Bankroll management rules based on strategy
        move_up_threshold = self.get_move_up_threshold(strategy, stake_level)
        move_down_threshold = self.get_move_down_threshold(strategy, stake_level)
        stop_loss_threshold = self.get_stop_loss_threshold(strategy, stake_level)
        
        for sim in range(num_simulations):
            bankroll = starting_bankroll
            current_stake = stake_level
            bankroll_history = [bankroll]
            stake_history = [current_stake]
            busted = False
            max_bankroll = bankroll
            max_drawdown = 0
            
            for session in range(sessions):
                if bankroll < stop_loss_threshold:
                    busted = True
                    break
                
                # Adjust stakes based on bankroll
                new_stake = self.adjust_stake_level(bankroll, current_stake, strategy)
                if new_stake != current_stake:
                    current_stake = new_stake
                    # Recalculate session parameters for new stake
                    session_mean_adj = session_mean * (current_stake / stake_level)
                    session_std_adj = session_std * (current_stake / stake_level)
                else:
                    session_mean_adj = session_mean
                    session_std_adj = session_std
                
                # Generate session result
                session_result = np.random.normal(session_mean_adj, session_std_adj)
                bankroll += session_result
                
                # Track metrics
                max_bankroll = max(max_bankroll, bankroll)
                current_drawdown = (max_bankroll - bankroll) / max_bankroll * 100
                max_drawdown = max(max_drawdown, current_drawdown)
                
                bankroll_history.append(bankroll)
                stake_history.append(current_stake)
            
            results.append({
                'final_bankroll': bankroll,
                'max_bankroll': max_bankroll,
                'max_drawdown_pct': max_drawdown,
                'busted': busted,
                'bankroll_history': bankroll_history,
                'stake_history': stake_history,
                'profit_loss': bankroll - starting_bankroll,
                'roi': (bankroll - starting_bankroll) / starting_bankroll * 100
            })
        
        return self.aggregate_simulation_results(results, starting_bankroll)
    
    def get_move_up_threshold(self, strategy: str, stake_level: float) -> float:
        """Get threshold for moving up in stakes"""
        multipliers = {
            'conservative': 40,
            'standard': 30,
            'aggressive': 20,
            'ultra_aggressive': 15
        }
        return stake_level * multipliers.get(strategy, 30)
    
    def get_move_down_threshold(self, strategy: str, stake_level: float) -> float:
        """Get threshold for moving down in stakes"""
        multipliers = {
            'conservative': 25,
            'standard': 20,
            'aggressive': 15,
            'ultra_aggressive': 10
        }
        return stake_level * multipliers.get(strategy, 20)
    
    def get_stop_loss_threshold(self, strategy: str, stake_level: float) -> float:
        """Get stop loss threshold"""
        multipliers = {
            'conservative': 10,
            'standard': 8,
            'aggressive': 5,
            'ultra_aggressive': 3
        }
        return stake_level * multipliers.get(strategy, 8)
    
    def adjust_stake_level(self, bankroll: float, current_stake: float, strategy: str) -> float:
        """Adjust stake level based on bankroll"""
        # Define stake levels (in big blinds)
        stake_levels = [25, 50, 100, 200, 500, 1000, 2000]
        
        # Find appropriate stake level
        move_up_mult = self.get_move_up_threshold(strategy, 1) 
        move_down_mult = self.get_move_down_threshold(strategy, 1)
        
        for stake in stake_levels:
            if bankroll >= stake * move_up_mult:
                continue
            elif bankroll >= stake * move_down_mult:
                return min(stake, current_stake)  # Don't move up too aggressively
            else:
                # Find lower stake level
                lower_stakes = [s for s in stake_levels if s < stake]
                if lower_stakes:
                    return lower_stakes[-1]
                else:
                    return stake_levels[0]
        
        return stake_levels[-1]  # Highest stakes
    
    def aggregate_simulation_results(self, results: List[Dict], starting_bankroll: float) -> Dict:
        """Aggregate Monte Carlo simulation results"""
        final_bankrolls = [r['final_bankroll'] for r in results]
        max_drawdowns = [r['max_drawdown_pct'] for r in results]
        rois = [r['roi'] for r in results]
        bust_rate = sum(1 for r in results if r['busted']) / len(results) * 100
        
        return {
            'success_rate': 100 - bust_rate,
            'bust_rate': bust_rate,
            'final_bankroll_stats': {
                'mean': np.mean(final_bankrolls),
                'median': np.median(final_bankrolls),
                'std': np.std(final_bankrolls),
                'min': np.min(final_bankrolls),
                'max': np.max(final_bankrolls),
                'percentiles': {
                    '10th': np.percentile(final_bankrolls, 10),
                    '25th': np.percentile(final_bankrolls, 25),
                    '75th': np.percentile(final_bankrolls, 75),
                    '90th': np.percentile(final_bankrolls, 90)
                }
            },
            'drawdown_stats': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'max': np.max(max_drawdowns),
                'percentiles': {
                    '90th': np.percentile(max_drawdowns, 90),
                    '95th': np.percentile(max_drawdowns, 95),
                    '99th': np.percentile(max_drawdowns, 99)
                }
            },
            'roi_stats': {
                'mean': np.mean(rois),
                'median': np.median(rois),
                'std': np.std(rois),
                'positive_roi_rate': sum(1 for roi in rois if roi > 0) / len(rois) * 100
            }
        }
    
    def analyze_simulation_results(self, simulation_results: Dict, bankroll_params: Dict) -> Dict:
        """Analyze simulation results for insights"""
        success_rate = simulation_results['success_rate']
        bust_rate = simulation_results['bust_rate']
        roi_stats = simulation_results['roi_stats']
        drawdown_stats = simulation_results['drawdown_stats']
        
        # Performance assessment
        if success_rate > 95:
            performance = 'excellent'
        elif success_rate > 85:
            performance = 'good'
        elif success_rate > 70:
            performance = 'acceptable'
        else:
            performance = 'poor'
        
        # Risk assessment
        max_drawdown_99th = drawdown_stats['percentiles']['99th']
        if max_drawdown_99th < 30:
            risk_level = 'low'
        elif max_drawdown_99th < 50:
            risk_level = 'medium'
        elif max_drawdown_99th < 70:
            risk_level = 'high'
        else:
            risk_level = 'very_high'
        
        return {
            'performance_rating': performance,
            'risk_rating': risk_level,
            'expected_roi': roi_stats['mean'],
            'roi_volatility': roi_stats['std'],
            'probability_of_profit': roi_stats['positive_roi_rate'],
            'worst_case_drawdown': max_drawdown_99th,
            'strategy_viability': self.assess_strategy_viability(success_rate, roi_stats['mean']),
            'key_insights': self.generate_key_insights(simulation_results, bankroll_params)
        }
    
    def assess_strategy_viability(self, success_rate: float, expected_roi: float) -> str:
        """Assess overall strategy viability"""
        if success_rate > 90 and expected_roi > 50:
            return 'highly_viable'
        elif success_rate > 80 and expected_roi > 20:
            return 'viable'
        elif success_rate > 70 and expected_roi > 0:
            return 'marginally_viable'
        else:
            return 'not_viable'
    
    def generate_key_insights(self, simulation_results: Dict, bankroll_params: Dict) -> List[str]:
        """Generate key insights from simulation"""
        insights = []
        
        success_rate = simulation_results['success_rate']
        expected_roi = simulation_results['roi_stats']['mean']
        max_drawdown = simulation_results['drawdown_stats']['percentiles']['99th']
        
        # Success rate insights
        if success_rate < 80:
            insights.append(f"High bust rate ({100-success_rate:.1f}%) indicates insufficient bankroll")
        
        # ROI insights
        if expected_roi < 0:
            insights.append("Negative expected ROI suggests unprofitable play")
        elif expected_roi > 100:
            insights.append("Very high expected ROI - verify win rate assumptions")
        
        # Drawdown insights
        if max_drawdown > 60:
            insights.append(f"Severe drawdowns possible (up to {max_drawdown:.1f}%)")
        
        # Strategy insights
        strategy = bankroll_params.get('strategy', 'conservative')
        if strategy == 'aggressive' and success_rate < 85:
            insights.append("Aggressive bankroll management too risky for this win rate")
        
        return insights
    
    def generate_simulation_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on simulation analysis"""
        recommendations = []
        
        performance = analysis['performance_rating']
        risk_level = analysis['risk_rating']
        viability = analysis['strategy_viability']
        
        # Performance-based recommendations
        if performance == 'poor':
            recommendations.extend([
                "Increase starting bankroll significantly",
                "Consider more conservative strategy",
                "Review win rate assumptions"
            ])
        elif performance == 'excellent':
            recommendations.append("Current bankroll strategy is well-optimized")
        
        # Risk-based recommendations
        if risk_level == 'very_high':
            recommendations.extend([
                "Reduce risk exposure",
                "Implement stricter stop-loss rules",
                "Consider professional bankroll coaching"
            ])
        elif risk_level == 'low':
            recommendations.append("Risk level allows for potential strategy optimization")
        
        # Viability recommendations
        if viability == 'not_viable':
            recommendations.extend([
                "Current strategy not recommended",
                "Fundamental changes needed",
                "Consider moving down in stakes"
            ])
        elif viability == 'highly_viable':
            recommendations.append("Strategy shows excellent long-term prospects")
        
        return recommendations