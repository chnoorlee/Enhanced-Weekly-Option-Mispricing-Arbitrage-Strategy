import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Try to add system font if available
try:
    matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
except:
    pass

class AmericanOptionPricer:
    """American Option Pricer using Binomial Tree Model"""
    
    def __init__(self, S, K, T, r, sigma, option_type='call', steps=100):
        self.S = S  # Underlying price
        self.K = K  # Strike price
        self.T = T  # Time to expiration
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.option_type = option_type
        self.steps = steps
        
    def price(self):
        """Calculate American option price"""
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        # Initialize price tree
        prices = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            for j in range(i + 1):
                prices[j, i] = self.S * (u ** (i - j)) * (d ** j)
        
        # Initialize option value tree
        option_values = np.zeros((self.steps + 1, self.steps + 1))
        
        # Option values at expiration
        for j in range(self.steps + 1):
            if self.option_type == 'call':
                option_values[j, self.steps] = max(0, prices[j, self.steps] - self.K)
            else:
                option_values[j, self.steps] = max(0, self.K - prices[j, self.steps])
        
        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Holding value
                hold_value = np.exp(-self.r * dt) * (p * option_values[j, i + 1] + 
                                                    (1 - p) * option_values[j + 1, i + 1])
                
                # Exercise value
                if self.option_type == 'call':
                    exercise_value = max(0, prices[j, i] - self.K)
                else:
                    exercise_value = max(0, self.K - prices[j, i])
                
                # American option takes maximum
                option_values[j, i] = max(hold_value, exercise_value)
        
        return option_values[0, 0]
    
    def delta(self):
        """Calculate Delta"""
        h = 0.01
        price_up = AmericanOptionPricer(self.S + h, self.K, self.T, self.r, 
                                      self.sigma, self.option_type, self.steps).price()
        price_down = AmericanOptionPricer(self.S - h, self.K, self.T, self.r, 
                                        self.sigma, self.option_type, self.steps).price()
        return (price_up - price_down) / (2 * h)
    
    def gamma(self):
        """Calculate Gamma"""
        h = 0.01
        delta_up = AmericanOptionPricer(self.S + h, self.K, self.T, self.r, 
                                      self.sigma, self.option_type, self.steps).delta()
        delta_down = AmericanOptionPricer(self.S - h, self.K, self.T, self.r, 
                                        self.sigma, self.option_type, self.steps).delta()
        return (delta_up - delta_down) / (2 * h)
    
    def theta(self):
        """Calculate Theta (time decay)"""
        h = 1/365  # One day
        if self.T <= h:
            return 0
        price_now = self.price()
        price_tomorrow = AmericanOptionPricer(self.S, self.K, self.T - h, self.r, 
                                            self.sigma, self.option_type, self.steps).price()
        return price_tomorrow - price_now
    
    def vega(self):
        """Calculate Vega (volatility sensitivity)"""
        h = 0.01
        price_up = AmericanOptionPricer(self.S, self.K, self.T, self.r, 
                                      self.sigma + h, self.option_type, self.steps).price()
        price_down = AmericanOptionPricer(self.S, self.K, self.T, self.r, 
                                        self.sigma - h, self.option_type, self.steps).price()
        return (price_up - price_down) / (2 * h)

class EnhancedWeeklyOptionsArbitrage:
    """Enhanced Weekly Options Arbitrage Strategy"""
    
    def __init__(self, symbol='SPY', risk_free_rate=0.05):
        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.options_data = []
        self.transaction_cost = 0.002  # 0.2% transaction cost
        
    def fetch_data(self, start_date='2022-01-01', end_date='2024-01-01'):
        """Fetch stock data"""
        print(f"Fetching data for {self.symbol}...")
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=start_date, end=end_date)
        
        # Calculate historical volatility
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate additional technical indicators
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = self.calculate_rsi(self.data['Close'], 14)
        
        print(f"Data fetched successfully, {len(self.data)} trading days")
        return self.data
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_implied_volatility(self, option_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility"""
        def objective(sigma):
            try:
                theoretical_price = AmericanOptionPricer(S, K, T, r, sigma, option_type).price()
                return abs(theoretical_price - option_price)
            except:
                return float('inf')
        
        result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
        return result.x if result.success else 0.2
    
    def generate_weekly_options(self, current_date, current_price):
        """Generate weekly options data (simulated with enhanced pricing)"""
        T = 7 / 365  # 7 days to expiration
        
        # Generate different strike prices
        strikes = np.arange(current_price * 0.95, current_price * 1.05, current_price * 0.01)
        options = []
        
        for K in strikes:
            # Find closest date index for volatility
            time_diff = abs(self.data.index - current_date)
            vol_idx = time_diff.argmin()
            current_vol = self.data.iloc[vol_idx]['Volatility']
            
            if pd.isna(current_vol):
                current_vol = 0.2
            
            # Calculate theoretical option prices
            call_pricer = AmericanOptionPricer(current_price, K, T, self.risk_free_rate, 
                                             current_vol, 'call')
            put_pricer = AmericanOptionPricer(current_price, K, T, self.risk_free_rate, 
                                            current_vol, 'put')
            
            call_price = call_pricer.price()
            put_price = put_pricer.price()
            
            # Calculate Greeks
            call_delta = call_pricer.delta()
            call_gamma = call_pricer.gamma()
            call_theta = call_pricer.theta()
            call_vega = call_pricer.vega()
            
            put_delta = put_pricer.delta()
            put_gamma = put_pricer.gamma()
            put_theta = put_pricer.theta()
            put_vega = put_pricer.vega()
            
            # Enhanced market noise based on volatility and market conditions
            rsi = self.data.iloc[vol_idx]['RSI']
            vol_factor = current_vol / 0.2  # Normalize to base volatility
            
            # Higher noise in high volatility and extreme RSI conditions
            noise_factor = 0.01 + 0.02 * vol_factor
            if not pd.isna(rsi):
                if rsi > 70 or rsi < 30:  # Overbought/oversold conditions
                    noise_factor *= 1.5
            
            call_market_noise = np.random.normal(0, noise_factor)
            put_market_noise = np.random.normal(0, noise_factor)
            
            call_market_price = call_price * (1 + call_market_noise)
            put_market_price = put_price * (1 + put_market_noise)
            
            options.append({
                'date': current_date,
                'strike': K,
                'call_theoretical': call_price,
                'call_market': call_market_price,
                'put_theoretical': put_price,
                'put_market': put_market_price,
                'underlying_price': current_price,
                'volatility': current_vol,
                'call_delta': call_delta,
                'call_gamma': call_gamma,
                'call_theta': call_theta,
                'call_vega': call_vega,
                'put_delta': put_delta,
                'put_gamma': put_gamma,
                'put_theta': put_theta,
                'put_vega': put_vega,
                'rsi': rsi if not pd.isna(rsi) else 50
            })
        
        return options
    
    def find_enhanced_arbitrage_opportunities(self, options):
        """Find enhanced arbitrage opportunities with multiple strategies"""
        arbitrage_ops = []
        
        for option in options:
            call_diff = option['call_market'] - option['call_theoretical']
            put_diff = option['put_market'] - option['put_theoretical']
            
            # Dynamic threshold based on volatility and Greeks
            vol_factor = option['volatility'] / 0.2
            base_threshold = 0.03  # 3% base threshold
            dynamic_threshold = base_threshold * (0.5 + 0.5 * vol_factor)
            
            # Strategy 1: Simple price discrepancy arbitrage
            if abs(call_diff) > option['call_theoretical'] * dynamic_threshold:
                profit_potential = abs(call_diff) - option['call_theoretical'] * self.transaction_cost
                if profit_potential > 0:
                    arbitrage_ops.append({
                        'date': option['date'],
                        'strategy': 'price_discrepancy',
                        'type': 'call',
                        'strike': option['strike'],
                        'action': 'sell' if call_diff > 0 else 'buy',
                        'market_price': option['call_market'],
                        'theoretical_price': option['call_theoretical'],
                        'profit_potential': profit_potential,
                        'underlying_price': option['underlying_price'],
                        'delta': option['call_delta'],
                        'gamma': option['call_gamma'],
                        'theta': option['call_theta'],
                        'vega': option['call_vega']
                    })
            
            if abs(put_diff) > option['put_theoretical'] * dynamic_threshold:
                profit_potential = abs(put_diff) - option['put_theoretical'] * self.transaction_cost
                if profit_potential > 0:
                    arbitrage_ops.append({
                        'date': option['date'],
                        'strategy': 'price_discrepancy',
                        'type': 'put',
                        'strike': option['strike'],
                        'action': 'sell' if put_diff > 0 else 'buy',
                        'market_price': option['put_market'],
                        'theoretical_price': option['put_theoretical'],
                        'profit_potential': profit_potential,
                        'underlying_price': option['underlying_price'],
                        'delta': option['put_delta'],
                        'gamma': option['put_gamma'],
                        'theta': option['put_theta'],
                        'vega': option['put_vega']
                    })
            
            # Strategy 2: Volatility arbitrage
            if option['call_vega'] != 0:  # Avoid division by zero
                implied_vol_call = self.calculate_implied_volatility(
                    option['call_market'], option['underlying_price'], 
                    option['strike'], 7/365, self.risk_free_rate, 'call'
                )
                implied_vol_put = self.calculate_implied_volatility(
                    option['put_market'], option['underlying_price'], 
                    option['strike'], 7/365, self.risk_free_rate, 'put'
                )
                
                vol_diff_call = implied_vol_call - option['volatility']
                vol_diff_put = implied_vol_put - option['volatility']
                
                # If implied volatility significantly differs from historical
                if abs(vol_diff_call) > 0.05:  # 5% volatility difference
                    profit_potential = abs(vol_diff_call * option['call_vega'] * 100) - \
                                     option['call_theoretical'] * self.transaction_cost
                    if profit_potential > 0:
                        arbitrage_ops.append({
                            'date': option['date'],
                            'strategy': 'volatility_arbitrage',
                            'type': 'call',
                            'strike': option['strike'],
                            'action': 'sell' if vol_diff_call > 0 else 'buy',
                            'market_price': option['call_market'],
                            'theoretical_price': option['call_theoretical'],
                            'profit_potential': profit_potential,
                            'underlying_price': option['underlying_price'],
                            'vol_difference': vol_diff_call,
                            'delta': option['call_delta'],
                            'vega': option['call_vega']
                        })
                
                # Add similar check for put options
                if abs(vol_diff_put) > 0.05:  # 5% volatility difference
                    profit_potential = abs(vol_diff_put * option['put_vega'] * 100) - \
                                     option['put_theoretical'] * self.transaction_cost
                    if profit_potential > 0:
                        arbitrage_ops.append({
                            'date': option['date'],
                            'strategy': 'volatility_arbitrage',
                            'type': 'put',
                            'strike': option['strike'],
                            'action': 'sell' if vol_diff_put > 0 else 'buy',
                            'market_price': option['put_market'],
                            'theoretical_price': option['put_theoretical'],
                            'profit_potential': profit_potential,
                            'underlying_price': option['underlying_price'],
                            'vol_difference': vol_diff_put,
                            'delta': option['put_delta'],
                            'vega': option['put_vega']
                        })
        
        return arbitrage_ops
    
    def backtest_enhanced_strategy(self, start_date='2022-01-01', end_date='2024-01-01'):
        """Backtest enhanced arbitrage strategy"""
        if self.data is None:
            self.fetch_data(start_date, end_date)
        
        print("Starting enhanced arbitrage strategy backtest...")
        
        # Select Friday trading dates (weekly options expiration)
        trading_dates = self.data[self.data.index.dayofweek == 4].index
        
        portfolio_value = [100000]  # Initial capital $100k
        daily_returns = []
        arbitrage_trades = []
        detailed_trades = []
        
        for i, date in enumerate(trading_dates[:min(len(trading_dates), 100)]):
            current_price = self.data.loc[date, 'Close']
            
            # Generate options data for current date
            options = self.generate_weekly_options(date, current_price)
            
            # Find arbitrage opportunities
            arbitrage_ops = self.find_enhanced_arbitrage_opportunities(options)
            
            if arbitrage_ops:
                # Select top 3 opportunities by profit potential
                top_ops = sorted(arbitrage_ops, key=lambda x: x['profit_potential'], reverse=True)[:3]
                
                total_profit = 0
                for op in top_ops:
                    # Enhanced position sizing using simplified Kelly criterion
                    win_prob = 0.6  # Estimated win probability
                    avg_win = op['profit_potential']
                    avg_loss = op['market_price'] * self.transaction_cost
                    
                    if avg_loss > 0:
                        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
                        kelly_fraction = max(0.01, min(0.1, kelly_fraction))  # Cap between 1% and 10%
                    else:
                        kelly_fraction = 0.05
                    
                    position_size = portfolio_value[-1] * kelly_fraction
                    contracts = max(1, int(position_size / (op['market_price'] * 100)))
                    
                    profit = op['profit_potential'] * contracts
                    total_profit += profit
                    
                    # Add symbol information for multi-stock tracking
                    trade_record = {
                        'symbol': self.symbol,  # Add symbol information
                        'date': op['date'],
                        'strategy': op['strategy'],
                        'type': op['type'],
                        'strike': op['strike'],
                        'action': op['action'],
                        'contracts': contracts,
                        'profit': profit,
                        'delta': op.get('delta', 0),
                        'gamma': op.get('gamma', 0),
                        'theta': op.get('theta', 0),
                        'vega': op.get('vega', 0)
                    }
                    detailed_trades.append(trade_record)
                
                arbitrage_trades.extend(top_ops)
                portfolio_value.append(portfolio_value[-1] + total_profit)
                daily_returns.append(total_profit / portfolio_value[-2])
            else:
                portfolio_value.append(portfolio_value[-1])
                daily_returns.append(0)
        
        self.portfolio_value = portfolio_value
        self.daily_returns = daily_returns
        self.arbitrage_trades = arbitrage_trades
        self.detailed_trades = detailed_trades
        self.trading_dates = trading_dates[:len(portfolio_value)-1]
        
        print(f"Enhanced backtest completed, found {len(arbitrage_trades)} arbitrage opportunities")
        return portfolio_value, daily_returns, arbitrage_trades
    
    def plot_enhanced_results(self):
        """Plot enhanced backtest results"""
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Portfolio value evolution
        ax1.plot(self.trading_dates, self.portfolio_value[1:], 'b-', linewidth=2, label='Strategy')
        
        # Add S&P 500 benchmark
        spy_returns = self.data.loc[self.trading_dates, 'Close'].pct_change().fillna(0)
        spy_portfolio = [100000]
        for ret in spy_returns:
            spy_portfolio.append(spy_portfolio[-1] * (1 + ret))
        ax1.plot(self.trading_dates, spy_portfolio[1:], 'r--', linewidth=2, label='S&P 500 Benchmark')
        
        ax1.set_title('Portfolio Value Evolution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Cumulative returns comparison
        strategy_returns = [(v / 100000 - 1) * 100 for v in self.portfolio_value[1:]]
        benchmark_returns = [(v / 100000 - 1) * 100 for v in spy_portfolio[1:]]
        
        ax2.plot(self.trading_dates, strategy_returns, 'g-', linewidth=2, label='Strategy')
        ax2.plot(self.trading_dates, benchmark_returns, 'r--', linewidth=2, label='S&P 500')
        ax2.set_title('Cumulative Returns Comparison (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Returns (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Strategy distribution
        if self.arbitrage_trades:
            strategies = [trade.get('strategy', 'price_discrepancy') for trade in self.arbitrage_trades]
            strategy_counts = pd.Series(strategies).value_counts()
            ax3.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%')
            ax3.set_title('Arbitrage Strategy Distribution', fontsize=14, fontweight='bold')
        
        # 4. Profit distribution
        if self.detailed_trades:
            profits = [trade['profit'] for trade in self.detailed_trades]
            ax4.hist(profits, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_title('Trade Profit Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Profit per Trade ($)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print enhanced statistics
        self.print_enhanced_statistics()
    
    def print_enhanced_statistics(self):
        """Print enhanced backtest statistics"""
        if not hasattr(self, 'portfolio_value'):
            print("Please run backtest first")
            return
        
        initial_value = 100000
        final_value = self.portfolio_value[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate annualized return
        days = len(self.portfolio_value) - 1
        annualized_return = ((final_value / initial_value) ** (252 / days) - 1) * 100
        
        print("\n" + "="*70)
        print("ENHANCED ARBITRAGE STRATEGY BACKTEST RESULTS")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Initial Capital: ${initial_value:,.2f}")
        print(f"Final Capital: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Return: {annualized_return:.2f}%")
        print(f"Total Trading Days: {days}")
        print(f"Total Arbitrage Opportunities: {len(self.arbitrage_trades)}")
        
        if self.daily_returns:
            returns_array = np.array(self.daily_returns)
            print(f"Average Daily Return: {np.mean(returns_array)*100:.4f}%")
            print(f"Return Volatility: {np.std(returns_array)*100:.4f}%")
            print(f"Sharpe Ratio: {np.mean(returns_array)/np.std(returns_array)*np.sqrt(252):.4f}" if np.std(returns_array) > 0 else "Sharpe Ratio: N/A")
            print(f"Maximum Daily Return: {np.max(returns_array)*100:.4f}%")
            print(f"Maximum Daily Loss: {np.min(returns_array)*100:.4f}%")
            
            # Calculate maximum drawdown
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100
            print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        if self.detailed_trades:
            profits = [trade['profit'] for trade in self.detailed_trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]
            
            print(f"\nTRADE ANALYSIS:")
            print(f"Total Trades Executed: {len(self.detailed_trades)}")
            print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(self.detailed_trades)*100:.1f}%)")
            print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(self.detailed_trades)*100:.1f}%)")
            print(f"Average Profit per Trade: ${np.mean(profits):.2f}")
            print(f"Average Winning Trade: ${np.mean(winning_trades):.2f}" if winning_trades else "Average Winning Trade: N/A")
            print(f"Average Losing Trade: ${np.mean(losing_trades):.2f}" if losing_trades else "Average Losing Trade: N/A")
            print(f"Largest Winning Trade: ${np.max(profits):.2f}")
            print(f"Largest Losing Trade: ${np.min(profits):.2f}")
            
            # Strategy breakdown
            strategies = [trade.get('strategy', 'price_discrepancy') for trade in self.detailed_trades]
            strategy_counts = pd.Series(strategies).value_counts()
            print(f"\nSTRATEGY BREAKDOWN:")
            for strategy, count in strategy_counts.items():
                strategy_profits = [trade['profit'] for trade in self.detailed_trades 
                                  if trade.get('strategy', 'price_discrepancy') == strategy]
                avg_profit = np.mean(strategy_profits)
                print(f"{strategy}: {count} trades, Avg Profit: ${avg_profit:.2f}")
        
        # Greeks analysis
        if self.detailed_trades:
            print(f"\nGREEKS ANALYSIS:")
            deltas = [trade.get('delta', 0) for trade in self.detailed_trades]
            gammas = [trade.get('gamma', 0) for trade in self.detailed_trades]
            thetas = [trade.get('theta', 0) for trade in self.detailed_trades]
            vegas = [trade.get('vega', 0) for trade in self.detailed_trades]
            
            print(f"Average Delta Exposure: {np.mean(deltas):.4f}")
            print(f"Average Gamma Exposure: {np.mean(gammas):.6f}")
            print(f"Average Theta (Daily Decay): ${np.mean(thetas):.4f}")
            print(f"Average Vega (Vol Sensitivity): {np.mean(vegas):.4f}")

def main():
    """Main function with enhanced strategy for all modes"""
    print("Enhanced Black-Scholes Weekly Options Arbitrage System")
    print("="*60)
    
    # Choose running mode
    print("\nSelect running mode:")
    print("1. Enhanced single stock backtest (SPY)")
    print("2. Enhanced single stock backtest with detailed analysis (SPY)")
    print("3. Enhanced multi-stock backtest (10 large US stocks)")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == '1' or choice == '2':
        # Enhanced single stock backtest
        detailed = choice == '2'
        print(f"\nRunning enhanced single stock backtest (SPY){'with detailed analysis' if detailed else ''}...")
        arbitrage = EnhancedWeeklyOptionsArbitrage(symbol='SPY', risk_free_rate=0.05)
        
        # Fetch data and run backtest
        arbitrage.fetch_data(start_date='2022-01-01', end_date='2024-01-01')
        arbitrage.backtest_enhanced_strategy()
        
        # Plot results
        arbitrage.plot_enhanced_results()
        
    elif choice == '3':
        # Enhanced multi-stock backtest
        print("\nRunning enhanced multi-stock backtest...")
        
        # List of large US stocks
        stocks = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JNJ']
        
        total_portfolio_value = []
        all_returns = []
        all_trades = []
        
        print(f"Analyzing {len(stocks)} stocks...")
        
        for i, symbol in enumerate(stocks):
            print(f"\nProcessing {symbol} ({i+1}/{len(stocks)})...")
            
            try:
                # Create arbitrage instance for each stock
                arbitrage = EnhancedWeeklyOptionsArbitrage(symbol=symbol, risk_free_rate=0.05)
                
                # Fetch data and run backtest
                arbitrage.fetch_data(start_date='2022-01-01', end_date='2024-01-01')
                portfolio_value, daily_returns, trades = arbitrage.backtest_enhanced_strategy()
                
                # Collect results
                if portfolio_value:
                    total_portfolio_value.append(portfolio_value)
                    all_returns.extend(daily_returns)
                    all_trades.extend(trades)
                    
                    final_return = (portfolio_value[-1] / 100000 - 1) * 100
                    print(f"{symbol}: {final_return:.2f}% return, {len(trades)} trades")
                    
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Calculate combined portfolio performance
        if total_portfolio_value:
            print("\n" + "="*70)
            print("ENHANCED MULTI-STOCK PORTFOLIO RESULTS")
            print("="*70)
            
            # Average portfolio performance
            avg_portfolio = np.mean(total_portfolio_value, axis=0)
            total_return = (avg_portfolio[-1] / 100000 - 1) * 100
            
            # Calculate annualized return
            days = len(avg_portfolio) - 1
            annualized_return = ((avg_portfolio[-1] / 100000) ** (252 / days) - 1) * 100
            
            print(f"Number of stocks analyzed: {len(total_portfolio_value)}")
            print(f"Average portfolio return: {total_return:.2f}%")
            print(f"Average annualized return: {annualized_return:.2f}%")
            print(f"Total arbitrage opportunities: {len(all_trades)}")
            
            if all_returns:
                returns_array = np.array(all_returns)
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
                print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.4f}")
                
                # Calculate maximum drawdown
                cumulative = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown) * 100
                print(f"Maximum Drawdown: {max_drawdown:.2f}%")
            
            # Plot multi-stock results
            plt.figure(figsize=(15, 10))
            
            # Plot individual stock performances
            plt.subplot(2, 2, 1)
            for i, (symbol, portfolio) in enumerate(zip(stocks[:len(total_portfolio_value)], total_portfolio_value)):
                returns = [(v / 100000 - 1) * 100 for v in portfolio]
                plt.plot(returns, label=symbol, alpha=0.7)
            plt.title('Individual Stock Strategy Performance')
            plt.xlabel('Trading Days')
            plt.ylabel('Cumulative Return (%)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Plot average portfolio performance
            plt.subplot(2, 2, 2)
            avg_returns = [(v / 100000 - 1) * 100 for v in avg_portfolio]
            plt.plot(avg_returns, 'b-', linewidth=2, label='Average Portfolio')
            plt.title('Average Portfolio Performance')
            plt.xlabel('Trading Days')
            plt.ylabel('Cumulative Return (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot trade distribution by stock
            plt.subplot(2, 2, 3)
            stock_trade_counts = {}
            for trade in all_trades:
                symbol = trade.get('symbol', 'Unknown')
                stock_trade_counts[symbol] = stock_trade_counts.get(symbol, 0) + 1
            
            if stock_trade_counts:
                plt.bar(stock_trade_counts.keys(), stock_trade_counts.values())
                plt.title('Arbitrage Opportunities by Stock')
                plt.xlabel('Stock Symbol')
                plt.ylabel('Number of Trades')
                plt.xticks(rotation=45)
            
            # Plot strategy distribution
            plt.subplot(2, 2, 4)
            strategies = [trade.get('strategy', 'price_discrepancy') for trade in all_trades]
            if strategies:
                strategy_counts = pd.Series(strategies).value_counts()
                plt.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%')
                plt.title('Strategy Distribution Across All Stocks')
            
            plt.tight_layout()
            plt.show()
        
        else:
            print("No successful backtests completed.")
    
    else:
        print("Invalid choice. Please select 1, 2, or 3.")
        return
    
    # Option pricing demonstration
    print("\n" + "="*60)
    print("AMERICAN OPTION PRICING DEMONSTRATION")
    print("="*60)
    
    # Example: SPY current price 400, strike 405, 7 days to expiration
    option_pricer = AmericanOptionPricer(
        S=400,      # Current price
        K=405,      # Strike price
        T=7/365,    # 7 days to expiration
        r=0.05,     # 5% risk-free rate
        sigma=0.2,  # 20% volatility
        option_type='call'
    )
    
    call_price = option_pricer.price()
    call_delta = option_pricer.delta()
    call_gamma = option_pricer.gamma()
    call_theta = option_pricer.theta()
    call_vega = option_pricer.vega()
    
    print(f"\nCALL OPTION (Strike: $405, 7 days to expiration):")
    print(f"Theoretical Price: ${call_price:.4f}")
    print(f"Delta: {call_delta:.4f}")
    print(f"Gamma: {call_gamma:.6f}")
    print(f"Theta (daily): ${call_theta:.4f}")
    print(f"Vega: {call_vega:.4f}")
    
    # Put option
    put_pricer = AmericanOptionPricer(
        S=400, K=405, T=7/365, r=0.05, sigma=0.2, option_type='put'
    )
    put_price = put_pricer.price()
    put_delta = put_pricer.delta()
    put_gamma = put_pricer.gamma()
    put_theta = put_pricer.theta()
    put_vega = put_pricer.vega()
    
    print(f"\nPUT OPTION (Strike: $405, 7 days to expiration):")
    print(f"Theoretical Price: ${put_price:.4f}")
    print(f"Delta: {put_delta:.4f}")
    print(f"Gamma: {put_gamma:.6f}")
    print(f"Theta (daily): ${put_theta:.4f}")
    print(f"Vega: {put_vega:.4f}")
    
    # Put-Call Parity check
    parity_diff = call_price - put_price - (400 - 405 * np.exp(-0.05 * 7/365))
    print(f"\nPUT-CALL PARITY CHECK:")
    print(f"Parity Difference: ${parity_diff:.6f}")
    print(f"(Should be close to 0 for European options)")

if __name__ == "__main__":
    main()