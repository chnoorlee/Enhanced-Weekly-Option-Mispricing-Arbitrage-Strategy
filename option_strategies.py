import numpy as np
import pandas as pd
from main import AmericanOptionPricer

class AdvancedArbitrageStrategies:
    """高级套利策略"""
    
    def __init__(self, risk_free_rate=0.05):
        self.risk_free_rate = risk_free_rate
    
    def box_spread_arbitrage(self, S, K1, K2, T, r, sigma):
        """箱式套利策略"""
        if K1 >= K2:
            raise ValueError("K1 must be less than K2")
        
        # 构建箱式套利：买入K1看涨，卖出K2看涨，卖出K1看跌，买入K2看跌
        call_K1 = AmericanOptionPricer(S, K1, T, r, sigma, 'call').price()
        call_K2 = AmericanOptionPricer(S, K2, T, r, sigma, 'call').price()
        put_K1 = AmericanOptionPricer(S, K1, T, r, sigma, 'put').price()
        put_K2 = AmericanOptionPricer(S, K2, T, r, sigma, 'put').price()
        
        # 箱式套利成本
        box_cost = call_K1 - call_K2 - put_K1 + put_K2
        
        # 理论价值（折现后的执行价差）
        theoretical_value = (K2 - K1) * np.exp(-r * T)
        
        # 套利利润
        arbitrage_profit = theoretical_value - box_cost
        
        return {
            'box_cost': box_cost,
            'theoretical_value': theoretical_value,
            'arbitrage_profit': arbitrage_profit,
            'is_profitable': arbitrage_profit > 0.01  # 考虑交易成本
        }
    
    def conversion_reversal_arbitrage(self, S, K, T, r, sigma, market_call_price, market_put_price):
        """转换套利和逆转换套利"""
        # 理论期权价格
        theoretical_call = AmericanOptionPricer(S, K, T, r, sigma, 'call').price()
        theoretical_put = AmericanOptionPricer(S, K, T, r, sigma, 'put').price()
        
        # Put-Call Parity for American options (approximation)
        synthetic_call = market_put_price + S - K * np.exp(-r * T)
        synthetic_put = market_call_price - S + K * np.exp(-r * T)
        
        # 转换套利机会（如果合成看涨期权便宜）
        conversion_profit = market_call_price - synthetic_call
        
        # 逆转换套利机会（如果合成看跌期权便宜）
        reversal_profit = market_put_price - synthetic_put
        
        return {
            'conversion_profit': conversion_profit,
            'reversal_profit': reversal_profit,
            'conversion_opportunity': conversion_profit > 0.01,
            'reversal_opportunity': reversal_profit > 0.01,
            'recommended_strategy': 'conversion' if conversion_profit > reversal_profit else 'reversal'
        }
    
    def volatility_arbitrage(self, S, K, T, r, market_price, historical_vol, implied_vol, option_type='call'):
        """波动率套利"""
        # 使用历史波动率计算理论价格
        theoretical_price_hist = AmericanOptionPricer(S, K, T, r, historical_vol, option_type).price()
        
        # 使用隐含波动率计算理论价格
        theoretical_price_impl = AmericanOptionPricer(S, K, T, r, implied_vol, option_type).price()
        
        # 波动率差异
        vol_diff = implied_vol - historical_vol
        
        # 套利机会
        if vol_diff > 0.05:  # 隐含波动率明显高于历史波动率
            strategy = 'sell_option'  # 卖出期权
            expected_profit = market_price - theoretical_price_hist
        elif vol_diff < -0.05:  # 隐含波动率明显低于历史波动率
            strategy = 'buy_option'  # 买入期权
            expected_profit = theoretical_price_hist - market_price
        else:
            strategy = 'no_arbitrage'
            expected_profit = 0
        
        return {
            'historical_vol': historical_vol,
            'implied_vol': implied_vol,
            'vol_difference': vol_diff,
            'strategy': strategy,
            'expected_profit': expected_profit,
            'theoretical_price_hist': theoretical_price_hist,
            'theoretical_price_impl': theoretical_price_impl
        }

# 示例使用
if __name__ == "__main__":
    strategies = AdvancedArbitrageStrategies()
    
    # 箱式套利示例
    box_result = strategies.box_spread_arbitrage(
        S=400, K1=395, K2=405, T=7/365, r=0.05, sigma=0.2
    )
    print("箱式套利结果:", box_result)
    
    # 波动率套利示例
    vol_result = strategies.volatility_arbitrage(
        S=400, K=405, T=7/365, r=0.05, 
        market_price=8.5, historical_vol=0.18, implied_vol=0.25
    )
    print("波动率套利结果:", vol_result)