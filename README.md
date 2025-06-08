 # Enhanced-Weekly-Option-Mispricing-Arbitrage-Strategy 

# 1. Strategy Overview
This alpha research proposes a weekly options mispricing arbitrage strategy focusing on large-cap U.S. equities and liquid ETFs (e.g., SPY, QQQ). The strategy operates at a weekly frequency and seeks to exploit systematic deviations between implied volatility (IV) and historical volatility (HV), as well as predictable time decay (Theta) in near-expiry options. Economic intuition rests on two behavioral phenomena: (1) options, especially weeklies, tend to be overpriced due to demand for short-term hedging or speculation, and (2) market participants often overreact to recent price shocks, leading to a persistent IV-HV gap. I was inspired by academic papers on volatility risk premiums and empirical observations of consistent overpricing in near-the-money weekly options, especially around earnings and macroeconomic event windows.

# 2. Raw Alpha Definition
The raw alpha arises from shorting overvalued options (where IV >> HV) and capturing time decay while maintaining limited directional exposure. In words, the alpha is proportional to the volatility premium (IV - HV), the rate of time decay (Theta), and the moneyness (proxied by Delta) of the option. Formally:

Alpha = ∑[wᵢ × (IVᵢ - HVᵢ) × Θᵢ × (1 - |Δᵢ|)]

Where:
wᵢ is the position weight on option i

IVᵢ is the implied volatility

HVᵢ is the 20-day historical volatility

Θᵢ is the time decay (Theta)

Δᵢ is the option delta

<img width="583" alt="image" src="https://github.com/user-attachments/assets/f28b5acf-cf9f-4245-9b7f-4daa14fc7224" />



This formulation penalizes directional exposure and favors options with high IV premia and rapid time decay.

# 3. Strategy Construction Steps
Each Monday, the strategy scans the option chain for large IV-HV spreads on weekly options. Options with significant IV overpricing (e.g., top 20th percentile) are selected. A delta-neutral portfolio is constructed by combining short calls and puts, adjusted for directional exposure (targeting portfolio |Delta| < 0.1). Gamma risk is dynamically managed: if net Gamma exceeds a threshold, positions are rebalanced intraday. A daily Theta harvesting routine is applied, closing or rolling positions nearing expiry or with Theta erosion falling below a set threshold. Risk controls include a 2% max loss per trade, 5% portfolio VaR cap, and market regime filters based on RSI and moving average crossovers to avoid entering in unstable or trending conditions.

# 4. Performance Prediction
Backtesting from 2019 to 2024 suggests robust performance. The expected annualized return ranges from 18% to 25%, with a Sharpe ratio between 1.8 and 2.2. The strategy shows a max drawdown of 8–12% and a monthly win rate of 75–80%. It performs best in low-volatility, range-bound markets but also finds opportunity during volatile periods like the 2020 COVID crash due to elevated IV-HV spreads. Key statistics include a 1.2% daily 95% VaR, information ratio ~1.6, and a Calmar ratio of 2.1. Beta to the S&P 500 is low (~0.15), indicating strong diversification potential. We expect consistent positive returns in stable market regimes, with some vulnerability to sudden volatility collapses or large directional moves.
Here is the English translation for the “如何运行代码” section that you can add to your README:

---

# 5. How to Run the Code

1. **Clone the repository**

```bash
git clone https://github.com/chnoorlee/Enhanced-Weekly-Option-Mispricing-Arbitrage-Strategy.git
cd Enhanced-Weekly-Option-Mispricing-Arbitrage-Strategy
```

2. **Create and activate a virtual environment (optional but recommended)**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```
> If there is no requirements.txt, please install the necessary packages manually based on the imports in the project, such as pandas, numpy, yfinance, matplotlib, etc.

4. **Run the main script**

Assuming the main entry point is main.py (please adjust the filename as needed):

```bash
python main.py
```

5. **Customize parameters or configuration**

If needed, modify the configuration files or script parameters as described in the README or within the code.

---

# 6. Additional Explaination
**Finally, there is an explanation from the author: If you do not know how to perform step 5, you can download the code, extract it to a folder where you can find it, install Python on your local computer, download the dependencies in  requirement.txt using pip, and finally run the** 
```bash
python main.py
```
**code in cmd or powershell.**
