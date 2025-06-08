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
## The result of the backtest is shown in the following figure.
![屏幕截图 2025-06-08 140158](https://github.com/user-attachments/assets/78b0113a-6f5d-4a09-9530-2d6e6f1f74e9)


---

# 周度期权错定价套利策略

# 1. 策略概述

本 Alpha 研究提出了一种**周度期权错定价套利策略**，聚焦于美国大型股和高流动性 ETF（如 SPY、QQQ）。该策略按周运行，旨在利用**隐含波动率（IV）与历史波动率（HV）之间的系统性偏差**，以及近到期期权的**可预测时间衰减（Theta）**。其经济直觉基于两个行为现象：（1）由于短期对冲或投机需求，尤其是周度期权常被高估；（2）市场参与者常对近期价格冲击反应过度，从而造成持续的 IV-HV 差距。此策略灵感来源于有关波动率风险溢价的学术论文以及对接近到期的近平值期权（尤其是在财报和宏观经济事件窗口）持续被高估的经验观察。

# 2. 原始 Alpha 定义

该 Alpha 源于做空高估期权（即 IV 明显高于 HV），通过时间衰减获利，同时保持有限的方向性风险。用文字表达：该 alpha 与以下三个因子成正比——**波动率溢价（IV - HV）**、**Theta 衰减速度** 和 **期权的虚实程度（由 Delta 表征）**。公式如下：

Alpha = ∑\[wᵢ × (IVᵢ - HVᵢ) × Θᵢ × (1 - |Δᵢ|)]

其中：
wᵢ：期权 i 的权重
IVᵢ：隐含波动率
HVᵢ：20 日历史波动率
Θᵢ：时间价值衰减（Theta）
Δᵢ：期权 Delta 值

<img width="583" alt="image" src="https://github.com/user-attachments/assets/f28b5acf-cf9f-4245-9b7f-4daa14fc7224" />

该公式惩罚方向性敞口，并偏好具有高波动率溢价和快速时间衰减的期权。

# 3. 策略构建步骤

每周一，策略会扫描大型标的的期权链，寻找 IV-HV 差值较大的周度期权。选择那些 IV 被明显高估的期权（例如排名前 20%）。通过组合做空 call 和 put 构建一个 delta 中性组合，调整方向性敞口（目标是组合的 |Delta| < 0.1）。Gamma 风险通过动态调整管理：若净 Gamma 超出阈值，策略会在日内再平衡。每日运行 Theta 收割机制，关闭或滚动即将到期或 Theta 收益下降的头寸。风控包括每笔交易最大亏损 2%、投资组合 VaR 上限 5%、以及基于 RSI 与均线交叉的市场状态过滤器，以避免在高波动或趋势市场中开仓。

# 4. 绩效预测

从 2019 到 2024 年的回测结果显示策略表现稳健。年化收益预期在 18% 至 25% 之间，夏普比率在 1.8 至 2.2 之间。最大回撤为 8–12%，月度胜率在 75–80%。策略在低波动、区间震荡市场中表现最佳，但在诸如 2020 年 COVID 暴跌等高波动时期也能从 IV-HV 扩张中获利。关键指标包括：95%日 VaR 为 1.2%，信息比率约为 1.6，Calmar 比率为 2.1。与 S\&P 500 的 Beta 约为 0.15，具有良好的多样化潜力。我们预计在稳定市场中将持续获得正收益，但需防范波动率崩溃或极端单边行情的风险。

---

# 5. 如何运行代码

1. **克隆本项目仓库**

```bash
git clone https://github.com/chnoorlee/Enhanced-Weekly-Option-Mispricing-Arbitrage-Strategy.git
cd Enhanced-Weekly-Option-Mispricing-Arbitrage-Strategy
```

2. **创建并激活虚拟环境（可选但推荐）**

```bash
python -m venv venv
# Windows 用户
venv\Scripts\activate
# macOS / Linux 用户
source venv/bin/activate
```

3. **安装依赖库**

```bash
pip install -r requirements.txt
```

> 如果没有 requirements.txt 文件，请根据项目中导入的包手动安装，例如 pandas、numpy、yfinance、matplotlib 等。

4. **运行主脚本**

假设入口为 `main.py`（请根据实际情况调整文件名）：

```bash
python main.py
```

5. **自定义参数或配置**

如有需要，可根据 README 或代码中描述的方法修改配置文件或脚本参数。

---

# 6. 附加说明

**作者提示**：如果你不知道如何执行第 5 步，可以先下载代码并解压到你能找到的本地文件夹，安装好 Python 后，在命令行中使用 pip 安装 `requirements.txt` 里的依赖，然后在 cmd 或 powershell 中执行以下命令：

```bash
python main.py
```

## 回测效果如下图
![屏幕截图 2025-06-08 140158](https://github.com/user-attachments/assets/78b0113a-6f5d-4a09-9530-2d6e6f1f74e9)
