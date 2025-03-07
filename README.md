Initial setup was configuring API Keys from ByBit Now Connecting MT5 Bybit Exchange to my Script and APIs

Stratergy based off

ML assessment and rescore—> Markov chain 2 —> ML validation —> Random Forest —> RNN strategy selection via pseudo-forest—> execution

Microstructure noise, Fractal dimensions, Hust exponent and Bid/ask volume imbalance + L2/L3

Next ive added a feature to lower my risk within trades, SL and TP on each trade Given a range of volume order of 0.01 - 0.02 - 0.03 - 0.04 - 0.05 - 0.06 - 0.07 - 0.08 - 0.08 - 0.09 0.10 Depending on factors within the script

Next added multiple timeframes to pick trends and recieve data from , 1min,3min,5min

Lastly added a Hourly/4Hourly/12Hourly/Daily/Weekly/Monthly Summarise of performance

Made the bot teach it to know when to take profit and cut losing rades more effectively, so we lose less wit normal stop losses

Dynamic Profit Target:
Added dynamic_profit_target that adjusts based on total_return from wallet.get_performance_summary().
If the wallet is up more than 5%, the profit target increases (e.g., from 3% to 3.5% or higher).
If the wallet is down more than 5%, the profit target decreases slightly (minimum 1%) to secure gains earlier.
Trend Reversal Check:
Added calculate_sma and calculate_rsi to compute moving averages and RSI.
is_trend_reversal checks for SMA crossovers (10-period vs. 30-period) and RSI confirmation:
Bullish reversal: Short SMA crosses above Long SMA and RSI > 50.
Bearish reversal: Short SMA crosses below Long SMA and RSI < 50.
Positions are closed if a reversal is detected and the position is in profit, avoiding low-price exits.
Integration:
Added SMA and RSI calculations to the df in the main loop.
Updated the position management logic to use the dynamic profit target and trend reversal checks.

Helper Functions:
calculate_sma: Computes the Simple Moving Average for a given period.
calculate_rsi: Calculates the Relative Strength Index (RSI) to assess momentum.
is_trend_reversal: Detects bullish or bearish reversals using SMA crossover and RSI confirmation.
Main Loop Updates:
Added SMA and RSI calculations to the df at the start of each iteration.
Introduced dynamic_profit_target that adjusts based on the wallet's total_return:
If up > 5%, increases the profit target (e.g., 3% → 3.5% or higher).
If down > 5%, reduces the target to secure gains (minimum 1%).
Integrated trend reversal checks to avoid closing positions at unfavorable prices unless in profit:
Closes "Buy" positions on bearish reversals if profitable.
Closes "Sell" positions on bullish reversals if profitable.
Logging:
Added logging for dynamic profit targets and trend reversal events to monitor the bot's behavior.





Next step is to enhance the bot to read bids/asks and possibly hook it up to a greed/fear index and liquidation levels
