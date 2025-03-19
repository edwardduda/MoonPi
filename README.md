# MoonPi
Jan 2025 - PresentJan 2025 - Present
MoonPi is a solo project that uses Deep Reinforcement Learning with a novel attention-based architecture for algorithmic trading. The system combines data engineering, combinatorics, risk mitigation and transformers to make trading decisions in financial markets.

## Methodology/Approach:
While most algorithmic trading relies on recurrent neural networks (RNNs) to predict continuous values like prices, my model takes a different approach by framing market prediction as a discrete action selection problem. Drawing inspiration from combinatorics—a branch of discrete mathematics focused on counting and arranging finite sets—I applied this principle to financial market forecasting. Instead of predicting continuous price movements, I used deep neural networks to forecast discrete actions, such as 'buy' or 'sell.' This method leverages the proven strength of neural networks in handling discrete domains, like classification and natural language processing, where inputs and outputs are structured and finite. By treating trading decisions as distinct choices within a combinatorial framework, this approach enhances predictive precision and aligns well with the capabilities of neural networks.

I gathered price action data using the Yahoo Finance API, using dating back to the early 1960s. I organized this data by sector and plotted the distribution of the time frames to minimize skew and create a more balanced balanced dataset. I don't want a bias towards the present. I also standardized all the data to bypass inflation. $1 in 1964 is worth more than $1 in 2025. TensorBoard was used to monitor model performance throughout training. Seaborn was usiing to plot attention heatmaps.

## Custom Architecture & Environment:
* Implemented a Deep Q-Learning Network to approximate a Q-function for decision-making.
* Defined a state as a segment of time, ~100 days. Working with a defined segment of time, I have 
 an enclosed environment for the model to train in. 
* Randomly shuffled segments to reduce temporal dependencies.
* Extended the utility of my full data set from ~60 years worth of data to ~7143 years worth of data. * Tokened all positions in the segment to represent a state.
* Designed specialized attention blocks analyzing both features and temporal positioning.

## Result and Outcome:
<img width="649" alt="Screenshot 2025-03-17 at 5 11 03 PM" src="https://github.com/user-attachments/assets/d93118c6-807d-4e3d-aa4b-a804913754bf" />

#### PnL Performance:

* X-axis: Training steps (1 step = 1 day)
* Y-axis: Net profit/loss (% relative to a $1,000 initial portfolio)
The chart plots the net profit of my RL model. I have a dynamic epsilon and learning rate with weight decay to help the model navigate out sub optimal local minma. The oscillation is not so sinusoidal so the change is a bit abrupt, hence why there are spikes in the pnl performance. 
<img width="920" alt="Screenshot 2025-03-17 at 5 11 14 PM" src="https://github.com/user-attachments/assets/f40994fd-46ca-4247-be2c-e50a39481a83" />

#### Model Training Performance:

These graphs showcase the training dynamics of my RL model as it learns to optimize trading decisions.

* Loss & Loss Std: The model’s error decreases over time, showing stable learning with minor fluctuations.

* Reward & Reward Std: The average reward is steadily increasing, meaning the model is improving its decision-making, while reward variance is decreasing—indicating more consistent performance.

* I have a dynamic epsilon, learning rate, and reward function that changes during training.
<img width="748" alt="Screenshot 2025-03-17 at 5 11 23 PM" src="https://github.com/user-attachments/assets/5c1bbd53-2dcc-4ee4-ac6c-9404c8cc43e0" />

#### Attention Heatmaps:

These heatmaps illustrate how my model processes different aspects of the input data using my custom architecture:
* Feature Layer Attention (Left) – Captures broad patterns across all input features.
* Technical Indicator Attention (Middle) – Highlights key market indicators that influence decision- 
   making.
* Temporal Attention (Right) – Focuses on sequential patterns in price action over time.
Takeaways:
<br>
The model learns to prioritize different market signals dynamically.
<br>
Temporal patterns show strong attention shifts, indicating potential predictive signals.
<br>
Next steps: Fine-tuning attention distribution and optimizing feature selection.
<br>
<img width="959" alt="Screenshot 2025-03-17 at 5 11 32 PM" src="https://github.com/user-attachments/assets/59b8b7fd-e01a-4f29-978e-38be50ed95a8" />

#### Q-value Distribution
These plots illustrate the distribution of Q-values over time:

* Left: Main Q-value distribution (predicted action values). My model's predicted q-value (action)
* Right: Target Q-value distribution (expected future rewards). The target q-value function I'm tuning the model to approximate

Take aways:
The standard deviation between q-values in both the main and target networks decreases which indicates stability and learning.
I have achieved a mean ~0.7% profitability per quarter. I gained a deep understanding of reinforcement learning, neural networks, and structured problem-solving process-from brainstorming to execution and validation/analysis. 


## Future: 
* Complete the conversion of my environment from Python to C++.
* Train model using GPU clusters instead of locally on my MacBook.
* Implement Mixture of Experts to enrich action space representation while maintaining time and 
 space efficiency on my MacBook.
