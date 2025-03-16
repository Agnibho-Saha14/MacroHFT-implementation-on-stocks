import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, macro_action, micro_action, reward, next_state, done):
        self.buffer.append((state, macro_action, micro_action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, macro_action, micro_action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, macro_action, micro_action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class TradingEnv:
    def __init__(self, data, window_size=10):
        # Check and clean data - remove rows with zero or NaN prices
        data = data.copy()
        data = data.dropna(subset=['close', 'open', 'high', 'low'])
        data = data[data['close'] > 0]  # Filter out zero prices
        
        # Normalize features
        self.price_columns = ['open', 'high', 'low', 'close']
        self.feature_columns = [col for col in data.columns if col not in ['timestamp', 'Trend', 'Volatility_Label']]
        
        # Store normalized data
        self.normalized_data = self.normalize_data(data)
        self.raw_data = data
        self.window_size = window_size
        self.n_features = len(self.feature_columns) * window_size  # Features × window
        self.trend_label = data['Trend'].values
        self.volatility_label = data['Volatility_Label'].values
        self.reset()

    def normalize_data(self, data):
        """Normalize features for better model training"""
        normalized = data.copy()
        
        # Min-max normalization for price data
        for col in self.price_columns:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                if std != 0:
                    normalized[col] = (data[col] - mean) / std
        
        # Min-max scaling for oscillators that already have bounds
        for col in ['RSI_14']:
            if col in data.columns:
                normalized[col] = data[col] / 100  # RSI is 0-100
                
        return normalized[self.feature_columns].values

    def get_state(self, step):
        """Create a state with window_size previous observations"""
        start = max(0, step - self.window_size + 1)
        window = self.normalized_data[start:step+1]
        
        # Pad with zeros if we don't have enough history
        padding = self.window_size - window.shape[0]
        if padding > 0:
            padding_shape = (padding, window.shape[1])
            padding_data = np.zeros(padding_shape)
            window = np.vstack((padding_data, window))
            
        return window.flatten()

    def reset(self):
        self.current_step = self.window_size - 1
        self.cash_balance = 10000.0  # Starting balance
        self.position = 0  # Number of assets held
        self.done = False
        self.portfolio_value_history = [self.cash_balance]
        
        state = self.get_state(self.current_step)
        trend = self.trend_label[self.current_step]
        volatility = self.volatility_label[self.current_step]
        
        additional_info = {
            'trend': trend,
            'volatility': volatility,
            'position': self.position,
            'balance': self.cash_balance
        }
        
        return state, additional_info

    def step(self, action):
        # 0: Hold, 1: Buy, 2: Sell
        if self.done:
            return None, None, 0, True, {}

        current_price = self.raw_data.iloc[self.current_step]['close']
        prev_portfolio_value = self.portfolio_value()
        
        # Execute action - with safety checks
        if action == 1:  # Buy
            # Safety check to prevent division by zero
            if current_price > 0:
                # Calculate max shares we can buy with 95% of our cash
                max_shares = int((self.cash_balance * 0.95) / current_price)
                if max_shares > 0:
                    shares_to_buy = max_shares  # Can be adjusted for position sizing
                    self.position += shares_to_buy
                    self.cash_balance -= shares_to_buy * current_price
                    # Include transaction cost (0.1%)
                    self.cash_balance -= shares_to_buy * current_price * 0.001
        
        elif action == 2:  # Sell
            if self.position > 0 and current_price > 0:
                shares_to_sell = self.position  # Sell all positions
                self.cash_balance += shares_to_sell * current_price
                # Include transaction cost (0.1%)
                self.cash_balance -= shares_to_sell * current_price * 0.001
                self.position = 0
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.normalized_data) - 1:
            self.done = True
        
        # Calculate reward based on portfolio value change
        current_portfolio_value = self.portfolio_value()
        self.portfolio_value_history.append(current_portfolio_value)
        
        # Calculate returns with safety check
        if prev_portfolio_value > 0:
            portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            portfolio_return = 0
        
        # Create a reward function that considers returns and risk
        reward = portfolio_return * 100  # Scale up for better learning
        
        # Get the next state
        next_state = self.get_state(self.current_step)
        
        # Safety check for trend and volatility indices
        trend_index = min(self.current_step, len(self.trend_label)-1)
        vol_index = min(self.current_step, len(self.volatility_label)-1)
        
        trend = self.trend_label[trend_index] if trend_index >= 0 else 0
        volatility = self.volatility_label[vol_index] if vol_index >= 0 else 0
        
        additional_info = {
            'trend': trend,
            'volatility': volatility,
            'position': self.position,
            'balance': self.cash_balance,
            'portfolio_value': current_portfolio_value,
            'return': portfolio_return
        }
        
        return next_state, additional_info, reward, self.done, {}
    
    def portfolio_value(self):
        """Calculate total portfolio value (cash + asset positions)"""
        if self.current_step < len(self.raw_data):
            current_price = self.raw_data.iloc[self.current_step]['close']
            # Safety check for price
            if np.isnan(current_price) or current_price <= 0:
                return self.cash_balance
            return self.cash_balance + self.position * current_price
        return self.cash_balance

class MacroAgent(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3):
        super(MacroAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class MicroAgent(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=3):
        super(MicroAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class MacroHFTAgent:
    def __init__(self, state_size, action_size=3, hidden_size=128, learning_rate=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=10, 
                 batch_size=64, buffer_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
        # Target network update frequency
        self.target_update = target_update
        self.update_count = 0
        
        # Create models
        self.macro_model = MacroAgent(state_size, hidden_size, action_size)
        self.macro_target = MacroAgent(state_size, hidden_size, action_size)
        self.macro_target.load_state_dict(self.macro_model.state_dict())
        
        # Create micro models for trend-based and volatility-based strategies
        self.micro_models = {
            'up_trend': MicroAgent(state_size),
            'down_trend': MicroAgent(state_size),
            'sideways': MicroAgent(state_size)
        }
        
        self.micro_targets = {
            'up_trend': MicroAgent(state_size),
            'down_trend': MicroAgent(state_size),
            'sideways': MicroAgent(state_size)
        }
        
        # Load weights from model to target
        for key in self.micro_models:
            self.micro_targets[key].load_state_dict(self.micro_models[key].state_dict())
        
        # Optimizers
        self.macro_optimizer = optim.Adam(self.macro_model.parameters(), lr=learning_rate)
        self.micro_optimizers = {
            key: optim.Adam(model.parameters(), lr=learning_rate) 
            for key, model in self.micro_models.items()
        }
    
    def select_macro_action(self, state, info):
        """Select which micro strategy to use based on trend and volatility"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                return self.macro_model(state_tensor).argmax().item()
    
    def select_micro_action(self, state, macro_action, trend):
        """Select specific action (buy/sell/hold) based on the chosen strategy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Map macro action to strategy key
        strategy_map = {
            0: 'up_trend',
            1: 'down_trend',
            2: 'sideways'
        }
        
        strategy = strategy_map[macro_action]
        model = self.micro_models[strategy]
        
        # Use epsilon-greedy for micro actions too
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                return model(state_tensor).argmax().item()
    
    def learn(self):
        """Update models using experiences from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample experiences
        states, macro_actions, micro_actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        macro_actions = torch.LongTensor(macro_actions)
        micro_actions = torch.LongTensor(micro_actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Update macro model
        macro_q_values = self.macro_model(states)
        next_macro_q_values = self.macro_target(next_states)
        
        # Get Q values for chosen actions
        macro_q_values = macro_q_values.gather(1, macro_actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = next_macro_q_values.max(1)[0]
            macro_targets = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        macro_loss = F.mse_loss(macro_q_values, macro_targets)
        
        # Update parameters
        self.macro_optimizer.zero_grad()
        macro_loss.backward()
        self.macro_optimizer.step()
        
        # Update micro models (grouped by macro action)
        strategy_map = {
            0: 'up_trend',
            1: 'down_trend',
            2: 'sideways'
        }
        
        for macro_value in range(self.action_size):
            # Get indices where this macro action was chosen
            indices = (macro_actions == macro_value).nonzero(as_tuple=True)[0]
            
            if len(indices) == 0:
                continue
                
            strategy = strategy_map[macro_value]
            micro_model = self.micro_models[strategy]
            micro_target = self.micro_targets[strategy]
            
            # Extract states and actions for this strategy
            strategy_states = states[indices]
            strategy_next_states = next_states[indices]
            strategy_micro_actions = micro_actions[indices]
            strategy_rewards = rewards[indices]
            strategy_dones = dones[indices]
            
            # Compute Q values and targets
            micro_q_values = micro_model(strategy_states)
            next_micro_q_values = micro_target(strategy_next_states)
            
            micro_q_values = micro_q_values.gather(1, strategy_micro_actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                max_next_q_values = next_micro_q_values.max(1)[0]
                micro_targets = strategy_rewards + (1 - strategy_dones) * self.gamma * max_next_q_values
            
            micro_loss = F.mse_loss(micro_q_values, micro_targets)
            
            # Update parameters
            optimizer = self.micro_optimizers[strategy]
            optimizer.zero_grad()
            micro_loss.backward()
            optimizer.step()
        
        # Update target networks
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.macro_target.load_state_dict(self.macro_model.state_dict())
            for key in self.micro_models:
                self.micro_targets[key].load_state_dict(self.micro_models[key].state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train_macro_hft(env, agent, episodes=100, max_steps=1000, print_interval=10, eval_interval=20):
    episode_rewards = []
    best_reward = 0
    best_sharpe = 0
    
    # Arrays to store performance metrics
    all_metrics = []
    portfolio_histories = []
    
    # Progress bar
    progress_bar = tqdm(range(episodes), desc="Training Progress")
    
    for episode in progress_bar:
        state, info = env.reset()
        total_reward = 0
        step_count = 0
        daily_returns = []
        
        while not env.done and step_count < max_steps:
            # Select macro action (strategy)
            macro_action = agent.select_macro_action(state, info)
            
            # Select micro action (specific trading decision)
            trend = info['trend']
            micro_action = agent.select_micro_action(state, macro_action, trend)
            
            # Take action and observe next state and reward
            next_state, next_info, reward, done, _ = env.step(micro_action)
            
            # Store daily return for metrics calculation
            if 'return' in next_info:
                daily_returns.append(next_info['return'])
            
            # Store experience in replay buffer
            agent.memory.push(state, macro_action, micro_action, reward, next_state, done)
            
            # Learn from experiences
            agent.learn()
            
            # Update state
            state = next_state
            info = next_info
            total_reward += reward
            step_count += 1
        
        episode_rewards.append(total_reward)
        portfolio_histories.append(env.portfolio_value_history)
        
        # Calculate performance metrics for this episode
        metrics = calculate_metrics(daily_returns)
        all_metrics.append(metrics)
        
        # Update progress bar with metrics
        progress_desc = f"Ep {episode+1}/{episodes} | Reward: {total_reward:.2f} | Sharpe: {metrics['sharpe_ratio']:.2f} | PF: {metrics['profit_factor']:.2f}"
        progress_bar.set_description(progress_desc)
        
        # Print detailed metrics at intervals
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics[-print_interval:]])
            avg_pf = np.mean([m['profit_factor'] for m in all_metrics[-print_interval:]])
            
            print(f"\nEpisode {episode+1} Statistics:")
            print(f"  Reward = {total_reward:.2f}, Avg Reward = {avg_reward:.2f}, Epsilon = {agent.epsilon:.4f}")
            print(f"  Final Portfolio Value: ${env.portfolio_value():.2f}")
            print(f"  Performance Metrics:")
            print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"    Sortino Ratio: {metrics['sortino_ratio']:.4f}")
            print(f"    Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"    Profit Factor: {metrics['profit_factor']:.4f}")
            print(f"    Win Rate: {metrics['win_rate']*100:.2f}%")
        
        # Save best model based on Sharpe ratio
        if metrics['sharpe_ratio'] > best_sharpe and metrics['sharpe_ratio'] != float('inf'):
            best_sharpe = metrics['sharpe_ratio']
            torch.save(agent.macro_model.state_dict(), "best_sharpe_macro_model.pth")
            for key, model in agent.micro_models.items():
                torch.save(model.state_dict(), f"best_sharpe_micro_model_{key}.pth")
            print(f"\nNew best model at episode {episode+1} with Sharpe ratio {best_sharpe:.4f}")
        
        # Also save model based on reward as before
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.macro_model.state_dict(), "best_reward_macro_model.pth")
            for key, model in agent.micro_models.items():
                torch.save(model.state_dict(), f"best_reward_micro_model_{key}.pth")
            print(f"\nNew best reward model at episode {episode+1} with reward {best_reward:.2f}")
    
    # Plot the learning curve and metrics
    plot_training_results(episode_rewards, all_metrics, portfolio_histories)
    
    return agent, episode_rewards, all_metrics



def calculate_metrics(returns, risk_free_rate=0.0):
    """Calculate trading performance metrics"""
    if len(returns) < 2:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'win_rate': 0
        }
    
    # Convert to numpy array if it's not already
    returns = np.array(returns)
    
    # Calculate daily returns (assuming the returns provided are sequential)
    daily_returns = returns
    
    # Total return
    total_return = np.prod(1 + daily_returns) - 1
    
    # Annualized return (assuming 252 trading days per year)
    # annualized_return = ((1 + total_return) ** (252 / len(daily_returns))) - 1
    
    # Sharpe ratio
    excess_returns = daily_returns - risk_free_rate
    sharpe_ratio = 0
    if np.std(daily_returns) > 0:
        sharpe_ratio = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252)
    
    # Sortino ratio
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = 0
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Profit factor
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gains / losses if losses != 0 else float('inf')
    
    # Win rate
    wins = len(daily_returns[daily_returns > 0])
    total_trades = len(daily_returns[daily_returns != 0])
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'win_rate': win_rate
    }

def plot_training_results(rewards, metrics, portfolio_histories):
    """Plot training results including rewards and key performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    
    # Plot Sharpe ratio
    sharpe_ratios = [m['sharpe_ratio'] for m in metrics]
    axes[0, 1].plot(sharpe_ratios)
    axes[0, 1].set_title('Sharpe Ratio')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    
    # Plot Profit Factor
    profit_factors = [min(m['profit_factor'], 10) for m in metrics]  # Cap at 10 for better visualization
    axes[1, 0].plot(profit_factors)
    axes[1, 0].set_title('Profit Factor (capped at 10)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Profit Factor')
    
    # Plot best portfolio equity curve
    best_idx = np.argmax(sharpe_ratios)
    if best_idx < len(portfolio_histories):
        best_portfolio = portfolio_histories[best_idx]
        axes[1, 1].plot(best_portfolio)
        axes[1, 1].set_title(f'Best Portfolio Equity Curve (Episode {best_idx+1})')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Portfolio Value ($)')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    try:
        # Load and prepare data
        data = pd.read_csv("./Dataset/Training/ETHUSDT_labeled.csv")
        
        # Verify that all required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'Trend', 'Volatility_Label']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            print(f"Error: Missing required columns: {missing}")
            exit(1)
            
        # Data quality checks and handling
        print("Data shape before cleaning:", data.shape)
        
        # Check for NaN values
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in dataset. Removing rows with NaNs in critical columns...")
            data = data.dropna(subset=['close', 'open', 'high', 'low'])
        
        # Check for zero prices
        zero_prices = (data['close'] <= 0).sum()
        if zero_prices > 0:
            print(f"Found {zero_prices} rows with zero or negative prices. Removing...")
            data = data[data['close'] > 0]
            
        print("Data shape after cleaning:", data.shape)
        
        # Create environment
        env = TradingEnv(data, window_size=10)
        
        # Get state size from environment
        state_size = env.n_features
        print(f"State size: {state_size}")
        
        # Create agent
        agent = MacroHFTAgent(
            state_size=state_size,
            action_size=3,
            hidden_size=128,
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            target_update=10,
            batch_size=64,
            buffer_size=10000
        )
        
        # Train agent
        try:
            trained_agent, rewards, metrics = train_macro_hft(
                env=env,
                agent=agent,
                episodes=100,
                max_steps=5000,
                print_interval=5,
                eval_interval=20
            )
            
            # Evaluate agent
            
            
            # Save models if needed
            torch.save(trained_agent.macro_model.state_dict(), "macro_model.pth")
            for key, model in trained_agent.micro_models.items():
                torch.save(model.state_dict(), f"micro_model_{key}.pth")
            
            print("Training complete and models saved!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error loading data or initializing: {e}")
        import traceback
        traceback.print_exc()