import numpy as np
min_assets = 5
max_assets = 25
class PortfolioSimulator:
    @staticmethod
    def adjust_weights(weights, action):
        num_assets = len(weights)
        asset_index = action % num_assets
        increase = action < num_assets

        if increase:
            weights[asset_index] *= 2
        else:
            weights[asset_index] /= 2

        # Ensure the weights sum to 1
        weights_sum = np.sum(weights)
        weights = weights / weights_sum

        return weights

    @staticmethod
    def calculate_sharpe_ratio(weights, stock_returns):
        # Calculate daily portfolio returns
        daily_portfolio_returns = np.dot(stock_returns, weights)
        # Adding a small epsilon value to avoid division by zero
        mean_return = np.mean(daily_portfolio_returns)
        std_return = np.std(daily_portfolio_returns) + 1e-4
        sharpe_ratio = mean_return / std_return
        return sharpe_ratio

    @staticmethod
    def calculate_correlation(weights, stock_returns):
        daily_portfolio_returns = np.dot(stock_returns, weights)
        portfolio_correlation = np.corrcoef(daily_portfolio_returns, rowvar=False)

        return portfolio_correlation


    @staticmethod
    def sample_portfolio_weights(num_stocks):
        weights = np.random.dirichlet(np.ones(num_stocks), size=1).flatten()
        while np.any(weights > 0.5):
            weights = np.random.dirichlet(np.ones(num_stocks), size=1).flatten()
        return weights

    @staticmethod
    def calculate_portfolio_std(weights, stock_returns):
        daily_portfolio_returns = np.dot(stock_returns, weights)
        portfolio_std = np.std(daily_portfolio_returns)
        return portfolio_std