#randomly sample a portfolio N assets size (5-25)
#randomly sample portfolio weighting (all less than 50%) - figure out scheme to do this so overall sum is 1
#randomly sample a t, features calculated on the past, result calculated on the future
#make a graph input with nodes (individual stock volatility + optional performance vector) and edges (correlations vector)
#to use continuous values or bin here
#sharpe/risk is the value, action is what stock to downweigh/upweigh, advantage is increase in sharpe/decrease in risk (need to figure out what makes most sense out of 1/variance, 1/sd, -risk)
#DQN with some epsilon greedy as choice


