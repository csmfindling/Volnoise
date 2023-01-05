
idx = 3
params = proposed_thetas[idx]

particles, log_inc_marglkd = smc(actions, rewards, params)

print log_inc_marglkd
print log_lkd_theta_move[idx]
print(idx)
print ('\n')

proposed_thetas = np.tile(np.array([proposed_thetas[1]]), (100, 1))

for idx in range(30):
	params = proposed_thetas[idx]

	particles, log_inc_marglkd = smc(actions, rewards, params)

	print log_inc_marglkd
	print log_lkd_theta_move[idx]
	print(idx)
	print ('\n')