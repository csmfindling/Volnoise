//
//  smc_functions.hpp
//  smc_wy
//
//  Created by Charles Findling on 10/11/16.
//  Copyright © 2016 csmfindling. All rights reserved.
//

#ifndef smc_functions_hpp
#define smc_functions_hpp

#include <stdio.h>

namespace smc {

    void smc_update_2q(double* log_lkd, double* logThetaLks, double* noisy_descendants, double* amount_noise_descendants, double* noisy_ancestors, double* amount_noise_ancestors, int* ancestorsIndexes, double* weights_unnorm, double* logThetaWeights, double* theta_samples, int n_theta, int N_samples, int t_idx, int prev_act, int* actions, double* prev_rew, int apply_cst_noise, int apply_inertie, int apply_weber, int nbAlpha, int temperature);

  	double smc_2q(double* Q_values, double* noise_values, double* Q_values_ancestors, double* noise_values_ancestors, double* weights_res, double* sample, int* ancestors_indexes, int N_samples, int T, int n_essay, int* actions, double* rewards, int apply_cst_noise, int apply_inertie, int apply_weber, int nbAlpha, int temperature);
}


#endif /* smc_functions_hpp */
