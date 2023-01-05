//
//  smc_functions.cpp
//  smc_wy
//
//  Created by Charles Findling on 10/11/16.
//  Copyright © 2016 csmfindling. All rights reserved.
//

#include "smc_functions.hpp"
#include <boost/random/mersenne_twister.hpp>
#include "../useful_functions/usefulFunctions.hpp"
#include "../useful_functions/usefulFunctions.cpp"
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <stdexcept>

namespace smc {
    
    using namespace std;
    
    boost::mt19937 generator(static_cast<unsigned int>(time(0)));
    boost::normal_distribution<> distribution(0., 1.);
    boost::math::normal s;

    double smc_2q(double* Q_values, double* noise_values, double* Q_values_ancestors, double* noise_values_ancestors, double* weights_res, double* sample, int* ancestors_indexes, int N_samples, int T, int n_essay, int* actions, double* rewards, int apply_cst_noise, int apply_inertie, int apply_weber, int nbAlpha, int temperature)
    {

        if (nbAlpha == 0)
        {
            throw std::logic_error("nbAlpha must be set to 1 or 2");
        }
        
        // Instantiate output vector and inner variables
        double marg_log_lkd = 0;
        int act             = -1;
        int ances_idx       = -1;
        int prev_act        = -1;
        double alpha_c, alpha_u, beta_softmax, epsilon, eta, mu_0, mu_1, prev_rew_0, prev_rew_1;
        double lambdaa = std::sqrt(3)/boost::math::constants::pi<double>();
        double noise_level0; double noise_level1;
        double inertie = 0;

        if ((nbAlpha == 2)&&(temperature==0))
        {
            alpha_c      = *(sample);           // alpha chosen
            alpha_u      = *(sample + 1);       // alpha unchosen
            beta_softmax = std::pow(10., *(sample + 2));
            epsilon      = *(sample + 3);

            if (apply_cst_noise == 1)
            {
                eta = *(sample + 4);
            }
            if (apply_inertie == 1)
            {
                inertie = *(sample + 5);
            }
        }
        else if ((nbAlpha == 1)&&(temperature==0))
        {
            alpha_c        = *(sample);       // alpha chosen
            alpha_u        = *(sample);       // alpha unchosen
            beta_softmax   = std::pow(10., *(sample + 1));
            epsilon        = *(sample + 2);

            if (apply_cst_noise == 1)
            {
                eta = *(sample + 3);
            }
            if (apply_inertie == 1)
            {
                inertie = *(sample + 4);
            }            
        }
        else if ((nbAlpha == 1)&&(temperature==1))
        {
            alpha_c        = *(sample);       // alpha chosen
            alpha_u        = *(sample);       // alpha unchosen
            beta_softmax   = 1./(*(sample + 1));
            epsilon        = *(sample + 2);

            if (apply_cst_noise == 1)
            {
                eta = *(sample + 3);
            }
            if (apply_inertie == 1)
            {
                inertie = *(sample + 4);
            }
        }
        else
        {
            alpha_c        = *(sample);       // alpha chosen
            alpha_u        = *(sample + 1);       // alpha unchosen
            beta_softmax   = 1./(*(sample + 2));
            epsilon        = *(sample + 3);

            if (apply_cst_noise == 1)
            {
                eta = *(sample + 4);
            }
            if (apply_inertie == 1)
            {
                inertie = *(sample + 5);
            }            
        }

        double weightsSum;

        // Loop over the time
        for (int t_idx = 0; t_idx < T; ++t_idx)
        {
            act        = *(actions + t_idx);
            weightsSum = 0.;

            if (t_idx > 0)
            {
                copy(Q_values, Q_values + 2 * N_samples, Q_values_ancestors);
                copy(noise_values, noise_values + 2 * N_samples, noise_values_ancestors);
                stratified_resampling(generator, weights_res, N_samples, ancestors_indexes);
                prev_rew_0 = *(rewards + t_idx - 1);
                prev_rew_1 = *(rewards + n_essay + t_idx - 1);

                for (int n_idx = 0; n_idx < N_samples; ++n_idx)
                {
                    ances_idx = ancestors_indexes[n_idx];

                    if (actions[t_idx - 1] == 0)
                    {
                        mu_0 = (1 - alpha_c) * (*(Q_values_ancestors + ances_idx * 2)) + alpha_c * prev_rew_0;
                        mu_1 = (1 - alpha_u) * (*(Q_values_ancestors + ances_idx * 2 + 1)) + alpha_u * prev_rew_1;
                    } 
                    else
                    {
                        mu_0 = (1 - alpha_u) * (*(Q_values_ancestors + ances_idx * 2)) + alpha_u * prev_rew_0;
                        mu_1 = (1 - alpha_c) * (*(Q_values_ancestors + ances_idx * 2 + 1)) + alpha_c * prev_rew_1;
                    }

                    if (apply_weber==1)
                    {
                        noise_level0 = std::log(1 + eta + epsilon * std::abs(prev_rew_0 - (*(Q_values_ancestors + ances_idx * 2))));
                        noise_level1 = std::log(1 + eta + epsilon * std::abs(prev_rew_1 - (*(Q_values_ancestors + ances_idx * 2 + 1))));
                    }
                    else
                    {
                        noise_level0 = epsilon;
                        noise_level1 = epsilon;
                    }

                    *(noise_values + 2 * n_idx)     = inertie * (*(noise_values_ancestors + 2 * ances_idx)) + (1 - inertie) * distribution(generator) * noise_level0;
                    *(noise_values + 2 * n_idx + 1) = inertie * (*(noise_values_ancestors + 2 * ances_idx + 1)) + (1 - inertie) * distribution(generator) * noise_level1;
                    *(Q_values + 2 * n_idx)         = *(noise_values + 2 * n_idx) + mu_0;
                    *(Q_values + 2 * n_idx + 1)     = *(noise_values + 2 * n_idx + 1) + mu_1;


                    *(weights_res + n_idx)  = log_logistic_proba(beta_softmax, *(Q_values + 2 * n_idx), *(Q_values + 2 * n_idx + 1), act);                
                }
            }
            else
            {
                for (int n_idx = 0; n_idx < N_samples; ++n_idx)
                {
                    *(Q_values + 2 * n_idx)     = 1./2;
                    *(Q_values + 2 * n_idx + 1) = 1./2;
                    *(noise_values + 2 * n_idx)     = 0.;
                    *(noise_values + 2 * n_idx + 1) = 0.;

                    // weights
                    *(weights_res + n_idx) = log(.5);
                }
            }

            double b   = *max_element(weights_res, weights_res + N_samples);
            for (int n_idx = 0; n_idx < N_samples; ++n_idx)
            {
                *(weights_res + n_idx) = exp(*(weights_res + n_idx) - b);
                weightsSum            += *(weights_res + n_idx);
            }

            marg_log_lkd += (b + log(weightsSum) - log(N_samples));

            for (int n_idx = 0; n_idx < N_samples; ++n_idx)
            {
                *(weights_res + n_idx) = *(weights_res + n_idx)/weightsSum;
            }
        }
        return marg_log_lkd;
    }


    void smc_update_2q(double* log_lkd, double* logThetaLks, double* noisy_descendants, double* amount_noise_descendants, double* noisy_ancestors, double* amount_noise_ancestors, int* ancestorsIndexes, double* weights_unnorm, double* logThetaWeights, double* theta_samples, int n_theta, int N_samples, int t_idx, int prev_act, int* actions, double* prev_rew, int apply_cst_noise, int apply_inertie, int apply_weber, int nbAlpha, int temperature)
    {

        if (nbAlpha == 0)
        {
            throw std::logic_error("nbAlpha must be set to 1 or 2");
        }

        // inner variables
        double marg_log_lkd = 0;
        int ances_idx       = -1;
        int act             = *(actions + t_idx);
        double alpha_c, alpha_u, beta_softmax, epsilon, eta, mu_0, mu_1, inertie;
        double lambdaa = std::sqrt(3)/boost::math::constants::pi<double>();
        double noise_level0; double noise_level1;

        copy(noisy_descendants, noisy_descendants + 2 * N_samples * n_theta, noisy_ancestors);
        copy(amount_noise_descendants, amount_noise_descendants + 2 * N_samples * n_theta, amount_noise_ancestors);

        for (int theta_idx = 0; theta_idx < n_theta; ++theta_idx)
        {
            if ((apply_cst_noise==1)&&(nbAlpha == 2)&&(temperature == 0))
            {
                alpha_c      = *(theta_samples + theta_idx * 5);
                alpha_u      = *(theta_samples + theta_idx * 5 + 1);
                beta_softmax = std::pow(10., *(theta_samples + theta_idx * 5 + 2));
                epsilon      = *(theta_samples + theta_idx * 5 + 3);
                eta          = *(theta_samples + theta_idx * 5 + 4);
            }
            else if ((apply_cst_noise==0)&&(nbAlpha == 2)&&(temperature == 0))
            {
                alpha_c      = *(theta_samples + theta_idx * 4);
                alpha_u      = *(theta_samples + theta_idx * 4 + 1);
                beta_softmax = std::pow(10., *(theta_samples + theta_idx * 4 + 2));
                epsilon      = *(theta_samples + theta_idx * 4 + 3);
            }
            else if ((apply_cst_noise==1)&&(nbAlpha == 1)&&(temperature == 0))
            {
                alpha_c      = *(theta_samples + theta_idx * 4);
                alpha_u      = *(theta_samples + theta_idx * 4);
                beta_softmax = std::pow(10., *(theta_samples + theta_idx * 4 + 1));
                epsilon      = *(theta_samples + theta_idx * 4 + 2);
                eta          = *(theta_samples + theta_idx * 4 + 3);
            }
            else if (temperature == 0)
            {
                alpha_c      = *(theta_samples + theta_idx * 3);
                alpha_u      = *(theta_samples + theta_idx * 3);
                beta_softmax = std::pow(10., *(theta_samples + theta_idx * 3 + 1));
                epsilon      = *(theta_samples + theta_idx * 3 + 2);                
            }
            else if ((temperature == 1)&&(nbAlpha==1))
            {
                if (apply_cst_noise == 0)
                {
                    alpha_c      = *(theta_samples + theta_idx * 3);
                    alpha_u      = *(theta_samples + theta_idx * 3);
                    beta_softmax = 1./(*(theta_samples + theta_idx * 3 + 1));
                    epsilon      = *(theta_samples + theta_idx * 3 + 2);
                    eta          = 0;
                    inertie      = 0;
                }
                else if (apply_inertie == 1)
                {
                    alpha_c      = *(theta_samples + theta_idx * 5);
                    alpha_u      = *(theta_samples + theta_idx * 5);
                    beta_softmax = 1./(*(theta_samples + theta_idx * 5 + 1));
                    epsilon      = *(theta_samples + theta_idx * 5 + 2);
                    eta          = *(theta_samples + theta_idx * 5 + 3);
                    inertie      = *(theta_samples + theta_idx * 5 + 4);
                }
                else
                {
                    alpha_c      = *(theta_samples + theta_idx * 4);
                    alpha_u      = *(theta_samples + theta_idx * 4);
                    beta_softmax = 1./(*(theta_samples + theta_idx * 4 + 1));
                    epsilon      = *(theta_samples + theta_idx * 4 + 2);
                    eta          = *(theta_samples + theta_idx * 4 + 3);
                    inertie      = 0;                    
                } 
            }
            else
            {
                if (apply_cst_noise == 0)
                {
                    alpha_c      = *(theta_samples + theta_idx * 4);
                    alpha_u      = *(theta_samples + theta_idx * 4 + 1);
                    beta_softmax = 1./(*(theta_samples + theta_idx * 4 + 2));
                    epsilon      = *(theta_samples + theta_idx * 4 + 3);
                }
                else
                {
                    alpha_c      = *(theta_samples + theta_idx * 5);
                    alpha_u      = *(theta_samples + theta_idx * 5 + 1);
                    beta_softmax = 1./(*(theta_samples + theta_idx * 5 + 2));
                    epsilon      = *(theta_samples + theta_idx * 5 + 3);
                    eta          = *(theta_samples + theta_idx * 5 + 4);
                }
            }
            
            double weightsSum = 0.;

            if (t_idx > 0)
            {                 
                stratified_resampling(generator, weights_unnorm + theta_idx * N_samples, N_samples, ancestorsIndexes);

                for (int n_idx = 0; n_idx < N_samples; ++n_idx)
                {
                    if (actions[t_idx - 1] == 0)
                    {
                        mu_0 = (1 - alpha_c) * (*(noisy_ancestors + theta_idx * 2 * N_samples + 2 * ancestorsIndexes[n_idx])) + alpha_c * prev_rew[0];
                        mu_1 = (1 - alpha_u) * (*(noisy_ancestors + theta_idx * 2 * N_samples + 2 * ancestorsIndexes[n_idx] + 1)) + alpha_u * prev_rew[1];
                    }
                    else
                    {
                        mu_0 = (1 - alpha_u) * (*(noisy_ancestors + theta_idx * 2 * N_samples + 2 * ancestorsIndexes[n_idx])) + alpha_u * prev_rew[0];
                        mu_1 = (1 - alpha_c) * (*(noisy_ancestors + theta_idx * 2 * N_samples + 2 * ancestorsIndexes[n_idx] + 1)) + alpha_c * prev_rew[1];
                    }

                    if (apply_weber == 1)
                    {
                        noise_level0 = std::log(1 + eta + epsilon * std::abs(prev_rew[0] - (*(noisy_ancestors + theta_idx * N_samples * 2 + 2 * ancestorsIndexes[n_idx]))));
                        noise_level1 = std::log(1 + eta + epsilon * std::abs(prev_rew[1] - (*(noisy_ancestors + theta_idx * N_samples * 2 + 2 * ancestorsIndexes[n_idx] + 1))));
                    }
                    else
                    {
                        noise_level0 = epsilon;
                        noise_level1 = epsilon;
                    }

                    *(amount_noise_descendants + theta_idx * 2 * N_samples + 2 * n_idx)     = inertie * (*(amount_noise_ancestors + theta_idx * N_samples * 2 + 2 * ancestorsIndexes[n_idx])) + (1 - inertie) * distribution(generator) * noise_level0;
                    *(amount_noise_descendants + theta_idx * 2 * N_samples + 2 * n_idx + 1) = inertie * (*(amount_noise_ancestors + theta_idx * N_samples * 2 + 2 * ancestorsIndexes[n_idx] + 1)) + (1 - inertie) * distribution(generator) * noise_level1;                    
                    *(noisy_descendants + theta_idx * 2 * N_samples + 2 * n_idx)     = *(amount_noise_descendants + theta_idx * 2 * N_samples + 2 * n_idx) + mu_0;
                    *(noisy_descendants + theta_idx * 2 * N_samples + 2 * n_idx + 1) = *(amount_noise_descendants + theta_idx * 2 * N_samples + 2 * n_idx + 1) + mu_1;

                    *(weights_unnorm + theta_idx * N_samples + n_idx) = log_logistic_proba(beta_softmax, *(noisy_descendants + theta_idx * 2 * N_samples + 2 * n_idx), *(noisy_descendants + theta_idx * 2 * N_samples + 2 * n_idx + 1), act);
                }     
            }
            else
            {
                for (int n_idx = 0; n_idx < N_samples; ++n_idx)
                {
                    *(noisy_descendants + theta_idx * 2 * N_samples + 2 * n_idx)     = 1./2;
                    *(noisy_descendants + theta_idx * 2 * N_samples + 2 * n_idx + 1) = 1./2;
                    *(amount_noise_descendants + theta_idx * 2 * N_samples + 2 * n_idx)     = 0.;
                    *(amount_noise_descendants + theta_idx * 2 * N_samples + 2 * n_idx + 1) = 0.;

                    *(weights_unnorm + theta_idx * N_samples + n_idx)                = std::log(1/2.);                    
                }
            }

            double b   = *max_element(weights_unnorm + theta_idx * N_samples, weights_unnorm + theta_idx * N_samples + N_samples);
            for (int n_idx = 0; n_idx < N_samples; ++n_idx)
            {
                *(weights_unnorm + theta_idx * N_samples + n_idx) = exp(*(weights_unnorm + theta_idx * N_samples + n_idx) - b);
                weightsSum                                       += *(weights_unnorm + theta_idx * N_samples + n_idx);
            }

            *(logThetaWeights + theta_idx) += (b + log(weightsSum) - log(N_samples));
            *(logThetaLks + theta_idx)     += (b + log(weightsSum) - log(N_samples));
            *(log_lkd + theta_idx)          = (b + log(weightsSum) - log(N_samples));

            if (std::isnan(*(log_lkd + theta_idx)))
            {
                std::cout <<b << std::endl;
                std::cout << weightsSum << std::endl;
            }

            for (int n_idx = 0; n_idx < N_samples; ++n_idx)
            {
                *(weights_unnorm + theta_idx * N_samples + n_idx) = *(weights_unnorm + theta_idx * N_samples + n_idx)/weightsSum;
            }
        }
        return ;
    }
}
















