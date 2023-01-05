//
//  smc_functions.cpp
//  smc_functions
//
//  Created by Charles Findling on 3/16/16.
//  Copyright © 2016 Charles Findling. All rights reserved.
//

#include "smc_functions.hpp"
#include "usefulFunctions.hpp"
#include "usefulFunctions.cpp"
#include <boost/random/mersenne_twister.hpp>

namespace smc {
    
    using namespace std;
    
    boost::mt19937 generator(static_cast<unsigned int>(time(0)));
    boost::uniform_real<> distribution(0,1);
    boost::normal_distribution<> distribution_normal(0, 1);

    double bootstrap_smc_step(double* logParamWeights, double* distances, int* currentTaskSetSamples, int* ancestorTaskSetSamples, double* weightsList, double* const paramDirichletSamples, 
                                    double* const paramBetaSamples, double lambda, double eta, double noise_inertie, int* mapping, int const currentStimulus, int const reward, int const action,
                                        int numberOfParamSamples, int numberOfLatentSamples, int const K, int t, int const numberOfStimuli, double* likelihoods, int* positiveStates, 
                                        double* ante_proba_local, double* post_proba_local, int* ancestorsIndexes, double* gammaAdaptedProba, double* sum_weightsList, double* currentNoises, double temperature)
    {
        //std::vector<double> likelihoods(K,1);
        //std::vector<double> ante_weights(K,0.);
        //std::vector<double> parameters_dirichlet(K,1);
        //std::vector<double> sum_transition_proba(K, 0.);
        //std::vector<bool> positiveStates(K,1);
        //std::vector<double> cumSum(K, 0.);
        //std::vector<double> post_weights(K * numberOfParamSamples, 0.);
        //std::vector<double> sum_post_weights(numberOfParamSamples, 0.);
        //double temperature        = 0.;
        double partMargLikelihood = 0.;
        double cumSum         = 0.;
        int index             = 0;
        double uniform_sample = 0.;
        double sum_gammaAdaptedProba = 0.;
        double distance           = 0.;
        //std::vector<double> distances(K * numberOfParamSamples, 0.);
        std::vector<double> ante_proba(K, 0.);
        std::vector<double> post_proba(K, 0.);
/*        std::vector<double> ante_proba_local(K, 0.);
        std::vector<double> post_proba_local(K, 0.);
        std::vector<int> ancestorsIndexes(numberOfLatentSamples, 0);*/
        //std::vector<double> gamma(K, 0);
/*        std::vector<double> gammaAdaptedProba(K, 0);
        std::vector<double> sum_weightsList(numberOfParamSamples);*/
        int ancestor;
        double local_temp = 0.5;
        //double distance;
        std::vector<double> noisy_post_proba(K, 0.);
        double distance_;
        //double alpha = 1.;
        //double beta  = 0.;
        //std::vector<double> temperature_vec(K);

        if (t > 0)
        {
            //positiveStates = isEqual_and_adapted_logical_xor(mapping, currentStimulus, numberOfStimuli, K, action, reward);
            isEqual_and_adapted_logical_xor(mapping, currentStimulus, numberOfStimuli, K, action, reward, positiveStates);

            double maxParamW   = *max_element(logParamWeights, logParamWeights + numberOfParamSamples); // weight rescaling with maximum of weights at time t-1
            double sumParamW_ante = 0;
            double sumParamW_post = 0;

            for (int param_idx = 0; param_idx < numberOfParamSamples; ++param_idx)
            {
/*                ante_proba_local.assign(K, 0.);
                post_proba_local.assign(K, 0.);*/

                for (int k = 0; k < K; ++k) 
                {
                    likelihoods[k]      = positiveStates[k] * (*(paramBetaSamples + param_idx)) + (1 - positiveStates[k]) * (1 - (*(paramBetaSamples + param_idx)));
                    ante_proba_local[k] = 0.;
                    post_proba_local[k] = 0.;
                }

                double sum_post_weights_local = 0;
                for (int traj_idx = 0; traj_idx < numberOfLatentSamples; ++traj_idx)
                {
                    ante_proba_local[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)] += 1;
                    post_proba_local[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)] += likelihoods[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)];
                    sum_post_weights_local += likelihoods[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)];

                    *(weightsList + param_idx * numberOfLatentSamples + traj_idx) = likelihoods[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)];
                }
                //divide(weightsList + param_idx * numberOfLatentSamples, sum_post_weights_local, numberOfLatentSamples);

                sum_weightsList[param_idx] = sum_post_weights_local;

                // double sum_weights_diff  = 0.;
                // *(distances + param_idx) = 0.;
                for (int k = 0; k < K; ++k)
                {
                    ante_proba_local[k] /= (1. * numberOfLatentSamples);
                    ante_proba[k]       += exp(*(logParamWeights + param_idx) - maxParamW) * ante_proba_local[k];
                    //*(distances + param_idx) += lambda * abs(ante_proba_local[k] - post_proba_local[k]);
                }

                //*(distances + param_idx) += eta;

                sumParamW_ante                 += exp(*(logParamWeights + param_idx) - maxParamW);
                partMargLikelihood              = sum_post_weights_local/numberOfLatentSamples;
                *(logParamWeights + param_idx) += log(partMargLikelihood);
                sumParamW_post                 += exp(*(logParamWeights + param_idx) - maxParamW);

                // sumParamW   += exp(*(logParamWeights + param_idx) - maxParamW);
                for (int k = 0; k < K; ++k)
                {
                    post_proba_local[k] /= sum_post_weights_local;
                    post_proba[k] += exp(*(logParamWeights + param_idx) - maxParamW) * post_proba_local[k];
                }
            }

            distance_ = 0.;
            for (int k = 0; k < K; ++k)
            {
                if (post_proba[k] > 0)
                {                
                    distance_     += abs(ante_proba[k]/sumParamW_ante - post_proba[k]/sumParamW_post); //- (post_proba[k]/sumParamW_post) * log((post_proba[k]/sumParamW_post)); ///(ante_proba[k]/sumParamW_ante)); //   //                
                    post_proba[k] = post_proba[k]/sumParamW_post;
                }
            }
            //std::cout << distance << std::endl;
            distance_ = lambda * distance_ + eta; // lambda * std::log(1 + distance) + eta; //lambda * std::log10(1 + distance); 

            *(distances) = distance_;
            local_temp   = distribution(generator) * distance_; //(*(distances + param_idx));

/*            if(post_proba[0] > 6000){
                std::cout << 'wtf' << std::endl;
                std::cout << sumParamW_post << std::endl;
                throw 10;
            }*/
            //std::cout << post_proba[0] << '\n' << post_proba[1] << std::endl;
            //std::cout << sumParamW_post << std::endl;
            //std::cout << '\n' << std::endl;
            //*(temperaturesList + t) = temperature;
            //temperature_vec = Sample_Uniform_Distribution(generator, K) * temperature;

            //copy(&temperature_vec[0], &temperature_vec[0] + K, temperaturesList + t * K);            
/*            if (K == 2)
                {
                            std::cout << '\n' <<std::endl; 
                    print(post_proba);
                    Sample_Dirichlet_Distribution(generator, &post_proba[0], 1., K, 1./(temperature), &noisy_post_proba[0]);
                }*/

            //distance = temperature * noise_inertie + distance;

            //local_temp = distribution(generator) * distance; //( (1 - *(weightsList + param_idx * numberOfLatentSamples + ancestorsIndexes[traj_idx])) * lambda + eta) ; //distribution(generator) * (*(distances + K * param_idx + ancestor)); // (*(distances + param_idx)); //
/*            while (local_temp < 0)
            {
                local_temp = distribution(generator) * distance;
            }*/

            //std::cout << t << "\s" << temperature << std::endl;
            
            //std::cout << local_temp << std::endl;
            //std::cout << '\n' << std::endl;       
                                    
        }        

    

        for (int param_idx = 0; param_idx < numberOfParamSamples; ++param_idx) 
        {
            if (t > 0)
            {   
                // Assign gamma vector
                // gamma.assign(paramDirichletSamples + param_idx * K, paramDirichletSamples + (param_idx + 1) * K);

                // Ancestors for theta sample i
                // ancestorsIndexes = Sample_Discrete_Distribution(generator, weightsList + param_idx * numberOfLatentSamples, numberOfLatentSamples, numberOfLatentSamples);
                // ancestorsIndexes = stratified_resampling(generator, weightsList + param_idx * numberOfLatentSamples, numberOfLatentSamples);
                //Sample_Dirichlet_Distribution(generator, weightsList + param_idx * numberOfLatentSamples, sum_weightsList[param_idx], numberOfLatentSamples, numberOfLatentSamples/(1. + temperature), weightsList + param_idx * numberOfLatentSamples);
                //std::cout << numberOfLatentSamples/(1. + temperature)<< std::endl;
                //sum_weightsList[param_idx] = 1.;
                //stratified_resampling(generator, distribution(generator), weightsList + param_idx * numberOfLatentSamples, numberOfLatentSamples, &ancestorsIndexes[0], sum_weightsList[param_idx], temperature);
                stratified_resampling(distribution(generator), weightsList + param_idx * numberOfLatentSamples, numberOfLatentSamples, &ancestorsIndexes[0], sum_weightsList[param_idx]);

                for (int traj_idx = 0; traj_idx < numberOfLatentSamples; ++traj_idx)
                {
                    if (K > 0)
                    {
                        ancestor   = *(ancestorTaskSetSamples + param_idx * numberOfLatentSamples + ancestorsIndexes[traj_idx]);                        
                        *(currentNoises + param_idx * numberOfLatentSamples + traj_idx) = local_temp;
                        if (distribution(generator) > local_temp)
                        {
                            *(currentTaskSetSamples + param_idx * numberOfLatentSamples + traj_idx) = ancestor;
                        }
                        else
                        {
                            if (K > 2)
                            {
                                sum_gammaAdaptedProba = isNotEqual_timesVector_notnormalise(mapping, currentStimulus, numberOfStimuli, K, mapping[currentStimulus * K + ancestor], paramDirichletSamples + param_idx * K, &gammaAdaptedProba[0]);
                                // gammaAdaptedProba = isNotEqual_timesVector_normalise(mapping, currentStimulus, numberOfStimuli, K, mapping[currentStimulus * K + ancestor], &gamma[0]);
                                // *(currentTaskSetSamples + param_idx * numberOfLatentSamples + traj_idx) = Sample_Discrete_Distribution(generator, gammaAdaptedProba);
                                index          = 0;
                                cumSum         = gammaAdaptedProba[0]/sum_gammaAdaptedProba;
                                uniform_sample = distribution(generator);

                                while (cumSum < uniform_sample){
                                    index  += 1;
                                    cumSum += gammaAdaptedProba[index]/sum_gammaAdaptedProba;
                                }
                                *(currentTaskSetSamples + param_idx * numberOfLatentSamples + traj_idx) = index;
                            }
                            else
                            {
                                *(currentTaskSetSamples + param_idx * numberOfLatentSamples + traj_idx) = 1 - ancestor;
                            }
                        }
                    }
                    else
                    {
                        *(currentNoises + param_idx * numberOfLatentSamples + traj_idx)         = temperature;
                        *(currentTaskSetSamples + param_idx * numberOfLatentSamples + traj_idx) = Sample_Discrete_Distribution(generator, &noisy_post_proba[0], K);
                    }

                }
            }
            else
            {
                generator.discard(70000);

                for (int traj_idx = 0; traj_idx < numberOfLatentSamples; ++traj_idx) 
                {
                    *(currentTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx) = Sample_Discrete_Distribution(generator, paramDirichletSamples + param_idx * K, K);
                }
            }
        }
        return local_temp;
    };
    
    double guided_smc(int* candidateTaskSetSamples, int const T, double* temperaturesList, double candidateParamBeta, double* candidateParamGamma,
                                     int* const mapping, int const numberOfStimuli, int K, int* const stimuli, int* const rewards, int* const actions, int const numberOfLatentSamples)
    {
        double logApproximateLikelihood = 0;
        
        double weightsSum;
        std::vector<double> likelihoods(K,1);
        // std::vector<double> ante_weights(K,0.);
        std::vector<double> post_weights(K,0.);
        std::vector<double> parameters_dirichlet(K,1.);
        std::vector<double> sum_transition_proba(K,0.);
        std::vector<bool> positiveStates(K,1);
        std::vector<double> cumSum(K, 0.);

        double jump = 1./numberOfLatentSamples;
        double upper_bound  = 0;
        int count           = 0;
        partial_sum(candidateParamGamma, candidateParamGamma + K, cumSum.begin());
        for (int traj_idx = 0; traj_idx < numberOfLatentSamples; ++traj_idx) 
        {
            upper_bound += jump;
            while ((cumSum[count] < upper_bound) && (count < (K - 1))) { ++count;}
            *(candidateTaskSetSamples + traj_idx) = count;
        }

        for (int t = 0; t < T; ++t) {
            
            int action          = *(actions + t);
            int currentStimulus = *(stimuli + t);
            int reward          = *(rewards + t);
            
            positiveStates = isEqual_and_adapted_logical_xor(mapping, currentStimulus, numberOfStimuli, K, action, reward);

            // ante_weights.assign(K, 0.);
            post_weights.assign(K, 0.);
            cumSum.assign(K, 0.);

            for (int k = 0; k < K; ++k)
            {
                likelihoods[k] = positiveStates[k] * candidateParamBeta + (1 - positiveStates[k]) * (1 - candidateParamBeta);
            }

            double sum_post_weights = 0;
            for (int traj_idx = 0; traj_idx < numberOfLatentSamples; ++traj_idx)
            {
                // ante_weights[*(candidateTaskSetSamples + traj_idx)] += 1;
                post_weights[*(candidateTaskSetSamples + traj_idx)] += likelihoods[*(candidateTaskSetSamples + traj_idx)];
                sum_post_weights                                    += likelihoods[*(candidateTaskSetSamples + traj_idx)];
            }

            double sum_param_dirichlet = 0;
            sum_transition_proba.assign(K, 0.);
            parameters_dirichlet.assign(K, 0.);

            for (int k = 0; k < K; ++k)
            {
                for (int k1 = 0; k1 < K; ++k1)
                {
                    if ( (*(mapping + currentStimulus * K + k1)) != (*(mapping + currentStimulus * K + k)))
                    {
                         sum_transition_proba[k1] += (*(candidateParamGamma + k));
                    }
                }
            }

            for (int k = 0; k < K; ++k)
            {
                parameters_dirichlet[k] = post_weights[k] / sum_post_weights * (1 - (*(temperaturesList + (t + 1) * K + k)));

                for (int k1 = 0; k1 < K; ++k1)
                {
                    if ( (*(mapping + currentStimulus * K + k1)) != (*(mapping + currentStimulus * K + k)))
                    {
                        parameters_dirichlet[k]      += post_weights[k1] / sum_post_weights * (*(temperaturesList + (t + 1) * K + k1)) * (*(candidateParamGamma + k)) / sum_transition_proba[k1];
                    }
                }
                sum_param_dirichlet     += parameters_dirichlet[k];
            }

            // cout << sum_param_dirichlet << endl;

            cumSum[0] = parameters_dirichlet[0] / sum_param_dirichlet;
            for (int k = 1; k < K; ++k)
            {
                cumSum[k] = cumSum[k-1] + parameters_dirichlet[k] / sum_param_dirichlet;   
            }

            double jump = 1./numberOfLatentSamples;
            double upper_bound  = 0;
            int count           = 0;
            for (int traj_idx = 0; traj_idx < numberOfLatentSamples; ++traj_idx) 
            {
                upper_bound += jump;
                while ((cumSum[count] < upper_bound) && (count < (K - 1))) { ++count;}
                *(candidateTaskSetSamples + traj_idx) = count;
            }

            logApproximateLikelihood     += log(sum_post_weights/numberOfLatentSamples);
        }
        return logApproximateLikelihood;
    };
}