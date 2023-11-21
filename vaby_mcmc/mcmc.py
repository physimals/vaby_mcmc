"""
VABY_MCMC - Markov Chain Monte Carlo inference method
"""
import numpy as np
import tensorflow as tf

from vaby import InferenceMethod, NP_DTYPE, TF_DTYPE
from vaby.parameter import NoiseParameter
import vaby.dist as dist

class Mcmc(InferenceMethod):
    """
    MCMC model fitting
    """

    def __init__(self, data_model, burnin=1000, samples=1000, **kwargs):
        """
        :param burnin: Number of burn-in iterations
        :param samples: Number of samples to extract
        """
        InferenceMethod.__init__(self, data_model, **kwargs)

        if len(data_model.structures) > 1:
            raise NotImplementedError("MCMC inference does not currently support multiple structures")

        self.burnin = burnin
        self.samples = samples
        self.infer_covar = False
        self.display_step = kwargs.get("display_step", 1)

    def run(self):
        """
        Run MCMC

        We use the posterior as the proposal distribution, but the actual posterior
        is defined by the post-burnin samples
        """
        self.log.info("Starting MCMC inference")
        self.log.info(" -  %i voxels of %i time points" , self.data_model.data_space.size, self.data_model.ntpts)
        self.log.info(" - Number of burn-in iterations: %i", self.burnin)
        self.log.info(" - Number of samples to collect: %i", self.samples)

        self._create_prior_post()

        struc = self.data_model.structures[0]
        model = struc.model
        n_params = model.nparams
        # f_params = [idx for idx, p in enumerate(model.params) if p.name.startswith("f")]
        accepted = 0
        MH_rejected = 0
        bounds_rejected = 0

        samples = np.zeros([struc.size, self.burnin + self.samples, n_params])  # [W, S, P]
        #sample_history = np.copy(samples)
        #decision_history = np.copy(samples)
        #likelihood_history = np.zeros(struc.size, self.burnin + self.samples)

        # MCMC iterations
        proposal = self.post.mean  # [W, P]
        print("p", proposal.shape)
        log_likelihood = self._log_likelihood(proposal)  # [W]
        for it in range(self.burnin + self.samples):
            samples[:, it, :] = proposal
            if it % self.display_step == 0:
                status = "BURNIN" if it < self.burnin else "SAMPLE"
                self.log.info(" - Iteration %04d - %s" % (it, status))
                if it >= self.burnin:
                    mean = self.log_avg(self.samples[:, self.burnin:it + 1, :], axis=(0, 1))
                    self.log.info(f"   - Mean: {mean}")
                    #self.log.info(f"   - Noise mean: %.4g variance: %.4g" % (self.log_avg(state["noise_mean"]), self.log_avg(state["noise_var"])))
                    #self.log.info(f"   - Cost: %.4g (latent %.4g, reconst %.4g)" % (state["cost"], state["latent"], state["reconst"]))

            # new_proposal = np.random.multivariate_normal(proposal, self.post.cov)  # [W, P]
            new_proposal = tf.reshape(self.post.sample(1), (-1, n_params))  # [W, P]
            print("newp", new_proposal.shape)
            # sample_history[i, :] = new_proposal
            # sumf = 0
            # sumf = sum([new_proposal[idx] for idx in f_params])
            # if np.all((new_proposal) > lb) and np.all((new_proposal) < ub) and sumf <= 1:
            if 1:
                new_log_likelihood = self._log_likelihood(new_proposal)

                # We cannot assume symmetric distribution, so the proposal distribution remains in the Metropolis condition
                q_new2old = self.prior.log_prob(proposal)
                q_old2new = self.prior.log_prob(new_proposal)
                p_accept = (log_likelihood - new_log_likelihood) - q_new2old + q_old2new
                accept = tf.exp(p_accept) > tf.random.uniform(p_accept.shape, dtype=TF_DTYPE)

                log_likelihood = tf.where(accept, new_log_likelihood, log_likelihood)
                proposal = tf.where(accept, new_proposal, proposal)

                self.post.set(proposal, self.post.cov)

        #acc_ratio = np.nan_to_num(100 * ((acc) / (bound_rej + MH_rej)), posinf=100)
        #bound_rej_ratio = np.nan_to_num(100 * ((bound_rej) / (acc + bound_rej + MH_rej)), posinf=100)
        #MH_rej = np.nan_to_num(100 * ((MH_rej) / (acc + bound_rej + MH_rej)), posinf=100)

        samples = samples[:, self.burnin:, :]  # [W, S, P]
        #idx = sort_f(samples[:, self.burnin:, :])
        #hist_samples = hist_samples[:, idx]
        #hist_decision = hist_decision[:, idx]

        self.log.info(" - End of inference. ")

        state = {
            "modelfit" : self._get_model_prediction(),
            f"{struc.name}_mean" : np.transpose(np.mean(samples, axis=1)),  # [P, W]
            f"{struc.name}_var" : np.transpose(np.variance(samples, axis=1)),  # [P, W]
            #f"{struc.name}_cov"] = tf.transpose(self.struc_posts[struc.name].cov, (1, 2, 0))  # [P, P, W]
        }
        return state

    def _create_prior_post(self):
        """
        Create voxelwise prior and posterior distribution tensors
        """
        self.log.info("Setting up prior and posterior")
        self.struc_posts, self.struc_priors = {}, {}
        for struc in self.data_model.structures:
            self.log.info("Structure %s", struc.name)

            # Create posterior distributions for model parameters
            # Note this can be initialized using the actual data
            model_posts, model_priors = [], []
            for idx, param in enumerate(struc.model.params):
                param.init_prior_post(self.data_model.data_space.srcdata.flat)
                model_posts.append(param.post)
                model_priors.append(param.prior)
                self.log.info("   - %s", param)
                self.log.info("     - Prior: %s", param.prior)
                self.log.info("     - Posterior: %s", param.post)

            # Create combined posterior
            all_normal = all([p.is_normal() for p in model_posts])
            if self.infer_covar and all_normal:
                if all_normal:
                    self.log.info(" - Inferring covariances (correlation) between %i model parameters" % len(model_posts))
                    post = dist.MVN(model_posts)
                else:
                    self.log.warn(" - Cannot infer covariance - not all model parameters have Normal distributiuon")
                    post = dist.FactorisedDistribution(model_posts)
            else:
                self.log.info(" - Not inferring covariances between parameters")
                post = dist.FactorisedDistribution(model_posts)

            self.struc_posts[struc.name] = post
            self.post = self.struc_posts[struc.name]

            # Create combined prior - always factorized
            self.struc_priors[struc.name] = dist.FactorisedDistribution(model_priors)
            self.prior = self.struc_priors[struc.name]

        # The noise parameter is defined separately to model parameters in
        # the acquisition data space
        self.noise_param = NoiseParameter(self.data_model.data_space)
        self.noise_param.init_prior_post(self.data_model.data_space.srcdata.flat)
        self.noise_post = self.noise_param.post
        self.noise_prior = self.noise_param.prior
        self.post.build()
        self.prior.build()
        self.noise_post.build()
        self.noise_prior.build()

        self.log.info("Noise")
        self.log.info(" - Prior: %s", self.noise_param.prior)
        self.log.info(" - Posterior: %s", self.noise_param.post)

    def _get_model_prediction(self, proposal):
        """
        Get a model prediction for the proposal parameters

        :return Tensor [W, N]
        """
        model_prediction = None
        for struc in self.data_model.structures:
            model_tpts = struc.model.tpts()  # [W/1, N]
            param_samples = proposal  # [W, P]
            param_samples = tf.transpose(param_samples)[..., tf.newaxis]  # [P, W, 1]
            print(param_samples.shape)
            pred = struc.model.evaluate(param_samples, model_tpts)  # [W, N]
            print(pred.shape)
            pred = struc.to_source_space(pred)
            if model_prediction is None:
                model_prediction = pred
            else:
                model_prediction += pred

            return model_prediction  # [W, N]

    def _log_likelihood(self, proposal):
        """
        Log-likelihood of the proposal
        """
        model_prediction = self._get_model_prediction(proposal)  # [W, N]
        ll = self.noise_param.log_likelihood(self.data_model.data_space.srcdata.flat, model_prediction[:, tf.newaxis, :], self.noise_param.post.mean, self.data_model.ntpts)  # [W]
        print(proposal.shape)
        ll += self.prior.log_prob(proposal)  # [W]

        return ll  # [W]
