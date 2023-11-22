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

        self._num_burnin = burnin
        self._num_samples = samples
        self._infer_covar = False
        self._display_step = kwargs.get("display_step", 10)
        self._target_accept = kwargs.get("target_accept", 0.5)
        self._review_steps = kwargs.get("review_steps", 30)

    def run(self):
        """
        Run MCMC

        We use the posterior as the proposal distribution, but the actual posterior
        is defined by the post-burnin samples
        """
        self.log.info("Starting MCMC inference")
        self.log.info(f" - {self.data_model.data_space.size} voxels of {self.data_model.ntpts} time points")
        self.log.info(f" - Number of burn-in iterations: {self._num_burnin}")
        self.log.info(f" - Number of samples to collect: {self._num_samples}")
        self.log.info(f" - Target acceptance rate: {self._target_accept}, reviewed every {self._review_steps} iterations")

        self._create_prior_post()

        struc = self.data_model.structures[0]
        model = struc.model
        num_params = model.nparams
        num_voxels = self.data_model.data_space.size
        accepted, rejected = [0] * num_params, [0] * num_params
        post_samples = np.zeros([struc.size, self._num_burnin + self._num_samples, num_params], dtype=NP_DTYPE)  # [W, S, P]
        if self._debug:
            get_log_likelihood = self._log_likelihood
        else:
            get_log_likelihood = tf.function(self._log_likelihood)

        # Set up proposal distribution
        proposal = self.post.mean  # [W, P]
        proposal_cov = np.zeros([num_voxels, num_params, num_params], dtype=NP_DTYPE)  # [W, P, P]
        for param_idx in range(num_params):
            proposal_cov[:, param_idx, param_idx] = 1
        self.post.set(proposal, proposal_cov)

        log_likelihood = get_log_likelihood(proposal)  # [W]

        # MCMC iterations
        for it in range(self._num_burnin + self._num_samples):
            post_samples[:, it, :] = proposal
            if it % self._display_step == 0:
                self._log_iter(it, post_samples)

            # Jitter all the parameters together but then iterate over each
            # so only one is varied at a time
            new_proposals = tf.reshape(self.post.sample(1), (-1, num_params))  # [W, P]
            for param_idx in range(num_params):
                param_selector = np.zeros(new_proposals.shape, dtype=bool)
                param_selector[:, param_idx] = 1
                new_proposal = tf.where(tf.constant(param_selector), new_proposals, proposal)
                new_log_likelihood = get_log_likelihood(new_proposal)

                # We cannot assume symmetric distribution, so the proposal distribution remains in the Metropolis condition
                q_new2old = self.prior.log_prob(proposal)
                q_old2new = self.prior.log_prob(new_proposal)
                p_accept = (log_likelihood - new_log_likelihood) - q_new2old + q_old2new
                accept = tf.exp(p_accept) > tf.random.uniform(p_accept.shape, dtype=TF_DTYPE)

                log_likelihood = tf.where(accept, new_log_likelihood, log_likelihood)
                proposal = tf.where(tf.reshape(accept, [-1, 1]), new_proposal, proposal)
                param_accepted = tf.reduce_sum(tf.cast(accept, TF_DTYPE))
                accepted[param_idx] += param_accepted
                rejected[param_idx] += (num_voxels - param_accepted)

            if it > 0 and it % self._review_steps == 0:
                for param_idx in range(num_params):
                    # CHECK THE ACCEPTANCE RATIO
                    # Following Rosenthal 2010 and so on...
                    # For RWM, sigma=l/sqrt(d), Acc_ratio(l_opt) is:
                    # approx 0.234 for d-dimensional with d-->inf
                    # 0.44 for 1 dimension
                    #
                    # For MALA, sigma=l/(d^(1/6)), Acc_ratio(l_opt) is:
                    # approx 0.574 for d-dimension in MALA
                    # FSL BedpostX (Behrens, 2003) --> Don't retain ergodicity
                    accepted_ratio = accepted[param_idx] / (accepted[param_idx] + rejected[param_idx])
                    # acc*0.25 to have ~80% acc ratio --- acc*3 to have ~25% acc_ratio
                    adjust_factor = np.sqrt((1 + accepted[param_idx]) / (1 + rejected[param_idx]))
                    proposal_cov[:, param_idx, param_idx] = np.minimum(1e10, proposal_cov[:, param_idx, param_idx] * adjust_factor)
                    self.log.debug(f" - Updating proposal variance {param_idx}: Current acceptance rate {accepted_ratio} target {self._target_accept} adjustment factor {adjust_factor}")
                accepted, rejected = [0] * num_params, [0] * num_params

            self.post.set(proposal, proposal_cov)

        post_samples = post_samples[:, self._num_burnin:, :]  # [W, S, P]
        self.log.info(" - End of inference. ")
        state = {
            "modelfit" : self._get_model_prediction(np.mean(post_samples, axis=1)),
            f"{struc.name}_mean" : np.transpose(np.mean(post_samples, axis=1)),  # [P, W]
            f"{struc.name}_var" : np.transpose(np.var(post_samples, axis=1)),  # [P, W]
        }
        cov = np.zeros([num_voxels, num_params, num_params])
        for param_idx in range(num_params):
            cov[:, param_idx, param_idx] = np.var(post_samples, axis=1)[:, param_idx]
        state[f"{struc.name}_cov"] = cov

        return state

    def _log_iter(self, it, post_samples):
        status = "BURNIN" if it < self._num_burnin else "SAMPLE"
        self.log.info(" - Iteration %04d - %s" % (it, status))
        if it >= self._num_burnin:
            mean = self.log_avg(post_samples[:, self._num_burnin:it + 1, :], axis=(0, 1))
            self.log.info(f"   - Mean: {mean}")

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
            all_normal = all([isinstance(p, dist.Normal) for p in model_posts])
            if self._infer_covar and all_normal:
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
            pred = struc.model.evaluate(param_samples, model_tpts)  # [W, N]
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
        prior_ll = self.prior.log_prob(proposal)  # [W]
        ll += tf.reshape(prior_ll, [-1])

        return ll  # [W]
