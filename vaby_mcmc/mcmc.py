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

    def __init__(self, data_model, burnin=1000, noise_burnin_fraction=0.5, samples=1000, **kwargs):
        """
        :param burnin: Number of burn-in iterations
        :param samples: Number of samples to extract
        """
        InferenceMethod.__init__(self, data_model, **kwargs)

        self._num_burnin = burnin
        self._noise_burnin = burnin * noise_burnin_fraction
        self._param_burnin = burnin - self._noise_burnin
        self._num_samples = samples
        self._infer_covar = False
        self._display_step = kwargs.get("display_step", 10)
        self._target_accept = 0.5
        self._review_step = kwargs.get("review_step", 30)

        if self._debug:
            self._get_log_likelihood = self._log_likelihood
        else:
            self._get_log_likelihood = tf.function(self._log_likelihood)

    def run(self):
        """
        Run MCMC

        We use the posterior as the proposal distribution, but the actual posterior
        is defined by the post-burnin samples
        """
        self.log.info("Starting MCMC inference")
        self.log.info(f" - {self.data_model.data_space.size} voxels of {self.data_model.ntpts} time points")
        self.log.info(f" - Number of burn-in iterations: {self._num_burnin}")
        self.log.info(f" - Parameter-only jumps for {self._param_burnin} burn-in iterations")
        self.log.info(f" - Number of samples to collect: {self._num_samples}")
        self.log.info(f" - Target acceptance rate: {self._target_accept}, reviewed every {self._review_step} iterations")

        self._init_structures()

        # MCMC iterations
        for it in range(self._num_burnin + self._num_samples):
            for name, struc_data in self.struc_data.items():
                self._jump_params(it, name, struc_data)

            if it % self._display_step == 0:
                self._log_iter(it)

        self.log.info(" - End of inference. ")
        return self._get_state()

    def _get_state(self):
        state = {
            "modelfit" : self._get_model_prediction(),
        }
        for name, struc_data in self.struc_data.items():
            struc = struc_data["struc"]
            samples = struc_data["samples"][:, self._num_burnin:, :]  # [W, S, P]
            if name == "noise":
                state[f"{name}_mean"] = np.mean(samples, axis=1).squeeze(axis=1)  # [W]
                state[f"{name}_var"] = np.var(samples, axis=1).squeeze(axis=1)  # [W]
            else:
                state[f"{name}_mean"] = np.transpose(np.mean(samples, axis=1))  # [P, W]
                state[f"{name}_var"] = np.transpose(np.var(samples, axis=1))  # [P, W]
                # We need to define covariance state even though we are not inferring it
                cov = np.zeros([struc.size, struc.model.nparams, struc.model.nparams])
                for param_idx in range(struc.model.nparams):
                    cov[:, param_idx, param_idx] = state[f"{name}_var"][param_idx, :]
                state[f"{struc.name}_cov"] = cov

        return state

    def _jump_params(self, it, name, struc_data):
        struc = struc_data["struc"]
        new_proposal_cov = struc_data["proposal_dist"].cov.numpy()
        struc_data["samples"][:, it, :] = struc_data["proposal"]
        if name == "noise":
            num_params = 1
            if it < self._param_burnin:
                # Keep noise constant during parameter-only burnin
                return
        else:
            num_params = struc.model.nparams

        # Jitter all the parameters together but then iterate over each
        # so only one is varied at a time
        new_proposals = tf.reshape(struc_data["proposal_dist"].sample(1), (-1, num_params))  # [W, P]
        for param_idx in range(num_params):
            current_proposal = struc_data["proposal"]
            current_log_likelihood = self._get_log_likelihood(struc_data, current_proposal)
            #current_log_likelihood = struc_data["log_likelihood"]
            param_selector = np.zeros(new_proposals.shape, dtype=bool)
            param_selector[:, param_idx] = 1
            new_proposal = tf.where(tf.constant(param_selector), new_proposals, current_proposal)
            new_log_likelihood = self._get_log_likelihood(struc_data, new_proposal)

            # We cannot assume symmetric distribution, so the proposal distribution remains in the Metropolis condition
            p_current = struc_data["prior"].log_prob(current_proposal)
            p_new = struc_data["prior"].log_prob(new_proposal)
            p_accept = (new_log_likelihood - current_log_likelihood) + (p_new - p_current)  # [W]
            accept = tf.exp(p_accept) > tf.random.uniform(p_accept.shape, dtype=TF_DTYPE)  # [W]

            self.log.debug(f" - structure {name} param {param_idx}")
            self.log.debug(f" - Current proposal: {current_proposal} {current_log_likelihood} {p_current}")
            self.log.debug(f" - New proposal: {new_proposal} {new_log_likelihood} {p_new}")
            self.log.debug(f" - Accept: {accept}")

            struc_data["log_likelihood"] = tf.where(accept, new_log_likelihood, current_log_likelihood)
            struc_data["proposal"] = tf.where(tf.reshape(accept, [-1, 1]), new_proposal, current_proposal)
            accept_num = tf.cast(accept, TF_DTYPE)  # [W]
            struc_data["accepted"][param_idx] += accept_num
            struc_data["rejected"][param_idx] += (1 - accept_num)

            if it > 0 and it % self._review_step == 0:
                # CHECK THE ACCEPTANCE RATIO
                # Following Rosenthal 2010 and so on...
                # For RWM, sigma=l/sqrt(d), Acc_ratio(l_opt) is:
                # approx 0.234 for d-dimensional with d-->inf
                # 0.44 for 1 dimension
                #
                # For MALA, sigma=l/(d^(1/6)), Acc_ratio(l_opt) is:
                # approx 0.574 for d-dimension in MALA
                # FSL BedpostX (Behrens, 2003) --> Don't retain ergodicity
                accepted, rejected = struc_data["accepted"][param_idx], struc_data["rejected"][param_idx]  # [W]
                accepted_ratio = accepted / (accepted + rejected)  # [W]
                self.log.info(f" - Updating proposal variance struc {name} param {param_idx}: Current acceptance rate {accepted_ratio} target {self._target_accept}")
                # acc*0.25 to have ~80% acc ratio --- acc*3 to have ~25% acc_ratio
                adjust_factor = np.sqrt((1 + accepted) / (1 + rejected))  # [W]
                current_variance = new_proposal_cov[:, param_idx, param_idx]  # [W]
                new_variance = current_variance * adjust_factor  # [W]
                self.log.debug(f" - Current variance {current_variance} adjustment factor {adjust_factor} new variance: {new_variance}")
                new_proposal_cov[:, param_idx, param_idx] = np.maximum(1e-6, np.minimum(1e6, new_variance))  # [W]
                struc_data["accepted"][param_idx] = 0
                struc_data["rejected"][param_idx] = 0

        struc_data["proposal_dist"].set(struc_data["proposal"], new_proposal_cov)

    def _log_iter(self, it):
        status = "PARAM-ONLY BURNIN" if it < self._param_burnin else "BURNIN" if it < self._num_burnin else "SAMPLE"
        self.log.info(" - Iteration %04d - %s" % (it, status))
        if it >= self._num_burnin:
            for name, struc_data in self.struc_data.items():
                samples = struc_data["samples"][:, self._num_burnin:it + 1, :]  # [W, S, P]
                node_means, node_vars = tf.math.reduce_mean(samples, axis=1), tf.math.reduce_variance(samples, axis=1)  # [W, P]
                mean = self.log_avg(node_means, axis=0)
                var = self.log_avg(node_vars, axis=0)
                self.log.info(f"   - {name} mean: {mean} variance: {var}")
                #for name, variable in self.struc_priors[struc.name].trainable_variables.items():
                #    self.log.info(f"   - Structure {struc.name} {name}: %s" % self.log_avg(variable.numpy()))

        else:
            for name, struc_data in self.struc_data.items():
                proposal = struc_data["proposal"] # [W, P]
                mean = self.log_avg(proposal, axis=0)
                self.log.info(f"   - {name} mean proposal: {mean}")

    def _init_structures(self):
        """
        Create voxelwise prior and posterior distribution tensors
        """
        self.log.info("Setting up data structures")
        self.struc_data = {}
        for struc in self.data_model.structures:
            self.struc_data[struc.name] = {"name" : struc.name, "struc" : struc}
            self.log.info("Structure %s", struc.name)

            # Create posterior distributions for model parameters
            # Note this can be initialized using the actual data
            # and that we initialize the variance on the posterior
            # to unity as this will be used as the proposal distribution
            model_posts, model_priors = [], []
            for idx, param in enumerate(struc.model.params):
                param.init_prior_post(self.data_model.data_space.srcdata.flat)
                param.post.build()
                param.prior.build()
                param.post.set(param.post.mean, 1.0)
                model_posts.append(param.post)
                model_priors.append(param.prior)
                self.log.info("   - %s", param)
                self.log.info("     - Prior: %s", param.prior)
                self.log.info("     - Posterior: %s", param.post)

            # Create combined prior and proposal dists - always factorised for now
            self.struc_data[struc.name]["proposal_dist"] = dist.FactorisedDistribution(model_posts)
            self.struc_data[struc.name]["prior"] = dist.FactorisedDistribution(model_priors)
            self.struc_data[struc.name]["proposal_dist"].build()
            self.struc_data[struc.name]["prior"].build()
            #print(struc.name, self.struc_data[struc.name]["proposal_dist"].cov.numpy().mean(axis=0))

            # Initialize MCMC structures
            self.struc_data[struc.name]["accepted"] = [0] * struc.model.nparams
            self.struc_data[struc.name]["rejected"] = [0] * struc.model.nparams
            self.struc_data[struc.name]["samples"] = np.zeros([struc.size, self._num_burnin + self._num_samples, struc.model.nparams], dtype=NP_DTYPE)  # [W, S, P]
            self.struc_data[struc.name]["proposal"] = self.struc_data[struc.name]["proposal_dist"].mean  # [W, P]

        # The noise parameter is defined separately to model parameters in
        # the acquisition data space
        self.noise_param = NoiseParameter(self.data_model.data_space)
        self.noise_param.init_prior_post(self.data_model.data_space.srcdata.flat)
        self.noise_param.post.build()
        self.noise_param.prior.build()
        #self.noise_param.post.set(self.noise_param.post.mean, 1.0)
        self.noise_param.post.set(1, 1.0)
        self.struc_data["noise"] = {"name" : "noise", "struc" : self.data_model.data_space}
        self.struc_data["noise"]["proposal_dist"] = dist.FactorisedDistribution([self.noise_param.post])
        self.struc_data["noise"]["prior"] = dist.FactorisedDistribution([self.noise_param.prior])
        self.struc_data["noise"]["proposal_dist"].build()
        self.struc_data["noise"]["prior"].build()

        # Initialize MCMC structures
        self.struc_data["noise"]["accepted"] = [0]
        self.struc_data["noise"]["rejected"] = [0]
        self.struc_data["noise"]["samples"] = np.zeros([self.data_model.data_space.size, self._num_burnin + self._num_samples, 1], dtype=NP_DTYPE)  # [W, S, 1]
        self.struc_data["noise"]["proposal"] = self.struc_data["noise"]["proposal_dist"].mean  # [W, 1]

        self.log.info("Noise")
        self.log.info(" - Prior: %s", self.noise_param.prior)
        self.log.info(" - Proposal: %s", self.noise_param.post)

        for name, struc_data in self.struc_data.items():
            self.struc_data[name]["log_likelihood"] = self._log_likelihood(struc_data)  # [W]

    def _get_model_prediction(self, struc_data=None, proposal=None):
        """
        Get a model prediction for the proposal parameters

        :return Tensor [W, N]
        """
        model_prediction = None
        for struc in self.data_model.structures:
            model_tpts = struc.model.tpts()  # [W/1, N]
            if struc_data is not None and struc.name == struc_data["name"] and proposal is not None:
                param_samples = proposal
            else:
                param_samples = self.struc_data[struc.name]["proposal"]  # [W, P]
            param_samples = tf.transpose(param_samples)[..., tf.newaxis]  # [P, W, 1]
            pred = struc.model.evaluate(param_samples, model_tpts)  # [W, N]
            pred = struc.to_source_space(pred)
            if model_prediction is None:
                model_prediction = pred
            else:
                model_prediction += pred

        return model_prediction  # [W, N]

    def _log_likelihood(self, struc_data, proposal=None):
        """
        Log-likelihood of the proposal
        """
        if proposal is None:
            proposal = struc_data["proposal"]

        if struc_data["name"] == "noise":
            noise_proposal = proposal
            #print("using noise proposal", proposal.shape)
        else:
            noise_proposal = self.struc_data["noise"]["proposal"]
            #print("existing noise proposal", noise_proposal.shape)
        model_prediction = self._get_model_prediction(struc_data, proposal)  # [V, N]
        ll = self.noise_param.log_likelihood(self.data_model.data_space.srcdata.flat, model_prediction[:, tf.newaxis, :], noise_proposal, self.data_model.ntpts)  # [V]

        return struc_data["struc"].from_source_space(ll)  # [W]
