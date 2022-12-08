# Importations.
import numpy as np
import pandas as pd
import torch
from torch.distributions import Beta
from torch.distributions import Bernoulli
from torch.distributions import Categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from utils import Utils


class BayesianDP(torch.nn.Module):
    
    '''
    This model estimates parameters alpha by maximum a posteriori 
    estimation, based on the maximum likelihood objective described 
    in Ratner et al (2016):
    
    Ratner, A. J., De Sa, C. M., Wu, S., Selsam, D., & Ré, C. (2016). 
    Data programming: Creating large training sets, quickly. Advances 
    in neural information processing systems, 29, 3567-3575.
    https:\\arxiv.org/abs/1605.07723.
    '''
    
    def __init__(self, 
                 alpha, 
                 beta, 
                 priors_alpha,
                 prior_strength = 1):
        
        '''
        Initialize model object with parameters alpha, beta.
        
        Ratner et al (2016) note the following about alpha, beta:
        "...we will assume here that 0.3 ≤ βi ≤ 0.5 and 0.8 ≤ αi ≤ 0.9. 
        We note that while these arbitrary constraints can be changed, 
        they are roughly consistent with our applied experience, where 
        users tend to write high-accuracy and high-coverage labeling 
        functions."
        
        Hence it ~might~ be useful to initialize alpha in these
        value ranges for some datasets.
        
        ---------------
        Parameters:
        ---------------
            alpha (list-like in R^m of floats in [0..1]): initial values 
                for probabilities of labeling function accuracies.
            beta (list-like in R^m of floats in [0..1]): initial values 
                for probabilities of labeling function voting rates. This
                should be computed directly from the labeling function
                output matrix used for training.
            priors_alpha (list of floats): means of Beta distributions for priors
                over alpha.
            prior_strength (numeric; default = 1): value by which to scale
                parameters to prior distributions.
        '''
        
        super().__init__()
        
        # Initialize Utils object for utility functions.
        self.utils = Utils()
        
        # Initialize prior distributions for alpha in R^m.
        # Use prior means to fetch beta distribution parameters.
        # Scale parameters based on prior strength value.
        if 1.0 in priors_alpha:
            raise ValueError("Prior distribution mean of 1.0 must be changed (e.g. to 0.9999).")
        prior_params = self.utils.get_prior_dist_params([100] * len(alpha), priors_alpha)
        prior_params = [(x[0] * prior_strength, x[1] * prior_strength) for x in prior_params]
        distributions = []
        for param_tuple in prior_params:
            distributions.append(Beta(param_tuple[0], param_tuple[1]))
        self.priors_alpha = distributions
        
        # Initialize model parameters alpha, beta.
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, requires_grad = True))
        self.beta = torch.tensor(beta, requires_grad = True)
        
        # Initialize priors over y with Bernoullis where p = 0.5 (uninformative).
        self.y = None
        self.prior_y = Bernoulli(torch.tensor([0.5]))
        self.prior_abstain = Bernoulli(torch.tensor([0.5]))
        
        
    def set_prior_y(self, p = 0.5):
        
        '''
        Reset parameter p to Bernoulli distribution that is the prior over y.
        
        ---------------
        Parameters:
        ---------------
            probs: list of floats summing to one.
        '''
        
        self.prior_y = Bernoulli(torch.tensor([p]))
        
        
    def set_prior_abstain(self, p = 0.5):
        
        '''
        Reset parameter p to Bernoulli distribution that is the prior over
        majority vote abstentions.
        
        ---------------
        Parameters:
        ---------------
            probs: list of floats summing to one.
        '''
        
        self.prior_abstain = Bernoulli(torch.tensor([p]))

    
    def compute_mu(self, lambda_x, decomposed = False):

        '''
        Compute probability of observing lambda(x_i) given an unobserved 
        groundtruth y_i, where lambda(x_i) is a vector of all m labeling function 
        outputs for a single observation x_i.

        ---------------
        Parameters:
        ---------------
            lambda_x (vector with elements in {-1, 0, 1}): vector containing
                all labeling function values for a single observation x_i.
            decomposed (boolean, default = False): flag indicating whether to
                return total mu or mu_pos and mu_neg (probability given gold
                label is 1 vs -1, respectively).
        
        ---------------
        Return values:
        ---------------
            mu (float in [0..1]): probability of observing lambda(x_i) given
                unobserved groundtruth y_i.
        '''
        
        # Assume unobserved y = 1.
        mu_pos = 1
        
        # Assume unobserved y = -1.
        mu_neg = 1
        
        # Clamp parameters so that values are in [0..1].
        alpha_clamped = self.clamp_params()
        
        # Take product of mu for each labeling function lambda_j.
        for j in range(len(lambda_x)):
            
            # Probability that labeling function abstains.
            if lambda_x[j] == 0:
                mu_pos *= (1 - self.beta[j])
                mu_neg *= (1 - self.beta[j])
            # Probability that labeling function votes.
            elif lambda_x[j] == 1:
                # Assume y = 1: labeling function agrees with y.
                mu_pos *= self.beta[j] * alpha_clamped[j]
                # Assume y = -1: labeling function disagrees with y.
                mu_neg *= self.beta[j] * (1 - alpha_clamped[j])
            else:
                # Assume y = 1: labeling function disagrees with y.
                mu_pos *= self.beta[j] * (1 - alpha_clamped[j])
                # Assume y = -1: labeling function agrees with y.
                mu_neg *= self.beta[j] * alpha_clamped[j]
        
        if decomposed:
            return mu_pos, mu_neg
        
        # Sum for both dummy values of unobserved y.
        mu = mu_pos + mu_neg
        
        return mu
    
    
    def compute_likelihood(self, L):

        '''
        Compute negative log-likelihood. This is the negative log of the maximum
        likelihood estimation derived in Ratner et al (2016).
        
        Note: this method is identical to the compute_loss() method in ratner_mle.py. 
        
        This function implements the maximum likelihood described in Ratner
        et al [equation 2]:
        (\hat{\alpha}, \hat{\beta}) = 
        \argmax_{\alpha,\beta} \sum_{x \in S} \log \mathbf{P}_{(\Lambda, Y) 
        \sim \mu_{\alpha,\beta}} (\Lambda = 
        \lambda(x))
        = \argmax\limits_{\alpha,\beta} \sum\limits_{x \in S} \log \left( 
        \sum\limits_{y' \in \{-1,1\}} \mu_{\alpha, \beta} (\lambda(x), y') 
        \right)
        
        For the model distribution described in Ratner et al [equation 1]:
        \mu_{\alpha, \beta}(\Lambda, Y) = 
        \frac{1}{2} \prod_{i=1}^{m} (\beta_i \alpha_i \mathbf{1}_{\{\Lambda_i = Y\}} 
        + \beta_i(1 - \alpha_i) \mathbf{1}_{\{\Lambda_i = -Y\}} + (1 - \beta_i) 
        \mathbf{1}_{\{\Lambda_i = 0\}})
        
        ---------------
        Parameters:
        ---------------
            L (matrix with elements in {-1, 0, 1}): labeling function matrix, where
                0 = abstain and {-1, 1} are class assignments.

        ---------------
        Return values:
        ---------------
            NLL (float): neg log-likelihood value for entire matrix.
        '''
        
        # Offset to prevent log of zero.
        epsilon = 1e-6
        
        NLL = 0
        for row in range(L.shape[0]):
            # Take log of mu for each row of labeling function matrix.
            NLL -= torch.log(self.compute_mu(L[row]) + epsilon)
        
        return NLL
    
    
    def compute_loss(self, L):
        
        '''
        This method computes the loss, i.e. the negative log likelihood minus the 
        log priors.
        
        ---------------
        Parameters:
        ---------------
            L (matrix with elements in {-1, 0, 1}): labeling function matrix, where
                0 = abstain and {-1, 1} are class assignments.

        ---------------
        Return values:
        ---------------
            MAP (float): loss value via maximum a posteriori estimation.
        '''
        
        # Offset to prevent log of zero.
        epsilon = 1e-6
        
        # Compute negative log likelihood.
        MAP = 0
        for row in range(L.shape[0]):
            # Take log of mu for each row of labeling function matrix.
            MAP -= torch.log(self.compute_mu(L[row]) + epsilon)
        
        # Subtract off log priors on alphas.
        alpha_clamped = self.clamp_params()
        for i in range(len(alpha_clamped)):
            log_prior = self.priors_alpha[i].log_prob(alpha_clamped[i] + epsilon)
            MAP -= log_prior

        return MAP
    
    
    def predict(self, 
                L, 
                y = None,
                p_y = 0.5,
                p_abstain = 0.5,
                force_abstain = True,
                tie_policy = "abstain", 
                return_proba = False):
        
        '''
        Use alpha, beta to predict class assignments for each observation in L.
        Each class is assigned by maximizing the joint probability of observing
        lambda(x_i) (the vector of labeling function outputs for input x_i) and
        a given value for unobserved y_i (in {-1, 1}).
        
        ---------------
        Parameters:
        ---------------
            L (matrix with elements in {-1, 0, 1}): labeling function matrix of
                dimension n x m (n rows for n observations, m columns for m
                labeling functions).
            y (default = None): reference vector for defining priors over y. If None,
                the majority vote over L will be used. The user may specify any vector 
                they desire with elements in {-1, 0, 1}.
            p_y (float; default = 0.5): p parameter to Bernoulli prior over majority vote
                labels. Default yields an uninformative prior.
            p_abstain(float; default = 0.5): p parameter to Bernoulli prior over majority 
                vote abstentions. Default yields an uninformative prior.
            force_abstain (boolean; default = True): indicates whether to force the model
                to abstain wherever majority vote abstains.
            tie_policy (string with default = "abstain"): tie-breaking policy in
                the event that both classes have the same probability of occurring.
                Options are "positive" (default to label = 1); "negative" (default
                to label = -1); "random" (randomly select from {-1, 1}); "abstain"
                (default to label = 0).
            return_proba (boolean, default = False): flag indicating whether to
                return probabilities along with integer class assignments.

        ---------------
        Return values:
        ---------------
            labels (numpy array with elements in {-1, 1}): predicted labels for
                observations in L. 
        '''
        
        # Offset to prevent log of zero.
        epsilon = 1e-6
        
        # Set referece y: either majority vote or user-specified vector.
        if y is None:
            # Compute majority vote predictions from training matrix L.
            y_mv = self.utils.vote_majority(L, tie_policy = tie_policy)
            self.y = np.array(y_mv)
        else:
            self.y = np.array(y)
            
        # Get p_abstain from y, if specified.
        if p_abstain is None:
            p_abstain = (self.y == 1).sum() / self.y.shape[0]
            
        # Set prior parameter for y. 
        self.set_prior_y(p = p_y)
        
        # Set prior parameter for abstentions. 
        self.set_prior_abstain(p = p_abstain)

        # Assign label that results in the highest probability.
        labels = np.zeros(shape = (L.shape[0], 1))
        probabilities = np.zeros(shape = (L.shape[0], 1))
        for row in range(L.shape[0]):
            
            # Compute log likelihood of given row.
            mu_pos, mu_neg = self.compute_mu(L[row], decomposed = True)
            mu_pos = torch.log(mu_pos + epsilon)
            mu_neg = torch.log(mu_neg + epsilon)

            # Add log prior over y.
            if self.y[row] == 1:
                mu_pos += self.prior_y.log_prob(1).reshape(mu_pos.shape)
                mu_neg += self.prior_y.log_prob(0).reshape(mu_neg.shape)
            elif self.y[row] == -1:
                mu_pos += self.prior_y.log_prob(0).reshape(mu_pos.shape)
                mu_neg += self.prior_y.log_prob(1).reshape(mu_neg.shape)
            else:
                # Uninformative prior if (p_abstain = 0.5).
                mu_pos += self.prior_abstain.log_prob(1).reshape(mu_pos.shape)
                mu_neg += self.prior_abstain.log_prob(0).reshape(mu_pos.shape)
                # Force model to abstain when majority vote abstains, if specified.
                if force_abstain:
                    mu_pos = mu_neg
            
            # Assign label with highest probability.
            if mu_pos > mu_neg:
                labels[row] = 1
                probabilities[row] = mu_pos.item()
            elif mu_pos < mu_neg:
                labels[row] = -1
                probabilities[row] = mu_neg.item()
            else:
                if tie_policy == "positive":
                    labels[row] = 1
                    probabilities[row] = mu_pos.item()
                elif tie_policy == "negative":
                    labels[row] = -1
                    probabilities[row] = mu_neg.item()
                elif tie_policy == "abstain":
                    labels[row] = 0
                    probabilities[row] = mu_neg.item()
                else:
                    labels[row] = np.random.choice((-1, 1))
                    if labels[row] == 1:
                        probabilities[row] = mu_pos.item()
                    else:
                        probabilities[row] = mu_neg.item()
        
        if return_proba:
            return labels, probabilities
        
        return labels

    
    def fit(self, 
            L, 
            L_val = None,
            y_val = None,
            optimizer = torch.optim.SGD,
            learning_rate = 0.001,
            tie_policy = "abstain",
            epochs = 1000, 
            early_stopping = True,
            patience = 5,
            clip_grads = 10,
            verbose = True):

        '''
        Training loop to estimate optimal alpha.
        
        ---------------
        Parameters:
        ---------------
            L (matrix with elements in {-1, 0, 1}): labeling function matrix of
                dimension n x m (n rows for n observations, m columns for m
                labeling functions).
            L_val (matrix with elements in {-1, 0, 1}): validation set.
            y_val (vector with elements in {-1, 1}): groundtruth labels for L_val.
            optimizer (default = torch.optim.SGD): torch optimizer.
            learning_rate (float; default = 0.001): learning rate for optimizer.
            tie_policy (string; default = "abstain"): tie-breaking policy.
            epochs (int; default = 1000): maximum training epochs.
            early_stopping (boolean; default = True): boolean indicating whether to
                exit training early if validation loss does not decrease for 
                (n = patience) epochs.
            patience (int; default = 5): total permissible epochs for no val loss
                decrease before exiting training.
            clip_grads (int; default = 10): max gradient norm allowable before
                clipping.
            verbose (boolean; default = True): boolean indicating whether to pretty
                print training progress.

        ---------------
        Return values:
        ---------------
            • loss_train_log if L_val is None.
            • [scores_log, epochs_log], [loss_train_log, loss_val_log] if L_val is not None.
        '''
        
        # Set random state seed for reproducibility.
        torch.manual_seed(500)
        
        # Initialize optimizer.
        optimizer = optimizer(self.parameters(), lr = learning_rate)

        # Init data structure to log training and validation data.
        loss_train_log = []
        loss_val_log = []
        scores_log = []
        
        # Set counter for early stopping.
        count = 0
        
        # Adjust print options for pretty print.
        torch.set_printoptions(linewidth = 200)
        
        if verbose:
            # View initial parameter values.
            print("\n----------------- INITIAL PARAMETERS -----------------\n",
                  "\n• INIT ALPHA   =", self.alpha.data, 
                  "\n• INIT BETA    =", self.beta.data,
                  "\n• PRIORS ALPHA =", [(float(x.concentration1.data), float(x.concentration0.data)) for x in self.priors_alpha])
        
        # Train model.
        for epoch in range(epochs):

            # Set to training mode and toggle gradients "on."
            self.train()
            self.toggle_grad(requires_grad = True)
            
            # Compute MAP.
            MAP = self.compute_loss(L)
            
            # Less relevant for this refactor, but for future reference:
            # Pass tensor of ones with an element for each item in MAP.
            # This allows backprop to be called on a non-scalar.
            MAP.backward(torch.ones_like(MAP))
            
            # Clip gradients to prevent vanishing / exploding, if specified.
            if clip_grads is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grads)
            
            # Perform a single optimization step (i.e. parameter update).
            optimizer.step()
            
            # Pretty print progress, if specified.
            if verbose and (epoch % (epochs / 10) == 0):
                alpha_clamped = self.clamp_params()
                print("\n----------------- EPOCH {} -----------------\n".format(epoch),
                      "\n• TRAIN LOSS =", MAP.item())
                if L_val is not None:
                    val_loss = self.compute_loss(L_val)
                    print("• VAL   LOSS =", val_loss.item())
                print("• ALPHA      =", alpha_clamped.data, 
                      "\n• BETA       =", self.beta.data)
                
                print("• ALPHA GRAD =", list(self.parameters())[0].grad)
            
            # Zero gradients.
            optimizer.zero_grad()
            
            # Log training loss.
            loss_train_log.append(MAP) 
            
            # Compute and log validation metrics.
            if L_val is not None:
                val_loss = self.compute_loss(L_val)
                loss_val_log.append(val_loss)
                
                # Predict labels.
                y_pred = self.predict(L_val, 
                                      return_proba = False, 
                                      tie_policy = tie_policy)
                # Score (output is [acc, f1, precision, recall, roc, coverage]).
                try:
                    scores = self.utils.score(y_val, 
                                              y_pred, 
                                              verbose = False,
                                              plot_confusion = False)
                    scores_log.append(scores)
                except ValueError:
                    print("Could not score: all predictions are abstentions.")

                # Early stopping, if specified.
                if early_stopping:
                    if epoch > 0:
                        if round(val_loss.item(), 1) >= round(prev_val_loss, 1):
                            count += 1
                        else:
                            count = 0
                        if count > patience:
                            print("\n--- EARLY STOPPING AT EPOCH {} OF {} ---\n".format(epoch, epochs))
                            break
                    else:
                        prev_val_loss = val_loss.item()
                    prev_val_loss = val_loss.item()
        
        if verbose:
            # Clamp parameters so that values are in [0..1].
            alpha_clamped = self.clamp_params()
            # View results.
            print("\n----------------- RESULTS -----------------\n",
                  "\n• FINAL TRAIN LOSS  =", MAP.item(), 
                  "\n• FINAL ALPHA =", alpha_clamped.data, 
                  "\n• FINAL BETA  =", self.beta.data)
            
        if L_val is not None:
            return scores_log, [loss_train_log, loss_val_log]
            
        return loss_train_log
    
    
    def clamp_params(self):
        
        '''
        Apply sigmoid function to parameters to constrain their values
        between [0..1], as alpha and beta are vectors of probabilities.
        '''
        
        alpha = torch.sigmoid(self.alpha)
        return alpha
    
    
    def toggle_grad(self, requires_grad = True):
        
        '''
        This method sets the requires_grad flag to true or false
        for all parameters associated with a model instance.
        '''
        
        for param in self.parameters():
            param.requires_grad_(requires_grad)