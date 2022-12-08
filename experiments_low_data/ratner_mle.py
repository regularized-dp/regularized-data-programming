# Importations.
import torch
import numpy as np
import sys
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from utils import Utils

class RatnerMLE(torch.nn.Module):
    
    '''
    This model estimates parameters alpha, beta by maximum likelihood 
    estimation as described in Ratner et al (2016):
    
    Ratner, A. J., De Sa, C. M., Wu, S., Selsam, D., & Ré, C. (2016). 
    Data programming: Creating large training sets, quickly. Advances 
    in neural information processing systems, 29, 3567-3575.
    https:\\arxiv.org/abs/1605.07723.
    '''
    
    def __init__(self, alpha, beta):
        
        '''
        Initialize model object with parameters alpha, beta.
        
        Ratner et al (2016) note the following about alpha, beta:
        "...we will assume here that 0.3 ≤ βi ≤ 0.5 and 0.8 ≤ αi ≤ 0.9. 
        We note that while these arbitrary constraints can be changed, 
        they are roughly consistent with our applied experience, where 
        users tend to write high-accuracy and high-coverage labeling 
        functions."
        
        Hence it ~might~ be useful to initialize alpha, beta in these
        value ranges, in the absence of a formal prior.
        '''
        
        super().__init__()
        
        # Initialize model parameters alpha, beta.
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, requires_grad = True))
        self.beta = torch.tensor(beta, requires_grad = True)
    
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
                return total mu or mu_pos1 and mu_neg1 (probability given gold
                label is 1 vs -1, respectively).
        
        ---------------
        Return values:
        ---------------
            mu (float in [0..1]): probability of observing lambda(x_i) given
                unobserved groundtruth y_i.
        '''
        
        # Assume unobserved y = 1.
        mu_pos1 = 1
        
        # Assume unobserved y = -1.
        mu_neg1 = 1
        
        # Clamp parameters so that values are in [0..1].
        alpha_clamped = self.clamp_params()
        
        # Take product of mu for each lambda_j.
        for j in range(len(lambda_x)):
            
            # Probability that labeling function abstains.
            if lambda_x[j] == 0:
                mu_pos1 *= (1 - self.beta[j])
                mu_neg1 *= (1 - self.beta[j])
            # Probability that labeling function votes.
            elif lambda_x[j] == 1:
                # Assume y = 1: labeling function agrees with y.
                mu_pos1 *= self.beta[j] * alpha_clamped[j]
                # Assume y = -1: labeling function disagrees with y.
                mu_neg1 *= self.beta[j] * (1 - alpha_clamped[j])
            else:
                # Assume y = 1: labeling function disagrees with y.
                mu_pos1 *= self.beta[j] * (1 - alpha_clamped[j])
                # Assume y = -1: labeling function agrees with y.
                mu_neg1 *= self.beta[j] * alpha_clamped[j]
        
        if decomposed:
            return mu_pos1, mu_neg1
        
        # Sum for both dummy values of unobserved y.
        mu = mu_pos1 + mu_neg1
        
        return mu
    
    
    def compute_loss(self, L):

        '''
        Compute negative log-likelihood. This is the negative log of the maximum
        likelihood estimation derived in Ratner et al (2016).
        
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
            NLL += torch.log(self.compute_mu(L[row]) + epsilon)
        NLL = -1 * NLL
        
        return NLL
    
    
    def predict(self,
                L, 
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

        # Assign label that results in the highest probability.
        labels = np.zeros(shape = (L.shape[0], 1))
        probabilities = np.zeros(shape = (L.shape[0], 1))
        for row in range(L.shape[0]):
            mu_pos1, mu_neg1 = self.compute_mu(L[row], decomposed = True)
            if mu_pos1 > mu_neg1:
                labels[row] = 1
                probabilities[row] = mu_pos1.item()
            elif mu_pos1 < mu_neg1:
                labels[row] = -1
                probabilities[row] = mu_neg1.item()
            else:
                if tie_policy == "positive":
                    labels[row] = 1
                    probabilities[row] = mu_pos1.item()
                elif tie_policy == "negative":
                    labels[row] = -1
                    probabilities[row] = mu_neg1.item()
                elif tie_policy == "abstain":
                    labels[row] = 0
                    probabilities[row] = mu_neg1.item()
                else:
                    labels[row] = np.random.choice((-1, 1))
                    if labels[row] == 1:
                        probabilities[row] = mu_pos1.item()
                    else:
                        probabilities[row] = mu_neg1.item()
        
        if return_proba:
            return labels, probabilities
        
        return labels
    
    
    def vote_majority(self, L, tie_policy = "abstain"):
        
        '''
        Predict class assignment per observation of a labeling function matrix
        by taking a simple majority vote.
        
        ---------------
        Parameters:
        ---------------
            L (matrix with elements in {-1, 0, 1}): labeling function matrix of
                dimension n x m (n rows for n observations, m columns for m
                labeling functions).
            tie_policy (string with default = "abstain"): tie-breaking policy in
                the event that both classes have the same probability of occurring.
                Options are "positive" (default to label = 1); "negative" (default
                to label = -1); "random" (randomly select from {-1, 1}); "abstain"
                (default to label = 0).

        ---------------
        Return values:
        ---------------
            labels (numpy array with elements in {-1, 1}): predicted labels for
                observations in L. 
        '''
        
        # Minor error-checking.
        policies = ["positive", "negative", "abstain", "random"]
        if tie_policy not in policies:
            print("Invalid tie-breaking policy. Valid options are:", policies)
        
        labels = np.zeros(shape = (L.shape[0], 1))
        
        # Take label that is most numerous in labeling function vector.
        for row in range(L.shape[0]):
            # Value counts per label (excluding abstention).
            total_pos = len(L[row][np.where(L[row] == 1)])
            total_neg = len(L[row][np.where(L[row] == -1)])
            # Take majority vote.
            if total_pos > total_neg:
                labels[row] = 1
            elif total_pos < total_neg:
                labels[row] = -1
            else:
                if tie_policy == "positive":
                    labels[row] = 1
                elif tie_policy == "negative":
                    labels[row] = -1
                elif tie_policy == "abstain":
                    labels[row] = 0
                else:
                    labels[row] = np.random.choice((-1, 1))
        
        return labels
        
    
    def clamp_params(self):
        
        '''
        Apply sigmoid function to parameters to constrain their values
        between [0..1], as alpha and beta are vectors of probabilities.
        '''
        
        alpha = torch.sigmoid(self.alpha)
        
        return alpha
    
    
    def toggle_grad(self, requires_grad = True):
        
        '''
        This method sets the requires_grad flag to true for all
        parameters associated with a model instance.
        '''
        
        for param in self.parameters():
            param.requires_grad_(requires_grad)
    
    
    def fit(self, 
            L, 
            optimizer,
            L_val = None,
            y_val = None,
            tie_policy = "abstain",
            epochs = 1000, 
            learning_rate = 0.001,
            early_stopping = True,
            delay = 50,
            patience = 5,
            clip_grads = None,
            verbose = True):

        '''
        Training loop to estimate optimal alpha.
        
        ---------------
        Parameters:
        ---------------
            L (matrix with elements in {-1, 0, 1}): labeling function matrix of
                dimension n x m (n rows for n observations, m columns for m
                labeling functions).

        ---------------
        Return values:
        ---------------
            loss_train_log or scores_log, [loss_train_log, loss_val_log].
        '''
        
        utils = Utils()
        
        # Set random state seed for reproducibility.
        torch.manual_seed(500)

        # Init data structure to log training information.
        loss_train_log = []
        loss_val_log = []
        scores_log = []
        
        # Set counter for early stopping.
        count = 0
        
        # Initialize optimizer.
        optimizer = optimizer(self.parameters(), lr = learning_rate)
        
        # Adjust print options for pretty print.
        torch.set_printoptions(linewidth = 200)
        
        if verbose:
            # View initial parameter values.
            print("\n----------------- INITIAL PARAMETERS -----------------\n",
                  "\n• INIT ALPHA  =", self.alpha.data, 
                  "\n• INIT BETA   =", self.beta.data)
        
        # Train model.
        for epoch in range(epochs):

            # Set to training mode and toggle gradients "on."
            self.train()
            self.toggle_grad(requires_grad = True)
            
            # Compute negative log likelihood.
            NLL = self.compute_loss(L)
            NLL.backward()
            
            # Clip gradients to prevent vanishing / exploding, if specified.
            if clip_grads is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grads)
            
            # Perform a single optimization step (i.e. parameter update).
            optimizer.step()
            
            # Pretty print progress, if specified.
            if verbose and (epoch % (epochs / 10) == 0):
                alpha_clamped = self.clamp_params()
                print("\n----------------- EPOCH {} -----------------\n".format(epoch),
                      "\n• TRAIN LOSS =", NLL.item())
                if L_val is not None:
                    val_loss = self.compute_loss(L_val)
                    print("• VAL   LOSS =", val_loss.item())
                print("• ALPHA      =", alpha_clamped.data, 
                      "\n• BETA       =", self.beta.data)
                
                print("• ALPHA GRAD =", list(self.parameters())[0].grad)
                
            # Zero gradients.
            optimizer.zero_grad()
            
            # Log training loss.
            loss_train_log.append(NLL) 
            
            # Compute and log validation metrics.
            if L_val is not None:
                val_loss = self.compute_loss(L_val)
                loss_val_log.append(val_loss)
                
                # Predict labels.
                val_pred = self.predict(L_val, 
                                        return_proba = False, 
                                        tie_policy = tie_policy)
                # Score (output is [acc, f1, precision, recall, roc, coverage]).
                try:
                    scores = utils.score(y_val, 
                                         val_pred, 
                                         verbose = False,
                                         plot_confusion = False)
                    scores_log.append(scores)
                except ValueError:
                    print("Could not score: all predictions are abstentions.")

                # Early stopping, if specified.
                if early_stopping and epoch > (delay - 1):
                    if epoch > delay:
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
                  "\n• FINAL TRAIN LOSS   =", NLL.item(), 
                  "\n• FINAL ALPHA  =", alpha_clamped.data, 
                  "\n• FINAL BETA   =", self.beta.data)
        
        if L_val is not None:
            return scores_log, [loss_train_log, loss_val_log]
        
        return loss_train_log