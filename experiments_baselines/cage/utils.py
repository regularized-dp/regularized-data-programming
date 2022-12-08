# Importations.
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



class Utils:
    
    '''
    Objects of this class provide utility functins for MAP and MLE models.
    '''
    
    def vote_majority(self, L, tie_policy = "positive"):
        
        '''
        Predict class assignment per observation of a labeling function matrix
        by taking a simple majority vote.
        
        ---------------
        Parameters:
        ---------------
            L (matrix with elements in {-1, 0, 1}): labeling function matrix of
                dimension n x m (n rows for n observations, m columns for m
                labeling functions).
            tie_policy (string with default = "positive"): tie-breaking policy in
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
            
            
    def score(self, y_true, y_pred, verbose = True, plot_confusion = True):
        
        '''
        Compute performance metrics.
        '''
        
        # Filter out abstained predictions.
        initial_size = y_pred.shape[0]
        filter_abstain = np.where(y_pred == 0, False, True)
        y_true = y_true.reshape(y_pred.shape)
        y_pred = y_pred[filter_abstain]
        y_true = y_true[filter_abstain]
        
        # Compute performance metrics.
        confusion = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division = 0)
        precision = precision_score(y_true, y_pred, zero_division = 0)
        recall = recall_score(y_true, y_pred, zero_division = 0)
        roc = 0.
        try:
            roc = roc_auc_score(y_true, y_pred)
        except ValueError:
            print("Could not compute ROC AUC.")
        coverage = y_pred.shape[0] / initial_size
        

        if verbose:
            print("\n---------------------------------------------")
            print("tn, fp, fn, tp =", confusion.ravel())
            print("F1             =", f1)
            print("Accuracy       =", acc)
            print("Precision      =", precision)
            print("Recall         =", recall)
            print("ROC AUC        =", roc)
            print("Coverage       =", coverage)
            print("---------------------------------------------\n")
        
        # Plot confusion matrix as heatmap, if specified.
        if plot_confusion and (len(confusion) > 0):
            plt.rcParams["figure.figsize"] = (5, 4)
            ax = plt.subplot()
            # (annot = True) to annotate cells.
            # (ftm = "g") to disable scientific notation.
            sns.heatmap(confusion, annot = True, fmt = "g", ax = ax);  
            # Labels, title, and ticks.
            ax.set_xlabel("\nPredicted labels");
            ax.set_ylabel("True labels\n"); 
            ax.set_title("Confusion Matrix\n"); 
            ax.xaxis.set_ticklabels(["-1 (negative)", "1 (positive)"]); 
            ax.yaxis.set_ticklabels(["-1 (negative)", "1 (positive)"]);
            plt.show()

        return [acc, f1, precision, recall, roc, coverage]
            
            
    # Beta distribution utility functions.
    def compute_beta_mean(self, x): 
        return x[0] / (x[0] + x[1])
    
    def compute_beta_mode(self, x):
        return (x[0] - 1) / (x[0] + x[1] - 2)

    def get_means(self, params):
        return [self.compute_beta_mean(x) for x in params]

    def get_beta(self, alpha, mean):
        beta = (alpha - (alpha * mean)) / mean
        return beta

    def get_prior_dist_params(self, alphas, means):
        betas = [self.get_beta(alpha, mean) for alpha,mean in zip(alphas, means)]
        params = [(alpha, beta) for alpha,beta, in zip(alphas, betas)]
        return params
    
    def get_priors_majority_vote(self, L, remove_zeros = True, default_value = 0.999):
    
        # Init majority vote model.
        majority_vote = self.vote_majority(L, tie_policy = "abstain")

        # Compute estimated accuracies.
        estimates = []
        for i in range(L.shape[1]):

            #print("np.unique(mv) initial =", np.unique(mv))
            #print("np.unique(column) initial =", np.unique(L[:,i]))

            # Filter out abstained predictions.
            mv = majority_vote.copy()
            keep_idx = np.where(mv != 0)[0]
            mv = mv[keep_idx, :]
            column = L[keep_idx, i]
            if column.shape[0] != 0:
                column = column.reshape(len(mv), -1).copy()
                if remove_zeros:
                    keep_idx2 = np.where(column != 0)[0]
                    column = column[keep_idx2, :]
                    mv = mv[keep_idx2, :]
                
            #print("np.unique(mv) after =", np.unique(mv))
            #print("np.unique(column) after =", np.unique(column))

            # Compute percent identical entries per column.
            total_same = (column == mv).sum()
            if (column.shape[0] == 0) or (total_same == 0):
                percent_same = default_value
            else:
                percent_same = total_same / column.shape[0]
            if percent_same == 1.0:
                percent_same = 0.999999
            estimates.append(percent_same)

        print("\n# Estimated accuracies via majority vote =", estimates)

        return estimates