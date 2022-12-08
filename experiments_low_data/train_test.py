# Importations.
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier # Linear SVM.

# Personal scripts.
from utils import Utils
from bayesian_dp import BayesianDP
from ratner_mle import RatnerMLE



class Trainer:
    
    '''
    Objects of this class provide utility functins for training and testing.
    '''
    
    
    def train_test_score(self,
                         L_train,
                         L_val,
                         L_test, 
                         y_val,
                         y_test,
                         init_alpha,
                         init_beta,
                         optimizer,
                         priors_alpha,
                         tie_policy = "abstain",
                         prior_strength = 10,
                         epochs = 1000,
                         learning_rate = 0.001,
                         early_stopping = True,
                         patience = 5,
                         clip_grads = 10,
                         verbose = False):
    
        '''
        This function trains a MAP model and tests its performance.

        -------------
        Parameters:
        -------------
        L_train: labeling function output matrix on which to train models.
        L_test: labeling function output matrix on which to test models.
        y_test: gold label vector for L_test.
        verbose (default = False): boolean indicating whether to print results.

        -------------
        Returns:
        -------------
            final_scores_map 
        '''

        MAP = BayesianDP(alpha = init_alpha, 
                         beta = init_beta,
                         priors_alpha = priors_alpha,
                         prior_strength = prior_strength)
        utils = Utils()

        # Estimate alpha.
        # Each map_scores[0] element = [acc, f1, precision, recall, roc, coverage].
        map_scores, map_logs = MAP.fit(L_train, 
                                       L_val = L_val,
                                       y_val = y_val,
                                       optimizer = optimizer,
                                       learning_rate = learning_rate,
                                       tie_policy = tie_policy,
                                       epochs = epochs, 
                                       early_stopping = early_stopping,
                                       patience = patience,
                                       clip_grads = clip_grads,
                                       verbose = verbose)

        if verbose:
            # Adjust print options for pretty print.
            torch.set_printoptions(linewidth = 200)

            # Visualize training and validation losses.
            self.plot_loss(map_logs[0], 
                           map_logs[1], 
                           L_train.shape[0],
                           L_val.shape[0],
                           title = "MAP loss curve")

            # Print final parameter values.
            map_alpha = MAP.clamp_params()
            print("\n• MAP ALPHA  =", map_alpha.data, 
                  "\n• MAP BETA   =", MAP.beta.data)

            # Compute L2 norms (priors - final alpha params).
            print("\n--- L2 norms: ---")
            l2 = np.linalg.norm(np.array(priors_alpha) - map_alpha.detach().numpy())
            print("\nL2 norm of (priors_alpha - final_alpha): %.3f" % l2)

            # Plot validation metric curves.
            self.plot_val_scores(map_scores, 
                                 title = "MAP validation metrics")

        # Predict labels on testing data.
        pred_map = MAP.predict(L_test, 
                               return_proba = False, 
                               tie_policy = tie_policy)

        # Score learned model.
        print("\n--- FINAL TEST SCORES ---\n")
        final_scores_map = utils.score(y_test, 
                                       pred_map, 
                                       verbose = verbose, 
                                       plot_confusion = verbose)

        return final_scores_map
    
    
    def plot_val_scores(self, 
                        scores, 
                        title = "Validation scores\n"):

        '''
        Plot validation scores for F1, accuracy, precision, recall, and AUC ROC.
        '''

        # scores = [acc, f1, precision, recall, roc, coverage]
        acc = [x[0] for x in scores]
        f1 = [x[1] for x in scores]
        prec = [x[2] for x in scores]
        rec = [x[3] for x in scores]
        roc = [x[4] for x in scores]

        # Plot validation curves.
        plt.rcParams["figure.figsize"] = (12, 8)
        ax = plt.subplot()
        x_axis = range(1, len(acc) + 1)
        plt.plot(x_axis, acc, label = "Accuracy")
        plt.plot(x_axis, f1, label = "F1")
        plt.plot(x_axis, prec, label = "Precision")
        plt.plot(x_axis, rec, label = "Recall")
        plt.plot(x_axis, roc, label = "AUC ROC")
        plt.legend()
        ax.set_title(title)
        ax.set_xlabel("\nEpochs")
        ax.set_ylabel("Value\n")
        plt.show()
    
    
    def plot_scores(self,
                    train_scores, 
                    val_scores, 
                    title = "Training vs validation scores\n"):

        '''
        Plot both training and validation curves for a single metrics, e.g. for F1, 
        accuracy, or precision, etc.

        Note that the output of model.score() = [acc, f1, precision, recall, roc, 
        coverage], i.e. to plot accuracy use train_scores[0] and val_scores[0].
        '''

        # Plot validation curves.
        plt.rcParams["figure.figsize"] = (12, 8)
        ax = plt.subplot()
        x_axis = epochs
        plt.plot(x_axis, train_scores, label = "Training")
        plt.plot(x_axis, val_scores, label = "Validation")
        plt.legend()
        ax.set_title(title)
        ax.set_xlabel("\nEpochs")
        ax.set_ylabel("Value\n")
        plt.show()
    

    def plot_loss(self,
                  loss_train_log, 
                  loss_val_log,
                  n_train, 
                  n_val, 
                  title = "MAP loss curve"):

        '''
        Plot training and validation loss.
        '''

        # Normalize loss values.
        loss_train_log = [x/n_train for x in loss_train_log]
        loss_val_log = [x/n_val for x in loss_val_log]

        # Plot.
        plt.rcParams["figure.figsize"] = (12, 8)
        ax = plt.subplot()
        x_axis = torch.arange(len(loss_train_log))
        #y = [loss.item() for loss in loss_log]
        plt.plot(x_axis, 
                 [loss.item() for loss in loss_train_log], 
                 label = "Training loss")
        plt.plot(x_axis, 
                 [loss.item() for loss in loss_val_log], 
                 label = "Validation loss")
        #plt.scatter(x, y, alpha = 0.4)
        plt.legend()
        ax.set_title(title)
        ax.set_xlabel("\nEpochs")
        ax.set_ylabel("Loss value\n")
        plt.show()
        
        
        
    def train_random_subset(self,
                            L_train,
                            L_val,
                            y_val,
                            L_test, 
                            y_test,
                            init_alpha_map,
                            init_alpha_mle,
                            init_beta,
                            priors,
                            optimizer = torch.optim.SGD,
                            prior_strength = 10,
                            p_y = 0.5,
                            force_abstain = True,
                            epochs_map = 200,
                            epochs_mle = 200,
                            learning_rate_map = 0.001,
                            learning_rate_mle = 0.001,
                            early_stopping = True,
                            patience = 5,
                            clip_grads = 10,
                            verbose = False):

        '''
        This function trains a MAP and MLE model and tests their performance.

        Parameters:
        -------------
        L_train: labeling function output matrix on which to train models.
        L_test: labeling function output matrix on which to test models.
        y_test: gold label vector for L_test.
        verbose (default = False): boolean indicating whether to print results.

        Returns:
        -------------
        scores_map, scores_mle, scores_majority: performance metrics for 
            MAP, MLE, and majority vote, respectively.
        '''


        # Initialize models.
        MAP = BayesianDP(alpha = init_alpha_map, 
                         beta = init_beta,
                         priors_alpha = priors,
                         prior_strength = prior_strength)
        MLE = RatnerMLE(alpha = init_alpha_mle, 
                        beta = init_beta)

        # Init Utils object for utility functions.
        utils = Utils()

        # Estimate alpha, beta.
        map_logs = MAP.fit(L_train, 
                           L_val = L_val,
                           y_val = y_val,
                           optimizer = optimizer,
                           learning_rate = learning_rate_map,
                           tie_policy = "abstain",
                           epochs = epochs_map, 
                           early_stopping = early_stopping,
                           patience = patience,
                           clip_grads = clip_grads,
                           verbose = verbose)
        mle_logs = MLE.fit(L_train, 
                           optimizer = optimizer,
                           L_val = L_val,
                           y_true = y_val,
                           tie_policy = "abstain",
                           epochs = epochs_mle, 
                           learning_rate = learning_rate_mle,
                           early_stopping = early_stopping,
                           patience = patience,
                           clip_grads = clip_grads,
                           verbose = False)

        if verbose:
            # Visualize training log.
            plot_loss(map_logs[0], 
                      map_logs[1], 
                      L_train.shape[0],
                      L_val.shape[0],
                      title = "MAP loss curve")

            # Print final parameter values.
            clamped_map = MAP.clamp_params()
            print("\n• FINAL MAP ALPHA  =", clamped_map.data, 
                  "\n• FINAL MAP BETA   =", MAP.beta.data)

            # Visualize training log.
            plot_loss(mle_logs[0], 
                      mle_logs[1], 
                      L_train.shape[0],
                      L_val.shape[0],
                      title = "MLE loss curve")

            # Print final parameter values.
            clamped_mle = MLE.clamp_params()
            print("\n• FINAL MLE ALPHA  =", clamped_mle.data, 
                  "\n• FINAL MLE BETA   =", MLE.beta.data)

        # Predict labels on testing data.
        pred_map = MAP.predict(L_test,
                               p_y = p_y,
                               force_abstain = force_abstain,
                               tie_policy = "abstain",
                               return_proba = False)
        pred_mle = MLE.predict(L_test, 
                               tie_policy = "abstain",
                               return_proba = False)

        # Score learned models.
        scores_map = utils.score(y_test, 
                                   pred_map, 
                                   verbose = verbose, 
                                   plot_confusion = verbose)
        scores_mle = utils.score(y_test, 
                                  pred_mle, 
                                  verbose = verbose, 
                                  plot_confusion = verbose)

        # Majority vote prediction.
        majority = utils.vote_majority(L_test, tie_policy = "abstain")

        # Score majority vote.
        scores_majority = utils.score(y_test, 
                                      majority,
                                      verbose = verbose, 
                                      plot_confusion = verbose)

        return scores_map, scores_mle, scores_majority


    def train_random_subset_map(self,
                                L_train,
                                L_val,
                                y_val,
                                L_test, 
                                y_test,
                                init_alpha,
                                init_beta,
                                priors,
                                optimizer = torch.optim.SGD,
                                prior_strength = 10,
                                epochs = 1000,
                                learning_rate = 0.001,
                                early_stopping = True,
                                patience = 5,
                                clip_grads = 10,
                                verbose = False):

        '''
        This function trains a MAP and MLE model and tests their performance.

        Parameters:
        -------------
        L_train: labeling function output matrix on which to train models.
        L_test: labeling function output matrix on which to test models.
        y_test: gold label vector for L_test.
        verbose (default = False): boolean indicating whether to print results.

        Returns:
        -------------
        scores_map, scores_mle, scores_majority: performance metrics for 
            MAP, MLE, and majority vote, respectively.
        '''

        MAP = BayesianDP(alpha = init_alpha, 
                         beta = init_beta,
                         priors_alpha = priors,
                         prior_strength = prior_strength)

        # Estimate alpha, beta.
        map_logs = MAP.fit(L_train, 
                           L_val = L_val,
                           y_val = y_val,
                           optimizer = optimizer,
                           learning_rate = learning_rate,
                           tie_policy = "abstain",
                           epochs = epochs, 
                           early_stopping = early_stopping,
                           patience = patience,
                           clip_grads = clip_grads,
                           verbose = verbose)
        if verbose:
            # Visualize training log.
            plot_loss(map_logs[0], 
                      map_logs[1], 
                      L_train.shape[0],
                      L_val.shape[0],
                      title = "MAP loss curve")

            # Print final parameter values.
            clamped_map = MAP.clamp_params()
            print("\n• FINAL MAP ALPHA  =", clamped_map.data, 
                  "\n• FINAL MAP BETA   =", MAP.beta.data)

        # Predict labels on testing data.
        pred_map = MAP.predict(L_test, 
                               return_proba = False, 
                               tie_policy = "abstain")

        # Score learned models.
        scores_map = utils.score(y_test, 
                                 pred_map, 
                                 verbose = verbose, 
                                 plot_confusion = verbose)

        return scores_map


    def train_random_subset_mle(self,
                                L_train,
                                L_val,
                                y_val,
                                L_test, 
                                y_test,
                                init_alpha,
                                init_beta,
                                optimizer = torch.optim.SGD,
                                epochs = 1000,
                                learning_rate = 0.001,
                                early_stopping = True,
                                delay = 20,
                                patience = 5,
                                clip_grads = 10,
                                verbose = False):

        '''
        This function trains a MLE and MLE model and tests their performance.

        Parameters:
        -------------
        L_train: labeling function output matrix on which to train models.
        L_test: labeling function output matrix on which to test models.
        y_test: gold label vector for L_test.
        verbose (default = False): boolean indicating whether to print results.

        Returns:
        -------------
        scores_mle, scores_mle, scores_majority: performance metrics for 
            MLE, MLE, and majority vote, respectively.
        '''

        MLE = RatnerMLE(alpha = init_alpha, 
                         beta = init_beta)

        # Estimate alpha, beta.
        mle_logs = MLE.fit(L_train, 
                           optimizer,
                           L_val = L_val,
                           y_true = y_val,
                           tie_policy = "abstain",
                           epochs = epochs, 
                           learning_rate = learning_rate,
                           early_stopping = early_stopping,
                           delay = delay,
                           patience = patience,
                           clip_grads = clip_grads,
                           verbose = False)
        if verbose:
            # Visualize training log.
            plot_loss(mle_logs[0], 
                      mle_logs[1], 
                      L_train.shape[0],
                      L_val.shape[0],
                      title = "MLE loss curve")

            # Print final parameter values.
            clamped_mle = MLE.clamp_params()
            print("\n• FINAL MLE ALPHA  =", clamped_mle.data, 
                  "\n• FINAL MLE BETA   =", MLE.beta.data)

        # Predict labels on testing data.
        pred_mle = MLE.predict(L_test, 
                                 return_proba = False, 
                                 tie_policy = "abstain")

        # Score learned models.
        scores_mle = utils.score(y_test, 
                                   pred_mle, 
                                   verbose = verbose, 
                                   plot_confusion = verbose)

        return scores_mle


    def loop_train_random_subset_map(self,
                                     L_train,
                                     L_val,
                                     y_val,
                                     L_test, 
                                     y_test,
                                     init_alpha,
                                     init_beta,
                                     prior_strength = 10,
                                     epochs = 1000,
                                     learning_rate = 0.001,
                                     early_stopping = True,
                                     patience = 5,
                                     clip_grads = 10,
                                     verbose = False,
                                     n_rows = 100, 
                                     n_iter = 25):

        '''
        This function trains a MAP model on a random subset of L of n rows.

        Parameters:
        -------------
        L_train: labeling function output matrix on which to train models.
        L_test: labeling function output matrix on which to test models.
        y_test: gold label vector for L_test.
        n_rows: list-like of integers indicating total rows to randomly subset.
        n_iter: total iterations to perform, i.e. total subsets to test.
        seed: integer indicating random state seed for reproducibility.

        Returns:
        -------------
        map_scores, mle_scores: performance metrics for MAP and MLE, respectively.
        '''

        # Store scores per iteration in lists.
        map_scores = []
        subset_indices = dict()

        for i in range(n_iter):

            # Fancy indexing to randomly subset training data.
            keep_indices = np.random.choice(L_train.shape[0], 
                                            n_rows,
                                            replace = False)
            L_subset = L_train[keep_indices, :]
            subset_indices[i] = keep_indices
            
            # Get majority vote priors from subset.
            utils = Utils()
            priors = utils.get_priors_majority_vote(L_subset, remove_zeros = True)

            # Train MAP models and return performance metrics.
            scores = self.train_random_subset_map(L_subset,
                                             L_val,
                                             y_val,
                                             L_test, 
                                             y_test,
                                             init_alpha,
                                             init_beta,
                                             priors,
                                             prior_strength = prior_strength,
                                             epochs = epochs,
                                             learning_rate = learning_rate,
                                             early_stopping = early_stopping,
                                             patience = patience,
                                             clip_grads = clip_grads,
                                             verbose = verbose)
            map_scores.append(scores)

        return map_scores, subset_indices



    def loop_train_random_subset(self,
                                 L_train,
                                 L_val,
                                 y_val,
                                 L_test, 
                                 y_test,
                                 init_alpha_map,
                                 init_alpha_mle,
                                 init_beta,
                                 prior_strength = 10,
                                 p_y = 0.5,
                                 force_abstain = True,
                                 epochs_map = 1000,
                                 epochs_mle = 1000,
                                 learning_rate_map = 0.001,
                                 learning_rate_mle = 0.001,
                                 early_stopping = True,
                                 patience = 5,
                                 clip_grads = 10,
                                 verbose = False,
                                 n_rows = 100, 
                                 n_iter = 25):

        '''
        This function trains a MAP and MLE model on a random subset of
        L of n rows.

        Parameters:
        -------------
        L_train: labeling function output matrix on which to train models.
        L_test: labeling function output matrix on which to test models.
        y_test: gold label vector for L_test.
        n_rows: list-like of integers indicating total rows to randomly subset.
        n_iter: total iterations to perform, i.e. total subsets to test.
        seed: integer indicating random state seed for reproducibility.

        Returns:
        -------------
        map_scores, mle_scores: performance metrics for MAP and MLE, respectively.
        '''

        # Store scores per iteration in lists.
        map_scores = []
        mle_scores = []
        majority_scores = []
        subset_indices = dict()

        for i in range(n_iter):

            # Fancy indexing to randomly subset training data.
            keep_indices = np.random.choice(L_train.shape[0], 
                                            n_rows,
                                            replace = False)
            L_subset = L_train[keep_indices, :]
            subset_indices[i] = keep_indices
            
            # Get majority vote priors from subset.
            utils = Utils()
            priors = utils.get_priors_majority_vote(L_subset, remove_zeros = True)

            # Train MAP and MLE models and return performance metrics.
            scores_map, scores_mle, scores_majority = self.train_random_subset(L_subset,
                                                                               L_val,
                                                                               y_val,
                                                                               L_test, 
                                                                               y_test,
                                                                               init_alpha_map,
                                                                               init_alpha_mle,
                                                                               init_beta,
                                                                               priors,
                                                                               prior_strength = prior_strength,
                                                                               p_y = p_y,
                                                                               force_abstain = force_abstain,
                                                                               epochs_map = epochs_map,
                                                                               epochs_mle = epochs_mle,
                                                                               learning_rate_map = learning_rate_map,
                                                                               learning_rate_mle = learning_rate_mle,
                                                                               early_stopping = early_stopping,
                                                                               patience = patience,
                                                                               clip_grads = clip_grads,
                                                                               verbose = verbose)
            map_scores.append(scores_map)
            mle_scores.append(scores_mle)
            majority_scores.append(scores_majority)

        return map_scores, mle_scores, majority_scores, subset_indices


    def compute_variance(self,
                         scores, 
                         model, 
                         n):

        '''
        n = subset size
        model = model name ("map", "mle", "maj", etc.)
        '''

        # Scores = [acc, f1, precision, recall, roc, coverage].
        acc = [x[0] for x in scores]
        f1 = [x[1] for x in scores]
        prec = [x[2] for x in scores]
        rec = [x[3] for x in scores]
        roc = [x[4] for x in scores]
        cov = [x[5] for x in scores]
        all_scores = [acc, f1, prec, rec, roc, cov]

        # Compute variance, standard deviation, mean, and median.
        var = []
        stdev = []
        mean = []
        median = []
        for i in range(len(all_scores)):
            var.append(np.var(all_scores[i]))
            stdev.append(np.std(all_scores[i]))
            mean.append(np.mean(all_scores[i]))
            median.append(np.median(all_scores[i]))

        # Print as R dataframe.
        print("\n# Scores = [accuracy, f1, precision, recall, auc roc, coverage]\n")
        print("df_{}_{} <- data.frame(Model = '{}', Subset = {}, Metric = c('Accuracy', 'F1', 'Precision', 'Recall', 'AUC ROC', 'Coverage'),".format(model, n, model, n))
        print("     Variance = c(")
        print(*var, sep = ", ")
        print("),")
        print("     SD = c(")
        print(*stdev, sep = ", ")
        print("),")
        print("     Mean = c(")
        print(*mean, sep = ", ")
        print("),")
        print("     Median = c(")
        print(*median, sep = ", ")
        print("))")

        return var, stdev, mean, median
    
    
    def get_optimal_hparams(self,
                            recalls,
                            f1s, 
                            accuracies, 
                            precisions, 
                            rocs):
        
        # Get hyparameter combinations with maximum metrics.
        max_recall = max(recalls)
        max_recall_indices = [i for i,j in enumerate(recalls) if j == max_recall]
        hparams_max_recall = [search_space[i] for i in max_recall_indices]
        print("\n--- HYPERPARAMETERS FOR MAX RECALL OF {}: ---\n".format(max_recall))
        print("Total models with optimal recall:", len(hparams_max_recall))
        print()
        print(hparams_max_recall)

        max_f1 = max(f1s)
        max_f1_indices = [i for i,j in enumerate(f1s) if j == max_f1]
        hparams_max_f1 = [search_space[i] for i in max_f1_indices]
        print("\n--- HYPERPARAMETERS FOR MAX F1 OF {}: ---\n".format(max_f1))
        print("Total models with optimal F1:", len(hparams_max_f1))
        print()
        print(hparams_max_f1)

        max_accuracies = max(accuracies)
        max_accuracies_indices = [i for i,j in enumerate(accuracies) if j == max_accuracies]
        hparams_max_accuracies = [search_space[i] for i in max_accuracies_indices]
        print("\n--- HYPERPARAMETERS FOR MAX ACCURACY OF {}: ---\n".format(max_accuracies))
        print("Total models with optimal accuracy:", len(hparams_max_accuracies))
        print()
        print(hparams_max_accuracies)

        max_precision = max(precisions)
        max_precision_indices = [i for i,j in enumerate(precisions) if j == max_precision]
        hparams_max_precision = [search_space[i] for i in max_precision_indices]
        print("\n--- HYPERPARAMETERS FOR MAX PRECISION OF {}: ---\n".format(max_precision))
        print("Total models with optimal precision:", len(hparams_max_precision))
        print()
        print(hparams_max_precision)

        max_roc = max(rocs)
        max_roc_indices = [i for i,j in enumerate(rocs) if j == max_roc]
        hparams_max_roc = [search_space[i] for i in max_roc_indices]
        print("\n--- HYPERPARAMETERS FOR MAX AUC ROC OF {}: ---\n".format(max_roc))
        print("Total models with optimal AUC ROC:", len(hparams_max_roc))
        print()
        print(hparams_max_roc)

        return hparams_max_recall, hparams_max_f1, hparams_max_accuracies, hparams_max_precision, hparams_max_roc
    
    
    def train_test_score_svm(self,
                             X_train,
                             X_test, 
                             y_train, 
                             y_test, 
                             alpha = 1e-4,
                             verbose = False):

        # Build dictionary of features and transform documents to feature vectors.
        # Value of a word in the vocabulary is its frequency in the whole training corpus.
        # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)

        # Regularize via “Term Frequency times Inverse Document Frequency.”
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        X_train_tfidf.shape

        # Call only transform on test data, as featurizers have already 
        # been fit to training data.
        X_test_counts = count_vect.transform(X_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)

        # Fit model.
        svm = SGDClassifier(loss = "hinge", 
                            penalty = "l2",
                            alpha = alpha, 
                            random_state = 42,
                            max_iter = 5, 
                            tol = None)
        svm = svm.fit(X_train_tfidf, y_train)

        # Predict on test set.
        y_svm = svm.predict(X_test_tfidf)

        # Evaluate performance.
        # Care about most about false negatives – discarding papers we want to keep.
        # Recall, F1, and accuracy rely on false negatives.
        confusion_svm = metrics.confusion_matrix(y_test, y_svm)
        acc_svm = metrics.accuracy_score(y_test, y_svm)
        f1_svm = metrics.f1_score(y_test, y_svm, zero_division = 0)
        precision_svm = metrics.precision_score(y_test, y_svm, zero_division = 0)
        recall_svm = metrics.recall_score(y_test, y_svm, zero_division = 0)
        roc_svm = metrics.roc_auc_score(y_test, y_svm)

        scores = [acc_svm, f1_svm, precision_svm, recall_svm, roc_svm, 1.0]

        if verbose:
            print("\n--- LINEAR SUPPORT VECTOR MACHINE ---\n")
            print("\n---------------------------------------------")
            print("tn, fp, fn, tp =", confusion_svm.ravel())
            print("F1             =", f1_svm)
            print("Accuracy       =", acc_svm)
            print("Precision      =", precision_svm)
            print("Recall         =", recall_svm)
            print("ROC AUC        =", roc_svm)
            print("---------------------------------------------\n")

        # Return: [acc, f1, precision, recall, roc, coverage]
        return scores

