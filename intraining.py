# Import custom tools
import fairness as fr
from fairness.utilities import PDF
# Scikit-learn
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn.metrics
import sklearn.model_selection
# Aif350
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.metrics import ClassificationMetric
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
import aif360.datasets
import aif360
# Themis-ml
import themis_ml
from themis_ml.postprocessing.reject_option_classification import \
    SingleROClassifier
from themis_ml.linear_model import LinearACFClassifier
from themis_ml.metrics import mean_difference
# Import generic tools
from io import StringIO
import datetime
# Data processing
import pandas as pd
import numpy as np
# Visualisationa and output generation
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import fpdf
# System level packages
import warnings
import sys
import os


class InTrainingFairness(fr.Fairness):

    def __init__(self,
                 input_dataframe:pd.DataFrame,
                 label_names,
                 protected_attribute_names,
                 privileged_groups=None,
                 unprivileged_groups=None):

        super().__init__(input_dataframe=input_dataframe,
                         label_names=label_names,
                         protected_attribute_names=protected_attribute_names,
                         privileged_groups=privileged_groups,
                         unprivileged_groups=unprivileged_groups)

    def intrain(self, protected_var, yvar, C_FP, C_FN):
        agep = self.data[protected_var]
        credit = self.data[yvar]
        # agep.value_counts()
        print("protected class : %0.03f, 95%% CI [%0.03f-%0.03f]" %
              mean_difference(credit, agep))
        print("")

        training_data = self.data
        X = training_data.loc[:, training_data.columns != yvar].values
        # Xacf= training_data.loc[:, (training_data.columns != yvar) & (training_data.columns != protected_var)].values
        y = training_data[yvar].values
        agep = agep.values

        # train baseline model
        logistic_clf = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.001, class_weight="balanced")
        baseline_clf = logistic_clf
        rpa_clf = logistic_clf
        roc_clf: SingleROClassifier = SingleROClassifier(estimator=logistic_clf)
        acf_clf = LinearACFClassifier(target_estimator=logistic_clf, binary_residual_type="absolute")

        baseline_clf.fit(X, y)
        baseline_preds = baseline_clf.predict(X)
        baseline_auc = roc_auc_score(y, baseline_preds)
        roc_clf.fit(X, y)
        roc_preds = roc_clf.predict(X, agep)
        acf_preds = acf_clf.fit(X, y, agep).predict(X, agep)

        #########

        df = pd.DataFrame()
        probs_acf= acf_clf.predict_proba(X, agep)
        probs_roc = roc_clf.predict_proba(X, agep)
        probs_bl = baseline_clf.predict_proba(X)

        df['probs_acf'] = pd.DataFrame(probs_acf)[1]
        df['probs_roc'] = pd.DataFrame(probs_roc)[1]
        df['probs_bl'] = pd.DataFrame(probs_bl)[1]

        df['priv_group_col'] = np.array(agep)
        df['actuals'] = np.array(y)

        df.to_csv('intrain.csv')

        #print(df.describe())

        ########




        print("--------------------------------------------------------------")
        print("AUC & Mean difference")
        print("--------------------------------------------------------------")

        print("")
        print("baseline                       :", baseline_auc)
        print("reject option classifier       :", roc_auc_score(y, roc_preds))
        print("additive counterfactually fair :", roc_auc_score(y, acf_preds))
        print("")
        print("mean difference baseline       :", mean_difference(baseline_preds, agep)[0])
        print("mean difference roc            :", mean_difference(roc_preds, agep)[0])
        print("mean difference acf            :", mean_difference(acf_preds, agep)[0])

        bl = confusion_matrix(y, baseline_preds)
        roc = confusion_matrix(y, roc_preds)
        auc = confusion_matrix(y, acf_preds)

        print("--------------------------------------------------------------")
        print("Confusion Matrix")
        print("--------------------------------------------------------------")

        print("")
        print("")
        print("Baseline")
        print(confusion_matrix(y, baseline_preds))
        print("")
        print("reject option classifier")
        print(confusion_matrix(y, roc_preds))
        print("")
        print("additive counterfactually fair")
        print(confusion_matrix(y, acf_preds))
        print("")
        print("")

        a, b, c, d = bl.ravel() ##(tn, fp, fn, tp)
        a1, b1, c1, d1 = roc.ravel() #(tn, fp, fn, tp)
        a2, b2, c2, d2 = auc.ravel() #(tn, fp, fn, tp)

        bl = b * C_FP + c * C_FN
        roc = b1 * C_FP + c1 * C_FN
        auc = b2 * C_FP + c2 * C_FN


        print("cost baseline: ",bl )
        print("cost RoC: ",roc )
        print("cost ACF: ",auc )

        metrics = self.generate_in_train_metrics_table(baseline_model=baseline_clf,
                                                       acf_model=acf_clf,
                                                       roc_model=roc_clf,
                                                       privileged=protected_var,
                                                       target=yvar,
                                                       false_positive_cost=C_FP,
                                                       false_negative_cost=C_FN,
                                                       test_set=training_data)


        ###### june 8
        cost_roc = fr.CostingFairness(input_dataframe=self.data,
                                      label_names=self.label_names,
                                      protected_attribute_names=self.protected_attribute_names,
                                      trained_model=roc_clf)
        costs_roc_table = cost_roc.return_cost_fairness_accuracy_optimised()


        cost_acf = fr.CostingFairness(input_dataframe=self.data,
                                      label_names=self.label_names,
                                      protected_attribute_names=self.protected_attribute_names,
                                      trained_model=acf_clf)
        costs_acf_table = cost_acf.return_cost_fairness_accuracy_optimised()

        return metrics, costs_roc_table, costs_acf_table

    def cost_matrix_acf(self, false_positive_cost, false_negative_cost):

        X = self.data.loc[:, self.data.columns != self.label_names[0]].values
        y = self.data[self.label_names[0]].values
        agep = self.data[self.protected_attribute_names[0]].values

        logistic_clf = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.001, class_weight="balanced")
        acf_clf = LinearACFClassifier(target_estimator=logistic_clf, binary_residual_type="absolute")
        acf_clf.fit(X, y, agep)

#### ***** June 8

        cost_acf = fr.CostingFairness(input_dataframe=self.data,
                                      label_names=self.label_names, #label_names=['credit'],
                                      protected_attribute_names=self.protected_attribute_names,
                                      trained_model=acf_clf)
        #costs_acf_table = cost_acf.return_cost_fairness_accuracy_optimised()
        costs_acf_table = cost_acf.calculate_cost_matrix(false_positive_cost=false_positive_cost,
                                         false_negative_cost=false_negative_cost)


        #print(costs_acf_table)




        return costs_acf_table

#####******

    def generate_in_train_metrics_table(self,
                                        baseline_model: sklearn.linear_model,
                                        roc_model: sklearn.linear_model,
                                        acf_model: sklearn.linear_model,
                                        test_set: pd.DataFrame,
                                        target: str,
                                        privileged: str,
                                        false_positive_cost: float,
                                        false_negative_cost: float) -> pd.DataFrame:
        """
        This function calculates the following table:

        | Metric         | Baseline | ROC | ACF |
        | AUC            |          |     |     |
        | Mean Diff      |
        | True Negative  |
        | False Positive |
        | False Negative |
        | True Positive  |
        | Cost           |

        :param baseline_model: Vanilla Logistic Regression
        :param roc_model:
        :param acf_model:
        :param test_set: Data-set used for testing the model
        :param target: Target test data-set
        :param privileged: Column with privileged attribute. 1 is privileged, 0 is not
        :param false_positive_cost: Cost of incorrectly classifying a data point as positive.
        :param false_negative_cost: Cost of incorrectly clasisfying a data point as negative.
        :return: A pd.DataFrame of the table above
        """

        def generate_comparison_metrics(model: sklearn.linear_model,
                                        x_test: pd.DataFrame,
                                        y_test: pd.DataFrame,
                                        y_predictions: pd.DataFrame = None):
            """
            This function returns a standard list of values of descriptive metrics, given an input model, features and target.
            :param model: The trained model object
            :param x_test: Feature test data-set
            :param y_test: Target test data-set
            :param y_predictions:(Optional) prediction values that have already been calculated
            :return: A list with the following: ['SUC', 'True Positive', 'True Negative',
                                                 'False Positive', 'False Negative', 'Cost']
            """
            if y_predictions is not None:
                if not y_predictions.isnull().values.all():
                    y_pred = y_predictions
            elif 'themis_ml' in model.__module__:
                y_pred = model.predict(x_test.values, x_test[privileged].values)
                y_pred_proba = model.predict_proba(x_test.values, x_test[privileged].values)[:, 1]
            else:
                y_pred = model.predict(x_test)
                y_pred_proba = model.predict_proba(x_test)[:, 1]

            mean_diff = themis_ml.metrics.mean_difference(y=y_pred, s=x_test[privileged])[0]









            accuracy = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred)
            recall = sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred)

            auc = sklearn.metrics.roc_auc_score(y_true=y_test, y_score=y_pred_proba)



            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
            tot_num = tn + fp + fn + tp
            tnp, fpp, fnp, tpp = tn / tot_num, fp / tot_num, fn / tot_num, tp / tot_num

            tpr = fr.utilities.true_positive_rate(true_positives=tp, false_negatives=fn)
            fpr = fr.utilities.false_positive_rate(false_positives=fp, true_negatives=tn)

            cost = false_positive_cost * fp + false_negative_cost * fn

            output_list = [mean_diff, accuracy, precision, recall, auc, tnp, fpp, fnp, tpp, tpr, fpr, cost]
            return output_list

        # Define output table
        output_table = pd.DataFrame()

        # Split input test_set into feature and target tables
        x = test_set.loc[:, (test_set.columns != target)].copy()
        y = test_set[target].copy()
        # x_test_privileged = test_set.loc[:, (test_set.columns != target)][test_set[privileged] == 1.0]
        # y_test_privileged = test_set[target][test_set[privileged] == 1.0]

        # Build the output table
        output_table['Metric'] = ['Mean Diff', 'Accuracy', 'Precision', 'Recall', 'AUC', 'True Negative',
                                  'False Positive', 'False Negative', 'True Positive', 'TPR', 'FPR', 'Cost']
        output_table['Baseline'] = generate_comparison_metrics(model=baseline_model,
                                                               x_test=x,
                                                               y_test=y)
        output_table['ROC Model'] = generate_comparison_metrics(model=roc_model,
                                                                x_test=x,
                                                                y_test=y)
        output_table['ACF Model'] = generate_comparison_metrics(model=acf_model,
                                                                x_test=x,
                                                                y_test=y)
        # Return output table
        return output_table
