# Import custom tools
import fairness as fr
from fairness.utilities import output_probabilities_to_csv, PDF, \
    generate_privileged_diff, generate_delta_table, true_positive_rate, false_positive_rate
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


class PreProcessingFairness(fr.Fairness):

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

    def evaluate_bias(self, **kwargs):
        """
        Evaluates bias within the dataset in self.
        :param kwargs: (optional) Can be passed if a different binary label data-set is to be evaluated. Should be
        passed as: bl_df=name_of_dataset
        :return: bias: Boolean indicsating bias if it is returned as True
                 aif: aif360 object with the following metrics:
                 https://aif360.readthedocs.io/en/latest/modules/metrics.html
                 for example: aif.base_rate()
                              aif.consistency()
                              aif.disparate_impact()
                              aif.mean_difference()
                              aif.statistical_parity_difference()

        """
        if kwargs:
            bl_df = kwargs['bl_df']
        else:
            bl_df = self.binary_label_df

        aif = aif360.metrics.BinaryLabelDatasetMetric(bl_df,
                                                      unprivileged_groups=self.unprivileged_groups,
                                                      privileged_groups=self.privileged_groups)

        if (aif.statistical_parity_difference() > 0.1) or (aif.statistical_parity_difference() < -0.1):
            bias = True
        else:
            bias = False

        return bias, aif

    def reweigh(self, **kwargs):
        """
        Re-weights the dataset to be fair before modelling.
        :param kwargs: (optional) binary label data-set to perform re-weighing on. Should be: bl_df=data-frame
        :return: The transformed data-set.
        """
        if kwargs:
            bl_df = kwargs['bl_df']
        else:
            bl_df = self.binary_label_df

        rw = Reweighing(unprivileged_groups=self.unprivileged_groups,
                        privileged_groups=self.privileged_groups)

        transformed_data = rw.fit_transform(bl_df)
        return transformed_data

    def model_performance_comparison(self, yvar, prev_group, C_FP, C_FN):
        """
        Compares the performance of a Vanilla Logistic Regression Model trained on both the weighted and un-weighted
        data.
        :param yvar:
        :param prev_group:
        :param C_FP:
        :param C_FN:
        :return:
        """
        trn_bl_df = self.reweigh()
        # sample_weights = self.reweigh(bl_df=trn_bl_df)

        s_weights = trn_bl_df.instance_weights
        print(s_weights)

        trainset = trn_bl_df.convert_to_dataframe()[0]
        testset = trainset

        X = trainset.loc[:, trainset.columns != yvar]
        y = trainset[yvar]
        #X_test = testset.loc[:, trainset.columns != yvar]
        #y_test = testset[yvar]

        X_test = X
        y_test = y

        clf_ww = sklearn.linear_model.LogisticRegression(random_state=999).fit(X, y, sample_weight=s_weights)
        clf_wow = sklearn.linear_model.LogisticRegression(random_state=999).fit(X, y)

        output_probabilities_to_csv(model=clf_ww, x_test=X_test, path='probs_ww_withprvgroup.csv',priv_group_col=trainset[prev_group], actuals=y_test)
        output_probabilities_to_csv(model=clf_wow, x_test=X_test, path='probs_wow_withprvgroup.csv', priv_group_col=trainset[prev_group], actuals=y_test)

        print("------------------------------------------")
        print("Accuracy of Vanila Logistic Model")
        print("------------------------------------------")
        print("Without Weights : ", round(clf_wow.score(X_test, y_test), 3))
        print("With Weights    : ", round(clf_ww.score(X_test, y_test), 3))

        X_test_age1 = testset.loc[:, trainset.columns != yvar][testset[prev_group] == 1.0]
        y_test_age1 = testset[yvar][testset[prev_group] == 1.0]
        X_test_age0 = testset.loc[:, trainset.columns != yvar][testset[prev_group] == 0.0]
        y_test_age0 = testset[yvar][testset[prev_group] == 0.0]

        wow = round(abs(clf_wow.score(X_test_age0, y_test_age0) - clf_wow.score(X_test_age1, y_test_age1)), 3)
        ww = round(abs(clf_ww.score(X_test_age0, y_test_age0) - clf_ww.score(X_test_age1, y_test_age1)), 3)

        #output_probabilities_to_csv(model=clf_ww, x_test=X_test_age0, path='probs_unpriv_ww.csv')
        #output_probabilities_to_csv(model=clf_ww, x_test=X_test_age1, path='probs_priv_ww.csv')
        #output_probabilities_to_csv(model=clf_wow, x_test=X_test_age0, path='probs_unpriv_wow.csv')
        #output_probabilities_to_csv(model=clf_wow, x_test=X_test_age1, path='probs_priv_wow.csv')

        print("")
        print("")
        print("--------------------------------------------------------------")
        print("Difference in accuracy between privileged and unprivileged")
        print("--------------------------------------------------------------")
        print("without weights : ", wow)
        print("with weights    : ", ww)

        Ypredclf = clf_ww.predict(X_test)
        Ypredclf2 = clf_wow.predict(X_test)
        withw = confusion_matrix(y_test, Ypredclf)
        without = confusion_matrix(y_test, Ypredclf2)
        print("")
        print("")
        print("--------------------------------------------------------------")
        print("Confusion Matrix")
        print("--------------------------------------------------------------")
        print("without weights")
        print(without)
        print("")
        print("")
        print("with weights")
        print(withw)

        a, b, c, d = without.ravel()  #(tn, fp, fn, tp)
        a1, b1, c1, d1 = withw.ravel()  #(tn, fp, fn, tp)

        withweights = b1 * C_FP + c1 * C_FN
        withoutweights = b * C_FP + c * C_FN

        print("")
        print("")
        print("cost with weights: ", withweights)
        print("cost without weights: ", withoutweights)
        print("Has cost decreased after reweighing?", withweights < withoutweights)

        print('')
        print('SUMMARY TABLE')

        cost = fr.CostingFairness(input_dataframe=self.data,
                                  label_names=['credit'],
                                  protected_attribute_names=['Age_previliged'],
                                  trained_model=clf_ww)

        metrics_table = self.generate_pre_train_metrics_table(model_without_weights=clf_wow,
                                                              model_with_weights=clf_ww,
                                                              test_set=testset,
                                                              target=yvar,
                                                              privileged=prev_group,
                                                              false_positive_cost=C_FP,
                                                              false_negative_cost=C_FN)
        priv_diff_table = generate_privileged_diff(metrics_table)
        delta_table = generate_delta_table(metrics_table)
        costs_table = cost.return_cost_fairness_accuracy_optimised()

        # pdf = PDF()
        # pdf.add_page()
        # pdf.write_table_to_pdf(metrics_table)
        # pdf.write_table_to_pdf(priv_diff_table)
        # pdf.write_table_to_pdf(delta_table)
        # pdf.output('TEST01.pdf', 'F')

        print("")
        print("What we see is interesting, after re-weighing the bias of the model has decreased significantly by {}%, "
              "with a very slight decrease in accuracy as shown earlier".format(round((wow - ww) * 100)))

        return metrics_table, priv_diff_table, delta_table, costs_table

    def metrics_for_weighted_model(self, false_positive_cost, false_negative_cost):

        trn_bl_df = self.reweigh()
        s_weights = trn_bl_df.instance_weights
        trainset = trn_bl_df.convert_to_dataframe()[0]

        X = trainset.loc[:, trainset.columns != self.label_names[0]]
        y = trainset[self.label_names]

        clf_ww = sklearn.linear_model.LogisticRegression(random_state=999).fit(X, y, sample_weight=s_weights)
        clf_wow = sklearn.linear_model.LogisticRegression(random_state=999).fit(X, y)

        metrics_table = self.generate_pre_train_metrics_table(model_without_weights=clf_wow,
                                                              model_with_weights=clf_ww,
                                                              test_set=trainset,
                                                              target=self.label_names[0],
                                                              privileged=self.protected_attribute_names[0],
                                                              false_positive_cost=false_positive_cost,
                                                              false_negative_cost=false_negative_cost)
        return metrics_table

    def cost_matrix_for_weighted_model(self, false_positive_cost, false_negative_cost):

        trn_bl_df = self.reweigh()
        s_weights = trn_bl_df.instance_weights
        trainset = trn_bl_df.convert_to_dataframe()[0]

        X = trainset.loc[:, trainset.columns != self.label_names[0]]
        y = trainset[self.label_names]

        clf_ww = sklearn.linear_model.LogisticRegression(random_state=999).fit(X, y, sample_weight=s_weights)

        cost = fr.CostingFairness(input_dataframe=self.data,
                                  label_names=['credit'],
                                  protected_attribute_names=['Age_previliged'],
                                  trained_model=clf_ww)

        cost_matrix = cost.calculate_cost_matrix(false_positive_cost=false_positive_cost,
                                                 false_negative_cost=false_negative_cost)

        return cost_matrix

    def generate_pre_train_metrics_table(self,
                                         model_without_weights: sklearn.linear_model,
                                         model_with_weights: sklearn.linear_model,
                                         test_set: pd.DataFrame,
                                         target: str,
                                         privileged: str,
                                         false_positive_cost: float,
                                         false_negative_cost: float,
                                         pred: pd.DataFrame = None,
                                         pred_fair: pd.DataFrame = None) -> pd.DataFrame:
        """
        This function calculates the following table:

        | Metric         | 'Overall' WoW | 'Overall' WW | 'Priv = 1' WoW | 'Priv = 0' WoW | 'Priv = 1' WW | 'Priv = 0' WW |
        | Accuracy       |               |              |                |                |               |               |
        | True Negative  |               |              |                |                |               |               |
        | False Positive |               |              |                |                |               |               |
        | False Negative |               |              |                |                |               |               |
        | True Positive  |               |              |                |                |               |               |
        | TPR            |               |              |                |                |               |               |
        | FPR            |               |              |                |                |               |               |

        :param model_without_weights: The model object trained without weights
        :param model_with_weights: The model object trained with weights
        :param test_set: Data-set used for testing the model
        :param target: Target test data-set
        :param privileged: Column with privileged attribute. 1 is privileged, 0 is not
        :param pred: (Optional) prediction values that have already been calculated
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
            :return: A list with the following: ['Accuracy', 'True Positive', 'True Negative',
                                                 'False Positive', 'False Negative', 'AOD', 'TPR', 'FPR']
            """

            if y_predictions is not None:
                if not y_predictions.isnull().values.all():
                    y_pred = y_predictions
            elif 'themis_ml' in model.__module__:
                y_pred = model.predict(x_test, x_test[privileged].values)
            else:
                y_pred = model.predict(x_test)

            accuracy = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred)
            recall = sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred)
            auc = sklearn.metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
            tot_num = tn + fp + fn + tp
            tnp, fpp, fnp, tpp = tn / tot_num, fp / tot_num, fn / tot_num, tp / tot_num
            tpr = true_positive_rate(true_positives=tp, false_negatives=fn)  # Also called 'sensitivity' and 'recall'
            fpr = false_positive_rate(false_positives=fp, true_negatives=tn)
            cost = false_positive_cost * fp + false_negative_cost * fn
            output_list = [accuracy, precision, recall, auc, tnp, fpp, fnp, tpp, tpr, fpr, cost]
            return output_list

        # Define output table
        output_table = pd.DataFrame()

        y_pred = None
        y_pred_privileged = None
        y_pred_not_privileged = None
        y_pred_fair = None
        y_pred_fair_privileged = None
        y_pred_fair_not_privileged = None

        # Split input test_set into feature and target tables
        x_test = test_set.loc[:, (test_set.columns != target)]
        y_test = test_set[target]
        x_test_privileged = test_set.loc[:, (test_set.columns != target)][test_set[privileged] == 1.0]
        y_test_privileged = test_set[target][test_set[privileged] == 1.0]
        x_test_not_privileged = test_set.loc[:, (test_set.columns != target)][test_set[privileged] == 0.0]
        y_test_not_privileged = test_set[target][test_set[privileged] == 0.0]

        if pred is not None:
            test_set['pred'] = list(pred)
            y_pred = test_set['pred']
            y_pred_privileged = test_set['pred'][test_set[privileged] == 1.0]
            y_pred_not_privileged = test_set['pred'][test_set[privileged] == 0.0]

        if pred_fair is not None:
            test_set['pred_fair'] = list(pred_fair)
            y_pred_fair = test_set['pred_fair']
            y_pred_fair_privileged = test_set['pred_fair'][test_set[privileged] == 1.0]
            y_pred_fair_not_privileged = test_set['pred_fair'][test_set[privileged] == 0.0]

        # Build the output table
        output_table['Metric'] = ['Accuracy', 'Precision', 'Recall', 'AUC', 'True Negative', 'False Positive',
                                  'False Negative', 'True Positive', 'TPR', 'FPR', 'Cost']
        output_table['Overall WoW'] = generate_comparison_metrics(model=model_without_weights,
                                                                  x_test=x_test,
                                                                  y_test=y_test,
                                                                  y_predictions=y_pred)
        output_table['Overall WW'] = generate_comparison_metrics(model=model_with_weights,
                                                                 x_test=x_test,
                                                                 y_test=y_test,
                                                                 y_predictions=y_pred_fair)
        output_table['Priv WoW'] = generate_comparison_metrics(model=model_without_weights,
                                                               x_test=x_test_privileged,
                                                               y_test=y_test_privileged,
                                                               y_predictions=y_pred_fair)
        output_table['Not Priv WoW'] = generate_comparison_metrics(model=model_without_weights,
                                                                   x_test=x_test_not_privileged,
                                                                   y_test=y_test_not_privileged,
                                                                   y_predictions=y_pred_fair)
        output_table['Priv WW'] = generate_comparison_metrics(model=model_with_weights,
                                                              x_test=x_test_privileged,
                                                              y_test=y_test_privileged,
                                                              y_predictions=y_pred_privileged)
        output_table['Not Priv WW'] = generate_comparison_metrics(model=model_with_weights,
                                                                  x_test=x_test_not_privileged,
                                                                  y_test=y_test_not_privileged,
                                                                  y_predictions=y_pred_not_privileged)

        print(type(output_table))
        # Return output table
        return output_table

