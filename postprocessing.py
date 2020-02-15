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


class PostProcessingFairness(fr.Fairness):

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

    def t_post_process(self,
                       yvar,
                       scores,
                       cost_constraint,
                       C_FP,
                       C_FN,
                       class_thresh=0.5,
                       randseed=12345679,
                       export: bool = False):

        trainset = self.binary_label_df.convert_to_dataframe()[0]
        training_data = self.data
        X = training_data.loc[:, training_data.columns != yvar].values
        y = training_data[yvar].values

        y_train_pred_prob = scores.values

        # dataset_orig_train_pred = scores.values
        dataset_orig_train_pred = self.binary_label_df.copy(deepcopy=True)

        # dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
        dataset_orig_train_pred.scores = scores.values

        y_train_pred = dataset_orig_train_pred.labels
        y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
        y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
        dataset_orig_train_pred.labels = y_train_pred
        cm_pred_train = ClassificationMetric(self.binary_label_df,
                                             dataset_orig_train_pred,
                                             unprivileged_groups=self.unprivileged_groups,
                                             privileged_groups=self.privileged_groups)

        print("")
        print("--------------------------------------------------------------")
        print("Original-Predicted dataset")
        print("--------------------------------------------------------------")

        print("")

        print("Difference in GFPR between unprivileged and privileged groups")
        print(cm_pred_train.difference(cm_pred_train.generalized_false_positive_rate))
        print("Difference in GFNR between unprivileged and privileged groups")
        print(cm_pred_train.difference(cm_pred_train.generalized_false_negative_rate))

        cpp = CalibratedEqOddsPostprocessing(privileged_groups=self.privileged_groups,
                                             unprivileged_groups=self.unprivileged_groups,
                                             cost_constraint=cost_constraint,
                                             seed=randseed)
        cpp = cpp.fit(self.binary_label_df, dataset_orig_train_pred)

        dataset_transf_train_pred = cpp.predict(dataset_orig_train_pred)

        if export is True:
            pd.DataFrame(dataset_transf_train_pred.labels).to_csv('ExportLabels.csv')
            pd.DataFrame(dataset_transf_train_pred.scores).to_csv('ExportScores.csv')

        cm_transf_test = ClassificationMetric(self.binary_label_df,
                                              dataset_transf_train_pred,
                                              unprivileged_groups=self.unprivileged_groups,
                                              privileged_groups=self.privileged_groups)

        print("")
        print("--------------------------------------------------------------")
        print("Original-Transformed dataset")
        print("--------------------------------------------------------------")

        print("")

        print("Difference in GFPR between unprivileged and privileged groups")
        print(cm_transf_test.difference(cm_transf_test.generalized_false_positive_rate))
        print("Difference in GFNR between unprivileged and privileged groups")
        print(cm_transf_test.difference(cm_transf_test.generalized_false_negative_rate))

        before_post_processing = dataset_orig_train_pred.labels.reshape([1, len(y)])[0].tolist()
        after_post_processing = dataset_transf_train_pred.labels.reshape([1, len(y)])[0].tolist()

        ppdf=pd.DataFrame()
        ppdf['before_pp']=before_post_processing
        ppdf['after_pp']=dataset_transf_train_pred.scores
        ppdf['protected_attributes']=dataset_orig_train_pred.protected_attributes
        ppdf.to_csv('post_processing.csv')

        print("")
        print("--------------------------------------------------------------")
        print("Confusion Matrix")
        print("--------------------------------------------------------------")

        print("")
        print("Before")
        print(confusion_matrix(y, before_post_processing))
        print("")
        print("")
        print("After")
        print(confusion_matrix(y, after_post_processing))

        a, b, c, d = confusion_matrix(y, before_post_processing).ravel()
        a1, b1, c1, d1 = confusion_matrix(y, after_post_processing).ravel()

        before = c * C_FP + b * C_FN
        after = c1 * C_FP + b1 * C_FN

        print ("cost before pp", before)
        print ("cost after pp", after)

        print("")
        print("")
        print("------------------------------------------------------------------------------------------")

        print("Has cost decreased after calibrated post processing?", after < before)

        metrics = self.generate_post_proc_metrics_table(baseline_predictions=dataset_orig_train_pred,
                                                        false_negative_cost=C_FN,
                                                        false_positive_cost=C_FP)

        return metrics

    # def cost_matrix_post_proc(self, cost_constraint, false_positive_cost, false_negative_cost, randseed=0):
    #
    #     cpp = CalibratedEqOddsPostprocessing(privileged_groups=self.privileged_groups,
    #                                          unprivileged_groups=self.unprivileged_groups,
    #                                          cost_constraint=cost_constraint,
    #                                          seed=randseed)
    #
    #     dataset_orig_train_pred = self.binary_label_df.copy(deepcopy=True)
    #
    #     cpp = cpp.fit(self.binary_label_df, dataset_orig_train_pred)
    #
    #     predictions_new = cpp.predict_proba(dataset_orig_train_pred)
    #
    #     y = self.data[self.label_names[0]].values
    #
    #     pred_after_post_processing = predictions_new.labels.reshape([1, len(y)])[0].tolist()
    #
    #     cost = fr.CostingFairness(input_dataframe=self.data,
    #                               label_names=self.label_names,
    #                               protected_attribute_names=self.protected_attribute_names,
    #                               predictions=pred_after_post_processing)
    #
    #     cost_matrix = cost.calculate_cost_matrix(false_positive_cost, false_negative_cost)
    #
    #     return cost_matrix

    def generate_post_proc_metrics_table(self,
                                         baseline_predictions: aif360.datasets.binary_label_dataset.BinaryLabelDataset,
                                         # target: str,
                                         # privileged: str,
                                         false_positive_cost: float,
                                         false_negative_cost: float) -> pd.DataFrame:
        """
        This function calculates the following table:

        | Metric         | Baseline | FNR | FPR | Weighted |
        | GFNR           |          |     |     |
        | GFPR           |
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

        def generate_comparison_metrics(self,
                                        cost_constraint: str = None):
            """
            This function returns a standard list of values of descriptive metrics, given an input model, features and
            target.
            :param model: The trained model object
            :param x_test: Feature test data-set
            :param y_test: Target test data-set
            :param y_predictions:(Optional) prediction values that have already been calculated
            :return: A list with the following: ['SUC', 'True Positive', 'True Negative',
                                                 'False Positive', 'False Negative', 'Cost']
            """
            if cost_constraint is not None:
                model = CalibratedEqOddsPostprocessing(privileged_groups=self.privileged_groups,
                                                       unprivileged_groups=self.unprivileged_groups,
                                                       cost_constraint=cost_constraint)

                model.fit(self.binary_label_df, baseline_predictions)
                predictions = model.predict(baseline_predictions)

            else:
                predictions = baseline_predictions

            y = self.data[self.label_names]

            metrics = ClassificationMetric(self.binary_label_df,
                                           predictions,
                                           unprivileged_groups=self.unprivileged_groups,
                                           privileged_groups=self.privileged_groups)

            gfnr_diff = metrics.difference(metrics.generalized_false_negative_rate)
            gfpr_diff = metrics.difference(metrics.generalized_false_positive_rate)

            pred_after_post_processing = predictions.labels.reshape([1, len(y)])[0].tolist()

            accuracy = sklearn.metrics.accuracy_score(y_true=y, y_pred=pred_after_post_processing)
            precision = sklearn.metrics.precision_score(y_true=y, y_pred=pred_after_post_processing)
            recall = sklearn.metrics.recall_score(y_true=y, y_pred=pred_after_post_processing)

            auc = sklearn.metrics.roc_auc_score(y_true=y, y_score=pred_after_post_processing)

            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=pred_after_post_processing).ravel()
            tot_num = tn + fp + fn + tp
            tnp, fpp, fnp, tpp = tn / tot_num, fp / tot_num, fn / tot_num, tp / tot_num

            tpr = fr.utilities.true_positive_rate(true_positives=tp, false_negatives=fn)
            fpr = fr.utilities.false_positive_rate(false_positives=fp, true_negatives=tn)

            cost = false_positive_cost * fp + false_negative_cost * fn

            output_list = [gfnr_diff, gfpr_diff, accuracy, precision, recall, auc, tnp, fpp, fnp, tpp, tpr, fpr, cost]
            return output_list

        # Define output table
        output_table = pd.DataFrame()

        # Build the output table
        output_table['Metric'] = ['GFNR Diff', 'GFPR Diff', 'Accuracy', 'Precision', 'Recall', 'AUC', 'True Negative',
                                  'False Positive', 'False Negative', 'True Positive', 'TPR', 'FPR', 'Cost']
        output_table['Baseline'] = generate_comparison_metrics(self=self)
        output_table['FNR Optimised'] = generate_comparison_metrics(self=self, cost_constraint='fnr')
        output_table['FPR Optimised'] = generate_comparison_metrics(self=self, cost_constraint='fpr')
        output_table['Weighted Optimised'] = generate_comparison_metrics(self=self, cost_constraint='weighted')
        # Return output table
        return output_table
