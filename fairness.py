# Import custom tools
from config import ROOT_DIR
import fairness as fr
from fairness.utilities import output_probabilities_to_csv, PDF, \
    generate_privileged_diff, generate_delta_table
# Scikit-learn
from sklearn.metrics import confusion_matrix
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
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


class Fairness(object):

    def __init__(self,
                 input_dataframe:pd.DataFrame,
                 label_names,
                 protected_attribute_names,
                 privileged_groups=None,
                 unprivileged_groups=None):

        self.data = input_dataframe
        self.label_names = label_names
        self.protected_attribute_names = protected_attribute_names
        self.new_df, self.binary_label_df = self.prepare_data(self.data, self.label_names, self.protected_attribute_names)
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups

    @staticmethod
    def prepare_data(df, label_names, protected_attribute_names):

        newdf = aif360.datasets.StructuredDataset(df,
                                                  label_names=label_names,
                                                  protected_attribute_names=protected_attribute_names)

        bl_df = aif360.datasets.BinaryLabelDataset(df=df,
                                                   label_names=label_names,
                                                   protected_attribute_names=protected_attribute_names)
        return newdf, bl_df

    def metrics(self, **kwargs):
        if kwargs:
            bl_df = kwargs['bl_df']
        else:
            bl_df = self.binary_label_df
        aif = aif360.metrics.BinaryLabelDatasetMetric(bl_df, unprivileged_groups=self.unprivileged_groups,
                                                      privileged_groups=self.privileged_groups)
        print("Base Rate", aif.base_rate())
        print("Consistency", aif.consistency())
        print("Disparate Impact", aif.disparate_impact())
        print("Mean Difference", aif.mean_difference())
        print("Statistical Parity Difference", aif.statistical_parity_difference())
        if (aif.statistical_parity_difference() > 0.1) or (aif.statistical_parity_difference() < -0.1):
            print("BIAS DETECTED")

        else:
            print("NO BIAS")

    def plot_heatmap(self, yvar, protected_variable, filepath:str):
        sns.heatmap(pd.crosstab(self.data[protected_variable], self.data[yvar]), annot=True, fmt="d")
        plt.title('Heat-map of Protected Variable and Target Variable')
        plt.savefig(filepath)
        plt.clf()

    def plot_proportions(self, yvar: str, protected_variable: str, filepath:str):
        summary = pd.crosstab(self.data[protected_variable], self.data[yvar])
        summary['perc_true'] = summary[1] / (summary[0] + summary[1])
        summary['perc_false'] = 1 - summary['perc_true']
        summary = summary.reset_index(inplace=False)
        plt.bar(summary[protected_variable], summary['perc_true'], color='b', label=str(yvar + ' is True'))
        plt.bar(summary[protected_variable], summary['perc_false'], color='r', bottom=summary['perc_true'],
                label=str(yvar + ' is False'))
        plt.xticks([0, 1])
        plt.xlabel(protected_variable)
        plt.ylabel('Percentage')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True)
        plt.savefig(filepath)
        plt.clf()

    def generate_heatmap_data(self):
        heatmap_data = pd.crosstab(self.data[self.protected_attribute_names[0]], self.data[self.label_names[0]])
        return heatmap_data

    def generate_report(self,
                        yvar,
                        protected_variable,
                        scores,
                        false_positive_cost,
                        false_negative_cost,
                        filepath='Fairness_Report.pdf'):

        # -------------------------------------------------------------------------------------------------------
        # Initialise PDF Report
        # -------------------------------------------------------------------------------------------------------
        pdf = PDF()

        # -------------------------------------------------------------------------------------------------------
        # Set title and Contents
        # -------------------------------------------------------------------------------------------------------
        pdf.add_page()
        pdf.set_y(15)
        pdf.set_font('Arial', 'B', 20)
        text = 'Fairness Report - ' + str(datetime.datetime.now().strftime('%d/%m/%Y'))
        pdf.cell(0, 0, txt=text, align='C')
        pdf.ln(20)

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 0, txt='Table of Contents', align='L')
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 12)
        pdf.cell(0, 0, txt='1.0 - Introduction to the Data', align='L')
        pdf.cell(0, 0, txt='2', align='R')
        pdf.ln(10)
        pdf.cell(0, 0, txt='    1.1 - Heatmap of the Distribution of Protected Variable vs Target Variable', align='L')
        pdf.cell(0, 0, txt='2', align='R')
        pdf.ln(10)
        pdf.cell(0, 0, txt='    1.2 - Plot of Proportions of Target Variable vs. Protected Variable', align='L')
        pdf.cell(0, 0, txt='2', align='R')
        pdf.ln(10)
        pdf.cell(0, 0, txt='2.0 - Bias Detection and Weighting', align='L')
        pdf.cell(0, 0, txt='3', align='R')
        pdf.ln(10)
        pdf.cell(0, 0, txt='    2.1 - Bias Detection', align='L')
        pdf.cell(0, 0, txt='3', align='R')
        pdf.ln(10)
        pdf.cell(0, 0, txt='    2.2 - Bias Detection Post Re-Weighting', align='L')
        pdf.cell(0, 0, txt='3', align='R')
        pdf.ln(10)
        pdf.cell(0, 0, txt='3.0 - Performance After Re-Weighting', align='L')
        pdf.cell(0, 0, txt='4', align='R')
        pdf.ln(10)
        pdf.cell(0, 0, txt='4.0 - In-Training Fairness', align='L')
        pdf.cell(0, 0, txt='5', align='R')
        pdf.ln(10)
        pdf.cell(0, 0, txt='5.0 - Post Prediction Fairness', align='L')
        pdf.cell(0, 0, txt='6', align='R')
        pdf.ln(10)

        # -------------------------------------------------------------------------------------------------------
        # 1.0 - Introduction to Data
        # -------------------------------------------------------------------------------------------------------
        pdf.add_page()
        pdf.set_y(15)
        pdf.set_font('Arial', 'B', 20)
        text = 'Fairness Report - ' + str(datetime.datetime.now().strftime('%d/%m/%Y'))
        pdf.cell(0, 0, txt=text, align='C')
        pdf.ln(20)

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 0, txt='1.0 - Introduction to the Data', align='L')
        pdf.ln(10)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 0, "    1.1 - The Target and Protected Columns", align='L')
        pdf.ln(5)
        pdf.set_x(55)
        pdf.cell(50, 10, yvar, 1, 0, 'C')
        pdf.cell(50, 10, protected_variable, 1, 0, 'C')
        pdf.ln(10)
        pdf.set_font('arial', '', 12)
        for i in range(0, 5):
            pdf.set_x(55)
            pdf.cell(50, 8, '%s' % (str(self.data[yvar].ix[i])), 1, 0, 'C')
            pdf.cell(50, 8, '%s' % (str(self.data[protected_variable].ix[i])), 1, 0, 'C')
            pdf.ln(8)
        pdf.ln(10)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 0, txt='    1.1 - Heatmap of the Distribution of Protected Variable vs Target Variable', align='L')
        pdf.ln(5)
        self.plot_heatmap(yvar=yvar, protected_variable=protected_variable, filepath='heatmap.png')
        pdf.image('heatmap.png', x=60, w=100)
        os.remove('heatmap.png')
        pdf.ln(5)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 0, txt='    1.2 - Plot of Proportions of Target Variable vs. Protected Variable', align='L')
        pdf.ln(5)
        self.plot_proportions(yvar=yvar, protected_variable=protected_variable, filepath='props.png')
        pdf.image('props.png', x=55, w=100)
        os.remove('props.png')

        # -------------------------------------------------------------------------------------------------------
        # 2.0 - Bias detection in Data Section
        # -------------------------------------------------------------------------------------------------------
        pre_proc = fr.PreProcessingFairness(input_dataframe=self.data,
                                            label_names=self.label_names,
                                            protected_attribute_names=self.protected_attribute_names,
                                            privileged_groups=self.privileged_groups,  # Young
                                            unprivileged_groups=self.unprivileged_groups  # Older
                                            )






        pdf.add_page()
        pdf.set_y(15)
        pdf.set_font('Arial', 'B', 20)
        text = 'Fairness Report - ' + str(datetime.datetime.now().strftime('%d/%m/%Y'))
        pdf.cell(0, 0, txt=text, align='C')
        pdf.ln(20)

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 0, txt='2.0 - Bias Detection and Mitigation in Data', align='L')
        pdf.ln(10)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 0, txt='    2.1 - Bias Detection', align='L')
        pdf.ln(10)

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        pdf.set_font('Arial', '', 12)
        self.metrics()
        pdf.write(5,mystdout.getvalue())
        pdf.ln(10)
        sys.stdout = old_stdout

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 0, txt='    2.2 - Bias Detection Post Re-Weighting', align='L')
        pdf.ln(10)

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        pdf.set_font('Arial', '', 12)
        new_bl_df = pre_proc.reweigh()
        self.metrics(bl_df=new_bl_df)
        pdf.write(5, mystdout.getvalue())
        pdf.ln(10)
        sys.stdout = old_stdout

        # -------------------------------------------------------------------------------------------------------
        # 3.0 - Performance Measurement after Bias Mitigation in Data
        # -------------------------------------------------------------------------------------------------------
        pdf.add_page()
        pdf.set_y(15)
        pdf.set_font('Arial', 'B', 20)
        text = 'Fairness Report - ' + str(datetime.datetime.now().strftime('%d/%m/%Y'))
        pdf.cell(0, 0, txt=text, align='C')
        pdf.ln(20)

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 0, txt='3.0 - Performance of Model after Mitigating Bias in Data', align='L')
        pdf.ln(10)
        print('')

        metrics_table, priv_diff_table, delta_table, costs_table = pre_proc.model_performance_comparison(yvar=yvar,
                                                                                                         prev_group=protected_variable,
                                                                                                         C_FP=false_positive_cost,
                                                                                                         C_FN=false_negative_cost)

        pdf.write_table_to_pdf(metrics_table)
        pdf.write_table_to_pdf(priv_diff_table)
        pdf.write_table_to_pdf(delta_table)
        pdf.write_table_to_pdf(costs_table)

        # -------------------------------------------------------------------------------------------------------
        # 4.0 - Bias Detection and Mitigation During Training
        # -------------------------------------------------------------------------------------------------------
        intrain = fr.InTrainingFairness(input_dataframe=self.data,
                                        label_names=self.label_names,
                                        protected_attribute_names=self.protected_attribute_names,
                                        privileged_groups=self.privileged_groups,  # Young
                                        unprivileged_groups=self.unprivileged_groups  # Older
                                        )

        pdf.add_page()
        pdf.set_y(15)
        pdf.set_font('Arial', 'B', 20)
        text = 'Fairness Report - ' + str(datetime.datetime.now().strftime('%d/%m/%Y'))
        pdf.cell(0, 0, txt=text, align='C')
        pdf.ln(20)

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 0, txt='4.0 - In-Training Fairness', align='L')
        pdf.ln(10)

        results, costs_roc_table, costs_acf_table = intrain.intrain(protected_var=protected_variable,
                                                                    yvar=yvar,
                                                                    C_FP=false_positive_cost,
                                                                    C_FN=false_negative_cost)

        pdf.write_table_to_pdf(results)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 0, txt='ROC Cost Optimisation', align='L')
        pdf.ln(10)

        pdf.write_table_to_pdf(costs_roc_table)

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 0, txt='ACF Cost Optimisation', align='L')
        pdf.ln(10)

        pdf.write_table_to_pdf(costs_acf_table)

        # -------------------------------------------------------------------------------------------------------
        # 5.0 - Bias Detection and Mitigation in Predictions
        # -------------------------------------------------------------------------------------------------------
        pdf.add_page()
        pdf.set_y(15)
        pdf.set_font('Arial', 'B', 20)
        text = 'Fairness Report - ' + str(datetime.datetime.now().strftime('%d/%m/%Y'))
        pdf.cell(0, 0, txt=text, align='C')
        pdf.ln(20)

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 0, txt='5.0 - Post Prediction Fairness', align='L')
        pdf.ln(10)

        post_proc = fr.PostProcessingFairness(input_dataframe=self.data,
                                            label_names=self.label_names,
                                            protected_attribute_names=self.protected_attribute_names,
                                            privileged_groups=self.privileged_groups,  # Young
                                            unprivileged_groups=self.unprivileged_groups  # Older
                                            )

        # scores_path = os.path.join(os.pardir, 'data', 'scores.csv')
        # scores = pd.read_csv(scores_path)


        post_proc_results = post_proc.t_post_process(cost_constraint="fpr",  # "fnr", "fpr" or "weighted"
                                                     yvar=yvar,
                                                     scores=scores,
                                                     C_FP=false_positive_cost,
                                                     C_FN=false_negative_cost,
                                                     export=False)



        pdf.write_table_to_pdf(post_proc_results)

        # -------------------------------------------------------------------------------------------------------
        # 6.0 - Conclusion
        # -------------------------------------------------------------------------------------------------------

        pdf.add_page()
        pdf.set_y(15)
        pdf.set_font('Arial', 'B', 20)
        text = 'Fairness Report FPR - ' + str(datetime.datetime.now().strftime('%d/%m/%Y'))
        pdf.cell(0, 0, txt=text, align='C')
        pdf.ln(20)

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 0, txt='6.0 - Comparing Costs over Approaches', align='L')
        pdf.ln(10)

        pdf.image(os.path.join(ROOT_DIR, 'data', 'cost_fairness.png'), x=5, w=200)

        pdf.output(filepath, 'F')
