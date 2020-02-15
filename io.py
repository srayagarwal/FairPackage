import fairness as fr
import fairness.utilities
import os
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

data_path = os.path.join(os.pardir,'data','germandata.csv')
data = pd.read_csv(data_path)
# scores_path = os.path.join(os.pardir, 'data', 'scores.csv')
scores = pd.read_csv(os.path.join(os.pardir, 'data', 'scores.csv'))


class OutputGenerator(object):

    def __init__(self,
                 fairness:fr.Fairness,
                 pre_proc:fr.PreProcessingFairness,
                 intrain:fr.InTrainingFairness,
                 post_proc:fr.PostProcessingFairness,
                 use_case: str, folderpath: str):

        self.timestamp = datetime.datetime.now()
        self.use_case = use_case
        self.folder_path = folderpath
        self.fairness = fairness
        self.pre_proc = pre_proc
        self.intrain = intrain
        self.post_proc = post_proc

        self.return_csv_files()
        self.return_json_files()

    def save_to_csv(self, dataframe: pd.DataFrame, filename):
        dataframe['Timestamp'] = self.timestamp
        dataframe['Use_Case'] = self.use_case
        dataframe.to_csv(self.folder_path + '/' + filename)

    def save_to_json(self, dataframe: pd.DataFrame, filename):
        dataframe['Timestamp'] = self.timestamp
        dataframe['Use_Case'] = self.use_case
        dataframe.to_json(self.folder_path + '/' + filename, orient='records')

    def return_csv_files(self):
        # 1
        heatmap = self.fairness.generate_heatmap_data()
        heatmap.reset_index(inplace=True)
        self.save_to_csv(dataframe=heatmap, filename="1_heatmap_data.csv")

        # 2
        wow_bias, wow_metrics = self.pre_proc.evaluate_bias()
        new_bl_df = self.pre_proc.reweigh()
        ww_bias, ww_metrics = self.pre_proc.evaluate_bias(bl_df=new_bl_df)
        bias_metrics = dict(wow=dict(bias=wow_bias, parity_sjsone=wow_metrics.statistical_parity_difference()),
                            ww=dict(bias=ww_bias, parity_score=ww_metrics.statistical_parity_difference()))
        bias_metrics_table = pd.DataFrame(bias_metrics)
        bias_metrics_table.reset_index(inplace=True)
        self.save_to_csv(dataframe=bias_metrics_table, filename="2_bias_metrics.csv")

        # 3
        table, delta_table, diff_table, cost_table = self.pre_proc.model_performance_comparison(
            yvar=self.pre_proc.label_names[0],
            prev_group=self.pre_proc.protected_attribute_names[0],
            C_FP=700,
            C_FN=300)
        self.save_to_csv(dataframe=table, filename="3_pre_proc_performance.csv")

        # 4
        results, cost_table_roc, cost_table_acf = self.intrain.intrain(
            protected_var=self.intrain.protected_attribute_names[0],
            yvar=self.intrain.label_names[0],
            C_FP=700,
            C_FN=300)
        self.save_to_csv(dataframe=results, filename="4_intrain_performance.csv")
        # 5
        post_proc_results = self.post_proc.t_post_process(
            cost_constraint="fpr",  # "fnr", "fpr" or "weighted"
            yvar=self.post_proc.label_names[0],
            scores=scores['0'],
            C_FP=700,
            C_FN=300,
            export=False)
        self.save_to_csv(dataframe=post_proc_results, filename="5_post_proc_performance.csv")
        # 6a
        cost_reweigh_matrix = self.pre_proc.cost_matrix_for_weighted_model(false_negative_cost=300, false_positive_cost=700)
        self.save_to_csv(dataframe=cost_reweigh_matrix, filename="6a_pre_proc_cost.csv")
        # 6b
        cost_acf_matrix = self.intrain.cost_matrix_acf(false_negative_cost=300, false_positive_cost=700)
        self.save_to_csv(dataframe=cost_acf_matrix, filename="6b_intrain_cost.csv")
        # 6c
        # N/A
        # 7a
        self.save_to_csv(dataframe=cost_table, filename="7a_reweighing_cost_optimisation.csv")
        # 7b
        self.save_to_csv(dataframe=cost_table_acf, filename="7b_acf_cost_optimisation.csv")
        # 7c
        # N/A

    def return_json_files(self):
        # 1
        heatmap = self.fairness.generate_heatmap_data()
        heatmap.reset_index(inplace=True)
        self.save_to_json(dataframe=heatmap, filename="1_heatmap_data.json")
        # 2
        wow_bias, wow_metrics = self.pre_proc.evaluate_bias()
        new_bl_df = self.pre_proc.reweigh()
        ww_bias, ww_metrics = self.pre_proc.evaluate_bias(bl_df=new_bl_df)
        bias_metrics = dict(wow=dict(bias=wow_bias, parity_score=wow_metrics.statistical_parity_difference()),
                            ww=dict(bias=ww_bias, parity_score=ww_metrics.statistical_parity_difference()))
        bias_metrics_table = pd.DataFrame(bias_metrics)
        bias_metrics_table.reset_index(inplace=True)
        self.save_to_json(dataframe=bias_metrics_table, filename="2_bias_metrics.json")
        # 3
        table, delta_table, diff_table, cost_table = self.pre_proc.model_performance_comparison(
            yvar=self.pre_proc.label_names[0],
            prev_group=self.pre_proc.protected_attribute_names[0],
            C_FP=700,
            C_FN=300)
        self.save_to_json(dataframe=table, filename="3_pre_proc_performance.json")
        # 4
        results, cost_table_roc, cost_table_acf = self.intrain.intrain(
            protected_var=self.pre_proc.protected_attribute_names[0],
            yvar=self.pre_proc.label_names[0],
            C_FP=700,
            C_FN=300)
        self.save_to_json(dataframe=results, filename="4_intrain_performance.json")
        # 5
        post_proc_results = self.post_proc.t_post_process(
            cost_constraint="fpr",  # "fnr", "fpr" or "weighted"
            yvar=self.pre_proc.label_names[0],
            scores=scores['0'],
            C_FP=700,
            C_FN=300,
            export=False)
        self.save_to_json(dataframe=post_proc_results, filename="5_post_proc_performance.json")
        # 6a
        cost_reweigh_matrix = self.pre_proc.cost_matrix_for_weighted_model(false_negative_cost=300, false_positive_cost=700)
        self.save_to_json(dataframe=cost_reweigh_matrix, filename="6a_pre_proc_cost.json")
        # 6b
        cost_acf_matrix = self.intrain.cost_matrix_acf(false_negative_cost=300, false_positive_cost=700)
        self.save_to_json(dataframe=cost_acf_matrix, filename="6b_intrain_cost.json")
        # 6c
        # N/A
        # 7a
        self.save_to_json(dataframe=cost_table, filename="7a_reweighing_cost_optimisation.json")
        # 7b
        self.save_to_json(dataframe=cost_table_acf, filename="7b_acf_cost_optimisation.json")
        # 7c
        # N/A
