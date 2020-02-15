import fairness as fr
import fairness.io
import pandas as pd
from datetime import datetime
import os

data_path = os.path.join(os.pardir,'data','germandata.csv')
data = pd.read_csv(data_path)


example = fr.Fairness(input_dataframe=data,
                      label_names=['credit'],
                      protected_attribute_names=['Age_previliged'],
                      privileged_groups=[{'Age_previliged': 1}], #Young
                      unprivileged_groups=[{'Age_previliged': 0}]  #Older
                      )

pre_proc = fr.PreProcessingFairness(input_dataframe=data,
                                    label_names=['credit'],
                                    protected_attribute_names=['Age_previliged'],
                                    privileged_groups=[{'Age_previliged': 1}], #Young
                                    unprivileged_groups=[{'Age_previliged': 0}]  #Older
                                    )

intrain = fr.InTrainingFairness(input_dataframe=data,
                                label_names=['credit'],
                                protected_attribute_names=['Age_previliged'],
                                privileged_groups=[{'Age_previliged': 1}], #Young
                                unprivileged_groups=[{'Age_previliged': 0}]  #Older
                                )

post_proc = fr.PostProcessingFairness(input_dataframe=data,
                                      label_names=['credit'],
                                      protected_attribute_names=['Age_previliged'],
                                      privileged_groups=[{'Age_previliged': 1}], #Young
                                      unprivileged_groups=[{'Age_previliged': 0}]  #Older
                                      )

costs = fr.CostingFairness(input_dataframe=data,
                           label_names=['credit'],
                           protected_attribute_names=['Age_previliged'],
                           )


if __name__ == '__main__':





    # # print(pre_proc.metrics_for_weighted_model())
    # op = fairness.io.OutputGenerator(use_case='test', folderpath='output_csv')
    # op.return_csv_files()
    # op.return_json_files()

    # print('Show Metrics:')
    # bias, metrics = pre_proc.evaluate_bias()
    # print('MEAN DIFFERENCE: ' + str(metrics.mean_difference()))
    # print('BIAS DETECTED: ' + str(bias))
    #
    # print('Show improvements')
    # new_bl_df = pre_proc.reweigh()
    # bias, metrics = pre_proc.evaluate_bias(bl_df=new_bl_df)
    # print('MEAN DIFFERENCE: ' + str(metrics.mean_difference()))
    # print('BIAS DETECTED: ' + str(bias))
    #
    # print('Show reweigh_model_perf')
    # table, delta_table, diff_table, cost_table = pre_proc.model_performance_comparison(yvar='credit',
    #                                                                                    prev_group='Age_previliged',
    #                                                                                    C_FP=700,
    #                                                                                    C_FN=300)
    #
    # print(table)
    # print(delta_table)
    # print(diff_table)
    #
    # print('-------------Show intrain--------------')
    # results, cost_table_roc, cost_table_acf = intrain.intrain(protected_var='Age_previliged',
    #                                                           yvar='credit',
    #                                                           C_FP=700,
    #                                                           C_FN=300)
    #
    # print(results)
    #
    # print('Show post_process')
    #
    scores_path = os.path.join(os.pardir,'data','scores.csv')
    scores=pd.read_csv(scores_path)
    # # print(scores)
    # #
    # # post_proc_results = post_proc.t_post_process(cost_constraint="fpr",  # "fnr", "fpr" or "weighted"
    # #                                              yvar='credit',
    # #                                              scores=scores['0'],
    # #                                              C_FP=700,
    # #                                              C_FN=300,
    # #                                              export=False)
    # # print(post_proc_results)
    # #
    name = "Credit_Fairness_Report__" + str(datetime.now()) + ".pdf"
    # #
    # # # print(costs.return_cost_fairness_accuracy_optimised(false_negative_cost=300, false_positive_cost=700))
    # #
    example.generate_report(yvar='credit',
                            protected_variable='Age_previliged',
                            scores=scores['0'],
                            false_positive_cost=700,
                            false_negative_cost=300,
                            filepath=name)
