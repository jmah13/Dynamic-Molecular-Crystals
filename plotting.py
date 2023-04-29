import numpy as np
import pandas as pd
import seaborn as sns
import statistical_analysis as sa
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics



def plot(data, col_types, x_axis, y_axis):
    num_categorical = np.sum(col_types.loc[[x_axis, y_axis]]['type'] == 'categorical')
    if num_categorical == 0: 
        regplot(data, col_types, x_axis, y_axis)
    elif num_categorical == 1:
        boxplot(data, col_types, x_axis, y_axis)
    elif num_categorical == 2:
        print('Plotting for 2 categoricals - Not Implemented')
    else:
        pass

def regplot(data, col_types, x_axis, y_axis):
    r_val, ignored = sa.r2(sa.scale_data(data[x_axis], x_axis, col_types),  
                           sa.scale_data(data[y_axis], y_axis, col_types))
    print(f'Pearson R2: {r_val} - Used: {len(data) - ignored} / {len(data)}')
    ax = sns.regplot(x=x_axis, y=y_axis, data=data)
    ax = set_scales(ax, col_types, x_axis, y_axis)


def boxplot(data, col_types, x_axis, y_axis):

    ax = sns.boxplot(x=x_axis, y=y_axis, 
                data=data, showmeans=True,meanline=True, showfliers = False)
    ax = sns.swarmplot(x=x_axis, y=y_axis, data=data, color=".25")
    ax = set_scales(ax, col_types, x_axis, y_axis)

def set_scales(ax, col_types, x_axis, y_axis):

    if col_types.loc[x_axis]['scale'] ==  'log':
        ax.set_xscale("log")

    if col_types.loc[y_axis]['scale'] ==  'log':
        ax.set_yscale("log")   
    
    return ax

def run_comparisons(df, selected_col_types, test, 
                    comparison_type = 'one_vs_all', 
                    plot_whole_caterogy = 1, display_whole_caterogy = 1,
                    plot_significant_comparisons = 1, display_significant_comparisons = 1):

    all_sig_results = []
    all_results = []

    for cat_name, cat_row in selected_col_types[selected_col_types['type'] == 'categorical'].iterrows():
        for con_name, con_row in selected_col_types[selected_col_types['type'] == 'continous'].iterrows():
            print(f'{cat_name} -vs- {con_name}')
            analysis = sa.CategoricalVsContinous(df, cat_name, con_name, test, plot=plot_whole_caterogy)
            
            results, result_objs = analysis.run_test(group_type=comparison_type)

            all_results.append(results)
            sig_results = results[results['overall_sig'] == 'Yes']
            all_sig_results.append(sig_results)
            
            if display_whole_caterogy:
                display(sig_results)
            
            for i, row in sig_results.iterrows():
                if plot_significant_comparisons:
                    result_objs[i].plot_data()
                    
                if display_significant_comparisons:
                    display(sig_results.loc[i:i])
    
    return pd.concat(all_sig_results), pd.concat(all_results)


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

def get_metrics_df(evals_result):
    metrics_dict = {}
    for col, metrics in evals_result.items():
        for metric_name, val in metrics.items():
            metrics_dict[f'{col}-{metric_name}'] = val
            
    metrics_df = pd.DataFrame(metrics_dict)
    return metrics_df


def plot_regression(model, dtest, y_test, y_train, quartiles, title=None):
    preds = model.predict(dtest)

    df = pd.DataFrame({'Predictions':preds, 'Groud Truth': y_test})
    df_train = pd.DataFrame({'Train1':y_train, 'Train2': y_train})

    r2_val = r2(df['Predictions'], df['Groud Truth'])
    print(f'R2: {r2_val}')
    p = sns.jointplot(data=df, x='Predictions', y='Groud Truth', kind="reg")
    
    p.x = df_train.Train1
    p.y = df_train.Train1
    p.plot_joint(plt.scatter, marker='o', c='b', s=50)
    for ax in (p.ax_joint, p.ax_marg_x):
        ax.axvline(quartiles['Q1'], color='crimson', ls='--', lw=1)
        #ax.axvline(quartiles['median'], color='crimson', ls='--', lw=1)
        ax.axvline(quartiles['Q2'], color='crimson', ls='--', lw=1)

    for ax in (p.ax_joint, p.ax_marg_y):
        ax.axhline(quartiles['Q1'], color='crimson', ls='--', lw=1)
        #ax.axhline(quartiles['median'], color='crimson', ls='--', lw=1)
        ax.axhline(quartiles['Q2'], color='crimson', ls='--', lw=1)
    #p.plot_joint(data=df_train, x='Train1', y='Train2', kind="scatter")

    if title is not None: 
        p.fig.tight_layout()
        p.fig.suptitle(title)

    plt.show()

def plot_AUC(model, dtest, y_test, title=None):
    pass
    # https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
    # preds = model.predict(dtest)
    # probs = model.predict_proba(dtest)

    # fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

    # skplt.metrics.plot_roc_curve(y_true, y_probas)
    # plt.show()
    # import pdb; pdb.set_trace()
    # df = pd.DataFrame({'Predictions':preds, 'Ground Truth': y_test})
    # p = sns.jointplot(data=df, x='Predictions', y='Ground Truth', kind="reg")#, stat_func=r2)
    # if title is not None: 
    #     p.fig.tight_layout()
    #     p.fig.suptitle(title)

    # plt.show()

def plot_training_metrics(evals_result, title=None):
    metrics_df = get_metrics_df(evals_result)

    sns.lineplot(data=metrics_df)
    if title is not None: plt.title(title)
    plt.show()