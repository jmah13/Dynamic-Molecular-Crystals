import numpy as np
import pandas as pd
import copy
from scipy import stats
import itertools
import plotting as pt
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.stats import t

def get_col_types(df):
    col_types = []
    for col in df.columns:
        vals = df[col].values

        types = {'name':col}

        if vals.dtype == 'float64':
            min_val = np.nanmin(vals)
            max_val = np.nanmax(vals)
            range_vals = max_val / min_val 

            col_type = 'continous'
            if  range_vals > 1e3 and min_val != 0:
                scale = 'log'
            else:
                scale = 'linear' 
                
            types.update({'type':col_type, 'scale':scale,'min':min_val, 'max':max_val, 'range':range_vals})
                
        if vals.dtype == 'object':
            col_type = 'categorical'
            num_categories = int(len(df[col].unique()))
            types.update({'type':col_type, 'num_categories':num_categories})
            

        col_types.append(types)

    col_types = pd.DataFrame(col_types)
    col_types = col_types.set_index('name')
    return col_types

def scale_data(vals, col_name, col_types):
    if col_types.loc[col_name]['scale'] ==  'log':
        vals = np.log(vals)

    return vals

def unique_vals(data, col_name):
    vals = [x for x in data[col_name] if str(x) != 'nan']
    return np.unique(vals)

def get_pairs(data, col_name):
    pairs = list(itertools.combinations(unique_vals(data, col_name), 2))
    return pairs
    
def r2(x, y):
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    return stats.pearsonr(x[~nas], y[~nas])[0] ** 2, np.sum(nas)

class StatTest():
    def __call__(self, group1, group2, col_name, col_types):
        pass

    def is_statistical(self, statistic):
        pass

    def scale_groups(self, group1, group2, col_name, col_types):
        group1 = scale_data(group1[col_name], col_name, col_types).dropna()
        group2 = scale_data(group2[col_name], col_name, col_types).dropna()
        return group1, group2

class KStest(StatTest):
    @property
    def name(self):
        return 'ks'

    @property
    def min_ks_statistic(self):
        return 0.2

    def __call__(self, group1, group2, col_name, col_types):
        group1, group2 = self.scale_groups(group1, group2, col_name, col_types)
        return stats.ks_2samp(group1, group2, mode='exact')

    def is_statistical(self, ks_statistic):
        return ks_statistic >= self.min_ks_statistic

class Ttest(StatTest):
    @property
    def name(self):
        return 't-test'

    def min_t_statistic(self, group1, group2):
        dgf = len(group1) + len(group2) - 2
        cv = t.ppf(1.0 - 0.05, dgf)
        return cv

    def __call__(self, group1, group2, col_name, col_types):
        group1, group2 = self.scale_groups(group1, group2, col_name, col_types)
        return stats.ttest_ind(group1, group2)

    def is_statistical(self, t_statistic, group1, group2):
        return abs(t_statistic) >= self.min_t_statistic(group1, group2),self.min_t_statistic(group1, group2)


class Analysis():
    def __init__(self, data):
        self.data = data
        self.col_types = get_col_types(data)
    
    @property
    def max_pvalue(self): return 0.05

    @property
    def min_sample_size(self): return 10

    def statistical_significance_pvalue(self, pvalue):
        return pvalue <= self.max_pvalue
    
    def statistical_significance_ks_statistic(self, ks_statistic):
        return ks_statistic >= self.min_ks_statistic

class CategoricalVsContinous(Analysis):
    def __init__(self, data, categorical_col, continous_col, test: StatTest, plot=False):
        super().__init__(data)
        self.cat_col = categorical_col
        self.con_col = continous_col
        self.cat_data = self.data[self.cat_col]
        self.con_data = self.data[self.con_col]
        self.test = test
        self.plot = plot

        assert self.col_types.loc[self.con_col]['type'] == 'continous'
        assert self.col_types.loc[self.cat_col]['type'] == 'categorical'
        self.scale = self.col_types.loc[self.con_col]['scale']

        if self.plot:
            self.plot_data()
    
    def plot_data(self):
        pt.boxplot(self.data, self.col_types, self.con_col, self.cat_col)
        plt.show()
    
    def one_vs_all_grouping(self):
        grouping = []
        for val in unique_vals(self.data, self.cat_col):
            group1 = self.data[self.cat_data == val]
            group2 = self.data[self.cat_data != val]
            group1_name = val
            group2_name = f'not {val}'
            grouping.append({group1_name: group1, group2_name: group2})
        
        return grouping

    def pair_grouping(self):
        grouping = []
        pairs = get_pairs(self.data, self.cat_col)

        for pair in pairs:
            group1 = self.data[self.cat_data == pair[0]]
            group2 = self.data[self.cat_data == pair[1]]
            grouping.append({pair[0]: group1, pair[1]: group2})

        return grouping
    
    def get_grouping(self, group_type='pairs'):
        if group_type == 'pairs':
            return self.pair_grouping()
        
        elif group_type == 'one_vs_all':
            return self.one_vs_all_grouping()
        
        else:
            raise RuntimeError(f'Type: {group_type} not defined')
    
    def run_test(self, group_type='pairs'):
        grouping = self.get_grouping(group_type=group_type)
        result_objs = []
        test_results = []

        for groups in grouping:
            group_analysis = TwoClassCategoricalVsContinous(groups, self.cat_col, self.con_col, self.test)
            result = group_analysis.run_test()
            test_results.append(result)
            result_objs.append(group_analysis)
        
        test_results = pd.DataFrame(test_results)

        return test_results, result_objs

class TwoClassCategoricalVsContinous(CategoricalVsContinous):
    def __init__(self, groups, categorical_col, continous_col, test, plot=False):
        assert len(groups) == 2
        copied_groups = {}

        for group_name, group in groups.items():
            group = copy.copy(group)
            group['group_name'] = group_name
            copied_groups[group_name] = group

        groups = copied_groups

        self.groups = groups
        self.group_names = list(groups.keys())

        data = pd.concat(self.groups.values())

        super().__init__(data, categorical_col, continous_col, test, plot=plot)

        if plot:
            self.plot_data()
            
    def get_group(self, index):
        return self.groups[self.group_names[index]]

    def plot_data(self):
        pt.boxplot(self.data, self.col_types, self.con_col, 'group_name')
        #plt.savefig("%s.pdf" % (self))
        plt.show()


    
    def run_test(self):
        group1 = self.get_group(0)
        group2 = self.get_group(1)

        results = self.test(group1, group2, self.con_col, self.col_types)

        pval_sig = self.statistical_significance_pvalue(results.pvalue)
        statistic_sig, critical_tscore = self.test.is_statistical(results.statistic, group1, group2)
        #critical_tscore = self.min_t_statistic

        sample_sig = True
        if len(group1[self.con_col].dropna()) < self.min_sample_size or len(group2[self.con_col].dropna()) < self.min_sample_size:
            sample_sig = False
        
        if pval_sig and sample_sig:
            overall_sig = 'Yes'
        else:
            overall_sig = 'No'

        
        test_result = { 'test_name':self.test.name, 
                        'categorical_col': self.cat_col, 'continous_col': self.con_col, 'scale':self.scale, 
                        'group1_name':self.group_names[0], 'group2_name':self.group_names[1],
                        'group1_length':len(group1[self.con_col].dropna()), 'group2_length':len(group2[self.con_col].dropna()),
                        'pvalue':results.pvalue, 'pvalue_sig': pval_sig,
                        'statistic':results.statistic, 'statistic_sig': statistic_sig, 'sample_sig': sample_sig,
                        'overall_sig':overall_sig, 'group1':critical_tscore, 'group2':group2}
        
        return test_result
  