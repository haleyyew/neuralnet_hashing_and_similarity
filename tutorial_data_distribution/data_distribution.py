import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import re
from numpy import array

def content_ratio(attr):
    return len(set(attr)) / len(attr)

# for one attr
def content_histogram(attr_df, attr_name):
    return attr_df.groupby(attr_name).agg({attr_name:'count'})

def visualize_histograms_pair(attr1_name, attr1_histog_re, attr2_name, attr2_histog_re, resample_dimension):
    pos = list(range(0, resample_dimension))
    plt.plot(pos, attr1_histog_re, 'go-', pos, attr2_histog_re, '.-')
    plt.legend([attr1_name, attr2_name], loc='best')
    plt.show()

def feature_extractor(pair_of_attrs, pair_of_attr_names):
    attr1 = pair_of_attrs[0]
    # attr2 = pair_of_attrs[1]
    attr1_name = pair_of_attr_names[0]
    # attr2_name = pair_of_attr_names[1]
    attr1_df = pd.DataFrame({attr1_name: attr1})
    # attr2_df = pd.DataFrame({attr2_name: attr2})

    content_ratio1 = content_ratio(attr1)
    # content_ratio2 = content_ratio(attr2)

    # print(content_ratio1)
    # print(content_ratio2)

    # all attrs in table1
    tbl1_attrs = attr1_df.keys()
    # all attrs in table2
    # tbl2_attrs = attr2_df.keys()

    distributions1 = {}
    # distributions2 = {}
    for attr in tbl1_attrs:
        content_histogram_tbl1 = content_histogram(attr1_df, attr)
        distributions1[attr] = content_histogram_tbl1

    # for attr in tbl2_attrs:
    #     content_histogram_tbl2 = content_histogram(attr2_df, attr)
    #     distributions2[attr] = content_histogram_tbl2

    # table1
    attr1_histog = []
    for distr in distributions1:
        # print(distr)
        if distr == attr1_name:
            # distributions1[distr].sort_values(distr)
            for index, row in distributions1[distr].iterrows():
                # print(row)
                # for key in row.keys():
                attr1_histog.append(row[distr])

    # # table2
    # attr2_histog = []
    # for distr in distributions2:
    #     # print(distr)
    #     if distr == attr2_name:
    #         # distributions2[distr].sort_values(distr)
    #         for index, row in distributions2[distr].iterrows():
    #             attr2_histog.append(row[distr])

    attr1_histog.sort()
    # attr2_histog.sort()

    # print(attr1_histog)
    # print(attr2_histog)

    # resample using fft
    resample_dimension = 20
    attr1_histog_re = signal.resample(attr1_histog, resample_dimension)
    # attr2_histog_re = signal.resample(attr2_histog, resample_dimension)

    # print(attr1_histog_re)
    # # print(attr2_histog_re)

    # visualize_histograms_pair(attr1_name, attr1_histog_re, attr2_name, attr2_histog_re, resample_dimension)

    # print(_get_col_dtype(attr1_df[attr1_name]))
    # print(_get_col_dtype(attr2_df[attr2_name]))

    average_cell_len1 = average_cell_len(attr1)
    # average_cell_len2 = average_cell_len(attr2)

    percentage_of_num1, percentage_of_alphabetic1 = percentage_of_num_alphabetic(attr1)
    # percentage_of_num2, percentage_of_alphabetic2 = percentage_of_num_alphabetic(attr2)

    attr1_features = [content_ratio1]
    attr1_features.extend(attr1_histog_re)
    attr1_features.append(average_cell_len1)
    attr1_features.append(percentage_of_num1)
    attr1_features.append(percentage_of_alphabetic1)

    # attr2_features = [content_ratio2]
    # attr2_features.extend(attr2_histog_re)
    # attr2_features.append(average_cell_len2)
    # attr2_features.append(percentage_of_num2)
    # attr2_features.append(percentage_of_alphabetic2)

    attr1_features_dict = ['attr_histogram']*resample_dimension
    attr1_features_dict = ['content_ratio', *attr1_features_dict, 'average_cell_len', 'percentage_of_num', 'percentage_of_alphabetic']

    return attr1_features, attr1_features_dict


def average_cell_len(cols):
    # cell is one string
    return sum([len(cell) for cell in cols])/len(cols)

def percentage_of_num_alphabetic(cols):
    total_nums = 0
    total_alphabets = 0
    total_chars = 0
    for cell in cols:
        total_nums += sum([len(s) for s in re.findall(r'-?\d+\.?\d*', cell)])
        total_alphabets += sum (len(s) for s in re.findall("[a-zA-Z]+", cell))
        total_chars += len(cell)
    return total_nums/total_chars, total_alphabets/total_chars

# # tokenize cell
# from nltk import word_tokenize

# pandas infer data type https://stackoverflow.com/questions/35003138/python-pandas-inferring-column-datatypes
def _get_col_dtype(col):
    """
    Infer datatype of a pandas column, process only if the column dtype is object.
    input:   col: a pandas Series representing a df column.
    """

    if col.dtype == "object":

        # try numeric
        try:
            col_new = pd.to_datetime(col.dropna().unique())
            return col_new.dtype
        except:
            try:
                col_new = pd.to_numeric(col.dropna().unique())
                return col_new.dtype
            except:
                try:
                    col_new = pd.to_timedelta(col.dropna().unique())
                    return col_new.dtype
                except:
                    return "object"

    else:
        return col.dtype



# attr from table1
attr1 = ['a', 'a', 'b', 'c', 'd', '-1', 'a1a']
# attr from table2
attr2 = ['a', 'a', 'a', 'a', 'a', 'e', '2.0', 'a1a']
# attr from table3
attr3 = ['abcd', 'bd bd bd', 'op op as as', 'qwert']
# attr from table4
attr4 = ['a', 'a', 'b', 'c', 'd']
# attr from table5
attr5 = ['a2a', 'a', '4.0', 'a', 'd']

attr1_name = 'attr1'
attr2_name = 'attr2'
attr3_name = 'attr3'
attr4_name = 'attr4'
attr5_name = 'attr5'

attr1_features, attr1_features_dict = feature_extractor([attr1], [attr1_name])
attr2_features, _ = feature_extractor([attr2], [attr2_name])
attr3_features, _ = feature_extractor([attr3], [attr3_name])
attr4_features, _ = feature_extractor([attr4], [attr4_name])

x12 = array(attr2_features) - array(attr1_features)
x13 = array(attr3_features) - array(attr1_features)
x14 = array(attr4_features) - array(attr1_features)
x23 = array(attr3_features) - array(attr2_features)
x24 = array(attr4_features) - array(attr2_features)
x34 = array(attr4_features) - array(attr3_features)

# random forest classifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
X = [x12, x13, x14, x23, x24, x34]
y = [True, False, True, False, True, False]
clf = RandomForestClassifier(n_estimators=15, max_depth=2,
                             random_state=0)
clf.fit(X, y)

print(clf.feature_importances_)

attr5_features, _ = feature_extractor([attr5], [attr5_name])
x25 = array(attr5_features) - array(attr2_features)

print(clf.predict([x25]))