# %%
import pathlib
import pandas as pd
import numpy as np
from itertools import combinations
import re

# plotting libraries
from matplotlib import pyplot as plt
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style

# machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score,
)

# stats
import statsmodels.api as sm
from statsmodels import stats as sms
import statsmodels.formula.api as smf
from scipy import stats

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# %%
# Parameters
DATA_DIRS = [
    "../data/3secondsafodoronset_afcondition",
    "../data/3secondsafonset_bfcondition",
]
RNG_SEED = 1
ODOR_MAPPING = {
    2: "Pentanol",
    3: "Hexanol",
    4: "Heptanol",
    5: "Octanol",
}


# %%
# Utility function for model fitting
def logistic_regression_performance(data, rng=np.random.default_rng(0)):
    """
    Fit a vanilla logistic regression and report f1 score per odor and overall accuracy score
    Fitted model is not saved or returned!

    Args:
        data: pd.DataFrame  # dataframe with columns 'odor', 'exp' and a number of 'feature' columns,
        'odor' column is the prediction target,
        columns prefixed with 'feature' will be used as predictors of the regression model,
        'exp' column is kept for record keeping purposes
        rng: numpy.random._generator.Generator  # random generator used in cross validation, fix it to reproduce the same results

    Return:
        f1score_df: pd.DataFrame  # dataframe recording cross validated f1 scores, each row is one cross validation sample
        accuracy_df: pd.DataFrame  # dataframe recording cross validated accuracy scores
    """
    # split data to X and y
    X = data.filter(like="feature").to_numpy()
    y = data.odor.to_numpy()
    # data quality control
    unique_odor, counts = np.unique(y, return_counts=True)
    assert len(np.unique(counts)) == 1, "Imbalanced experimental design!"
    assert data.exp.nunique() == 1, "More than one experiment is found!"
    assert data.isna().sum().sum() == 0, "NaN values found in dataset!"
    # log information about the dataset
    logging.info(
        "There are %d trials where %d odors are presented %d times",
        len(y),
        len(unique_odor),
        np.unique(counts)[0],
    )
    logging.info(
        "%d features were extracted from population neural responses",
        X.shape[1],
    )
    # configure model
    model = make_pipeline(
        StandardScaler(), LogisticRegression()
    )  # classification pipeline combines a standardization step (substract by mean and divide by std) and a vanilla logistic regression model
    # estimate variance of model performance through 5x2 cross validation
    cv = RepeatedStratifiedKFold(
        n_splits=2, n_repeats=5, random_state=np.random.RandomState(rng.bit_generator)
    )
    f1score = []
    accuracy = []
    true_labels = []  # List to store true labels
    pred_labels = []  # List to store predicted labels
    for train_idx, test_idx in cv.split(X, y):
        # train model on the training set
        model = model.fit(X[train_idx], y[train_idx])
        true_labels.extend(y[test_idx])
        # obtain predicted labels
        pred_labels.extend(model.predict(X[test_idx]))
        # evaluate on test set
        f1score.append(f1_score(y[test_idx], model.predict(X[test_idx]), average=None))
        accuracy.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
    f1score = np.stack(f1score)
    accuracy = np.stack(accuracy)
    f1score_df = pd.DataFrame(
        data=f1score,
        index=np.arange(f1score.shape[0]),
        columns=model.classes_,
    )
    f1score_df = pd.melt(f1score_df, var_name="odor", value_name="f1_score")
    f1score_df["odor"] = pd.Categorical(f1score_df["odor"])
    f1score_df["exp"] = data.exp.unique()[0]
    accuracy_df = pd.Series(accuracy).to_frame("accuracy")
    accuracy_df["exp"] = data.exp.unique()[0]
    return f1score_df, accuracy_df, true_labels, pred_labels


# %%
# random seed used for all downstream analysis
rng = np.random.default_rng(RNG_SEED)
# load data
data = []
for d in DATA_DIRS:
    for filename in pathlib.Path(d).glob("*.csv"):
        _data = pd.read_csv(filename, header=None)  # load csv file
        _data.columns = [f"feature_{i}" for i in range(len(_data.columns) - 1)] + [
            "odor"
        ]  # rename columns
        _data["exp"] = filename.stem  # log experiment name as file name
        _data["animal"] = filename.stem.split("_")[1]
        # extract before or after condition with regex (two characters after "odor")
        _data["condition"] = re.findall(r"(?<=odor).{2}", filename.stem)[0]
        data.append(_data)
        logging.info(f"loaded dataset: {filename}")
data = pd.concat(data)

# %%
# Initialize lists to store performance metrics for all odor combinations
all_f1score_df = []
all_accuracy_df = []


for (exp, animal, condition), exp_data in data.groupby(["exp", "animal", "condition"]):
    logging.info(f"Training models for {exp}:")
    exp_data = exp_data.dropna(axis=1)  # drop feature columns containing nans
    exp_data = exp_data.query("odor != 1")  # remove odor 1
    exp_data["odor"] = exp_data["odor"].map(ODOR_MAPPING)  # map odor numbers to names
    exp_data["odor"] = pd.Categorical(
        exp_data["odor"], categories=ODOR_MAPPING.values()
    )
    # Loop over all possible combinations of 2 out of 4 odors
    for odor1, odor2 in combinations(exp_data.odor.unique(), 2):
        # Prepare data for logistic regression
        odor_data = exp_data.query("odor == @odor1 | odor == @odor2")
        # Get model performance using logistic_regression_performance function
        f1score_df, accuracy_df, _, _ = logistic_regression_performance(odor_data, rng)
        # Append the results to the lists
        all_f1score_df.append(
            f1score_df.assign(
                odor1=odor1,
                odor2=odor2,
                comp=f"{odor1}\n{odor2}",
                animal=animal,
                condition=condition,
            )
        )
        all_accuracy_df.append(
            accuracy_df.assign(
                odor1=odor1,
                odor2=odor2,
                comp=f"{odor1}\n{odor2}",
                animal=animal,
                condition=condition,
            )
        )

# Concatenate all results into single DataFrames
all_f1score_df = pd.concat(all_f1score_df)
all_accuracy_df = pd.concat(all_accuracy_df)
all_accuracy_df.to_csv(
    "../notebooks/predict_from_feature_binary_accuracy.csv", index=False
)

# %%
mean_accuracy_df = (
    all_accuracy_df.groupby(["animal", "comp", "condition"])["accuracy"]
    .mean()
    .reset_index()
)


# correct the odor pair order
def from_odor_to_num(odor):
    # reverse mapping of odor names to numbers
    return list(ODOR_MAPPING.keys())[list(ODOR_MAPPING.values()).index(odor)]


def from_odor_pair_to_num(odor_pair):
    # reverse mapping of odor pairs to numbers
    return "".join([str(from_odor_to_num(odor)) for odor in odor_pair.split("\n")])


all_odor_pairs = all_f1score_df["comp"].unique()
mean_accuracy_df["comp"] = pd.Categorical(
    mean_accuracy_df["comp"],
    categories=all_f1score_df["comp"].unique()[
        np.argsort([
            from_odor_pair_to_num(odor_pair)
            for odor_pair in all_f1score_df["comp"].unique()
        ])
    ],
    ordered=True,
)


# %%
# stats
# Perform pairwise t-tests between before and after conditions for each odor pair
ttest_results = []
for comp, group in mean_accuracy_df.groupby(["comp"]):
    bf = group.query("condition == 'bf'")["accuracy"]
    af = group.query("condition == 'af'")["accuracy"]
    ttest_results.append(
        stats.ttest_rel(bf, af)
    )  # paired t-test for before and after conditions
ttest_results = pd.DataFrame(
    ttest_results, columns=["t_statistic", "p_value"], index=all_odor_pairs
)

# %%
# visualization
sns.barplot(
    data=mean_accuracy_df,
    x="comp",
    y="accuracy",
    hue="condition",
    # change color palette to grey and red
    palette={"bf": "#989898", "af": "#FD7F82"},
    hue_order=["bf", "af"],
    # add black frame around bars
    edgecolor="black",
    errorbar="se",
    # add caps to error bars
    capsize=0.3,
)
sns.stripplot(
    data=mean_accuracy_df,
    x="comp",
    y="accuracy",
    hue="condition",
    palette={"bf": "#2E2E2E", "af": "#FC0307"},
    hue_order=["bf", "af"],
    dodge=True,
)


def p2stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


# add asterisks for significant differences
for i, (_, row) in enumerate(ttest_results.iterrows()):
    if row["p_value"] < 0.05:
        plt.text(
            i,
            1,
            p2stars(row["p_value"]),
            ha="center",
            va="center",
            color="black",
            fontsize=12,
            fontweight="bold",
        )

# legend outside of the plot
plt.legend(title="Condition", bbox_to_anchor=(1, 1), loc="upper left")
plt.xlabel("")
sns.despine()
plt.savefig(
    "../notebooks/predict_from_feature_binary_accuracy.svg", dpi=300, bbox_inches="tight"
)

# %%
