from pathlib import Path
from typing import Counter, Dict, Callable

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ltr_utility import ModelParam
from ltr_utility.dataset import LtrDataset
from ltr_utility.model_selection import extract_query_configs
from ruletreerank import QueryRanker


def show_distr_conf(df:pd.DataFrame):
    conf_dist = {}

    df = df[df["results"].apply(func=lambda x: not all(item == x[0] for item in x))]

    for i in df.qxm.unique():
        idx_best = pd.DataFrame(
            Counter(df[df.qxm == i].idx_best).items(), columns=["Configs", "Count"]
        )
        idx_best["Configs"] = idx_best["Configs"].apply(lambda x: "#" + str(x))
        idx_best["Dist"] = idx_best["Count"] / idx_best["Count"].sum()
        idx_best = idx_best.sort_values("Count", ascending=False)
        conf_dist[i] = idx_best

    f, axs = plt.subplots(nrows=2, ncols=4, figsize=(30, 10))
    axs = axs.flatten()

    for (k,v), ax in zip(conf_dist.items(),axs):
        v1 = v.head(12)
        sns.barplot(data=v1, x="Configs", y="Dist", ax=ax, hue="Configs", legend=False, order=v1["Configs"],
                    palette="crest", edgecolor="black", linewidth=0.6)

        for i, row in v1.iterrows():
            ax.text(x=v1["Configs"].tolist().index(row["Configs"]), y=row["Dist"], s=str(row["Count"]),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6, color="#a0a0a0")
        ax.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.2)
        ax.set_title(f"{k}", fontsize=18)

    plt.tight_layout()
    plt.show()
    return pd.concat([v.assign(qxm=k) for k,v in conf_dist.items()])

def retrain_evaluate(train_valid:LtrDataset, test:LtrDataset,
                     configs:Dict, model: Callable, name:str):
    predictions = {}
    for i in configs.keys():
        q_ranker = QueryRanker(ranker=ModelParam(model=model, param=configs[i][1]),
                               q_per_model=i, batch_query=configs[i][0]).fit(train=train_valid)
        predictions[i] = (
            q_ranker.predict(X=train_valid.x, q=train_valid.q),
            q_ranker.predict(X=test.x, q=test.q)
        )
    return (pd
            .DataFrame(predictions, index=["pred_train", "pred_test"])
            .T
            .reset_index()
            .rename(columns={"index": "qxm"})
            .assign(model=name)
            )

def RTR_retrain_evaluate(train_valid:LtrDataset, test:LtrDataset,  configs:Dict, model: Callable, name:str):

    predictions_full, predictions_score, predictions_eucl = {}, {}, {}

    for i in configs.keys():
        print(i)
        if configs[i][1] and isinstance(configs[i][1], list): tmp_config = [*map(lambda x: {**x, "n_jobs_leaf": 10}, configs[i][1])]
        else: tmp_config = {**configs[i][1], "n_jobs_leaf": 10}

        q_ranker = QueryRanker(ranker=ModelParam(model=model, param=tmp_config),
                               q_per_model=i, batch_query=configs[i][0]).fit(train=train_valid)
        print("Eval")
        predictions_score[i] = (
            q_ranker.predict(X=train_valid.x, q=train_valid.q, output="score"),
            q_ranker.predict(X=test.x, q=test.q, output="score")
        )
        predictions_eucl[i] = (
            q_ranker.predict(X=train_valid.x, q=train_valid.q, output="euclidian"),
            q_ranker.predict(X=test.x, q=test.q, output="euclidian")

        )
        predictions_full[i] = (
            q_ranker.predict(X=train_valid.x, q=train_valid.q),
            q_ranker.predict(X=test.x, q=test.q)
        )
        del q_ranker

    full = (pd.DataFrame(predictions_full, index=["pred_train", "pred_test"]).T.reset_index()
            .rename(columns={"index": "qxm"}).assign(model=name))

    score = (pd.DataFrame(predictions_score, index=["pred_train", "pred_test"]).T.reset_index()
             .rename(columns={"index": "qxm"}).assign(model=name + "_score"))

    eucl = (pd.DataFrame(predictions_eucl, index=["pred_train", "pred_test"]).T.reset_index()
            .rename(columns={"index": "qxm"}).assign(model=name + "_eucl"))

    return pd.concat([full, score, eucl], axis=0)


def custom_train(base_result:Path, train_valid:LtrDataset, test:LtrDataset, indices:Dict[int,int],
                 path:str, ms:pd.DataFrame, name:str, class_:Callable, rtr:bool=False):

    configs = extract_query_configs(base_result / path)
    configs = {k: (configs[k][0], ms[ms.qxm == k].iloc[0]["configs"][v]) for k,v in indices.items()}
    if rtr:
        return RTR_retrain_evaluate(train_valid=train_valid, test=test, configs=configs,
                                    model=class_, name=name)
    else:
        return retrain_evaluate(train_valid=train_valid, test=test, configs=configs,
                                model=class_, name=name)