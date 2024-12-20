from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure
from tsxp.forecaster import ForecasterMsExog

##
import pandas as pd
import numpy as np

### Feature importance evaluators
import shap
from sklearn.inspection import permutation_importance

from lightgbm import LGBMRegressor
from grouped_permutation_importance import grouped_permutation_importance


class ForecasterMsExogFeatureImportance:

    def __init__(self, forecaster: ForecasterMsExog):
        self.forecaster = forecaster
        self.feature_importance = None

        self.model = forecaster.forecaster.regressor  # trained model
        self.data = forecaster.data  # data object  containing the series and exog data
        (self.X_train, self.y_train, self.X_test, self.y_test) = self.forecaster.create_train_xy()

        # TODO: Implement the feature importance calculation
        self.feature_importance = self.calculate_feature_importance()
        self.feature_rank = self.__calculate_rank()
        self.relative_feature_importance = self.__calculate_percentage()
        self.hierarchies = self.create_hierarchy()

    def calculate_feature_importance(self) -> pd.DataFrame:
        importances_pfi = {
            "PFI_MSE": self.__calculate_permutation_importance(self.X_train, self.y_train),
            "PFI_MSE_TEST": self.__calculate_permutation_importance(self.X_test, self.y_test),
            "PFI_R2": self.__calculate_permutation_importance(self.X_train, self.y_train, ["r2"]),
            "PFI_R2_TEST": self.__calculate_permutation_importance(self.X_test, self.y_test, ["r2"]),
            "TREE_GAIN": self.__calculate_gain_feature_importance(),
            "TREE_SPLIT": self.__calculate_split_feature_importance(),
        }

        if isinstance(self.model, LGBMRegressor):
            shap_fun = self.__calculate_fast_tree_shap_importance
            x = self.X_train
            y = self.y_train
        else:
            shap_fun = self.__calculate_fast_tree_shap_importance
            x = self.X_train.sample(1000)
            y = self.y_train.loc[x.index]

        importances_shap = {
            # "TREE_SHAP_TRAIN": shap_fun(x, y),
            # "FAST_TREE_SHAP": self.__calculate_fast_tree_shap_importance(self.X_test, self.y_test),
            "TREE_SHAP_TEST": shap_fun(self.X_test, self.y_test),
            "TREE_PATH_SHAP": shap_fun(x, y, perturbation="tree_path_dependent"),  # tree_path_dependent_shap_values
            # "KERNEL_SHAP": self.__calculate_shap_kernel_importance(self.X_test, self.y_test),  # TODO: Test if works
        }

        importances = {**importances_pfi, **importances_shap}

        df = pd.concat(importances, axis=1)
        df.columns = importances.keys()
        return df

    def __calculate_permutation_importance(self, data_x, data_y, scoring=["neg_mean_squared_error"]):
        results = permutation_importance(
            self.model,
            data_x,
            data_y,
            n_repeats=100,
            random_state=42,
            n_jobs=-1,
            scoring=scoring,
        )

        importance_metrics = {}
        feature_names = data_x.columns
        for metric in results:
            r = results[metric]
            importances_mean = {}
            for i in np.argsort(r.importances_mean)[::-1]:
                # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                importances_mean[feature_names[i]] = r.importances_mean[i]
                # feature_importances = pd.DataFrame.from_dict(importances_mean, orient="index")
            importance_metrics[metric] = importances_mean
        feature_importances = pd.DataFrame(importance_metrics)
        return feature_importances

    def __calculate_shap_tree_importance(self, data_x, data_y, perturbation="interventional"):
        x = (
            self.X_train if perturbation == "interventional" else None
        )  #  perturbation="tree_path_dependent" or  interventional(if ) - use data only if interventional
        explainer = shap.TreeExplainer(
            self.model,
            data=x,
            # random_state=42
        )
        # shap_values = explainer.shap_values(data_x, data_y)
        shap_values = explainer.shap_values(data_x, data_y)
        feature_importances = pd.DataFrame(shap_values, columns=data_x.columns)
        global_feature_importance = feature_importances.abs().mean().sort_values(ascending=False)
        return pd.DataFrame(global_feature_importance)

    def __calculate_fast_tree_shap_importance(self, data_x, data_y, perturbation="interventional"):
        import fasttreeshap

        print(f"Calculating fast TreeSHAP values with perturbation: {perturbation}")
        x = data_x if perturbation == "interventional" else None
        # print("Sample size: ", x.shape)
        explainer = fasttreeshap.TreeExplainer(
            self.model,
            data=x,
            feature_perturbation=perturbation,
            algorithm="v2",
        )
        shap_values_v1 = explainer.shap_values(data_x, data_y)
        # print(shap_values_v1)

        feature_importances = pd.DataFrame(shap_values_v1, columns=data_x.columns)
        self.train_shap_values = shap_values_v1
        global_feature_importance = feature_importances.abs().mean().sort_values(ascending=False)
        return pd.DataFrame(global_feature_importance)

    def __calculate_shap_kernel_importance(self, data_x, data_y):
        explainer = shap.KernelExplainer(self.model, data_x)
        shap_values = explainer(data_x)
        feature_importances = pd.DataFrame(shap_values, columns=data_x.columns)
        global_feature_importance = feature_importances.abs().mean().sort_values(ascending=False)
        return pd.DataFrame(global_feature_importance)

    def __calculate_rank(self):
        ir_order = self.feature_importance.copy()
        for ir in ir_order.columns:
            ir_order[f"{ir}"] = ir_order[ir].rank(ascending=False).fillna(-1).astype(int) # -1 -means couldn't calculate

        ir_order = pd.concat([ir_order], keys=["Rank"], axis=1)
        return ir_order

    def __calculate_percentage(self):
        ir_perc = self.feature_importance.copy()
        for ir in ir_perc:
            ir_perc[f"{ir}"] = ir_perc[ir] / ir_perc[ir].sum() * 100
        ir_perc = pd.concat([ir_perc], keys=["Perc"], axis=1)
        return ir_perc

    def __calculate_gain_feature_importance(self):  # todo
        # if lightgbm
        gain_imp = None
        if not isinstance(self.model, LGBMRegressor):
            gain_imp = self.model.feature_importances_
        else:
            gain_imp = self.model.booster_.feature_importance(importance_type="gain")
        df_imp = pd.DataFrame({"gain": gain_imp}, index=self.X_train.columns)
        return df_imp

    def __calculate_split_feature_importance(self):  #
        split_imp = None
        if not isinstance(self.model, LGBMRegressor):
            # not implemented in RandomForest
            split_imp = [0] * len(self.X_train.columns)
        else:
            split_imp = self.model.booster_.feature_importance(importance_type="split")
        df_imp = pd.DataFrame({"split": split_imp}, index=self.X_train.columns)
        return df_imp

    ####################

    def calculate_individual_shap_series(self, x, y):
        explainer = shap.TreeExplainer(  # TODO: Check difference based on the feature_perturbation parameter
            self.model, x  # random_state=42
        )  # self.trainx,  #https://github.com/shap/shap/issues/1366#issuecomment-756863719
        shap_values = explainer(x, y, check_additivity=False)
        return shap_values

    def plot_importance_for_series(self):
        shap_values = self.calculate_individual_shap_series(self.X_test, self.y_train)
        cohorts = [
            self.hierarchies["series"][shap_values[i, "_level_skforecast"].data] for i in range(len(shap_values))
        ]
        # print(cohorts)
        self.grouped_shap_plot(shap_values, cohorts)
        self.plot_feature_shap(shap_values, cohorts)

    # def plot_feature_shap(self, shap_values, cohorts):
    #     import matplotlib.pyplot as plt

    #     shap.plots.bar(shap_values.abs.mean(0), max_display=10)
    #     plt.show()

    def create_hierarchy(self):
        def extract_series_id(series_id, index):
            return {
                "index": index,
                "series": series_id,
            }

        mapping = self.forecaster.forecaster.encoding_mapping_
        reversed = [extract_series_id(k, v) for k, v in mapping.items()]
        reversed = pd.DataFrame(reversed)
        reversed.set_index("index", inplace=True)
        return reversed

    def grouped_shap_plot(self, shap_values, group_mapping, max_display=10):
        import matplotlib.pyplot as plt

        shap.plots.bar(
            shap_values.cohorts(group_mapping).abs.mean(0),
            show=False,
            max_display=max_display,
        )
        plt.show()

    def plot_feature_shap(self, shap_values, group_mapping, max_display=10):
        # for groups in range(len(group_mapping.unique())):
        cohorts = set(group_mapping)  # distinct series
        coh = shap_values.cohorts(group_mapping)
        columns = self.X_train.columns
        # fig, ax = plt.subplots(len(columns), figsize=(20, 10))
        plt_df = pd.DataFrame()
        for i, c in enumerate(columns):
            # for c in columns:
            cc = coh[..., c]  # .values
            vals = {}
            vals["feature"] = c
            for coh_id in cohorts:
                vals[coh_id] = [cc.cohorts[coh_id].abs.mean(0).values]
            df = pd.DataFrame.from_dict(vals)
            plt_df = pd.concat([plt_df, df], axis=0)

        plt_df.set_index("feature", inplace=True)

        for k, coh_id in enumerate(cohorts):
            # plt.subplot(1, len(cohorts), k+1 )
            # shap.summary_plot(coh.cohorts[coh_id], max_display=max_display, plot_type="layered_violin", title=coh_id)
            # sf = fig.add_subfigure()
            plt.figure(figsize=(10, 8))
            # shap.summary_plot(
            shap.plots.violin(
                # shap.plots.beeswarm(
                coh.cohorts[coh_id],
                color_bar_label=coh_id,
                plot_type="violin",
                # plot_type="layered_violin",
                # ax= axs[k],
                show=False,
                title=coh_id,
            )
            fig = plt.gcf()
            fig.set_size_inches(5, 4)
            plt.show()

        ################ Layout ovelaps the plots
        # fig = plt.figure(figsize=(10*len(cohorts),10), layout='constrained')
        # num_cohorts = len(cohorts)
        # fig, axs = plt.subplots(1, num_cohorts, figsize=(10 * num_cohorts, 10), constrained_layout=True)
        # fig = plt.figure( figsize=(150, 100), layout='compressed')
        # subfigs = fig.subfigures(1, len(cohorts))
        # for k, coh_id in enumerate(cohorts):
        #     # plt.subplot(1, len(cohorts), k+1 )
        #     # shap.summary_plot(coh.cohorts[coh_id], max_display=max_display, plot_type="layered_violin", title=coh_id)
        #     # sf = fig.add_subfigure()
        #     sf:SubFigure =subfigs[k]
        #     ax = sf.subplots(1, 1)
        #     # shap.summary_plot(
        #     shap.plots.violin(
        #         # shap.plots.beeswarm(
        #         coh.cohorts[coh_id],
        #         # color_bar_label=coh_id,
        #         plot_type="violin",
        #         # plot_type="layered_violin",
        #         # ax= axs[k],
        #         show=False,
        #         title=coh_id
        #     )
        # plt.show()
        # plt_df = plt_df.T  # .reset_index()
        # plt.tight_layout()
        # plt_df.plot(
        #     kind="barh",
        #     subplots=True,
        #     # color= cm.viridis(np.linspace(0, 1, len(plt_df))),
        #     figsize=(20, 30),
        #     layout=(int(np.ceil(len(columns) / 2).astype(int)), 2),
        #     legend=True,
        #     sharey=True,
        #     sharex=False,
        # )

    ################ Grouped permutation importance

    def plot_grouped_importance(self, X, y):

        r = grouped_permutation_importance(
            self.model,
            X=X,
            y=y,
            groups=self.data.series_dict_train,
            n_repeats=100,
            random_state=42,
            n_jobs=-1,
        )
        sorted_idx = r.importances_mean.argsort()[::-1]
        box = ax[0].boxplot(
            r.importances[sorted_idx].T,
            patch_artist=True,
            vert=True,
            showfliers=False,
            notch=True,
            labels=np.array(columns)[sorted_idx],
        )
        for patch in box["boxes"]:
            patch.set_facecolor("blue")
            patch.set_alpha(0.5)

    ### SHAP on FUll dataset vs on the series

    def series_shap(self, series_id):
        mapping = self.forecaster.forecaster.encoding_mapping_
        id = mapping[series_id]
        print(f"Series id: {series_id} - {id}")
        x_series = self.X_train[self.X_train["_level_skforecast"] == id]
        y_series = self.y_train[self.X_train["_level_skforecast"] == id]
        shap_values = self.calculate_individual_shap_series(x_series, y_series)
        shap.plots.bar(
            shap_values.abs.mean(0),
            show=False,
            # max_display=max_display,
        )
        plt.show()
        shap.summary_plot(shap_values, x_series, plot_type="violin")

    ### Scale the shap values
    def scaled_shap(self):
        scalers = self.forecaster.forecaster.transformer_series_
        scalers_exog = self.forecaster.forecaster.transformer_exog
        shap_values = self.calculate_individual_shap_series(self.X_test, self.y_test)

        return shap_values
