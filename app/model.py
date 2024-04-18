from ufc_func import *
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def trained_model():
    df = pd.read_csv("./data/ufc.csv")
    df = pd.DataFrame(df)

    df = df.dropna()
    df_with_time = df.copy()

    df_with_time["Seconds"] = df_with_time["Time"].apply(time_converter) + df_with_time["Round"].apply(round_converter)

    df_with_time = df_with_time.loc[
        (df_with_time["Fighter 1"] == df_with_time["Winner"]) | 
        (df_with_time["Fighter 1"] == df_with_time["Winner"])].reset_index(drop=True)

    df_fil = methods_destroyer(df_with_time)
    df_methods = df_fil.copy()
    df_methods["Method"] = df_fil["Method"].apply(replace_method)
    df_fil = weight_destroyer(df_fil)

    three, five = df_breaker_by_rounds(df_fil)

    df_fil = three.copy()

    df_fil = de_remover(df_fil)

    df_fil["Gender"] = np.where(df_fil["Weight_Class"].str.contains("Women"), 1, 0)
    df_man = df_fil.loc[df_fil["Gender"] == 0].drop(["Gender"], axis=1).reset_index(drop=True)
    df_woman = df_fil.loc[df_fil["Gender"] != 0].drop(["Gender"], axis=1).reset_index(drop=True)

    df_man_clear = weight_breaker(df_man)
    winrates_df = history_winrate(df_man_clear)

    names = [ele.replace(" ", "_") for ele in winrates_df.columns.to_list()]
    winrates_df.columns = names

    categories_all = [ele for ele in winrates_df.columns if "Fighter" in ele or "winrate" in ele]
    categories = [(categories_all[i], categories_all[i+1]) for i in range(0, len(categories_all), 2)]

    swapper(winrates_df, categories)

    winrates_df['Winrate_1_is_0'] = np.where(winrates_df["Current_winrate_F1"] == 0, 1, 0)
    winrates_df['Winrate_2_is_0'] = np.where(winrates_df["Current_winrate_F2"] == 0, 1, 0)

    winrates_df["Winner"] = pd.to_numeric(winrates_df["Winner"])

    winrates_df_model_logit = winrates_df.drop(['Fighter_1', 'Fighter_2', 'Winner', 'Method', 'Seconds', 'Lightweight'], axis=1).copy()

    model_logit=smf.logit(f"Winner ~ {' + '.join(winrates_df_model_logit)}", data=winrates_df)

    return model_logit.fit()

def fight_prediction(figther1, figther2, fighters_df, winrates_df, model):
  first = get_stats_of_fighter(fighters_df, figther1)
  second = get_stats_of_fighter(fighters_df, figther2)
  weight = get_fights_of_fighter(winrates_df, figther1)

  predict_df = pd.DataFrame({
      'Fighter_1_KD': first["Average_knockdowns"].iloc[0],
      'Fighter_2_KD': second["Average_knockdowns"].iloc[0],
      'Fighter_1_STR': first["Average_significant_strikes"].iloc[0],
      'Fighter_2_STR': second["Average_significant_strikes"].iloc[0],
      'Fighter_1_TD': first["Average_takedowns"].iloc[0],
      'Fighter_2_TD': second["Average_takedowns"],
      'Fighter_1_SUB': first["Average_submission_attempts"].iloc[0],
      'Fighter_2_SUB': second["Average_submission_attempts"].iloc[0],
      'Bantamweight': weight['Bantamweight'].iloc[0],
      'Welterweight': weight['Welterweight'].iloc[0],
      'Middleweight': weight['Middleweight'].iloc[0],
      'Light_Heavyweight': weight['Light_Heavyweight'].iloc[0],
      'Featherweight': weight['Featherweight'].iloc[0],
      'Heavyweight': weight['Heavyweight'].iloc[0],
      'Flyweight':weight['Flyweight'].iloc[0],
      'Current_winrate_F1': first["Winrate"].iloc[0],
      'Current_winrate_F2': second["Winrate"].iloc[0],
      'Winrate_1_is_0': int(first["Winrate"].iloc[0] == 0),
      'Winrate_2_is_0': int(second["Winrate"].iloc[0] == 0)
  })
  predictions = model.predict(predict_df)

  return predictions.iloc[0]

def get_winrates_df():
    df = pd.read_csv("./data/ufc.csv")
    df = pd.DataFrame(df)

    df = df.dropna()
    df_with_time = df.copy()

    df_with_time["Seconds"] = df_with_time["Time"].apply(time_converter) + df_with_time["Round"].apply(round_converter)

    df_with_time = df_with_time.loc[
        (df_with_time["Fighter 1"] == df_with_time["Winner"]) | 
        (df_with_time["Fighter 1"] == df_with_time["Winner"])].reset_index(drop=True)

    df_fil = methods_destroyer(df_with_time)
    df_methods = df_fil.copy()
    df_methods["Method"] = df_fil["Method"].apply(replace_method)
    df_fil = weight_destroyer(df_fil)

    three, five = df_breaker_by_rounds(df_fil)

    df_fil = three.copy()

    df_fil = de_remover(df_fil)

    df_fil["Gender"] = np.where(df_fil["Weight_Class"].str.contains("Women"), 1, 0)
    df_man = df_fil.loc[df_fil["Gender"] == 0].drop(["Gender"], axis=1).reset_index(drop=True)
    df_woman = df_fil.loc[df_fil["Gender"] != 0].drop(["Gender"], axis=1).reset_index(drop=True)

    df_man_clear = weight_breaker(df_man)
    winrates_df = history_winrate(df_man_clear)

    names = [ele.replace(" ", "_") for ele in winrates_df.columns.to_list()]
    winrates_df.columns = names

    categories_all = [ele for ele in winrates_df.columns if "Fighter" in ele or "winrate" in ele]
    categories = [(categories_all[i], categories_all[i+1]) for i in range(0, len(categories_all), 2)]

    swapper(winrates_df, categories)

    winrates_df['Winrate_1_is_0'] = np.where(winrates_df["Current_winrate_F1"] == 0, 1, 0)
    winrates_df['Winrate_2_is_0'] = np.where(winrates_df["Current_winrate_F2"] == 0, 1, 0)

    winrates_df["Winner"] = pd.to_numeric(winrates_df["Winner"])

    return winrates_df