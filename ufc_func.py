import pandas as pd
import numpy as np
from random import random
from math import ceil

def methods_destroyer(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[~df["Method"].str.contains("Overturned|DQ|Other")]
    return filtered

def time_converter(time: str) -> int:
    """
    The time is given in the following pattern "minutes:seconds"
    and converted to time in seconds
    """
    first, second = time.split(":")

    return int(first) * 60 + int(second)

def round_converter(rnd: str) -> int:
    """
    Convert the round amounts into time in seconds
    """
    return (rnd-1) * 300

def replace_method(method: str) -> str:
    """
    Makes type of fight finishes more generic
    """
    if "DEC" in method:
        return "DEC"
    elif "SUB" in method:
        return "SUB"
    elif "KO" in method or "CNC" in method:
        return "KO/TKO"

def weight_destroyer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all outdated values
    """
    filtered = df[~df["Weight_Class"].str.contains("Open|Catch|Super")]
    return filtered

def de_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes unneccassary columns
    """
    if df.columns.str.contains("Date|Event Name|Round|Time|Location").any():
        return df.drop(["Date", "Event Name", "Round", "Time", "Location"], axis=1)
    
def weight_breaker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Breakes Weight_Class category to 
    on columns that display separate weight categories
    """
    for ele in df["Weight_Class"].unique():
        df[ele] = np.where(df["Weight_Class"].str.contains(ele), 1, 0)

    return df if not df.columns.str.contains("Weight_Class").any() else df.drop(['Weight_Class'], axis=1)

def second_to_round(time: int) -> int:
    """
    Convert time to round
    """
    return ceil(time/300)

def history_winrate(df: pd.DataFrame) -> None:
    """
    Destructive method

    Calculates the win rate of each fighter before their fight 
    """
    nrows = df.size//df.columns.size
    winrate = []

    for i in range(nrows):
        current = df.iloc[i,]

        for j in range(1,3):
            current_fighter = current[f"Fighter {j}"]

            stats = pd.concat([df.iloc[i+1:,].loc[df[f"Fighter {k}"] == current_fighter] for k in range(1,3)])

            if stats.empty:
                current_winrate = 0

            else:
                wins = 0 if stats.loc[stats["Winner"] == current_fighter].empty else (stats["Winner"].value_counts())[current_fighter]
                all_fights = len(stats.index)

                current_winrate = 1 if not (wins - all_fights) else round(wins/all_fights, 2)
            
            winrate += [current_winrate]

    winrate = np.array(winrate)
    winrate2d = winrate.reshape(nrows, 2)
    
    winrate_df = pd.DataFrame({"Current winrate F1": winrate2d[:, 0], "Current winrate F2": winrate2d[:, 1]})

    res = pd.concat([df, winrate_df], axis=1, ignore_index=True)
    res.columns = df.columns.to_list() + winrate_df.columns.to_list()
    
    return res

def swapper(df: pd.DataFrame, to_swap: list[tuple]) -> None:
    """
    Destructive method

    Swaps fighter statistics in randomly chosen fights
    """
    nrows = df.shape[0]

    def swap(df: pd.DataFrame, index: int, category_1: str, category_2: str):
        temp = df.loc[index, category_1]
        df.loc[index, category_1] = df.loc[index, category_2]
        df.loc[index, category_2] = temp

    for i in range(nrows):
        decision = random()

        if decision > 0.5:
            for ele in to_swap:
                swap(df, i, ele[0], ele[1])

            df.loc[i, "Winner"] = 0
        else:
            df.loc[i, "Winner"] = 1

def replace_stats_with_avg(df: pd.DataFrame) -> None:
    """
    Destructive method

    Caclulate the fighters average statistics before their fight
    """
    for i in range(df.index.stop):
        for j in range(1, 3):
            current = df.iloc[i]

            fighter = current[f"Fighter_{j}"]

            fighter_1_df = df.iloc[i+1:].loc[(df["Fighter_1"] == fighter)].reset_index(drop=True)
            fighter_2_df = df.iloc[i+1:].loc[(df["Fighter_2"] == fighter)].reset_index(drop=True)

            if fighter_1_df.empty and fighter_2_df.empty:
                avg_kd = 0
                avg_strk = 0
                avg_tkd = 0
                avg_subs = 0
            elif fighter_1_df.empty and (not fighter_2_df.empty): 
                avg_kd = fighter_2_df["Fighter_2_KD"].mean()
                avg_strk = fighter_2_df["Fighter_2_STR"].mean()
                avg_tkd = fighter_2_df["Fighter_2_TD"].mean()
                avg_subs = fighter_2_df["Fighter_2_SUB"].mean()
            elif fighter_2_df.empty and (not fighter_1_df.empty): 
                avg_kd = fighter_1_df["Fighter_1_KD"].mean()
                avg_strk = fighter_1_df["Fighter_1_STR"].mean()
                avg_tkd = fighter_1_df["Fighter_1_TD"].mean()
                avg_subs = fighter_1_df["Fighter_1_SUB"].mean()
            else:
                avg_kd = (fighter_1_df["Fighter_1_KD"].sum() + fighter_2_df["Fighter_2_KD"].sum()) / (fighter_1_df.shape[0] + fighter_2_df.shape[0])
                avg_strk = (fighter_1_df["Fighter_1_STR"].sum() + fighter_2_df["Fighter_2_STR"].sum()) / (fighter_1_df.shape[0] + fighter_2_df.shape[0])
                avg_tkd = (fighter_1_df["Fighter_1_TD"].sum() + fighter_2_df["Fighter_2_TD"].sum()) / (fighter_1_df.shape[0] + fighter_2_df.shape[0])
                avg_subs = (fighter_1_df["Fighter_1_SUB"].sum() + fighter_2_df["Fighter_2_SUB"].sum()) / (fighter_1_df.shape[0] + fighter_2_df.shape[0])

            df.loc[i, f"Fighter_{j}_KD"] = round(avg_kd, 2)
            df.loc[i, f"Fighter_{j}_STR"] = round(avg_strk, 2)
            df.loc[i, f"Fighter_{j}_TD"] = round(avg_tkd, 2)
            df.loc[i, f"Fighter_{j}_SUB"] = round(avg_subs, 2)

def fighter_wrapper(func):
    """
    Wrapper for getting info
    """

    def wrapper(df: pd.DataFrame, name: str) -> pd.Series | pd.DataFrame:
        """
        Get fighter info from the dataframe
        """
        section = func()
        return df.loc[(df[section] == name).any(axis=1)]
    
    return wrapper

@fighter_wrapper
def get_stats_of_fighter() -> list[str]:
    """
    Returns the statistic of fighter
    """
    return ["Fighter_name"]

@fighter_wrapper
def get_fights_of_fighter() -> list[str]:
    """
    Returns the fight history of fighter
    """
    return ["Fighter_1","Fighter_2"]

def calculate_outliers(series: pd.Series) -> tuple[float]:
    """
    Calculate outliers lower bounds (for lower and higher)
    """
    q1,q3 = series.quantile([0.25, 0.75])

    iqr = q3 - q1

    return q3 + 1.5 * iqr, q1 - 1.5 * iqr

def df_breaker_by_rounds(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Breakes the dataframe in two parts:
    Fights with 3 round and fights with 5 rounds
    """
    three_rounds = df.loc[df["Round"] <= 3]
    five_rounds = df.loc[df["Round"] > 4]

    return three_rounds, five_rounds