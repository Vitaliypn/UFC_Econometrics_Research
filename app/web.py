from flask import Flask, render_template, request
from model import *
import statsmodels.formula.api as smf
import pandas as pd

app = Flask(__name__)

read_df = pd.read_csv("./data/fighter.csv")
fighter_df = pd.DataFrame(read_df)
winrates_df = get_winrates_df()
model = trained_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    weights = ['Lightweight', 'Bantamweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', "Women's Flyweight",
                'Featherweight', 'Heavyweight', 'Flyweight',
                "Women's Featherweight"]
    return render_template('index.html', weights = weights)

@app.route('/fight', methods=["GET", "POST"])
def start_page():
    selected_weight = request.args.get('weight_chooser', '')
    fighter_weight = fighter_df[fighter_df["Weight_class"].apply(lambda x: selected_weight in x)].reset_index(drop=True)
    names = fighter_weight['Fighter_name'].to_list()
    length = len(names)

    return render_template('fight.html', names=names, length=length, selected_weight=selected_weight) 

@app.route('/championship', methods=["GET", "POST"])
def fight():
    fighter1 = request.args.get('fighter1', '')
    fighter2 = request.args.get('fighter2', '')
    prediction = fight_prediction(fighter1, fighter2, fighter_df, winrates_df, model)

    if prediction < 0.5:
        winner = fighter2
        prediction = 1 - prediction
    else: 
        winner = fighter1

    return render_template('championship.html', prediction=round(prediction*100,2), winner = winner)

if __name__ == "__main__":
    app.run(debug=True, port="5050")