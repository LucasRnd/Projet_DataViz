# Import des bibliothèques
import pandas as pd
import dash
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import joblib


########## Charger le modèle et le scaler ########
# Import du modèle de régression logistique
best_lr_model = joblib.load('best_logistic_model.pkl')

# Import du modèle Random Forest
best_rf_model = joblib.load('best_rf_model.pkl')

    

##########  Pré-traitement  ##########
#Import des données
df = pd.read_csv("C:/Users/Lucas/Downloads/archive/dataset_olympics.csv", sep=",", encoding="utf-8")

#Filtrer les Jeux olympiques d'été depuis 1994
df = df[(df['Year'] >= 1994) & (df['Year'] % 4 == 0)]

### Pour les moyennes de tailles, poid et age
# Convertir les colonnes 'Height', 'Weight' et 'Age' en numérique (au cas où elles seraient initialement des chaînes)
df[['Height', 'Weight', 'Age']] = df[['Height', 'Weight', 'Age']].apply(pd.to_numeric, errors='coerce')

# Supprimer les lignes avec des valeurs manquantes dans les colonnes 'Height', 'Weight' et 'Age'
df = df.dropna(subset=['Height', 'Weight', 'Age', 'Sex'])


##########  Fonctions utiles  ##########
#Générer le graphique du nombre de médailles par pays
def generate_medailles_par_pays(selected_year):
    # Filtrer les données en fonction de l'année sélectionnée
    filtered_df = df[df['Year'] == selected_year]
    medailles_par_pays = filtered_df['NOC'].value_counts().sort_values(ascending=False)[:10]
    fig = px.bar(medailles_par_pays, y=medailles_par_pays.index, x=medailles_par_pays.values, orientation='h',
                 labels={'y': 'Pays', 'x': 'Nombre de médailles'},
                 title=f'Top 10 des pays avec le plus grand nombre de médailles en {selected_year}')
    return(fig)
   
#Générer une BDD utile pour la carte & faire le graph 
def generate_world_map(selected_year):
    #Filtrer les données pour l'année sélectionnée
    filtered_df = df[df['Year'] == selected_year]

    #Exclure les lignes avec des valeurs 'Medal' manquantes (nan)
    filtered_df = filtered_df.dropna(subset=['Medal'])

    #Créer une matrice pour l'affichage
    df_carte = filtered_df.groupby(['NOC', 'Team'])['Medal'].count().reset_index()
    
    fig = px.scatter_geo(df_carte,
                         locations="NOC",  # Colonne contenant les codes des pays
                         size="Medal",  # Colonne contenant le nombre de médailles
                         projection="natural earth",  # Projection de la carte (d'autres options disponibles)
                         title=f"Carte du nombre de médailles par pays en {selected_year}")
        
    fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="white")

    return(fig)

# Fonction pour générer le graphique d'analyse des caractéristiques moyennes par sport et par année
def generate_average_chart(selected_feature, selected_gender):
    # Filtrer les données en fonction du genre
    filtered_df = df[df['Sex'] == selected_gender]

    # Grouper les données par sport, année et calculer la moyenne de la caractéristique sélectionnée
    grouped_df = filtered_df.groupby(['Sport', 'Year'])[selected_feature].mean().reset_index()

    # Créer le graphique avec des points
    fig = px.scatter(grouped_df, x='Year', y=selected_feature, color='Sport', title=f"{selected_feature} moyen par sport et par année ({selected_gender})")

    return fig



######## Charger le modele de regression ######


##########  Dash  ##########
#Créer une application Dash
app = dash.Dash(__name__)

#Créer les sections de la première page
page1_layout = html.Div(style={'backgroundColor': '#f2f2f2'}, children=[
    
    ### Mettre un sommaire ###
    html.H1("Tableau de bord des Jeux olympiques"),
    # Sommaire
    html.Ul([
        html.Li(html.A('Accueil', href='#accueil')),
        html.Li(html.A('Médailles par année', href='#medailles-par-annee-section')),
        html.Li(html.A('Médailles par pays', href='#pays-dropdown-page1')),
        html.Li(html.A('Médailles par pays', href='#pays-dropdown-page1')),
        # Ajoutez d'autres sections à la table des matières ici
    ]),
    # Section Accueil
    html.H2("Accueil", id='accueil'),
    html.P("Bienvenue dans le tableau de bord des Jeux olympiques. Explorez les données et découvrez des informations intéressantes."),
    
    
    ### Titre partie 1 ###
    html.H1("Partie I - Premier regard"),
        
        
    ### Medaille par annee ###
    dcc.Dropdown(
        id='annee-dropdown-page1',
        options=[
            {'label': str(year), 'value': year} for year in sorted(df['Year'].unique())
        ],
        value=df['Year'].max()
    ),
    # Visualisation du nombre de médailles par année
    html.H2("Médailles par annee", id='medailles-par-annee-section'),
    dcc.Graph(id='medailles-par-annee-page1'),
    
    
    ### Double graph medaille par an + Carte des pays ###
    html.H1("Nombre de médaille par pays aux Jeux olympiques"),
    dcc.Dropdown(
        id='pays-dropdown-page1',
        options=[
            {'label': str(year), 'value': year} for year in sorted(df['Year'].unique())
        ],
        value=df['Year'].max()  # Par défaut, sélectionnez la première année
    ),
    # Div pour afficher les graphiques sur la même ligne
    html.Div([
        # Graphique pour le nombre de médailles par pays
        html.Div([dcc.Graph(id='medailles-par-pays-page1')], style={'flex': '1'}),

        # Carte mondiale interactive
        html.Div([dcc.Graph(id='world-map', style={'width': '100%'})], style={'flex': '1'}),
    ], style={'display': 'flex'}),
          
    
    ### Double graph participants + medailles ###
    html.H1("Nombre de participants et de médailles par année aux Jeux olympiques"),
    
    # Composant interactif pour sélectionner la plage d'années
    dcc.RangeSlider(
        id='year-slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        marks={year: str(year) for year in range(df['Year'].min(), df['Year'].max() + 1, 4)},
        step=1,
        value=[1996, 2016],  # Plage d'années par défaut
    ),
    # Div pour afficher les graphiques sur la même ligne
    html.Div([
        # Graphique pour le nombre de participants
        html.Div([dcc.Graph(id='participants-par-annee')], style={'flex': '1'}),
    
        # Graphique pour le nombre de médailles
        html.Div([dcc.Graph(id='medailles-par-annee2')], style={'flex': '1'}),
    ], style={'display': 'flex'}),
    
])

######### Créer les sections de la deuxième page ###########
page2_layout = html.Div([  
    # Titre
    html.H1("Partie 2 - XXXX"),
    # Section Description
    html.H2("Description", id='accueil'),
    html.P("Bienvenue dans le tableau de bord des Jeux olympiques. Explorez les données et découvrez des informations intéressantes."),
    
        
    #### Moyennes taille, poids et age par genre
    html.H1("Analyse des caractéristiques moyennes par sport et par année"),

    # Bouton de mise à jour
    html.Button('Mettre à jour les graphiques', id='update-button'),

    # Graphiques générés par callback
    html.Div([
        # Graphiques pour les hommes
        html.Div([dcc.Graph(id='average-height-chart-M', style={'width': '33%', 'display': 'inline-block'}),
                  dcc.Graph(id='average-weight-chart-M', style={'width': '33%', 'display': 'inline-block'}),
                  dcc.Graph(id='average-age-chart-M', style={'width': '33%', 'display': 'inline-block'}),
        ], style={'display': 'flex'}),

        # Graphiques pour les femmes
        html.Div([dcc.Graph(id='average-height-chart-F', style={'width': '33%', 'display': 'inline-block'}),
                  dcc.Graph(id='average-weight-chart-F', style={'width': '33%', 'display': 'inline-block'}),
                  dcc.Graph(id='average-age-chart-F', style={'width': '33%', 'display': 'inline-block'}),
        ], style={'display': 'flex'}),
    ]),
    
    
])




######### Créer les sections de la troisième page ###########
page3_layout = html.Div([
    # Titre
    html.H1("Partie 3 - XXXX"),
    # Section Description
    html.H2("Description", id='accueil'),
    html.P("Bienvenue dans le tableau de bord des Jeux olympiques. Explorez les données et découvrez des informations intéressantes."),
    
    dcc.Input(id='input-age', type='number', placeholder='Age'),
    dcc.Input(id='input-height', type='number', placeholder='Height'),
    dcc.Input(id='input-weight', type='number', placeholder='Weight'),
    html.Button('Prédire', id='predict-button'),
    
    # Section pour Régression Logistique
    html.H2("Régression Logistique", id='regression-logistique'),
    html.Div(id='prediction-output-lr'),
    dcc.Graph(id='probability-bar-chart-lr'),  # Graphique à barres pour la Régression Logistique
    
    # Section pour Random Forest
    html.H2("Random Forest", id='random-forest'),
    html.Div(id='prediction-output-rf'),
    dcc.Graph(id='probability-bar-chart-rf'),  # Graphique à barres pour le Random Forest

])


######## Créer un système de navigation pour passer d'une page à l'autre #####
app.layout = html.Div([
    dcc.Tabs(id='pages', value='page-1', children=[
        dcc.Tab(label='Analyse basique', value='page-1'),
        dcc.Tab(label='Analyse poussée', value='page-2'),
        dcc.Tab(label='Prédiction', value='page-3'),
    ]),
    html.Div(id='page-content')
])

######### Sert à bouger entre les pages ##########
### Callback pour mettre à jour le contenu de la page en fonction de la sélection de l'utilisateur
@app.callback(
    Output('page-content', 'children'),
    Input('pages', 'value')
)
def update_page(selected_page):
    if selected_page == 'page-1':
        return page1_layout
    elif selected_page == 'page-2':
        return page2_layout
    elif selected_page == 'page-3':
        return page3_layout



######
### Callback pour mettre à jour le graphique du nombre de médailles par année
@app.callback(
    Output('medailles-par-annee-page1', 'figure'),
    Input('annee-dropdown-page1', 'value')
)
def update_medailles_par_annee(selected_year):
    #Filtrer les données en fonction de l'année sélectionnée
    filtered_df = df[df['Year'] == selected_year]
    
    #Agréger les données par sport et calculer la somme des médailles
    medailles_par_sport = filtered_df.groupby('Sport')['Medal'].count().reset_index()
    
    #Trier les sports en fonction du nombre de médailles (du plus élevé au plus bas)
    medailles_par_sport = medailles_par_sport.sort_values(by='Medal', ascending=False)
    
    #Sélectionner les 10 premiers sports
    top_10_sports = medailles_par_sport.head(10)
    
    #Créer un graphique avec Plotly Express
    fig = px.bar(top_10_sports, x='Sport', y='Medal', title=f'10 sports avec le plus de médailles en {selected_year}')
    
    return(fig)

### Callback pour mettre à jour le graphique du nombre de médailles par pays
@app.callback(
    Output('medailles-par-pays-page1', 'figure'),
    Input('pays-dropdown-page1', 'value')
)
def update_medailles_par_pays(selected_year):
    # Utilisez la fonction pour générer le graphique du nombre de médailles par pays
    fig = generate_medailles_par_pays(selected_year)
    
    return(fig)
    
### Callback pour mettre à jour la carte mondiale en fonction de l'année sélectionnée
@app.callback(
    Output('world-map', 'figure'),
    Input('pays-dropdown-page1', 'value')
)
def update_world_map(selected_year):
    # Utilisez la fonction generate_world_map pour créer la carte pour l'année sélectionnée
    fig = generate_world_map(selected_year)
    
    return fig


### Nombre d'athletes par année ###
# Callback pour mettre à jour les graphiques
@app.callback(
    [Output('participants-par-annee', 'figure'),
     Output('medailles-par-annee2', 'figure')],
    Input('year-slider', 'value')
)
def update_graphs(selected_years):
    # Filtrer les données en fonction de la plage d'années sélectionnée
    filtered_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
    
    ### Partie pour les participants ###
    # Exclure les lignes avec des valeurs 'Athlete' manquantes (nan)
    filtered_df = filtered_df.dropna(subset=['ID'])

    # Agréger les données par année pour le nombre de participants
    participants_par_annee = filtered_df['Year'].value_counts().sort_index()

    # Créer un graphique pour le nombre de participants
    fig_participants = px.line(x=participants_par_annee.index, y=participants_par_annee.values,
                               markers=True, line_shape='linear', labels={'x': 'Année', 'y': 'Nombre de participants'},
                               title="Nombre de participants par année aux Jeux olympiques")

    ### Partie pour les médailles ###
    # Filtrer les données avec médailles
    filtered_df = filtered_df[filtered_df['Medal'].notna()]
    
    # Agréger les données par année pour le nombre de médailles
    medailles_par_annee = filtered_df['Year'].value_counts().sort_index()

    # Créer un graphique pour le nombre de médailles
    fig_medailles = px.line(x=medailles_par_annee.index, y=medailles_par_annee.values,
                          markers=True, line_shape='linear', labels={'x': 'Année', 'y': 'Nombre de médailles'},
                          title='Nombre de médailles par année aux Jeux olympiques')

    return fig_participants, fig_medailles

####### Moyenne des tailles, poids et age pour chaque genre
# Callback pour mettre à jour les graphiques lorsqu'on clique sur le bouton
@app.callback(
    [
        Output('average-height-chart-M', 'figure'),
        Output('average-weight-chart-M', 'figure'),
        Output('average-age-chart-M', 'figure'),
        Output('average-height-chart-F', 'figure'),
        Output('average-weight-chart-F', 'figure'),
        Output('average-age-chart-F', 'figure'),
    ],
    [
        Input('update-button', 'n_clicks'),
    ]
)
def update_graphs(n_clicks):
    fig_height_M = generate_average_chart('Height', 'M')
    fig_weight_M = generate_average_chart('Weight', 'M')
    fig_age_M = generate_average_chart('Age', 'M')

    fig_height_F = generate_average_chart('Height', 'F')
    fig_weight_F = generate_average_chart('Weight', 'F')
    fig_age_F = generate_average_chart('Age', 'F')

    return fig_height_M, fig_weight_M, fig_age_M, fig_height_F, fig_weight_F, fig_age_F




####### Callback Page 3 ########
# Callback pour la prédiction avec affichage des probabilités sous forme de graphique à barres pour Régression Logistique
@app.callback(
    Output('prediction-output-lr', 'children'),
    Output('probability-bar-chart-lr', 'figure'),
    [Input('predict-button', 'n_clicks')],
    [
        Input('input-age', 'value'),
        Input('input-height', 'value'),
        Input('input-weight', 'value')
    ]
)
def predict_medal_with_probabilities_and_chart_lr(n_clicks, age, height, weight):
    if n_clicks is None:
        return '', {}

    # Préparer les données d'entrée pour la prédiction
    input_data = pd.DataFrame({'Age': [age], 'Height': [height], 'Weight': [weight]})
    input_data[['Age', 'Height', 'Weight']] = imputer.transform(input_data[['Age', 'Height', 'Weight']])
    input_data_scaled = scaler.transform(input_data)

    # Faire la prédiction avec le modèle Régression Logistique
    prediction_proba_lr = best_lr_model.predict_proba(input_data_scaled)

    # Convertir les indices en entiers
    class_names_lr = ['No Medal', 'Bronze', 'Silver', 'Gold']
    predicted_class_lr = int(best_lr_model.predict(input_data_scaled)[0])

    # Afficher le résultat avec les noms des classes
    prediction_str_lr = ', '.join([f"{class_names_lr[i]}: {prob:.4f}" for i, prob in enumerate(prediction_proba_lr[0])])

    # Créer un graphique à barres avec les probabilités prédites pour Régression Logistique
    fig_lr = px.bar(x=class_names_lr, y=prediction_proba_lr[0], labels={'x': 'Medal Class', 'y': 'Probability'},
                    title='Probabilités prédites pour chaque classe (Régression Logistique)',
                    color=class_names_lr, color_discrete_map={'No Medal': 'lightgray', 'Bronze': 'peru', 'Silver': 'silver', 'Gold': 'gold'})

    return prediction_str_lr, fig_lr



# Callback pour la prédiction avec affichage des probabilités sous forme de graphique à barres pour Random Forest
@app.callback(
    Output('prediction-output-rf', 'children'),
    Output('probability-bar-chart-rf', 'figure'),
    [Input('predict-button', 'n_clicks')],
    [
        Input('input-age', 'value'),
        Input('input-height', 'value'),
        Input('input-weight', 'value')
    ]
)
def predict_medal_with_probabilities_and_chart_rf(n_clicks, age, height, weight):
    if n_clicks is None:
        return '', {}

    # Préparer les données d'entrée pour la prédiction
    input_data = pd.DataFrame({'Age': [age], 'Height': [height], 'Weight': [weight]})
    input_data[['Age', 'Height', 'Weight']] = imputer.transform(input_data[['Age', 'Height', 'Weight']])
    input_data_scaled = scaler.transform(input_data)

    # Faire la prédiction avec le modèle Random Forest
    prediction_proba_rf = best_rf_model.predict_proba(input_data_scaled)

    # Convertir les indices en entiers
    class_names_rf = ['No Medal', 'Bronze', 'Silver', 'Gold']
    predicted_class_rf = int(best_rf_model.predict(input_data_scaled)[0])

    # Afficher le résultat avec les noms des classes
    prediction_str_rf = ', '.join([f"{class_names_rf[i]}: {prob:.4f}" for i, prob in enumerate(prediction_proba_rf[0])])

    # Créer un graphique à barres avec les probabilités prédites pour Random Forest
    fig_rf = px.bar(x=class_names_rf, y=prediction_proba_rf[0], labels={'x': 'Medal Class', 'y': 'Probability'},
                    title='Probabilités prédites pour chaque classe (Random Forest)',
                    color=class_names_rf, color_discrete_map={'No Medal': 'lightgray', 'Bronze': 'peru', 'Silver': 'silver', 'Gold': 'gold'})

    return prediction_str_rf, fig_rf

    
##########  Test  ##########
if __name__ == '__main__':
    app.run_server(debug=True)
