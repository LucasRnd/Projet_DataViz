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
#best_lr_model = joblib.load('best_logistic_model.pkl')

# Import du modèle Random Forest
#best_rf_model = joblib.load('best_rf_model.pkl')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib

# Import des données
df = pd.read_csv("C:/Users/Lucas/Downloads/archive/dataset_olympics.csv", sep=",", encoding="utf-8")

# Filtrer les Jeux olympiques d'été depuis 1994
df = df[(df['Year'] >= 1994) & (df['Year'] % 4 == 0)]

# Convertir les colonnes 'Height', 'Weight' et 'Age' en numérique
df[['Height', 'Weight', 'Age']] = df[['Height', 'Weight', 'Age']].apply(pd.to_numeric, errors='coerce')

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna(subset=['Height', 'Weight', 'Age', 'Sex'])

# Remplacer les valeurs manquantes par la médiane
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(df[['Age', 'Height', 'Weight']])

# Convertir les étiquettes de médaille en valeurs numériques
medal_mapping = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
df['Medal'] = df['Medal'].replace(medal_mapping).fillna(0)

# Séparer les caractéristiques (X) et la cible (y)
X = df[['Age', 'Height', 'Weight']]
y = df['Medal']

# Suréchantillonnage des classes avec SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Diviser l'ensemble de données
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardiser les caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialiser le modèle de régression logistique avec des paramètres réduits
logistic_model = LogisticRegression(random_state=42, C=1)

# Entraîner le modèle
logistic_model.fit(X_train_scaled, y_train)

# Sauvegarder le modèle
joblib.dump(logistic_model, 'best_logistic_model.pkl')

# Initialiser le modèle de forêt aléatoire avec des paramètres réduits
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)

# Entraîner le modèle
rf_model.fit(X_train_scaled, y_train)

# Sauvegarder le modèle
joblib.dump(rf_model, 'best_rf_model.pkl')

# Faire des prédictions sur l'ensemble de test
y_pred_logistic = logistic_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# Évaluer les performances des modèles
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

# Afficher les performances des deux modèles
print("Régression Logistique:")
print(f'Accuracy: {accuracy_logistic}')
print('Classification Report:')
print(classification_rep_logistic)

print("\nForêt Aléatoire:")
print(f'Accuracy: {accuracy_rf}')
print('Classification Report:')
print(classification_rep_rf)


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
html.P(
    "Ce graphique en barres présente la répartition du nombre de médailles par sport en fonction de "
    "l’année sélectionnée. Chaque barre représente un sport différent. En 2016 par exemple, 64 médailles "
    "ont récompensé les pratiquants de l’athlétisme, 46 médailles pour la natation… "
    "En analysant le graphique, on peut conclure que certains sports ont été plus performants en termes "
    "de médailles que d'autres comme l’athlétisme, la natation et l’aviron qui se hissent dans le podium "
    "en 2016, 2012, 2008, 2004 et 2000. "
    "Cela peut être dû à plusieurs raisons, notamment le nombre d’épreuves qui peut-être plus grand en "
    "fonction des sports. L’athlétisme par exemple propose une variété d'épreuves de sprint, de saut, de "
    "lancer, de demi-fond, etc. De ce fait, la diversité des épreuves peut être un élément justifiant le "
    "nombre élevé de médailles obtenus dans un sport. Également, ces jeux sont pratiqués à travers le "
    "monde et ont une grande cote de popularité et une visibilité médiatique importante."
),
    
    
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
    html.P(
        "En fonction de l’année sélectionnée, nous avons le top 10 des pays avec le plus grand nombre de "
        "médailles remporté. On peut observer visuellement quels pays ont eu le plus de succès en termes de "
        "médailles. L'analyse de ce graphique permet de tirer des conclusions sur les performances relatives "
        "des différents pays en termes de médailles. Par exemple en 2016, c’est le Brésil (229) qui se hissent à "
        "la première place suivie de près par les États-Unis (220). En 2012, c’est les USA qui remportent la "
        "première place avec 170 médailles. "
        "On peut prétendre en général l’existence d’une corrélation entre les performances sportives d'un "
        "pays hôte aux Jeux Olympiques et le fait que ce pays se classe dans le top 10 des médailles. Par "
        "exemple, la Grèce qui ne fait généralement pas partie du Top 10, s’est classée à la dixième place lors "
        "des JO 2004 organisés à Athènes. Le Brésil classé 9 ème en 1996, puis 10 ème en 2000, 7 ème en 2004, 2008 "
        "et 2012, s’est hissé à la première place en 2016 (année à laquelle les JO se sont déroulés à Rio de "
        "Janeiro). Idem pour le Royaume-Uni."
    ),

         
    
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
    
     html.P(
        "Ce graphique permet de visualiser l’évolution du nombre de participants en fonction des années. Il "
        "suffit dans un premier temps de sélectionner l’intervalle d’année souhaité. Si l’on sélectionne toutes "
        "les années, de 1996 à 2016, on peut voir qu’en 1996, il y avait 2985 participants. Par la suite le "
        "nombre de participants se situait autour de 3400 et 3490. En 2012, ce nombre s’affaiblit à hauteur de "
        "3277 pour enfin arriver à 3598 en 2016."
    ),
    
    html.P(
        "Ce graphique permet de visualiser le nombre de médailles par année. On peut voir une croissance "
        "quasiment linéaire de ce nombre. Cela peut être dû notamment à l’expansion des Jeux Olympiques. "
        "En effet avec l'ajout de nouveaux sports, disciplines et catégories. L'inclusion de ces nouvelles "
        "épreuves a naturellement conduit à une augmentation du nombre total de médailles disponibles. "
        "L'augmentation de la représentation des femmes dans les Jeux Olympiques, avec l'introduction de "
        "nouvelles épreuves et disciplines féminines, a contribué à l'augmentation du nombre total de "
        "médailles décernées."
    ),

])

######### Créer les sections de la deuxième page ###########
page2_layout = html.Div([  
    # Titre
    html.H1("Partie 2 - Analyse de l'âge et des mensurations"),
    
    # Section Description
    html.H2("Description", id='accueil'),
    html.P("Bienvenue dans la deuxième partie du tableau de bord des Jeux olympiques. Cette section est dédiée à l'analyse des données relatives à l'âge et aux mensurations des athlètes participant aux Jeux olympiques."),
    
    html.P("L'objectif de cette analyse est de fournir des insights intéressants sur les tendances générales de l'âge, de la taille et du poids des athlètes, en se concentrant sur la distinction entre hommes et femmes."),
    
    html.P("Explorez ces graphiques interactifs pour découvrir des informations utiles sur les caractéristiques physiques des athlètes dans différentes disciplines."),
    
    #### Moyennes taille, poids et age par genre
    html.H1("Analyse des caractéristiques moyennes par sport et par année"),

    # Bouton de mise à jour
    html.Button('Mettre à jour les graphiques', id='update-button'),

    # Graphiques générés par callback
    html.Div([
        # Graphiques pour les hommes
        html.Div([
            html.H2("Concernant les hommes", style={'text-align': 'center'}),
            
            dcc.Graph(id='average-height-chart-M', style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id='average-weight-chart-M', style={'width': '33%', 'display': 'inline-block'}),
            dcc.Graph(id='average-age-chart-M', style={'width': '33%', 'display': 'inline-block'}),
        ]),
        
        # Commentaires pour les graphiques des hommes
        html.P("Les basketteurs ont la plus grande taille, suivis des volleyeurs et enfin des joueurs de water polo. Les athlètes les plus petits pratiquent le trampoline."),
        html.P("En termes de poids, les basketteurs, les handballeurs et les joueurs de water-polo sont les plus lourds, tandis que les athlètes de trampoline sont les plus légers."),
        html.P("En ce qui concerne l'âge, les cavaliers, les tireurs et les joueurs de beach-volley sont parmi les plus âgés, tandis que les joueurs de football sont parmi les plus jeunes."),
    ]),

    # Graphiques pour les femmes
    html.Div([
        html.H2("Concernant les femmes", style={'text-align': 'center'}),
        
        dcc.Graph(id='average-height-chart-F', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='average-weight-chart-F', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='average-age-chart-F', style={'width': '33%', 'display': 'inline-block'}),
    ]),
    
    # Commentaires pour les graphiques des femmes
    html.P("Les basketteuses, les volleyeuses et les joueuses de beach-volley ont la plus grande taille, tandis que les gymnastes ont la plus petite."),
    html.P("En termes de poids, les basketteuses, les handballeuses et les volleyeuses sont les plus lourdes, tandis que les gymnastes sont les plus légères."),
    html.P("En ce qui concerne l'âge, les cavalières, les tireuses et les joueuses de beach-volley sont parmi les plus âgées, tandis que les gymnastes sont parmi les plus jeunes."),
])




######### Créer les sections de la troisième page ###########
page3_layout = html.Div([
    # Titre
    html.H1("Partie 3 - Prédiction de médaille"),
    # Section Description
    html.H2("Description", id='accueil'),
    
    html.P("Bienvenue dans la troisième partie du tableau de bord des Jeux olympiques. Dans cette section, vous avez la possibilité de faire des prédictions de médailles pour un athlète en particulier."),
    
    html.P("Utilisez les champs ci-dessous pour entrer l'âge, la taille et le poids de l'athlète. Ensuite, cliquez sur le bouton 'Prédire' pour obtenir les probabilités de remporter une médaille dans chaque catégorie (Bronze, Argent, Or) selon deux modèles de machine learning : Régression Logistique et Random Forest."),
    
    dcc.Input(id='input-age', type='number', placeholder='Age'),
    dcc.Input(id='input-height', type='number', placeholder='Height'),
    dcc.Input(id='input-weight', type='number', placeholder='Weight'),
    html.Button('Prédire', id='predict-button'),
    
    # Section pour Régression Logistique
    html.H2("Régression Logistique", id='regression-logistique'),
    html.Div(id='prediction-output-lr'),
    dcc.Graph(id='probability-bar-chart-lr'),  # Graphique à barres pour la Régression Logistique
    
html.Div("Le graphique ci-dessus illustre les chances de médailles suivant les mensurations et "
         "l'âge de l'utilisateur à l'aide d'une Régression Logistique. On remarque que les probabilités ne fluctuent pas "
         "beaucoup et semblent erronées, cela est dû à l'accuracy qui est de 0.3."),  # Commentaire sous le graphique

    
    # Section pour Random Forest
    html.H2("Random Forest", id='random-forest'),
    html.Div(id='prediction-output-rf'),
    dcc.Graph(id='probability-bar-chart-rf'),  # Graphique à barres pour le Random Forest

html.Div("Le graphique ci-dessus illustre les chances de médailles suivant les mensurations et "
         "l'âge de l'utilisateur à l'aide d'un random forest. On remarque que les probabilités ici fluctuent bien au regard d'un changement dans les mensurations, "
         "c'est notamment grâce à l'accuracy qui est de 0.9."),  # Commentaire sous le graphique

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


