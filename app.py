import os
import sys
import re
import datetime
import yaml
import logging
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from langchain_openai import AzureChatOpenAI
import shutil
from flask import flash


# Add the utils directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import custom modules
from route_finder import RouteFinder
from bike_availability_predictor import BikeAvailabilityPredictor
from accessibility import Accessibility
from drinking_water import DrinkingWater
from openagenda_events import OpenAgendaEvents
from rag_agent import RAGAgent
from collection import CollectionUtils
from data_processing import DataProcessor
from recommander import RecommendationSystem
from route_finder_app import RouteFinderApp  
from llm_utils import LLMUtils
from verbatims import VerbatimClassifier
from pathlib import Path

with open('config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
    
with open('config/prompts.yaml', 'r', encoding='utf-8') as file:
    prompts = yaml.safe_load(file)

llm = LLMUtils().get_llm()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize data processor and load data
data_processor = DataProcessor(config['data']['lieux_visite_path'])
logging.info('Loading data...')
data_processor.load_data()
logging.info('Processing data...')
lieux_visite = data_processor.process_addresses()

resto = pd.read_excel("data/Restaurants.xlsx")

# Initialize recommender system
embeddings_file_path = 'data/saved_embeddings.pkl'

recommender = RecommendationSystem(
    embedding_model_name=config["embedding_model"]["name"],
    data=lieux_visite,
    config_path='config/config.yaml',
    embeddings_path=embeddings_file_path
)

if not Path(embeddings_file_path).exists():
    logging.info("Calculating and saving embeddings...")
    recommender.preprocess_data()
    recommender.extract_features()
else:
    logging.info("Loading precomputed embeddings...")
    recommender.load_embeddings()

# Initialize utility classes
collection_utils = CollectionUtils()
accessibility = Accessibility(config['data']['acceslibre_file'], config['data']['datatour_file'])
drinking_water = DrinkingWater(config['overpass']['query'])
openagenda = OpenAgendaEvents(os.getenv('OVERPASS_API_KEY'), config['openagenda']['agenda_uids'])
rag_agent = RAGAgent(
    embedding_model_name=config['embedding_model']['name'],
    data_path=config['data']['datatour_file'],
    prompts=prompts['llm']
)

lieux_76 = pd.read_csv("data/lieux_76.csv", sep=";")

lieux_visite_data = lieux_visite[["NOM", "LATITUDE", "LONGITUDE", "DESCRIPTIF", "PHOTO", "TSID"]].dropna(subset=["NOM", "LATITUDE", "LONGITUDE", "TSID"]).fillna("").to_dict(orient='records')
restaurants_data = resto[["NOM", "LATITUDE", "LONGITUDE", "DESCRIPTIF", "PHOTO", "TSID"]].dropna(subset=["NOM", "LATITUDE", "LONGITUDE", "TSID"]).fillna("").to_dict(orient='records')



# Initialize RouteFinderApp
route_finder_app = RouteFinderApp(config)


@app.route('/monuments')
def monuments():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')
    page = request.args.get('page', 1, type=int)
    per_page = 10

    lieux_visite_sorted = lieux_76.sort_values(by='FREQUENTATION_SCORE', ascending=True)

    total_monuments = len(lieux_visite_sorted)
    total_pages = (total_monuments + per_page - 1) // per_page  

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    monuments = []
    
    for idx in range(start_idx, min(end_idx, total_monuments)):
        row = lieux_visite_sorted.iloc[idx]
        photos = str(row['PHOTO']).split(' ## ') if 'PHOTO' in lieux_visite_sorted.columns and pd.notna(row['PHOTO']) and row['PHOTO'] else []
        photo_url = None
        if photos:
            photo_url = photos[0].strip()
            if " " in photo_url:
                photo_url = photo_url.split(" ")[0]  
            if not re.match(r'^https?://', photo_url):
                photo_url = None
        
        descriptif = row['DESCRIPTIF'] if pd.notna(row['DESCRIPTIF']) else ""
        
        monuments.append({
            'TSID': row['TSID'],
            'NOM': row['NOM'],
            'COMMUNE': row['COMMUNE'],
            'DESCRIPTIF': descriptif,
            'PHOTO': photo_url or url_for('static', filename='imgs/no_photo.jpg')
        })
    
    return render_template('monuments.html', monuments=monuments, page=page, total_pages=total_pages)


@app.route('/recommandation')
def recommandation():
    users = pd.read_csv("data/id_login.csv", sep=";")
    users = users[users.username == session.get('username', "Guest")] 
    if users.shape[0] != 0:
        tsid = users.last_tsid_monu.iloc[0]
    else:
        return redirect(url_for('login'))
    
    user_lat = lieux_visite.loc[lieux_visite['TSID'] == tsid, 'LATITUDE'].values[0]
    user_lon = lieux_visite.loc[lieux_visite['TSID'] == tsid, 'LONGITUDE'].values[0]
    
    result = recommender.get_recommendations_with_geo(user_lat=user_lat, user_lon=user_lon, tsid=tsid)
    recommendations = result['recommendations'].head(8)
    
    recommendations_list = recommendations.to_dict(orient='records')
    return render_template('recommandation.html', recommendations=recommendations_list, ref_name=result['ref_name'], tsid=tsid)

@app.route("/find-routes", methods=["POST"])
def find_routes():
    data = request.json
    result = route_finder_app.find_routes(data)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)

@app.route("/")
def login():
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', False)
    return redirect(url_for('login'))

@app.route('/data')
def data():
    return jsonify({"lieux_visite": lieux_visite_data, "restaurants": restaurants_data})

@app.template_filter('ensure_string')
def ensure_string(value):
    return str(value) if value is not None else ''

@app.route('/tables')
def tables():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')

    if os.path.isfile(config['data']['feedback_file']):
        df = pd.read_pickle(config['data']['feedback_file'])[["username", 'date', 'TSID', 'NOM', 'verbatim', "theme", 'sentiment']]
    else:
        df = pd.DataFrame(columns=["username", 'date', 'TSID', 'NOM', 'verbatim', "theme", 'sentiment'])

    df = df.sort_values('date', ascending=False)

    table_data = df.to_dict(orient='records')

    page = request.args.get('page', 1, type=int)
    per_page = 8
    total = len(table_data)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = table_data[start:end]

    next_url = f"/tables?page={page + 1}" if end < total else None
    prev_url = f"/tables?page={page - 1}" if start > 0 else None

    return render_template('tables.html', table_data=paginated_data, next_url=next_url, prev_url=prev_url, name=name)

@app.route('/reset_feedback_file', methods=['POST'])
def reset_feedback_file():
    current_file = config['data']['feedback_file']  
    backup_file = os.path.join('data', 'Verbatims_bak.pkl')
    try:
        shutil.copy(backup_file, current_file)
        flash('Le fichier des feedbacks a été réinitialisé avec succès.', 'success')
    except Exception as e:
        flash(f'Erreur lors de la réinitialisation : {e}', 'danger')
    return redirect(url_for('tables'))

@app.route('/feedback', methods=['POST'])
def feedback():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 403
    
    data = request.get_json()
    
    feedback_text = data['text']
    nom = data['nom']
    longitude = data['longitude']
    latitude = data['latitude']
    username = session.get('username')
    
    classifier = VerbatimClassifier(llm=llm)

    feedback_file = config['data']['feedback_file']
    feedback_df = pd.DataFrame([[username, longitude, latitude, nom, feedback_text]], columns=['username', 'LONGITUDE', 'LATITUDE', 'NOM', 'FEEDBACK'])
    
    try:
        feedback_df['verbatims'] = feedback_df['FEEDBACK'].map(lambda x : classifier.classify_comment(x))
        feedback_df = feedback_df.explode('verbatims')

        feedback_df[['verbatim', 'theme', 'sentiment']] = feedback_df['verbatims'].apply(
            lambda x: pd.Series({'verbatim': x['verbatim'], 'theme': x['theme'].value, 'sentiment': x['sentiment'].value})
        )

        feedback_df.drop(columns=['verbatims'], inplace=True)
    except:
        print('UNE ERREUR EST SURVENUE')
        feedback_df = pd.DataFrame([[username, longitude, latitude, nom, feedback_text, feedback_text, "Autres", "Neutre"]], columns=['username', 'LONGITUDE', 'LATITUDE', 'NOM', 'FEEDBACK', "verbatim", "theme", "sentiment"])

    try:
        tsid = lieux_visite[(lieux_visite.LONGITUDE == longitude) & (lieux_visite.LATITUDE == latitude)].TSID.iloc[0]
    except:
        try:
            tsid = resto[(resto.LONGITUDE == longitude) & (resto.LATITUDE == latitude)].TSID.iloc[0]
        except:
            tsid = 'XXXXXX'


    feedback_df["TSID"] = tsid
    feedback_df['date'] = datetime.datetime.now().strftime("%Y-%m-%d")

    feedbacks = pd.concat([feedback_df, pd.read_pickle(feedback_file)], axis=0, ignore_index=True)
    feedbacks.to_pickle(feedback_file)
    
    return jsonify({'status': 'success'})


@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username'].lower()
    password = request.form['password']
    
    df = pd.read_csv(config['data']['id_login_file'], sep=";")
    df = df.applymap(lambda x : str(x))
    
    user = df[(df['username'] == username) & (df['password'] == password)]

    if not user.empty:
        session['logged_in'] = True
        session['username'] = user.iloc[0]['username']
        session['name'] = user.iloc[0]['name']
        return redirect(url_for('index'))
    else:
        return 'Invalid credentials'

@app.route('/index')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')
    return render_template('index.html', name=name)

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def do_register():
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']
    
    if password != confirm_password:
        return 'Passwords do not match'
    
    username = email.lower()
    
    df = pd.read_csv(config['data']['id_login_file'], sep=";")
    df = df.applymap(lambda x: str(x))
    
    if not df[(df['username'] == username)].empty:
        return 'Username already exists'
    
    new_user = pd.DataFrame([[first_name + ' ' + last_name, username, password]], columns=['name', 'username', 'password'])
    df = pd.concat([df, new_user], ignore_index=True)
    
    df.to_csv(config['data']['id_login_file'], sep=";", index=False)
    
    return redirect(url_for('login'))

@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/forgot_password', methods=['POST'])
def do_forgot_password():
    email = request.form['email'].lower()
    
    df = pd.read_csv(config['data']['id_login_file'], sep=";")
    df = df.applymap(lambda x: str(x))
    
    user = df[df['username'] == email]

    if user.empty:
        return 'Email not found'
    
    return 'Password reset instructions have been sent to your email'

@app.route('/charts')
def charts():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')

    df = pd.read_pickle("data/Verbatims.pkl")

    sentiment_counts = df['sentiment'].value_counts()
    bar_labels = [label.capitalize() for label in sentiment_counts.index.tolist()]  
    bar_values = sentiment_counts.values.tolist()

    theme_counts = df['theme'].value_counts()
    pie_labels = theme_counts.index.tolist()
    pie_values = theme_counts.values.tolist()

    top_negative = (
        df[df['sentiment'] == 'Négatif']
        .groupby('NOM')
        .size()
        .sort_values(ascending=False)
        .head(3)
    )

    top_negative_labels = top_negative.index.tolist()
    top_negative_values = top_negative.values.tolist()

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d') 

    recent_negatives = (
        df[df['sentiment'] == 'Négatif']
        .sort_values(by='date', ascending=False)
        .head(5)[['date', 'NOM', 'FEEDBACK']]  
    )
    recent_negatives_list = recent_negatives.to_dict(orient='records')

    now_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    now = datetime.datetime.now()
    four_months_ago = now - datetime.timedelta(days=120)
    df_recent = df[df['date'] < four_months_ago]

    df_recent['month'] = df_recent['date'].dt.to_period('M')  
    trends = df_recent.groupby(['month', 'sentiment']).size().unstack(fill_value=0)

    trend_labels = [str(month) for month in trends.index]  
    trend_positive = trends['Positif'].tolist() if 'Positif' in trends else [0] * len(trends)
    trend_neutral = trends['Neutre'].tolist() if 'Neutre' in trends else [0] * len(trends)
    trend_negative = trends['Négatif'].tolist() if 'Négatif' in trends else [0] * len(trends)

    chart_data = {
        'bar_labels': bar_labels,
        'bar_values': bar_values,
        'pie_labels': pie_labels,
        'pie_values': pie_values,
        'trend_labels': trend_labels,
        'trend_positive': trend_positive,
        'trend_neutral': trend_neutral,
        'trend_negative': trend_negative,
        'top_negative_labels': top_negative_labels,
        'top_negative_values': top_negative_values,
        'recent_negatives': recent_negatives_list,
        'updated_at': now_str
    }

    synthesis = "Chargement en cours..."

    return render_template('charts.html', name=name, chart_data=chart_data, synthesis=synthesis)

@app.route('/synthesis', methods=['GET'])
def generate_synthesis():
    try:
        df = pd.read_pickle('data/Verbatims.pkl')
        verbs = ";".join(df.FEEDBACK.values)

        synthesis = llm.invoke(f"""      
                            Answer only in French, not in English. Analyze the verbatims about all the places to visit found between the XML tags <verbatim>. These verbatims are delimited by semicolons, and people discuss the following themes
                                - Historical and cultural heritage,
                                - Landscapes and nature,
                                - Activities and leisure,
                                - Events and festivals,
                                - Gastronomy and local products,
                                - Accommodation and services,
                                - Transport and accessibility,
                                - Reception and information,
                                - Conservation and maintenance,
                                - User experience and general impressions,
                                - Digital accessibility,
                                - Other

                            Provide a global summary in French, using a maximum of three sentences (negative and positive).
                            <verbatim> {verbs} </verbatim>
                            """).content
    except Exception:
        synthesis = "Une erreur s'est produite"
    return {"synthesis": synthesis}

@app.route('/agenda')
def agenda():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')
    return render_template('agenda.html', name=name)

@app.route('/get_events')
def get_events():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    df = pd.read_csv('data/agenda.csv', sep=";")
    
    events = []
    event_count = {}
    for _, row in df.iterrows():
        event = {
            'title': row['title.fr'],
            'start': row['firstTiming.begin'],
            'end': row['lastTiming.end'],
            'description': row['description.fr'],
            'location': row['location.address']
        }
        events.append(event)
        date_str = row['firstTiming.begin'].split('T')[0]
        if date_str not in event_count:
            event_count[date_str] = 0
        event_count[date_str] += 1
    
    return jsonify(events=events, event_count=event_count)

@app.route("/rag_agent")
def serve_rag_agent():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')
    return render_template("rag_agent.html", name=name)

@app.route('/api/rag_agent', methods=['POST'])
def rag_api():
    data = request.json
    query = data.get("query", "")
    try:
        response, _ = rag_agent.unified_query(query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tiers/<user_id>", methods=["GET"])
def get_tiers(user_id):
    tiers_info = collection_utils.get_tiers(user_id)
    if not tiers_info:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    return jsonify(tiers_info)

@app.route("/api/purchase_tier", methods=["POST"])
def purchase_tier_item():
    data = request.json
    user_id = data.get("user_id")
    item_cost = data.get("cost")

    if user_id is None or item_cost is None:
        return jsonify({"error": "Paramètres manquants"}), 400

    remaining_coins = collection_utils.purchase_tier_item(user_id, item_cost)
    if remaining_coins is None:
        return jsonify({"error": "Utilisateur non trouvé ou fonds insuffisants"}), 400

    return jsonify({
        "message": "Achat réussi",
        "remaining_coins": remaining_coins
    })

@app.route("/api/user/<user_id>", methods=["GET"])
def get_user_details(user_id):
    user_details = collection_utils.calculate_user_details(user_id)
    if not user_details:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    return jsonify(user_details)

@app.route("/collection")
def collection():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')
    return render_template("collection.html", name=name)

@app.route('/lieu/<tsid>')
def lieu(tsid):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')
    lieu = lieux_visite[lieux_visite['TSID'] == tsid].dropna(subset=["NOM", "LATITUDE", "LONGITUDE","TSID"]).fillna("").to_dict(orient='records')[0]
    accessibility_info = accessibility.return_accessibility(lieu['NOM'])
    return render_template('lieu.html', lieu=lieu, name=name, accessibility_info=accessibility_info)
    
@app.route('/restaurant/<tsid>')
def restaurant(tsid):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')
    restaurant_data = resto[resto['TSID'] == tsid].dropna(subset=["NOM", "LATITUDE", "LONGITUDE", "TSID"]).fillna("").to_dict(orient='records')[0]
    restaurant_data['ADDR'] = f"{restaurant_data['ADRESSE1']} {restaurant_data['ADRESSE2']} {restaurant_data['CODEPOSTAL']} {restaurant_data['COMMUNE']}"
    accessibility_info = accessibility.return_accessibility(restaurant_data['NOM'])
    return render_template('restaurant.html', restaurant=restaurant_data, name=name, accessibility_info=accessibility_info)

@app.route('/restaurants')
def restaurants():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name = session.get('name', 'Guest')
    page = request.args.get('page', 1, type=int)
    per_page = 10

    restaurants_data = resto.dropna(subset=["NOM", "LATITUDE", "LONGITUDE", "TSID"]).fillna("")
    restaurants_data = restaurants_data[restaurants_data.CODEPOSTAL.astype(str).str[:2] == "76"]

    total_restaurants = len(restaurants_data)
    total_pages = (total_restaurants + per_page - 1) // per_page  

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    restos = []
    
    for idx in range(start_idx, min(end_idx, total_restaurants)):
        row = restaurants_data.iloc[idx]
        photos = str(row['PHOTO']).split(' ## ') if 'PHOTO' in restaurants_data.columns and pd.notna(row['PHOTO']) and row['PHOTO'] else []
        photo_url = None
        if photos:
            photo_url = photos[0].strip()
            if " " in photo_url:
                photo_url = photo_url.split(" ")[0]  
            if not re.match(r'^https?://', photo_url):
                photo_url = None
        
        descriptif = row['DESCRIPTIF'] if pd.notna(row['DESCRIPTIF']) else ""
        
        restos.append({
            'TSID': row['TSID'],
            'NOM': row['NOM'],
            'COMMUNE': row['COMMUNE'],
            'DESCRIPTIF': descriptif,
            'PHOTO': photo_url or url_for('static', filename='imgs/no_photo.jpg'),
            'SPECIALITES': row['SPECIALITES'] if 'SPECIALITES' in row and pd.notna(row['SPECIALITES']) else ""
        })
    
    return render_template('restaurants.html', restaurants=restos, page=page, total_pages=total_pages, name=name)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if query:
        results = lieux_visite[lieux_visite['NOM'].str.contains(query, case=False, na=False)]
        suggestions = results[['NOM', 'TSID']].to_dict(orient='records')
    else:
        suggestions = []
    return jsonify(suggestions)

@app.route('/api/drinking_water', methods=['GET'])
def get_drinking_water():
    geojson_data = drinking_water.fetch_drinking_water()
    return jsonify(geojson_data)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run()