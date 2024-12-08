import logging
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import great_circle
import pickle  
from pathlib import Path
import yaml

class RecommendationSystem:
    def __init__(self, embedding_model_name, data, config_path, device=None, embeddings_path=None):
        """
        Initialize the recommendation system with model path and data.

        Args:
            embedding_model_name (str): Name of the embedding model.
            data (pd.DataFrame): DataFrame containing the data.
            config_path (str): Path to the configuration file.
            device (str, optional): Device to run the model on. Defaults to 'cuda' if available.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.data = data
        self.embeddings_path = embeddings_path
        self.features = None
        self.cosine_sim = None
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def preprocess_data(self):
        """
        Preprocess the data by combining relevant columns into a single string.
        """
        combined_columns = self.config['combined_features_columns']
        self.data['combined_features'] = self.data[combined_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).fillna('')

    def save_embeddings(self):
        """
        Save extracted features and cosine similarity matrix.
        """
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump((self.features, self.cosine_sim), f)
        logging.info("Embeddings and similarity matrix saved.")

    def load_embeddings(self):
        """
        Load extracted features and cosine similarity matrix from a file.
        """
        with open(self.embeddings_path, 'rb') as f:
            self.features, self.cosine_sim = pickle.load(f)
        logging.info("Embeddings and similarity matrix loaded.")

    def extract_features(self):
        """
        Extract features if not already extracted and load from disk if available.
        """
        if self.embeddings_path and Path(self.embeddings_path).exists():
            self.load_embeddings()
        else:
            self.data['vector'] = self.data['combined_features'].apply(lambda x: self.embedding_model.encode(str(x)))
            self.features = np.vstack(self.data['vector'].values)
            self.cosine_sim = cosine_similarity(self.features)
            self.save_embeddings()

    @staticmethod
    def geographic_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the geographic distance between two points.
        """
        coord1 = (lat1, lon1)
        coord2 = (lat2, lon2)
        return great_circle(coord1, coord2).km

    def get_recommendations_with_geo(self, user_lat=None, user_lon=None, tsid=None):
        """
        Get recommendations considering both similarity and geographic proximity.
        """
        alpha = self.config['alpha']
        
        if tsid is not None:
            idx = self.data[self.data['TSID'] == tsid].index[0]
            ref_lat = self.data.at[idx, 'LATITUDE']
            ref_lon = self.data.at[idx, 'LONGITUDE']
            ref_name = self.data.at[idx, 'NOM']
        elif user_lat is not None and user_lon is not None:
            ref_lat = user_lat
            ref_lon = user_lon
            idx = None
            ref_name = "Unknown location"
        else:
            raise ValueError("Either TSID or (user_lat, user_lon) must be provided.")

        if idx is not None and self.cosine_sim is not None:
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            for i in range(len(sim_scores)):
                place_idx = sim_scores[i][0]
                place_lat = self.data.iloc[place_idx]['LATITUDE']
                place_lon = self.data.iloc[place_idx]['LONGITUDE']
                distance = self.geographic_distance(ref_lat, ref_lon, place_lat, place_lon)
                sim_scores[i] = (place_idx, sim_scores[i][1] - alpha * distance)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            if self.data['TSID'].iloc[sim_scores[0][0]] == tsid:
                sim_scores = sim_scores[1:11]
            else:
                sim_scores = sim_scores[:10]
            monument_indices = [i[0] for i in sim_scores]
        else:
            distances = [
                (idx, self.geographic_distance(ref_lat, ref_lon, row['LATITUDE'], row['LONGITUDE']))
                for idx, row in self.data.iterrows()
            ]
            distances = sorted(distances, key=lambda x: x[1])
            monument_indices = [i[0] for i in distances[:10]]

        recommendations = []
        for idx in monument_indices:
            place_lat = self.data.iloc[idx]['LATITUDE']
            place_lon = self.data.iloc[idx]['LONGITUDE']
            distance = self.geographic_distance(ref_lat, ref_lon, place_lat, place_lon) * 1000  # Convert km to meters
            photos = str(self.data.iloc[idx]['PHOTO']).split(' ## ') if 'PHOTO' in self.data.columns and pd.notna(self.data.iloc[idx]['PHOTO']) else []
            photo_url = photos[0] if photos else 'static/imgs/no_photo.jpg'
            recommendations.append({
                'TSID': self.data.iloc[idx]['TSID'],
                'NOM': self.data.iloc[idx]['NOM'],
                'LATITUDE': place_lat,
                'LONGITUDE': place_lon,
                'DISTANCE_METERS': distance,
                'PHOTO': photo_url
            })

        return {'ref_name': ref_name, 'recommendations': pd.DataFrame(recommendations)}