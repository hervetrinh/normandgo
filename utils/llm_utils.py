import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class LLMUtils:
    def __init__(self, embedding_model_name=None, data_path=None, prompts={}):
        """
        Initialize the LLMUtils class with the embedding model name, data path, and prompts.
        
        :param embedding_model_name: (Optional) Name of the embedding model to use.
        :param data_path: (Optional) Path to the data files.
        :param prompts: Dictionary of prompts for LLM interactions.
        """
        self.model_id = os.getenv("AZURE_NAME")
        self.api_key = os.getenv("AZURE_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_PUBLIC_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.api_version = os.getenv('AZURE_API_VERSION')
        if embedding_model_name:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = None
        self.data_path = data_path
        self.prompts = prompts
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """
        Initialize the LLM model.
        
        :return: Instance of AzureChatOpenAI.
        """
        return AzureChatOpenAI(
            model=self.model_id,
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            openai_api_version=self.api_version,
            azure_deployment=self.azure_deployment,
            temperature=0.7
        )
        
    def get_llm(self):
        """
        Get the LLM model instance.
        
        :return: Instance of AzureChatOpenAI.
        """
        return self.llm

    def predict_llm(self, prompt):
        """
        Predict a response using the LLM.

        :param prompt: Prompt to send to the LLM.
        :return: Response from the LLM.
        """
        return self.llm.invoke(prompt).content

    def load_files(self):
        """
        Charge les fichiers de données ou les vecteurs s'ils sont déjà sauvegardés.
        """
        try:
            return self.load_embeddings()
        except FileNotFoundError:
            logging.info("Aucun fichier d'embeddings trouvé, génération en cours...")
            files = [self.data_path]
            dataframes = [pd.read_excel(file) for file in files]
            all_events_df = pd.concat(dataframes, ignore_index=True)

            def combine_columns(row):
                return f"""{row['DESCRIPTIF']} {row.get('COMMUNE', '')} {row.get('CODEPOSTAL', '')} {row.get('ADRESSE1', '')}
                {row.get('ENVIES', '')} {row.get('EQUIPEMENTS', '')} {row.get('TYPE', '')}"""

            all_events_df['combined_text'] = all_events_df.apply(combine_columns, axis=1)
            all_events_df['vector'] = all_events_df['combined_text'].apply(lambda x: self.embedding_model.encode(str(x)))
            
            vectors = np.vstack(all_events_df['vector'].values)
            dimension = vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(vectors)

            self.save_embeddings(all_events_df, index)  # Sauvegarde des données
            return all_events_df, index, self.embedding_model

    @staticmethod
    def extract_location(query, known_locations):
        """
        Extract a location (city or place) from the user query.

        :param query: The user query.
        :param known_locations: List of known cities.
        :return: The detected city, or None if none is found.
        """
        for location in known_locations:
            if re.search(rf"{re.escape(location)}", query, re.IGNORECASE):
                return location
        return None

    @staticmethod
    def postal_code_distance(code1, code2):
        """
        Calculate an approximate distance between postal codes.

        :param code1: Postal code 1.
        :param code2: Postal code 2.
        :return: Approximate distance (numerical).
        """
        if not (str(code1).isdigit() and str(code2).isdigit()):
            return float('inf')  # Infinite distance if a code is missing
        return abs(int(code1) - int(code2))  # Approximate numerical difference

    def search_rag(self, query, df, index, top_k=5, postal_weight=0.5):
        """
        Search for the most relevant events in the database using geographic weighting.

        :param query: User query.
        :param df: Event database.
        :param index: Vector index.
        :param top_k: Number of results to return.
        :param postal_weight: Weighting for geographic distance (0 to 1).
        :return: Relevant results.
        """
        if not self.embedding_model:
            raise ValueError("Embedding model is not provided.")

        # Convert top_k to int if it's not already
        if not isinstance(top_k, int):
            try:
                top_k = int(top_k)
            except ValueError:
                raise ValueError(f"Invalid top_k value: {top_k}")

        known_locations = df['COMMUNE'].dropna().unique().tolist()
        user_location = self.extract_location(query, known_locations)
        user_postal_code = None
        if user_location:
            user_postal_code = df.loc[df['COMMUNE'].str.contains(user_location, case=False, na=False), 'CODEPOSTAL'].iloc[0]

        query_vector = self.embedding_model.encode(query)
        query_vector = np.array([query_vector], dtype='float32')

        distances, indices = index.search(query_vector, top_k * 10)

        results = df.iloc[indices[0]].copy()
        results['embedding_distance'] = distances[0]

        if user_postal_code:
            results['geo_distance'] = results['CODEPOSTAL'].apply(lambda x: self.postal_code_distance(user_postal_code, x))
        else:
            results['geo_distance'] = 0

        results['combined_score'] = (
            (1 - postal_weight) * results['embedding_distance'] +
            postal_weight * results['geo_distance']
        )

        results = results.sort_values(by='combined_score').head(top_k)
        return results
    
    def save_embeddings(self, df, index, embedding_file='embeddings.npy', index_file='faiss_index.bin'):
        """
        Sauvegarde les embeddings et l'index FAISS sur disque.
        """
        df.to_pickle('data/events.pkl')  # Sauvegarde des données
        np.save(embedding_file, np.vstack(df['vector'].values))  # Sauvegarde des vecteurs d'embeddings
        faiss.write_index(index, index_file)  # Sauvegarde de l'index FAISS

    def load_embeddings(self, embedding_file='embeddings.npy', index_file='faiss_index.bin'):
        """
        Charge les embeddings et l'index FAISS depuis le disque.
        """
        if os.path.exists('data/events.pkl') and os.path.exists(embedding_file) and os.path.exists(index_file):
            df = pd.read_pickle('data/events.pkl')  # Chargement des données
            embeddings = np.load(embedding_file)  # Chargement des vecteurs d'embeddings
            index = faiss.read_index(index_file)  # Chargement de l'index FAISS
            return df, index, self.embedding_model
        else:   
            raise FileNotFoundError("Fichiers d'embeddings ou d'index non trouvés.")