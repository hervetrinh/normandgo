import logging
from llm_utils import LLMUtils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class RAGAgent:
    def __init__(self, embedding_model_name=None, data_path=None, prompts={}):
        self.llm_utils = LLMUtils(
            embedding_model_name=embedding_model_name,
            data_path=data_path,
            prompts=prompts
        )
        if data_path and embedding_model_name:
            self.df, self.index, self.embedding_model = self.llm_utils.load_files()
        else:
            self.df, self.index, self.embedding_model = None, None, None

    def search(self, query):
        if self.df is None or self.index is None:
            raise ValueError("Data files or embedding model are not loaded.")
        if self.df.empty or self.index.ntotal == 0:
            raise ValueError("Data files or embedding model are empty.")
        return self.llm_utils.search_rag(query, self.df, self.index, top_k=10)  # Ensure top_k is an integer

    def generate_llm_response(self, prompt):
        response = self.llm_utils.predict_llm(prompt)
        return response

    def is_recommendation_request(self, query):
        prompt = (
            "Tu es un LLM spécialisé dans la recommandation de sites touristiques de la région normande."
            "Ton but est de comprendre si l'utilisateur attend de toi une ou plusieurs recommandations touristiques."
            "Réponds simplement par 'oui' ou 'non'. Voici la requête :\n" + query
        )
        response = self.generate_llm_response(prompt).strip().lower()
        return response == "oui"

    def rag_agent(self, query):
        recommendation_needed = self.is_recommendation_request(query)

        if recommendation_needed:
            retrieved_events = self.llm_utils.search_rag(query, self.df, self.index)
            if not retrieved_events.empty:
                event_details = []
                for _, row in retrieved_events.iterrows():
                    event_detail = (
                        "Nom : " + str(row.get("NOM", "N/A")) + ", " +
                        "Description : " + str(row.get("DESCRIPTIF", "N/A")) + ", " +
                        "Lieu : " + str(row.get("COMMUNE", "N/A"))
                    )
                    event_details.append(event_detail)
                context = "\n".join(event_details)
                prompt = (
                    "Voici une demande utilisateur : \"" + query + "\"\n\n" +
                    "Voici la liste des sites touristiques pour répondre à cette demande. Ils ont été choisis avec soin et tu dois tous les inclure dans la réponse en précisant leur description et leur lieu :\n\n" +
                    context + "\n\n" +
                    "Génère une réponse pour l'utilisateur qui récapitule tous les sites mentionnés dans la liste dans l'ordre. Peu importe si tu considères qu'ils ne sont pas pertinents, fais le récapitulatif pour tous."
                )
                return self.generate_llm_response(prompt), retrieved_events[["TSID"]].values.tolist()
            else:
                prompt = (
                    "Voici une demande utilisateur : \"" + query + "\"\n\n" +
                    "Je n'ai trouvé aucun site touristique correspondant à cette demande. " +
                    "Rédige une réponse pour l'utilisateur qui explique cela clairement et propose des alternatives générales."
                )
                return self.generate_llm_response(prompt), []
        else:
            prompt = (
                "Voici une demande utilisateur : \"" + query + "\"\n\n" +
                "Cette demande ne semble pas nécessiter de recommandation spécifique. " +
                "Crée une réponse conversationnelle adaptée pour répondre de manière utile et engageante à cette demande."
            )
            return self.generate_llm_response(prompt), []

if __name__ == '__main__':
    logging.info("Initialization...")
    agent = RAGAgent(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        data_path='data/Lieux de visite.xlsx',
        prompts={}
    )
    query = "Quels sont les événements disponibles à Rouen ?"
    
    logging.info("Performing search...")
    try:
        results = agent.search(query)
        logging.info(results)
    except Exception as e:
        logging.error(f"Error: {e}")