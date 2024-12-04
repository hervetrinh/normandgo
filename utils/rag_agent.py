import logging
import time
from llm_utils import LLMUtils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class RAGAgent:
    def __init__(self, embedding_model_name=None, data_path=None, prompts={}):
        self.llm_utils = LLMUtils(
            embedding_model_name=embedding_model_name,
            data_path=data_path,
            prompts=prompts,
        )
        self.df, self.index, self.embedding_model = None, None, None
        if data_path and embedding_model_name:
            logging.info("Preloading data and model...")
            start_time = time.time()
            self._load_resources()
            elapsed_time = time.time() - start_time
            logging.info(f"Data and model preloaded in {elapsed_time:.2f} seconds.")

    def _load_resources(self):
        """Load data and model resources efficiently."""
        self.df, self.index, self.embedding_model = self.llm_utils.load_files()

    def _ensure_resources_loaded(self):
        """Ensure the resources are loaded before proceeding."""
        if self.df is None or self.index is None:
            raise ValueError("Data files or embedding model are not loaded.")
        if self.df.empty or self.index.ntotal == 0:
            raise ValueError("Data files or embedding model are empty.")

    def unified_query(self, query):
        """Handle the query entirely within a single LLM call."""
        self._ensure_resources_loaded()

        # Step 1: Perform the search in the local data
        logging.info("Starting search in local data...")
        start_time = time.time()
        retrieved_events = self.llm_utils.search_rag(query, self.df, self.index, top_k=10)
        elapsed_time = time.time() - start_time
        logging.info(f"Search completed in {elapsed_time:.2f} seconds.")

        if not retrieved_events.empty:
            context = "\n".join(
                f"Nom : {row.get('NOM', 'N/A')}, "
                f"Description : {row.get('DESCRIPTIF', 'N/A')}, "
                f"Lieu : {row.get('COMMUNE', 'N/A')}"
                for _, row in retrieved_events.iterrows()
            )
        else:
            context = "Aucun site touristique correspondant à cette demande n'a été trouvé."

        # Step 2: Create a comprehensive prompt
        logging.info("Creating LLM prompt...")
        start_time = time.time()
        prompt = (
            f"Voici une demande utilisateur : \"{query}\"\n\n"
            "Voici les informations contextuelles pour répondre à cette demande :\n\n"
            f"{context}\n\n"
            "Génère une réponse complète et conversationnelle. Si des lieux sont listés, "
            "récapitule-les de manière engageante. Sinon, explique l'absence de résultats "
            "et propose des alternatives générales."
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Prompt creation completed in {elapsed_time:.2f} seconds.")

        # Step 3: Generate the response using the LLM
        logging.info("Generating LLM response...")
        start_time = time.time()
        response = self.llm_utils.predict_llm(prompt)
        elapsed_time = time.time() - start_time
        logging.info(f"LLM response generated in {elapsed_time:.2f} seconds.")

        tsids = retrieved_events[["TSID"]].values.tolist() if not retrieved_events.empty else []
        return response, tsids

if __name__ == "__main__":
    logging.info("Initialization...")
    start_time = time.time()
    agent = RAGAgent(
        embedding_model_name="models/all-MiniLM-L6-v2",
        data_path="data/Lieux de visite.xlsx",
        prompts={},
    )
    elapsed_time = time.time() - start_time
    logging.info(f"Agent initialized in {elapsed_time:.2f} seconds.")

    query = "Quels sont les événements disponibles à Rouen ?"
    logging.info("Generating unified response...")
    try:
        start_time = time.time()
        response, tsids = agent.unified_query(query)
        elapsed_time = time.time() - start_time
        logging.info(f"Unified query processed in {elapsed_time:.2f} seconds.")
        logging.info(f"Response: {response}")
        logging.info(f"TSIDs: {tsids}")
    except Exception as e:
        logging.error(f"Error: {e}")