import requests
import pandas as pd
from datetime import datetime
import logging
from config import config 


class OpenAgendaEvents:
    def __init__(self, api_key, agenda_uids):
        """
        Initialise la classe OpenAgendaEvents avec l'API key et les UIDs des agendas.
        
        :param api_key: Clé API publique d'OpenAgenda.
        :param agenda_uids: Liste des UIDs des agendas.
        """
        self.api_key = api_key
        self.agenda_uids = agenda_uids

    def fetch_events(self):
        """
        Récupère les événements à venir pour une liste d'agendas via l'API OpenAgenda.

        :return: DataFrame contenant les informations sur les événements.
        """
        current_date = datetime.now().isoformat()
        events_list = []

        for uid in self.agenda_uids:
            events_url = f'https://api.openagenda.com/v2/agendas/{uid}/events'
            params = {
                'key': self.api_key,
                'timings[gte]': current_date,
                'size': 100,
                'detailed': 1
            }

            while True:
                response = requests.get(events_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    events = data.get('events', [])
                    events_list.extend(events)
                    after = data.get('after')
                    if after:
                        params['after'] = after
                    else:
                        break
                else:
                    logging.error(f"Erreur lors de la récupération des événements pour UID {uid}: {response.status_code}")
                    break

        return pd.json_normalize(events_list)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    api_key = config['openagenda']['api_key']
    agenda_uids = config['openagenda']['agenda_uids']

    logging.info("Initialisation de OpenAgendaEvents...")
    openagenda = OpenAgendaEvents(api_key, agenda_uids)

    logging.info("Récupération des événements...")
    events_df = openagenda.fetch_events()

    cols = ['title.fr', 'description.fr', 'location.longitude', 'location.latitude', 'location.address', 'firstTiming.begin', 'lastTiming.end']

    if not events_df.empty:
        logging.info("Événements récupérés avec succès !")
        logging.info("Filtrage des événements pour le département 76...")
        events_df = events_df[events_df['location.postalCode'].str[:2] == '76'][cols]
        events_df.drop_duplicates(subset=['title.fr', 'description.fr']).to_csv("data/agenda.csv", sep=";")
    else:
        logging.info("Aucun événement trouvé.")
