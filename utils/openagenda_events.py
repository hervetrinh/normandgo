import requests
import pandas as pd
from datetime import datetime
import logging
import yaml


class OpenAgendaEvents:
    def __init__(self, api_key, agenda_uids):
        """
        Initialize the OpenAgendaEvents class with the API key and agenda UIDs.
        
        :param api_key: OpenAgenda public API key.
        :param agenda_uids: List of agenda UIDs.
        """
        self.api_key = api_key
        self.agenda_uids = agenda_uids

    def fetch_events(self):
        """
        Retrieve upcoming events for a list of agendas via the OpenAgenda API.

        :return: DataFrame containing event information.
        """
        current_date = datetime.now().isoformat()
        events_list = []

        for uid in self.agenda_uids:
            events_url = f'https://api.openagenda.com/v2/agendas/{uid}/events'
            params = {
                'key': self.api_key,
                'timings[gte]': current_date,  # Events from today
                'size': 100,  # Maximum number of events to retrieve per call
                'detailed': 1  # Get detailed information for each event
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
                    logging.error(f"Error retrieving events for agenda UID {uid}: {response.status_code}")
                    break

        df_openag = pd.json_normalize(events_list)
        return df_openag


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    api_key = config['openagenda']['api_key']
    agenda_uids = config['openagenda']['agenda_uids']

    logging.info("Initializing OpenAgendaEvents...")
    openagenda = OpenAgendaEvents(api_key, agenda_uids)

    logging.info("Fetching events...")
    events_df = openagenda.fetch_events()

    cols = ['title.fr', 'description.fr', 'location.longitude', 'location.latitude', 'location.address', 'firstTiming.begin', 'lastTiming.end']

    if not events_df.empty:
        logging.info("Events fetched successfully!")
        logging.info("Dep 76 only for the MVP")
        events_df[events_df['location.postalCode'].str[:2] == '76'][cols].drop_duplicates(subset=['title.fr', 'description.fr']).to_csv("data/agenda.csv", sep=";")
    else:
        logging.info("No events found.")
        