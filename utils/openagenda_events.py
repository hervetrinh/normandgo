import requests
import pandas as pd
from datetime import datetime
import logging


class OpenAgendaEvents:
    def __init__(self, api_key, agenda_uids):
        """
        Initializes the OpenAgendaEvents class with the API key and the UIDs of the agendas.
        
        :param api_key: Public API key for OpenAgenda.
        :param agenda_uids: List of agenda UIDs.
        """
        self.api_key = api_key
        self.agenda_uids = agenda_uids

    def fetch_events(self):
        """
        Retrieves upcoming events for a list of agendas via the OpenAgenda API.

        :return: DataFrame containing information about the events.
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
                    logging.error(f"Error while retrieving events for UID {uid}: {response.status_code}")
                    break

        return pd.json_normalize(events_list)