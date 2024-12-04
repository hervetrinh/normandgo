import pandas as pd

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.lieux_visite = None

    def load_data(self):
        """Load and preprocess the data."""
        self.lieux_visite = pd.read_excel(self.data_path)
        self.lieux_visite = self.lieux_visite[['TSID', 'NOM', 'LONGITUDE', 'LATITUDE', 'DESCRIPTIF', 
                                                'PHOTO', 'ADRESSE1', 'ADRESSE2', 'CODEPOSTAL', 'COMMUNE']].drop_duplicates()

    def concatenate_address(self, adresse1, adresse2, codepostal, commune):
        """Concatenate address fields into a single address field."""
        adresse1 = adresse1.fillna('').astype(str)
        adresse2 = adresse2.fillna('').astype(str)
        codepostal = codepostal.fillna('').astype(str)
        commune = commune.fillna('').astype(str)

        addr = adresse1 + " " + adresse2 + " " + codepostal + " " + commune
        addr = addr.str.replace(r'\s+', ' ', regex=True).str.strip()
        return addr

    def process_addresses(self):
        """Process addresses in the lieux_visite DataFrame."""
        self.lieux_visite['ADDR'] = self.concatenate_address(
            self.lieux_visite['ADRESSE1'],
            self.lieux_visite['ADRESSE2'],
            self.lieux_visite['CODEPOSTAL'],
            self.lieux_visite['COMMUNE']
        )
        return self.lieux_visite