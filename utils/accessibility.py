import pandas as pd

class Accessibility:
    def __init__(self, acceslibre_file, datatour_file):
        self.df_acces = pd.read_csv(acceslibre_file)
        self.df_datatour = pd.read_excel(datatour_file)

    @staticmethod
    def remove_nan(value, mode="bool"):
        if value != value:  # Check for NaN
            if mode == "bool":
                return False
            if mode == "list":
                return []
            if mode == "str":
                return 'Non disponible'
        return value

    def return_accessibility(self, name):
        dic_mode = {
            "transport_station_presence": "bool",
            "stationnement_presence": "bool",
            "stationnement_pmr": "bool",
            'entree_ascenseur': "bool",
            'entree_pmr': "bool",
            'accueil_personnels': "str",
            'accueil_audiodescription_presence': "bool",
            'accueil_audiodescription': "list",
            'accueil_equipements_malentendants_presence': "bool",
            'sanitaires_presence': "bool",
            'sanitaires_adaptes': "bool",
            "labels": "list",
            'labels_familles_handicap': "list"
        }
        dic_res = {field: self.remove_nan(None, mode) for field, mode in dic_mode.items()}
        df_restrict = self.df_acces[self.df_acces.name == name]

        if len(df_restrict) < 1:
            return dic_res

        row = df_restrict.iloc[0]

        for field, mode in dic_mode.items():
            dic_res[field] = self.remove_nan(row[field], dic_mode[field])

        return dic_res