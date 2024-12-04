import overpy


class DrinkingWater:
    def __init__(self, overpass_query):
        """
        Initialize the DrinkingWater class with the Overpass query.
        
        :param overpass_query: Overpass query to fetch drinking water locations.
        """
        self.api = overpy.Overpass()
        self.query = overpass_query

    def fetch_drinking_water(self, verbose=False):
        """
        Fetch public drinking water fountains in Normandy using Overpass API.

        :param verbose: If True, print details of the found fountains.
        :return: GeoJSON data of the fountains.
        """
        result = self.api.query(self.query)

        if verbose:
            print("Public fountains found:")
            for node in result.nodes:
                print(f"ID: {node.id}, Latitude: {node.lat}, Longitude: {node.lon}, Tags: {node.tags}")

        features = []
        for node in result.nodes:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(node.lon), float(node.lat)],  # Convert to float
                },
                "properties": node.tags,
            }
            features.append(feature)

        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
        }

        return geojson_data