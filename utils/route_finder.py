import requests
from geopy.distance import geodesic
import polyline
import pandas as pd
import math

class RouteFinder:
    def __init__(self, osrm_url, gtfs_path, station_info_url, station_status_url):
        """
        Initialize the RouteFinder with OSRM URL and GTFS data files.

        :param osrm_url: OSRM API URL.
        :param gtfs_path: Path to the GTFS data files.
        """
        self.OSRM_URL = osrm_url
        self.GTFS_PATH = gtfs_path
        self.STATION_INFO_URL = station_info_url
        self.STATION_STATUS_URL = station_status_url
        self.routes_df = pd.read_csv(self.GTFS_PATH + "routes.txt")
        self.stops_df = pd.read_csv(self.GTFS_PATH + "stops.txt")
        self.stop_times_df = pd.read_csv(self.GTFS_PATH + "stop_times.txt")
        self.trips_df = pd.read_csv(self.GTFS_PATH + "trips.txt")
        self.shapes_df = pd.read_csv(self.GTFS_PATH + "shapes.txt")

    def get_route(self, start, end):
        """
        Get the route between two points using OSRM.

        :param start: Tuple of (latitude, longitude) for the start point.
        :param end: Tuple of (latitude, longitude) for the end point.
        :return: List of points representing the route.
        """
        start_str = f"{start[1]},{start[0]}"  # longitude, latitude
        end_str = f"{end[1]},{end[0]}"  # longitude, latitude
        response = requests.get(f"{self.OSRM_URL}/{start_str};{end_str}?overview=full")
        if response.status_code == 200:
            routes = response.json()["routes"]
            if routes:
                return polyline.decode(routes[0]["geometry"])
        return None

    def find_nearest_stop(self, point, max_distance=2000):
        """
        Find the nearest stop to a given point within a maximum distance.

        :param point: Tuple of (latitude, longitude) for the point.
        :param max_distance: Maximum distance in meters to search for stops.
        :return: Dictionary representing the nearest stop.
        """
        self.stops_df["distance"] = self.stops_df.apply(
            lambda row: geodesic(point, (row["stop_lat"], row["stop_lon"])).meters, axis=1
        )
        nearby_stops = self.stops_df[self.stops_df["distance"] <= max_distance]
        if nearby_stops.empty:
            return None
        nearest_stop = nearby_stops.loc[nearby_stops["distance"].idxmin()]
        return nearest_stop.to_dict()

    def find_nearest_stop_with_shared_routes(self, point, reference_stop_id, max_distance=2000):
        """
        Find the nearest stop to the point that shares at least one route with the reference stop.
        """
        # Get all trips passing through the reference stop
        reference_trips = self.stop_times_df[self.stop_times_df["stop_id"] == reference_stop_id]["trip_id"].unique()

        # Get all route IDs for the reference stop
        reference_routes = self.trips_df[self.trips_df["trip_id"].isin(reference_trips)]["route_id"].unique()

        # Filter stops that belong to these routes
        candidate_trips = self.trips_df[self.trips_df["route_id"].isin(reference_routes)]["trip_id"].unique()
        candidate_stops = self.stop_times_df[self.stop_times_df["trip_id"].isin(candidate_trips)]["stop_id"].unique()

        # Filter stops_df for candidates
        filtered_stops = self.stops_df[self.stops_df["stop_id"].isin(candidate_stops)]

        # Compute distances
        filtered_stops["distance"] = filtered_stops.apply(
            lambda row: geodesic(point, (row["stop_lat"], row["stop_lon"])).meters, axis=1
        )

        # Find the nearest stop within the max distance
        nearby_stops = filtered_stops[filtered_stops["distance"] <= max_distance]
        if nearby_stops.empty:
            return None
        nearest_stop = nearby_stops.loc[nearby_stops["distance"].idxmin()]
        return nearest_stop.to_dict()

    def find_trips_between_stops(self, start_stop_id, end_stop_id):
        """
        Find trips that connect the start and end stops.

        :param start_stop_id: ID of the start stop.
        :param end_stop_id: ID of the end stop.
        :return: Tuple of (connecting_trips, trips_from_start, trips_to_end).
        """
        start_stops = self.stop_times_df[self.stop_times_df["stop_id"] == start_stop_id][["trip_id", "stop_sequence"]]
        end_stops = self.stop_times_df[self.stop_times_df["stop_id"] == end_stop_id][["trip_id", "stop_sequence"]]

        valid_trips = pd.merge(start_stops, end_stops, on="trip_id", suffixes=("_start", "_end"))

        connecting_trips = valid_trips["trip_id"].tolist()
        trips_from_start = []
        trips_to_end = []

        if not connecting_trips:
            trips_from_start = start_stops["trip_id"].unique()
            trips_to_end = end_stops["trip_id"].unique()

        return connecting_trips, trips_from_start, trips_to_end

    def get_bus_route(self, start_point, end_point):
        """
        Get the bus route between two points.

        :param start_point: Tuple of (latitude, longitude) for the start point.
        :param end_point: Tuple of (latitude, longitude) for the end point.
        :return: Dictionary representing the bus route.
        """
        start_stop = self.find_nearest_stop(start_point)
        if not start_stop:
            return {"error": "No nearby stop found for the start point."}

        end_stop = self.find_nearest_stop_with_shared_routes(
            end_point, start_stop["stop_id"]
        )
        if not end_stop:
            return {"error": "No nearby stop with shared routes found for the end point."}

        walk_to_start = self.get_route(start_point, (start_stop["stop_lat"], start_stop["stop_lon"]))
        walk_to_end = self.get_route((end_stop["stop_lat"], end_stop["stop_lon"]), end_point)

        connecting_trips, trips_from_start, trips_to_end = self.find_trips_between_stops(
            start_stop["stop_id"], end_stop["stop_id"]
        )
        print(start_stop)
        print(end_stop)
        if not connecting_trips:
            return {"error": "No direct bus route connects the selected points."}

        selected_trip_id = connecting_trips[0]
        trip_info = self.trips_df[self.trips_df["trip_id"] == selected_trip_id].iloc[0]

        shape_id = trip_info["shape_id"]
        shape_points = self.shapes_df[self.shapes_df["shape_id"] == shape_id][["shape_pt_lat", "shape_pt_lon"]].values.tolist()

        start_index = self.find_closest_index(start_stop["stop_lat"], start_stop["stop_lon"], shape_points)
        end_index = self.find_closest_index(end_stop["stop_lat"], end_stop["stop_lon"], shape_points)
        sliced_shape = shape_points[start_index:end_index + 1] if start_index < end_index else shape_points[end_index:start_index + 1][::-1]

        # Calculate total distance in km
        total_distance_km = sum(
            geodesic(sliced_shape[i], sliced_shape[i+1]).km for i in range(len(sliced_shape) - 1)
        )

        # Determine the transport mode for this route
        transport_mode = self.get_transport_mode(trip_info["route_id"])

        # Calculate CO2 emissions for the public transport route
        co2_emissions = self.calculate_co2(total_distance_km, mode=transport_mode)

        return {
            "start_stop": {
                "name": start_stop["stop_name"],
                "latitude": start_stop["stop_lat"],
                "longitude": start_stop["stop_lon"]
            },
            "end_stop": {
                "name": end_stop["stop_name"],
                "latitude": end_stop["stop_lat"],
                "longitude": end_stop["stop_lon"]
            },
            "routes": {
                "walk_to_start": walk_to_start,
                "public_transport_route": sliced_shape,
                "walk_to_end": walk_to_end
            },
        "co2_emissions": co2_emissions,
        "transport_mode": transport_mode
        }

    @staticmethod
    def find_closest_index(stop_lat, stop_lon, shape_points):
        """
        Find the closest index in the shape points to a given stop.

        :param stop_lat: Latitude of the stop.
        :param stop_lon: Longitude of the stop.
        :param shape_points: List of points representing the shape.
        :return: Index of the closest point in the shape.
        """
        return min(
            range(len(shape_points)),
            key=lambda i: geodesic((stop_lat, stop_lon), shape_points[i]).meters
        )

    def get_station_information(self):
        response = requests.get(self.STATION_INFO_URL)
        if response.status_code == 200:
            return response.json()['data']['stations']
        else:
            raise Exception("Erreur lors de la récupération des informations des stations.")

    def get_station_status(self):
        response = requests.get(self.STATION_STATUS_URL)
        if response.status_code == 200:
            return response.json()['data']['stations']
        else:
            raise Exception("Erreur lors de la récupération du statut des stations.")

    def find_nearest_station(self, location, stations, available_bikes=True):
        nearest_station = None
        min_distance = float('inf')
        for station in stations:
            station_location = (station['latitude'], station['longitude'])
            distance = geodesic(location, station_location).meters
            if available_bikes and station['available_bikes'] == 0:
                continue
            if not available_bikes and station['available_stands'] == 0:
                continue
            if distance < min_distance:
                min_distance = distance
                nearest_station = station
        return nearest_station

    def get_vls_stations(self):
        station_info = self.get_station_information()
        station_status = self.get_station_status()
        status_dict = {station['station_id']: station for station in station_status}
        combined_data = []
        for station in station_info:
            station_id = station['station_id']
            if station_id in status_dict:
                combined_data.append({
                    "station_id": station_id,
                    "name": station['name'],
                    "latitude": station['lat'],
                    "longitude": station['lon'],
                    "available_bikes": status_dict[station_id]['num_bikes_available'],
                    "available_stands": status_dict[station_id]['num_docks_available']
                })
        return combined_data

    def route_coin(self, km, mode):
        bike_coins_per_km = 150
        bus_coins_per_km = 100
        if mode=="bike":
            coins_earned = int(km * bike_coins_per_km)
        elif mode=="bus":
            coins_earned = int(km * bus_coins_per_km)
        else:
            coins_earned = 50 * km
        return coins_earned

    def clean_dict(self, data):
        """
        Recursively clean a dictionary to ensure all values are JSON-serializable.
        Replace nan with None and remove unnecessary fields.
        """
        if isinstance(data, dict):
            return {k: self.clean_dict(v) for k, v in data.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
        elif isinstance(data, list):
            return [self.clean_dict(item) for item in data]
        elif isinstance(data, float) and math.isnan(data):
            return None
        return data

    def calculate_co2(self, distance_km, mode):
        """
        Calculate CO2 emissions based on the distance and transportation mode.
        :param distance_km: Distance in kilometers
        :param mode: Transportation mode ("bike", "bus", "tram", "metro")
        :return: CO2 emissions in grams
        """
        # https://www.reseau-astuce.fr/fr/aide-et-accessibilite/54/bilan-carbone/15
        co2_per_km = {
            "bike": 0,    # Bicycle is emission-free
            "bus": 55.9,    # Bus average emission: 68 g CO2/km
            "tram": 1.2,   # Tram average emission: 14 g CO2/km
            "metro": 1.42    # Metro average emission: 6 g CO2/km
        }
        return distance_km * co2_per_km.get(mode, 0)  # Default to 0 if mode not listed

    def get_transport_mode(self, route_id):
        """
        Determine the transport mode based on the route_id.
        :param route_id: The route_id from trips_df or stop_times_df
        :return: Transport mode ("bus", "tram", "metro")
        """
        route_info = self.routes_df[self.routes_df["route_id"] == route_id].iloc[0]
        route_short_name = route_info["route_short_name"].lower()

        if "métro" in route_short_name:
            return "metro"
        elif route_short_name.startswith("t") and route_short_name[1:].isdigit():
            return "tram"
        else:
            return "bus"
