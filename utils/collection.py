import pandas as pd
import yaml


class CollectionUtils:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.users_file = self.config['collection']["database"]["users_file"]
        self.badges_info = self.config['collection']["badges"]
        self.tiers = self.config['collection']["tiers"]
        
    def load_users(self):
        return pd.read_csv(self.users_file)

    def save_users(self, users):
        users.to_csv(self.users_file, index=False)

    def calculate_user_details(self, user_id):
        users = self.load_users()
        user_data = users[users["user_id"] == user_id]
        if user_data.empty:
            return None

        user = user_data.iloc[0]
        coins = int(user["coins"])
        experience = int(user["experience"])
        badges = user["badges"].split(";") if pd.notna(user["badges"]) else []
        level = experience // 1000
        coins_for_next_level = (level + 1) * 1000 - experience

        badges_with_descriptions = [
            {"name": badge, "description": self.badges_info.get(badge, "Description non disponible")}
            for badge in badges
        ]

        return {
            "coins": coins,
            "level": int(level),
            "coins_for_next_level": coins_for_next_level,
            "badges": badges_with_descriptions
        }

    def get_tiers(self, user_id):
        users = self.load_users()
        user_data = users[users["user_id"] == user_id]
        if user_data.empty:
            return None

        user = user_data.iloc[0]
        coins = int(user["coins"])

        tiers_with_access = []
        highest_accessible_tier = 0

        for tier in self.tiers:
            max_nc = tier["max_nc"]
            accessible = tier["min_nc"] <= coins and (max_nc is None or coins < max_nc)
            if accessible:
                highest_accessible_tier = tier["tier"]
            tiers_with_access.append({
                **tier,
                "max_nc": "∞" if max_nc is None else max_nc,
                "accessible": accessible
            })

        for tier in tiers_with_access:
            if tier["tier"] <= highest_accessible_tier:
                tier["accessible"] = True

        return {
            "current_coins": coins,
            "tiers": tiers_with_access
        }

    def purchase_tier_item(self, user_id, item_cost):
        users = self.load_users()
        user_index = users.index[users["user_id"] == user_id].tolist()
        if not user_index:
            return None

        user_index = user_index[0]
        user = users.iloc[user_index]

        current_coins = int(user["coins"])
        if current_coins < item_cost:
            return None

        users.at[user_index, "coins"] = current_coins - item_cost
        self.save_users(users)

        return int(users.at[user_index, "coins"])
