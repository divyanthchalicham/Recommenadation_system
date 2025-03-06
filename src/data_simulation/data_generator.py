
import json
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
from faker import Faker

from src.config import (
    NUM_USERS, NUM_PRODUCTS, NUM_CATEGORIES, NUM_BRANDS,
    SIMULATION_DAYS, AVG_ACTIONS_PER_USER_PER_DAY
)

fake = Faker()

class DataGenerator:
    """Generate synthetic e-commerce data."""
    
    def __init__(self):
        self.categories = [
            "Electronics", "Clothing", "Home & Kitchen", "Books", 
            "Sports", "Beauty", "Toys", "Grocery", "Automotive", "Health"
        ][:NUM_CATEGORIES]
        
        self.brands = [fake.company() for _ in range(NUM_BRANDS)]
        self.users = [f"U{str(uuid.uuid4())[:8]}" for _ in range(NUM_USERS)]
        
        self.products = []
        self.user_activities = []
    
    def generate_products(self):
        """Generate product catalog."""
        products = []
        
        for _ in range(NUM_PRODUCTS):
            product_id = f"P{str(uuid.uuid4())[:8]}"
            category = random.choice(self.categories)
            brand = random.choice(self.brands)
            price = round(random.uniform(9.99, 999.99), 2)
            
            product = {
                "product_id": product_id,
                "product_name": f"{fake.word().capitalize()} {brand.split()[0]} {random.choice(['Device', 'Product', 'Item'])}",
                "category": category,
                "brand": brand,
                "price": price,
                "description": fake.paragraph(nb_sentences=3),
                "available": random.choice([True, True, True, False])  # 75% chance of being available
            }
            
            products.append(product)
        
        self.products = products
        return products
    
    def generate_user_activities(self):
        """Generate user activity data."""
        activities = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=SIMULATION_DAYS)
        
        
        user_preferences = {}
        for user_id in self.users:
            user_preferences[user_id] = {
                "preferred_categories": random.sample(self.categories, k=min(3, len(self.categories))),
                "preferred_brands": random.sample(self.brands, k=min(3, len(self.brands))),
                "price_sensitivity": random.uniform(0.1, 0.9)  # 0.1: not sensitive, 0.9: very sensitive
            }
        
      
        for day in range(SIMULATION_DAYS):
            current_date = start_date + timedelta(days=day)
            
            for user_id in self.users:
              
                num_actions = np.random.poisson(AVG_ACTIONS_PER_USER_PER_DAY)
                
                
                prefs = user_preferences[user_id]
                
            
                preferred_products = [
                    p for p in self.products
                    if p["category"] in prefs["preferred_categories"] or 
                    p["brand"] in prefs["preferred_brands"]
                ]
                
                if not preferred_products:
                    preferred_products = self.products
                
                
                for _ in range(num_actions):
                    
                    if random.random() < 0.7:
                        product = random.choice(preferred_products)
                    else:
                        product = random.choice(self.products)
                    
                    
                    price_factor = 1 - (product["price"] / 1000 * prefs["price_sensitivity"])
                    buy_probability = 0.2 * price_factor
                    
                    action_type = "BUY" if random.random() < buy_probability else "VIEW"
                    
                    
                    hour = random.randint(8, 22)  
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    timestamp = current_date.replace(hour=hour, minute=minute, second=second)
                    
                    activity = {
                        "user_id": user_id,
                        "action_type": action_type,
                        "product_id": product["product_id"],
                        "timestamp": timestamp.isoformat(),
                        "category": product["category"],
                        "price": product["price"]
                    }
                    
                    activities.append(activity)
        
        
        activities.sort(key=lambda x: x["timestamp"])
        self.user_activities = activities
        return activities
    
    def save_to_files(self, output_dir="data"):
        """Save generated data to JSON files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        
        with open(f"{output_dir}/product_catalog.json", "w") as f:
            json.dump(self.products, f, indent=2)
        
        
        with open(f"{output_dir}/user_activity.json", "w") as f:
            json.dump(self.user_activities, f, indent=2)
        
        print(f"Generated {len(self.products)} products and {len(self.user_activities)} user activities.")
        print(f"Data saved to {output_dir}/product_catalog.json and {output_dir}/user_activity.json")

def main():
    """Generate and save synthetic data."""
    generator = DataGenerator()
    generator.generate_products()
    generator.generate_user_activities()
    generator.save_to_files()

if __name__ == "__main__":
    main()