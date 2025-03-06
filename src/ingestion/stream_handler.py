
import json
import time
import threading
import random
from datetime import datetime

from src.database.mongo_handler import insert_activity, insert_products

class StreamingService:
    """Simulates streaming user activity data."""
    
    def __init__(self, activity_file="data/user_activity.json", product_file="data/product_catalog.json"):
        self.activity_file = activity_file
        self.product_file = product_file
        self.activities = []
        self.streaming = False
        self.stream_thread = None
    
    def load_data(self):
    
        try:
            with open(self.product_file, "r") as f:
                products = json.load(f)
                insert_products(products)
                print(f"Loaded {len(products)} products.")
            
            with open(self.activity_file, "r") as f:
                self.activities = json.load(f)
                print(f"Loaded {len(self.activities)} user activities.")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _stream_activities(self, speed_factor=10):
 
        if not self.activities:
            print("No activities to stream.")
            return
        
        print(f"Starting streaming with speed factor {speed_factor}x...")
        
     
        self.activities.sort(key=lambda x: x["timestamp"])
        
        prev_timestamp = None
        
        for activity in self.activities:
            if not self.streaming:
                break
            
            curr_timestamp = datetime.fromisoformat(activity["timestamp"])
            
    
            if prev_timestamp:
                time_diff = (curr_timestamp - prev_timestamp).total_seconds()
                sleep_time = time_diff / speed_factor
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            
            activity_copy = activity.copy()
            activity_copy["ingestion_timestamp"] = datetime.now().isoformat()
            insert_activity(activity_copy)
            
       
            if random.random() < 0.01:  
                print(f"Ingested: {activity_copy}")
            
            prev_timestamp = curr_timestamp
        
        self.streaming = False
        print("Streaming completed.")
    
    def start_streaming(self, speed_factor=10):
       
        if self.streaming:
            print("Streaming is already running.")
            return False
        
        if not self.activities:
            if not self.load_data():
                return False
        
        self.streaming = True
        self.stream_thread = threading.Thread(
            target=self._stream_activities,
            args=(speed_factor,)
        )
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        return True
    
    def stop_streaming(self):

        if not self.streaming:
            print("Streaming is not running.")
            return False
        
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        
        print("Streaming stopped.")
        return True

def main():
   
    service = StreamingService()
    service.load_data()
    service.start_streaming(speed_factor=100)  
    
    try:
 
        while service.streaming:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping streaming...")
        service.stop_streaming()

if __name__ == "__main__":
    main()