import json
import random

def check_local_inventory(item_id: str) -> str:
    """
    Mocks a backend ERP/Inventory SQL database check.
    Includes a deterministic override for executive demos to guarantee 
    a zero-stock scenario and trigger the HA/DR RAG fallback.
    """

    
    DEMO_STOCKOUT_ID = "14325" 
    
    if str(item_id) == DEMO_STOCKOUT_ID:
        # Force a total stock of 0 across all locations to trigger the fallback
        inventory_data = {
            "Downtown": 0,
            "Uptown": 0,
            "Suburbs": 0
        }
        
        return json.dumps({
            "status": "success",
            "stock_by_location": inventory_data,
            "instruction": "If total stock is 0, apologize and present the fallback_recommendations provided in the context."
        })
    # ---------------------------------------------------------

    # Default Behavior: Random inventory for all other items
    inventory_data = {
        "Downtown": random.choice([0, 1, 5, 12]),
        "Uptown": random.choice([0, 0, 4, 2]),
        "Suburbs": random.choice([0, 2, 8, 10])
    }

    return json.dumps({
        "status": "success",
        "stock_by_location": inventory_data,
        "instruction": "Present the stock levels to the user."
    })