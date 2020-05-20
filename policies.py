from abc import ABC, abstractmethod

class Inventory_Policy(ABC):
    @abstractmethod
    def get_order_quantity(self, states):
        pass
    
    
class Base_Stock_Policy(Inventory_Policy):
    
    def __init__(self, target_inventory):
        self.target_inventory = target_inventory
    
    def get_order_quantity(self, states):
        
        quantity = self.target_inventory - (states['inventory']  + states['on_order'] - states['unfilled_demand'])
        
        return max(0, quantity)
    