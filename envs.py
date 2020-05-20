import numpy as np
from demands import DemandGenerator
from policies import Base_Stock_Policy
from supplychain import Node, Arc, Supply_chain_network


class InventoryManagementEnv():

    def __init__(self, supply_chain_network):
        self.scn = supply_chain_network
        self.terminal = False

    def reset(self):
        self.terminal = False
        self.scn.reset()
        self.period = 0

        self.scn.before_action(self.period)

        states = self.scn.get_states(self.scn.player, self.period)
        #         states['period'] = self.period

        return states

    def step(self, quantity, verbose=True):

        self.scn.player_action(self.period, quantity)
        self.scn.after_action(self.period)

        cost = self.scn.cost_keeping()

        self.period += 1

        if self.period < self.scn.max_period:
            self.scn.before_action(self.period)
        else:
            self.terminal = True

        states = self.scn.get_states(self.scn.player, self.period)
        #         states['period'] = self.period

        return states, cost, self.terminal


def build_beer_game(player='wholesaler', demand_type='classic_beer_game'):

    if demand_type == 'classic_beer_game':
        demand_gen = DemandGenerator('classic_beer_game')
    elif demand_type == 'deterministic_random':
        demand_gen = DemandGenerator((8 + 2 * np.random.randn(35)).astype(int))
    else:
        demand_gen = DemandGenerator('normal', mean=8, sd=2, size=35)

    bs_32 = Base_Stock_Policy(32)
    bs_24 = Base_Stock_Policy(24)

    demand_source = Node(name='demand_source', demand_source=True, demands=demand_gen)
    retailer = Node(name='retailer', policy=bs_32, initial_previous_orders=[4, 4, 4, 4])
    wholesaler = Node(name='wholesaler', policy=bs_32, initial_previous_orders=[4, 4, 4, 4])
    distributor = Node(name='distributor', policy=bs_32, initial_previous_orders=[4, 4, 4, 4])
    manufacturer = Node(name='manufacturer', policy=bs_24, initial_previous_orders=[4, 4, 4, 4])
    supply_source = Node(name='supply_source', supply_source=True)

    nodes = [demand_source, retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [Arc('supply_source', 'manufacturer', 1, 2, initial_shipments=[(4, 1), (4, 2)], initial_SOs=[(4, 0, 1)]),
            Arc('manufacturer', 'distributor', 2, 2, initial_shipments=[(4, 1), (4, 2)],
                initial_SOs=[(4, 0, 1), (4, 0, 2)]),
            Arc('distributor', 'wholesaler', 2, 2, initial_shipments=[(4, 1), (4, 2)],
                initial_SOs=[(4, 0, 1), (4, 0, 2)]),
            Arc('wholesaler', 'retailer', 2, 2, initial_shipments=[(4, 1), (4, 2)], initial_SOs=[(4, 0, 1), (4, 0, 2)]),
            Arc('retailer', 'demand_source', 0, 0)]

    scn = Supply_chain_network(nodes=nodes, arcs=arcs, player=player)
    scn.max_period = 35

    return InventoryManagementEnv(scn)
