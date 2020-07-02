import numpy as np
from demands import DemandGenerator
from policies import BaseStockPolicy
from supplychain import Node, Arc, SupplyChainNetwork


class InventoryManagementEnv:

    def __init__(self, supply_chain_network):
        self.scn = supply_chain_network
        self.terminal = False

    def reset(self):
        self.terminal = False
        self.scn.reset()
        self.period = 0

        self.scn.before_action(self.period)

        states = self.scn.get_states(self.scn.player, self.period)
        states['period'] = self.period

        return states

    def step(self, quantity, verbose=True):
        """
        return:
            a tuple of stats (dict), cost (float) and terminal (bool)
        """

        if self.terminal:
            raise ValueError("Cannot take action when the state is terminal.")

        self.scn.player_action(self.period, quantity)
        self.scn.after_action(self.period)

        cost = self.scn.cost_keeping()

        self.period += 1

        if self.period < self.scn.max_period:
            self.scn.before_action(self.period)
        else:
            self.terminal = True

        states = self.scn.get_states(self.scn.player, self.period)
        states['period'] = self.period

        return states, cost, self.terminal


def build_beer_game(player='wholesaler', demand_type='classic_beer_game', max_period=35):
    if demand_type == 'classic_beer_game':
        demand_generator = DemandGenerator('classic_beer_game')
    elif demand_type == 'deterministic_random':
        demand_generator = DemandGenerator((8 + 2 * np.random.randn(max_period)).astype(int))
    else:
        demand_generator = DemandGenerator('normal', mean=8, sd=2, size=max_period)

    bs_32 = BaseStockPolicy(32)
    bs_24 = BaseStockPolicy(24)

    demand_source = Node(name='demand_source', demand_source=True, demands=demand_generator)
    retailer = Node(name='retailer', initial_inventory=12, policy=bs_32)
    wholesaler = Node(name='wholesaler', initial_inventory=12, policy=bs_32)
    distributor = Node(name='distributor', initial_inventory=12, policy=bs_32)
    manufacturer = Node(name='manufacturer', initial_inventory=12, policy=bs_24)
    supply_source = Node(name='supply_source', supply_source=True)
    nodes = [demand_source, retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [Arc('supply_source', 'manufacturer', 1, 2, initial_shipments=[(4, 1), (4, 2)],
                initial_sales_orders=[(4, 0, 1)], initial_previous_orders=[4, 4, 4, 4]),
            Arc('manufacturer', 'distributor', 2, 2, initial_shipments=[(4, 1), (4, 2)],
                initial_sales_orders=[(4, 0, 1), (4, 0, 2)], initial_previous_orders=[4, 4, 4, 4]),
            Arc('distributor', 'wholesaler', 2, 2, initial_shipments=[(4, 1), (4, 2)],
                initial_sales_orders=[(4, 0, 1), (4, 0, 2)], initial_previous_orders=[4, 4, 4, 4]),
            Arc('wholesaler', 'retailer', 2, 2, initial_shipments=[(4, 1), (4, 2)],
                initial_sales_orders=[(4, 0, 1), (4, 0, 2)], initial_previous_orders=[4, 4, 4, 4]),
            Arc('retailer', 'demand_source', 0, 0)]

    scn = SupplyChainNetwork(nodes=nodes, arcs=arcs, player=player)
    scn.max_period = max_period

    return InventoryManagementEnv(scn)


def build_newsvendor(c=1, p=4, mu=100, sigma=30):
    """
    Page 13, Porteus, E.L.(2002). Foundations of stochastic inventory theory. Stanford University Press.
    Args:
        c: unit cost, default 1
        p: sales_price, unit sales price 4
        mu: mean demand, default 100
        sigma: standard deviation of demand, default 30

    The optimal stockout probability zeta is 0.25 when the default parameters are used.
    """

    demand_generator = DemandGenerator(demands_pattern='normal', mean=mu, sd=sigma, size=1)

    demand_source = Node(name='demand_source', demand_source=True, demands=demand_generator)
    newsvendor = Node(name='newsvendor', holding_cost=c, stockout_cost=p-c)
    supply_source = Node(name='supply_source', supply_source=True)
    nodes = [demand_source, newsvendor, supply_source]

    arcs = [Arc(source='supply_source', target='newsvendor', information_leadtime=0, shipment_leadtime=0),
            Arc(source='newsvendor', target='demand_source', information_leadtime=0, shipment_leadtime=0)]

    scn = SupplyChainNetwork(nodes=nodes, arcs=arcs, player='newsvendor')
    scn.max_period = 1
    return InventoryManagementEnv(scn)
