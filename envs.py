import numpy as np
import time
from .demands import DemandGenerator
from .policies import BaseStockPolicy
from .supplychain import Node, Arc, SupplyChainNetwork


class InventoryManagementEnv:

    def __init__(self, supply_chain_network, visible_states=None, return_dict=False):
        """
        Args:
            supply_chain_network:
            visible_states: A string or a list of strings. Limit the states that are visible to the agent. Return
            all states when not provided.
            return_dict: whether the return states is a numpy array(Default) or a dictionary
        """
        self.scn = supply_chain_network
        self.visible_states = visible_states
        self.return_dict = return_dict
        self.period = 0
        self.terminal = False

    def reset(self):
        self.terminal = False
        self.scn.reset()
        self.period = 0

        self.scn.before_action(self.period)

        states = self.scn.get_states(self.scn.player, self.period)
        # states['period'] = self.scn.max_period - self.period

        if not self.return_dict:
            states = np.array(list(states.values()))
        return states

    def step(self, quantity):
        """
        return:
            a tuple of stats (dict), cost (float) and terminal (bool)
        """

        if self.terminal:
            raise ValueError("Cannot take action when the state is terminal.")

        self.scn.player_action(self.period, quantity)
        self.scn.after_action(self.period)

        cost = self.scn.cost_keeping()

        # self.scn.summary()
        self.period += 1

        if self.period >= self.scn.max_period:
            self.terminal = True
        else:
            self.scn.before_action(self.period)

        states = self.scn.get_states(self.scn.player, self.period)
        # states['period'] = self.scn.max_period - self.period

        # if self.visible_states is not None:

        if not self.return_dict:
            states = np.array(list(states.values()))

        return states, cost, self.terminal, {}


def build_beer_game_basic(player='retailer', max_period=100, return_dict=False, seed=None):
    """
    A basic scenario described in Oroojlooyjadid et al. page 19.
    Args:
        player:
        max_period:
        return_dict:
        seed:
    Returns:

    """
    if seed:
        np.random.seed(seed)
    # else:
        # make sure the environment is random in parallel processing
        # np.random.seed(int(time.time()))

    demand_generator = DemandGenerator('uniform', low=0, high=2, size=max_period)

    bs_8 = BaseStockPolicy(8)
    bs_0 = BaseStockPolicy(0)

    demand_source = Node(name='demand_source', demand_source=True, demands=demand_generator)
    retailer = Node(name='retailer', holding_cost=2.0, stockout_cost=2.0, policy=bs_8)
    wholesaler = Node(name='wholesaler', holding_cost=2.0, stockout_cost=0.0, policy=bs_8)
    distributor = Node(name='distributor', holding_cost=2.0, stockout_cost=0.0, policy=bs_0)
    manufacturer = Node(name='manufacturer', holding_cost=2.0, stockout_cost=0.0, policy=bs_0)
    supply_source = Node(name='supply_source', supply_source=True)
    nodes = [demand_source, retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [Arc('supply_source', 'manufacturer', 2, 2, initial_previous_orders=[0, 0, 0, 0]),
            Arc('manufacturer', 'distributor', 2, 2, initial_previous_orders=[0, 0, 0, 0]),
            Arc('distributor', 'wholesaler', 2, 2, initial_previous_orders=[0, 0, 0, 0]),
            Arc('wholesaler', 'retailer', 2, 2, initial_previous_orders=[0, 0, 0, 0]),
            Arc('retailer', 'demand_source', 1, 0)]

    scn = SupplyChainNetwork(nodes=nodes, arcs=arcs, player=player, max_period=max_period)

    return InventoryManagementEnv(scn, return_dict=return_dict)


def build_beer_game_uniform(player='retailer', max_period=100, return_dict=False, seed=None):
    if seed:
        np.random.seed(seed)
    # else:
        # make sure the environment is random in parallel processing
        # np.random.seed(int(time.time()))

    demand_generator = DemandGenerator('uniform', low=0, high=8, size=max_period)
    # demand_generator = DemandGenerator('uniform_50', size=50)

    bs_19 = BaseStockPolicy(19)
    bs_20 = BaseStockPolicy(20)
    bs_14 = BaseStockPolicy(14)

    demand_source = Node(name='demand_source', demand_source=True, demands=demand_generator)
    retailer = Node(name='retailer', initial_inventory=12, holding_cost=0.5, stockout_cost=1, policy=bs_19)
    wholesaler = Node(name='wholesaler', initial_inventory=12, holding_cost=0.5, stockout_cost=1, policy=bs_20)
    distributor = Node(name='distributor', initial_inventory=12, holding_cost=0.5, stockout_cost=1, policy=bs_20)
    manufacturer = Node(name='manufacturer', initial_inventory=12, holding_cost=0.5, stockout_cost=1, policy=bs_14)
    supply_source = Node(name='supply_source', supply_source=True)
    nodes = [demand_source, retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [Arc('supply_source', 'manufacturer', 2, 1, initial_previous_orders=[0, 0, 0, 0]),
            Arc('manufacturer', 'distributor', 2, 2, initial_previous_orders=[0, 0, 0, 0]),
            Arc('distributor', 'wholesaler', 2, 2, initial_previous_orders=[0, 0, 0, 0]),
            Arc('wholesaler', 'retailer', 2, 2, initial_previous_orders=[0, 0, 0, 0]),
            Arc('retailer', 'demand_source', 1, 0)]

    # arcs = [Arc('supply_source', 'manufacturer', 1, 2, initial_shipments=[(4, 1), (4, 2)],
    #             initial_sales_orders=[(4, 0, 1)], initial_previous_orders=[4, 4, 4, 4]),
    #         Arc('manufacturer', 'distributor', 2, 2, initial_shipments=[(4, 1), (4, 2)],
    #             initial_sales_orders=[(4, 0, 1), (4, 0, 2)], initial_previous_orders=[4, 4, 4, 4]),
    #         Arc('distributor', 'wholesaler', 2, 2, initial_shipments=[(4, 1), (4, 2)],
    #             initial_sales_orders=[(4, 0, 1), (4, 0, 2)], initial_previous_orders=[4, 4, 4, 4]),
    #         Arc('wholesaler', 'retailer', 2, 2, initial_shipments=[(4, 1), (4, 2)],
    #             initial_sales_orders=[(4, 0, 1), (4, 0, 2)], initial_previous_orders=[4, 4, 4, 4]),
    #         Arc('retailer', 'demand_source', 1, 0)]

    scn = SupplyChainNetwork(nodes=nodes, arcs=arcs, player=player, max_period=max_period)

    return InventoryManagementEnv(scn, return_dict=return_dict)


def build_beer_game(player='wholesaler', demand_type='classic_beer_game', max_period=100, return_dict=False, seed=None):
    if seed:
        np.random.seed(seed)
    # else:
        # make sure the environment is random in parallel processing
        # np.random.seed(int(time.time()))

    if demand_type == 'classic_beer_game':
        demand_generator = DemandGenerator('classic_beer_game')
    elif demand_type == 'deterministic_random':
        demand_generator = DemandGenerator((8 + 2 * np.random.randn(max_period)).astype(int))
    elif demand_type == 'normal':
        demand_generator = DemandGenerator('normal', mean=10, sd=2, size=max_period)
    elif demand_type == 'uniform':
        demand_generator = DemandGenerator('uniform', low=0, high=2, size=max_period)

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

    scn = SupplyChainNetwork(nodes=nodes, arcs=arcs, player=player, max_period=max_period)

    return InventoryManagementEnv(scn, return_dict=return_dict)


def build_newsvendor(c=1, p=4, mu=100, sigma=30, return_dict=False):
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
    zero_demand = DemandGenerator(demands_pattern=np.array([0], dtype=int))

    demand_source = Node(name='demand_source', demand_source=True, demands=zero_demand)
    newsvendor = Node(name='newsvendor', holding_cost=c, stockout_cost=p-c)
    supply_source = Node(name='supply_source', supply_source=True)
    nodes = [demand_source, newsvendor, supply_source]

    arcs = [Arc(source='supply_source', target='newsvendor', information_leadtime=0, shipment_leadtime=0),
            Arc(source='newsvendor', target='demand_source', information_leadtime=0, shipment_leadtime=0,
                initial_sales_orders=[(demand_generator,)])]

    scn = SupplyChainNetwork(nodes=nodes, arcs=arcs, player='newsvendor', max_period=1)
    scn.visible_states = ['inventory']

    return InventoryManagementEnv(scn, return_dict=return_dict)
