#!/usr/bin/python
# Author: Tyler Gamvrelis

# Standard library imports
from collections import defaultdict
import json
import logging
from queue import PriorityQueue, Empty
import random
from threading import Lock
from time import time

# Local application imports
from cost_func import PowerFunc
from resource_manager_proxy import ResourceManagerProxy
from record import LinkRecord, ResourceRecord
from request_response import ZoneAndLinkSupplyInfo
from resource_type import ResourceType
from utils import get_metrics_log_path

# Globals
logger = logging.getLogger(__name__)

class ResourceManagerSim(ResourceManagerProxy):
    """
    Implements a resource manager that simulates an underlying infrastructure.
    """

    def __init__(self):
        super().__init__()
        self._rng = random.Random(1)
        self._has_loaded = False
        # Time tracking for consumed resources. We need to know when to release
        # the consumed resources for each record
        self._history_lock = Lock()
        self._history = defaultdict(PriorityQueue)
        self._t_s = time() # Start time

        # Initialize!
        self.collect_resource_info()

        # Clear metrics log
        metric_file = get_metrics_log_path()
        with open(metric_file, 'w') as log_file:
            log_file.write('')
    
    def _load_data(self):
        """
        Gets the resource info from its physical storage location.

        Note: in this case, the info is just stored in objects in the local
        system's memory. Could update this function later to load a description
        from the disk, if we get there.
        """
        logger.debug('Creating infrastructure data')
        # Cost funcs
        cpu_cost_func = PowerFunc(0.223, 3)
        # mem_cost_func = PowerFunc(8.38e-6, 1.2)
        mem_cost_func = PowerFunc(0.5, 1.2)
        gpu_cost_func = PowerFunc(0.223, 1.2)
        link_cost_func = PowerFunc(0.6, 2)
        # Helper to map cost func to pbar bounds
        # TODO: what do we want pbar to be?
        get_bounds = lambda f,m: (f.get_min_marginal_cost(), m * f.get_max_marginal_cost())
        cpu_bounds = get_bounds(cpu_cost_func, 2)
        mem_bounds = get_bounds(mem_cost_func, 2)
        gpu_bounds = get_bounds(gpu_cost_func, 2)
        link_bounds = get_bounds(link_cost_func, 2)
        # Add resources to zones
        zlsi = ZoneAndLinkSupplyInfo()
        zlsi.add_resource(1, ResourceType.CPU, 1000, self._rng.uniform(*cpu_bounds), cpu_cost_func)
        zlsi.add_resource(1, ResourceType.MEM, 2000, self._rng.uniform(*mem_bounds), mem_cost_func)
        zlsi.add_resource(2, ResourceType.CPU, 1000, self._rng.uniform(*cpu_bounds), cpu_cost_func)
        zlsi.add_resource(2, ResourceType.MEM, 1000, self._rng.uniform(*mem_bounds), mem_cost_func)
        zlsi.add_resource(2, ResourceType.GPU, 1000, self._rng.uniform(*gpu_bounds), gpu_cost_func)
        zlsi.add_resource(3, ResourceType.CPU, 2000, self._rng.uniform(*cpu_bounds), cpu_cost_func)
        zlsi.add_resource(3, ResourceType.MEM, 1000, self._rng.uniform(*mem_bounds), mem_cost_func)
        # Add links between zones
        zlsi.add_link(1, 2, 2000, self._rng.uniform(*link_bounds), link_cost_func)
        zlsi.add_link(2, 1, 1000, self._rng.uniform(*link_bounds), link_cost_func)
        zlsi.add_link(2, 3, 2000, self._rng.uniform(*link_bounds), link_cost_func)
        zlsi.add_link(3, 2, 1000, self._rng.uniform(*link_bounds), link_cost_func)
        return zlsi

    def collect_resource_info(self):
        # Load infrastructure data
        if len(self._res_list) > 0:
            # We don't want to go through the process of recreating everything
            # since that's wasteful given that this is just a simulation. Plus,
            # that would mess up the RNG (we don't want resource properties to
            # change each time this is called)
            return True
        zlsi = self._load_data()
        res_list = []
        # Zones
        zones = zlsi.copy_zones()
        for zone in zones:
            zone_id = zone.zone_id
            for resource in zone.get_resources():
                res_type = resource.get_res_type()
                qty = resource.get_value()
                cost_func = resource.get_cost_func()
                pbar = resource.get_pbar()
                res = ResourceRecord(zone_id, res_type, qty, cost_func, pbar)
                res_list.append(res)
        # Links
        links = zlsi.copy_links()
        for link in links:
            src_zone_id, dst_zone_id = link.get_src_and_dst()
            qty = link.get_value()
            cost_func = link.get_cost_func()
            pbar = link.get_pbar()
            res = LinkRecord(src_zone_id, dst_zone_id, qty, cost_func, pbar)
            res_list.append(res)
        self._res_list = res_list
        return True

    def evolve_time(self):
        """
        Causes time-dependent updates to be performed, such as releasing
        resources for bundles whose duration has elapsed.
        """
        t = time()
        with self._history_lock:
            for record, hist_q in self._history.items():
                qty_to_add_back = 0
                while True:
                    try:
                        pri, qty = hist_q.get_nowait()
                    except Empty:
                        # Nothing in queue, so nothing to do
                        break
                    if pri < t:
                        qty_to_add_back += qty
                    else:
                        # Add back if this item's time has not yet come
                        hist_q.put((pri, qty))
                        break
                assert qty_to_add_back >= 0, 'Negative qty!'
                if qty_to_add_back > 0:
                    record.update_usage(-1 * qty_to_add_back)
        # Log prices and usage for each resource
        data = defaultdict(dict)
        data['d_t'] = t - self._t_s
        for res in self._res_list:
            price = res.post_price()
            res_type = res.get_res_type()
            usage = res.get_usage()
            supply = res.get_max_qty()
            if res_type == ResourceType.LINK:
                src_zone_id, dst_zone_id = res.get_src_and_dst()
                data[src_zone_id][dst_zone_id] = (
                    res_type.name, price, usage, supply
                )
            else:
                zone_id = res.get_zone_id()
                data[zone_id][res_type.name] = (
                    res_type.name, price, usage, supply
                )
        metric_file = get_metrics_log_path()
        with open(metric_file, 'a') as log_file:
            log_file.write(json.dumps(data) + '\n')

    def update_res_usage(self, record, qty, duration):
        record.update_usage(qty)
        # Time tracking for consumed resources. This is the responsibility of
        # the resource manager because these time values are simulated. In a
        # real deployment, keeping track of the lease time may be the
        # responsibility of a different service
        priority = time() + duration
        with self._history_lock:
            self._history[record].put((priority, qty))
