#!/usr/bin/python
# Author: Tyler Gamvrelis
# Virtual network resource records to drive pricing

# Standard library imports
from abc import ABC
import logging

# Local application imports
from pricing_functions import *
from resource_type import ResourceType

# Globals
logger = logging.getLogger(__name__)

class Record(ABC):
    """Base class for bookkeeping records."""
    cnt = 0

    def __init__(self, res_type, max_qty, val_func, max_val):
        """
        Initializes the record.

        Args:
            res_type : ResourceType
                Indicates the resource type
            max_qty : int
                Total supply
            val_func : ValuationFunc
                Function that retrieves the current valuation
            max_val : float
                Maximum that customers are willing to pay for a unit of this
                resource
        """
        self._id = Record.cnt
        Record.cnt += 1
        logger.debug(f'Creating record with ID = {self._id}')
        self._res_type = res_type
        self._max_qty = max_qty
        self._val_func = val_func
        self._max_val = max_val
        self._qty_in_use = 0
        self._price_func = PriceFuncFactory.get_price_func(
            self._val_func, pbar=self._max_val
        )
        # Caching
        self._last_price = None
        self._dirty_dict = {
            'price': True
        }

    def _mark_dirty(self, key):
        """
        Marks attribute dirty.

        Args:
            key : str
                Indicates a specific attribute to be marked dirty
        """
        self._dirty_dict[key] = True

    def _clear_dirty(self, key):
        """
        Marks attribute clean.

        Args:
            key : str
                Indicates a specific attribute to be marked clean
        """
        self._dirty_dict[key] = False

    def _is_dirty(self, key):
        """
        Returns True if there have been changes in the system such that the
        manager's current state should be recomputed.

        Args:
            key : str
                Indicates a specific attribute whose dirtiness is to be checked

        Raises:
            KeyError: if a dirtiness check has not been implemented for the
                given key
        """
        dirty = self._dirty_dict[key]
        return dirty

    def can_hold_additional_qty(self, qty):
        """
        Check whether the usage of this quantity can be increased by qty units
        without exceeding the supply limit.

        Args:
            qty : int
                The number of resources we may be interested in adding. We
                check whether this quantity can fit
        """
        retval = True
        if self._qty_in_use + qty > self._max_qty:
            retval = False
        return retval

    def get_id(self):
        """Returns the ID of this resource record."""
        return self._id

    def get_res_type(self):
        """Returns the type of this resource."""
        return self._res_type

    def get_max_qty(self):
        """Returns the total supply for this resource."""
        return self._max_qty

    def get_max_valuation(self):
        """Returns the maximum valuation for this resource."""
        return self._max_val

    def get_usage(self):
        """Returns the number of units currently in use."""
        return self._qty_in_use

    def get_utilization(self):
        """
        Returns the current utilization of this resource, as a fraction of the
        total supply.
        """
        utilization = self._qty_in_use / self._max_qty
        return utilization

    def get_supply_cost(self):
        """
        Returns the total supply cost incurred by providing this resource at
        the current utilization level.
        """
        utilization = self.get_utilization()
        return self._val_func(utilization)

    def post_price(self):
        """
        Returns a dict that specifies the per-unit price for this resource
        type.
        """
        if self._is_dirty('price'):
            logger.debug(f'Record {self._id}: (re)computing price...')
            self._last_price = self._price_func(self.get_utilization())
            self._clear_dirty('price')
        return self._last_price

    def update_usage(self, qty_delta):
        """
        Updates the resource usage record.

        Args:
            qty_delta : int
                Amount of resources newly consumed. Added to the total. A
                positive value indicates that resources have been consumed,
                while a negative value indicates that resources have been
                released
        """
        self._qty_in_use += qty_delta
        self._mark_dirty('price')
        logger.debug(
            f'Record {self._id}: usage updated by {qty_delta}. '
            f'Utilization: {self._qty_in_use} / {self._max_qty} '
            '(%0.2f' % (100 * self.get_utilization()) + '%)'
        )
        assert self._qty_in_use <= self._max_qty, "Resource supply exceeded"

    def update_supply(self, supply_delta):
        """
        Updates the total supply for this resource.

        Args:
            supply_delta : int
                Amount of resources added to the supply. A positive value
                indicates that the supply has grown, while a negative value
                indicates that the supply has shrunk (possibly due to equipment
                failures)
        """
        self._max_qty += supply_delta
        self._mark_dirty('price')
        logger.debug(f'Record {self._id}: updated supply by {supply_delta}')

    def __str__(self):
        sb = ''
        sb += f'Record ID: {self._id}\n'
        sb += str(self._res_type) + ':\n'
        sb += '\tUtilization: %d / %d (%.2f %%)\n' % (
            self._qty_in_use, self._max_qty, self.get_utilization()
        )
        sb += '\tValuation function: %s\n' % str(self._val_func)
        sb += '\tMax valuation: %f\n' % self._max_val
        return sb

    def __key(self):
        return (
            self._id,
            self._res_type,
            self._max_qty,
            self._max_val,
            self._qty_in_use
        )

    def __eq__(self, other):
        return (type(self) == type(other)) and \
            (self.__key() == other.__key())

    def __hash__(self):
        # Do NOT hash self.__key(). The reason being that if the record is in a
        # dict, then if any of its values change so will its hash. Although
        # this is valid behaviour (because the object is actually changing), we
        # want the hash to remain the same in this scenario so that the record
        # can conveniently be used as a key
        return hash(str(self._id))

class LinkRecord(Record):
    """
    Captures the essential properties of a unidirectional link between zones.
    """

    def __init__(self, src_zone_id, dst_zone_id, max_qty, val_func, max_val):
        """
        Initializes the link record.

        Args:
            src_zone_id : int
                Zone ID of the source node
            dst_zone_id : int
                Zone ID of the destination node
        """
        logging.debug(
            f'Creating link record {src_zone_id} --> {dst_zone_id} '
            f'(supply = {max_qty}, pbar = {max_val}, {val_func})'
        )
        super().__init__(ResourceType.LINK, max_qty, val_func, max_val)
        self.src_zone_id = src_zone_id
        self.dst_zone_id = dst_zone_id

    def get_src_and_dst(self):
        return self.src_zone_id, self.dst_zone_id

class ResourceRecord(Record):
    """
    Captures the essential properties of a set of resources of a single type.
    """

    def __init__(self, zone_id, res_type, max_qty, val_func, max_val):
        """
        Initializes the resource record.

        Args:
            zone_id : int
                The ID of the zone the resource resides in
        """
        logging.debug(
            f'Creating resource record {res_type} in zone {zone_id} '
            f'(supply = {max_qty}, pbar = {max_val}, {val_func})'
        )
        super().__init__(res_type, max_qty, val_func, max_val)
        self._zone_id = zone_id

    def get_zone_id(self):
        """Returns the ID of the zone this resource resides in."""
        return self._zone_id
    