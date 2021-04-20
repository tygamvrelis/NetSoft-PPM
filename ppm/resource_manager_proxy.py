#!/usr/bin/python
# Author: Tyler Gamvrelis
# Base class defining a resource manager interface

# Standard library imports
from abc import ABC, abstractmethod
from collections import defaultdict
import copy
import logging

# Local application imports
from resource_type import ResourceType

# Globals
logger = logging.getLogger(__name__)

class ResourceManagerProxy(ABC):
    """
    Base class for interfacing with a resource manager. A resource manager may
    be an object in local application code, or it may be a service running on a
    remote machine. A class which inherits from this interface will implement
    the mechanisms used to communicate with the chosen resource manager
    deployment scenario.
    """

    def __init__(self):
        self._res_list = [] # List of Record objects
        self._res_to_record_mapping = None
        self._res_to_price_mapping = None

    @abstractmethod
    def collect_resource_info(self):
        """
        Collects information about all available resources in the system. For
        each resource, this includes (but is not necessarily limited to) the
        total supply and cost function.

        Note: the intention of this function is to perform a full query of
        network resources in the system, so as to update the current state. If
        there are no changes to the system resources, then this function should
        always return the same thing. However, if some hardware fails, then
        it's possible our view of the system could be out of date; calling this
        function would remedy such a situation.

        Returns: True if successful, otherwise False
        """
        pass

    def get_resource_info(self):
        """Returns the last collected resource info."""
        return self._res_list

    # TODO: would be nice if the returned mapping was somehow read-only
    def get_res_to_price_mapping(self):
        """
        Returns a dictionary that can be used to retrieve prices given
        resource-specific (or link-specific) info.

        For a resource, you can get the price by indexing the dictionary first
        by its zone, then by the resource type of interest.

        For a link, you can get the price by indexing the dictionary first by
        the source zone, then the destination zone.
        """
        # Build mapping
        logger.debug('Building price mapping')
        self._res_to_price_mapping = defaultdict(dict)
        for res in self._res_list:
            price = res.post_price()
            res_type = res.get_res_type()
            if res_type == ResourceType.LINK:
                src_zone_id, dst_zone_id = res.get_src_and_dst()
                self._res_to_price_mapping[src_zone_id][dst_zone_id] = price
            else:
                zone_id = res.get_zone_id()
                self._res_to_price_mapping[zone_id][res_type] = price
        return self._res_to_price_mapping

    # TODO: would be nice if the returned mapping was somehow read-only
    def get_res_to_record_mapping(self):
        """
        Returns a dictionary that can be used to retrieve resource records
        given resource-specific (or link-specific) info.

        For a resource, you can get the record by indexing the dictionary first
        by its zone, then by the resource type of interest.

        For a link, you can get the record by indexing the dictionary first by
        the source zone, then the destination zone.
        """
        # Build mapping
        logger.debug('Building Record mapping')
        self._res_to_record_mapping = defaultdict(dict)
        for res in self._res_list:
            price = res.post_price()
            res_type = res.get_res_type()
            if res_type == ResourceType.LINK:
                src_zone_id, dst_zone_id = res.get_src_and_dst()
                self._res_to_record_mapping[src_zone_id][dst_zone_id] = res
            else:
                zone_id = res.get_zone_id()
                self._res_to_record_mapping[zone_id][res_type] = res
        return self._res_to_record_mapping

        @abstractmethod
        def update_res_usage(self, record, qty, duration):
            """
            Updates the usage of a particular resource/link record.

            Args:
                record : Record
                    The record to be updated
                qty : int
                    Amount of resources newly consumed
                duration : float
                    Amount of time this resource should be reserved for before
                    being released back into the pool
            """
            pass
