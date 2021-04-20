#!/usr/bin/python
# Author: Tyler Gamvrelis
# JSON requests and responses

# Standard library imports
from abc import ABC
import copy
import json
import logging
import os

# Third party imports
from jsonschema import validate
from resource_type import ResourceType

# Local application imports
from cost_func import CostFunc, CostFuncFactory
from utils import get_schema_dir

# Globals
logger = logging.getLogger(__name__)
SCHEMA_DIR = get_schema_dir()
with open(os.path.join(SCHEMA_DIR, 'bundle_request.schema.json')) as f:
    BUNDLE_REQUEST_SCHEMA = json.load(f)

with open(os.path.join(SCHEMA_DIR, 'bid_result.schema.json')) as f:
    BID_RESULT_SCHEMA = json.load(f)

with open(os.path.join(SCHEMA_DIR, 'price_response.schema.json')) as f:
    PRICE_RESPONSE_SCHEMA = json.load(f)

class BidResult:
    """Contains the results of a bid."""
    
    def __init__(self, success=False, bid_id=-1, msg=None, json=None):
        """
        Constructs a container for bid results
        
        Args:
            success : bool
                True if the bid has been accepted (False otherwise)
            bid_id : int
                ID generated to uniquely identify this bid attempt
            msg : string
                Message describing the reason for failure, if applicable
        """
        if json == None:
            self.success = success
            self.bid_id = bid_id
            self.msg = msg
        else:
            validate(json, BID_RESULT_SCHEMA)
            self.success = json['success']
            self.bid_id = json['bid_id']
            if 'msg' in json.keys():
                self.msg = json['msg']
            else:
                self.msg = None

    def to_json(self):
        """
        Returns the bid result information in JSON.
        
        Raises:
            ValidationError: If the json description of the object does not
                conform to the schema
        """
        json = {}
        json['success'] = self.success
        json['bid_id'] = self.bid_id
        if self.msg is not None:
            json['msg'] = self.msg
        validate(json, BID_RESULT_SCHEMA)
        return json

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.success == other.success) and \
            (self.bid_id == other.bid_id) and \
            (self.msg == other.msg)
        return retval

    def __str__(self):
        sb = ''
        sb += f'Bid {self.bid_id} '
        if self.success:
            sb += 'succeeded.'
            if self.msg is not None:
                sb += ' Info: '
                sb += self.msg
        else:
            sb += 'failed.'
            if self.msg is not None:
                sb += ' Reason: '
                sb += self.msg
        sb += '\n'
        return sb

class LinkItem:
    """
    Container for information on a specific source-destination link
    (unidirectional).
    """

    def __init__(
        self, src_zone_id=-1, dst_zone_id=-1, value=-1, pbar=None,
        json=None, json_value_keyword='value'
    ):
        self.json_value_keyword = json_value_keyword
        if json == None:
            self.src_zone_id = src_zone_id
            self.dst_zone_id = dst_zone_id
            self._value = value
            self.pbar = pbar
        else:
            self.src_zone_id = json['src_zone_id']
            self.dst_zone_id = json['dst_zone_id']
            self._value = json[self.json_value_keyword]
            try:
                self.pbar = json['pbar']
            except KeyError:
                self.pbar = None

    def set_value(self, value):
        self._value = value

    def get_value(self):
        return self._value

    def get_src_and_dst(self):
        return self.src_zone_id, self.dst_zone_id

    def set_pbar(self, pbar):
        self.pbar = pbar

    def get_pbar(self):
        return self.pbar

    def to_json(self):
        """Returns the contained information in JSON."""
        json = {}
        json['src_zone_id'] = self.src_zone_id
        json['dst_zone_id'] = self.dst_zone_id
        json[self.json_value_keyword] = self.get_value()
        if self.pbar is not None:
            json['pbar'] = self.pbar
        return json

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.src_zone_id == other.src_zone_id) and \
            (self.dst_zone_id == other.dst_zone_id) and \
            (self.get_value() == other.get_value()) and \
            (self.pbar == other.pbar) and \
            (self.json_value_keyword == other.json_value_keyword)
        return retval

    def __str__(self):
        sb = ''
        sb += f'{self.src_zone_id} --> {self.dst_zone_id}, '
        sb += self.json_value_keyword + f' = {self._value}'
        if self.pbar is not None:
            sb += f', pbar = {self.pbar}'
        return sb

class LinkSupplyItem(LinkItem):
    """
    Container for resource supply information on a specific source-destination
    link (unidirectional).
    """

    def __init__(
        self, src_zone_id=-1, dst_zone_id=-1, value=-1,
        pbar=-1, cost_func=None, json=None
    ):
        super().__init__(
            src_zone_id=src_zone_id,
            dst_zone_id=dst_zone_id,
            value=value,
            pbar=pbar,
            json=json,
            json_value_keyword='supply'
        )
        if json == None:
            self.cost_func = cost_func
        else:
            self.cost_func = CostFuncFactory.get_cost_func(json['cost_func'])

    def get_cost_func(self):
        return self.cost_func

    def to_json(self):
        """Returns the contained information in JSON."""
        json = super().to_json()
        json['cost_func'] = self.cost_func.to_json()
        return json

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.cost_func == other.cost_func) and \
            super().__eq__(other)
        return retval
    
    def __str__(self):
        sb = ''
        sb += super().__str__()
        sb += ' (' + str(self.cost_func) + ')'
        return sb

class ResourceItem:
    """Container for information on a specific resource."""

    def __init__(
        self, res_type=None, value=-1, pbar=None,
        json=None, json_value_keyword='value'
    ):
        self.json_value_keyword = json_value_keyword
        if json == None:
            self.res_type = res_type
            self._value = value
            self.pbar = pbar
        else:
            self.res_type = ResourceType[json['res_type']]
            self._value = json[self.json_value_keyword]
            try:
                self.pbar = json['pbar']
            except KeyError:
                self.pbar = None

    def set_value(self, value):
        self._value = value

    def get_value(self):
        return self._value

    def get_res_type(self):
        return self.res_type

    def set_pbar(self, pbar):
        self.pbar = pbar

    def get_pbar(self):
        return self.pbar

    def to_json(self):
        """Returns the contained information in JSON."""
        json = {}
        json['res_type'] = self.res_type.name
        json[self.json_value_keyword] = self.get_value()
        if self.pbar is not None:
            json['pbar'] = self.pbar
        return json

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.res_type == other.res_type) and \
            (self.get_value() == other.get_value()) and \
            (self.pbar == other.pbar) and \
            (self.json_value_keyword == other.json_value_keyword)
        return retval

    def __str__(self):
        sb = ''
        sb += f'Type: {self.res_type.name}, '
        sb += self.json_value_keyword + f' = {self._value}'
        if self.pbar is not None:
            sb += f', pbar = {self.pbar}'
        return sb

class ResourceSupplyItem(ResourceItem):
    """Container for supply information on a specific resource."""
    def __init__(
        self, res_type=None, value=-1, pbar=-1, cost_func=None, json=None
    ):
        super().__init__(
            res_type=res_type,
            value=value,
            pbar=pbar,
            json=json,
            json_value_keyword='supply'
        )
        if json == None:
            self.cost_func = cost_func
        else:
            self.cost_func = CostFuncFactory.get_cost_func(json['cost_func'])

    def get_cost_func(self):
        return self.cost_func

    def to_json(self):
        """Returns the contained information in JSON."""
        json = super().to_json()
        json['cost_func'] = self.cost_func.to_json()
        return json

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.cost_func == other.cost_func) and \
            super().__eq__(other)
        return retval
    
    def __str__(self):
        sb = ''
        sb += super().__str__()
        sb += ' (' + str(self.cost_func) + ')'
        return sb

class ZoneItemBase(ABC):
    """Base class for a zone information container."""

    def __init__(self, zone_id, resources):
        self.zone_id = zone_id
        if resources is not None:
            self.resources = resources
        else:
            self.resources = []

    def contains_res_type(self, res_type):
        """
        Returns True if the given resource type is contained in the resource
        list, otherwise False.

        Args:
            res_item : ResourceItem
                The resource item we wish to check for membership in this zone
        """
        retval = False
        for resource in self.resources:
            if resource.get_res_type() == res_type:
                retval = True
                break
        return retval

    def _contains(self, res_item):
        """
        Returns True if the given item is contained in the resource list,
        otherwise False.

        Args:
            res_item : ResourceItem|ResourceSupplyItem
                The item we wish to check for membership in this zone
        """
        retval = False
        for resource in self.resources:
            if resource == res_item:
                retval = True
                break
        return retval

    def get_num_resources(self):
        """Returns the number of resource types contained in this zone."""
        return len(self.resources)

    def get_resources(self):
        return self.resources

    def get_resource_item_by_type(self, res_type):
        assert isinstance(res_type, ResourceType), \
            f'Resource {res_type} is invalid'
        retval = None
        for resource in self.resources:
            if resource.res_type == res_type:
                retval = resource
                break
        return retval
    
    def to_json(self):
        """Returns the contained information in JSON."""
        json = {}
        json['zone_id'] = self.zone_id
        resource_list = []
        for resource in self.resources:
            resource_list.append(
                resource.to_json()
            )
        json['resources'] = resource_list
        return json

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.zone_id == other.zone_id) and \
            (self.resources == other.resources)
        return retval

    def __str__(self):
        sb = ''
        sb += f'Zone ID: {self.zone_id}\n'
        has_resources = len(self.resources) > 0
        if has_resources:
            for resource in self.resources:
                sb += '\t' + str(resource) + '\n'
        else:
            sb += '\t(no resources)\n'
        return sb

class ZoneItem(ZoneItemBase):
    """Container for zone information."""

    def __init__(
        self, zone_id=-1, resources=None, json=None, json_value_keyword='value'
    ):
        super().__init__(zone_id, resources)
        self.json_value_keyword = json_value_keyword
        if json is not None:
            self.zone_id = json['zone_id']
            self.resources = []
            for resource_item in json['resources']:
                resource = ResourceItem(
                    json=resource_item,
                    json_value_keyword=self.json_value_keyword
                )
                self.resources.append(resource)

    def contains(self, res_item):
        """
        Returns True if the given resource item is contained in the resource
        list, otherwise False.

        Args:
            res_item : ResourceItem
                The resource item we wish to check for membership in this zone
        """
        assert type(res_item) == ResourceItem, f'{res_item} is an invalid type'
        return self._contains(res_item)

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.json_value_keyword == other.json_value_keyword) and \
            super().__eq__(other)
        return retval

class ZoneSupplyItem(ZoneItemBase):
    """Container for zone supply information."""
    
    def __init__(
        self, zone_id=-1, resources=None, json=None
    ):
        super().__init__(zone_id, resources)
        if json is not None:
            self.zone_id = json['zone_id']
            self.resources = []
            for resource_supply_item in json['resources']:
                resource = ResourceSupplyItem(json=resource_supply_item)
                self.resources.append(resource)

    def contains(self, res_item):
        """
        Returns True if the given resource item is contained in the resource
        list, otherwise False.

        Args:
            res_item : ResourceItem
                The resource item we wish to check for membership in this zone
        """
        assert type(res_item) == ResourceSupplyItem, \
            f'{res_item} is an invalid type'
        return self._contains(res_item)

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            super().__eq__(other)
        return retval

class ZoneAndLinkInfoBase(ABC):
    """
    Base class for a zone and link info container which provides methods to
    access and modify the data.
    """

    def __init__(self, zones, links):
        if zones is not None:
            self.zones = zones
        else:
            self.zones = []
        if links is not None:
            self.links = links
        else:
            self.links = []

    def copy_zones(self):
        """
        Returns a copy of all zone information as a list.

        Note: we want this to be a deep copy so that the info can only be
        modified through the provided setters.
        """
        return copy.deepcopy(self.zones)

    def copy_zone_item_by_id(self, zone_id):
        """
        Retrieves the information for the specified zone, if it exists.

        Note: we want this to be a deep copy so that the info can only be
        modified through the provided setters.

        Args:
            zone_id : int
                ID of the zone of interest
        """
        retval = None
        for zone in self.zones:
            if zone.zone_id == zone_id:
                retval = copy.deepcopy(zone)
                break
        return retval

    def copy_links(self):
        """
        Returns a copy of all zone information as a list.

        Note: we want this to be a deep copy so that the info can only be
        modified through the provided setters.
        """
        return copy.deepcopy(self.links)

    def copy_link(self, src_zone_id, dst_zone_id):
        """
        Returns a link whose source node and destination done have the
        specified zone IDsh.

        Note: we want this to be a deep copy so that the info can only be
        modified through the provided setters.

        Args:
            src_zone_id : int
                Zone ID of the source node
            dst_zone_id : int
                Zone ID of the destination node
        """
        retval = None
        for link in self.links:
            cond = (link.src_zone_id == src_zone_id)
            cond &= (link.dst_zone_id == dst_zone_id)
            if cond:
                retval = copy.deepcopy(link)
                break
        return retval

    def copy_links_starting_at(self, src_zone_id):
        """
        Returns a list of links whose source node has the specified zone ID.

        Note: we want this to be a deep copy so that the info can only be
        modified through the provided setters.

        Args:
            src_zone_id : int
                Zone ID of the source node
        """
        links_cp = []
        for link in self.links:
            if link.src_zone_id == src_zone_id:
                links_cp.append(copy.deepcopy(link))
        return links_cp
    
    def copy_links_ending_at(self, dst_zone_id):
        """
        Returns a list of links whose destination node has the specified
        zone ID.

        Note: we want this to be a deep copy so that the info can only be
        modified through the provided setters.

        Args:
            dst_zone_id : int
                Zone ID of the destination node
        """
        links_cp = []
        for link in self.links:
            if link.dst_zone_id == dst_zone_id:
                links_cp.append(copy.deepcopy(link))
        return links_cp

    def get_num_zones(self):
        """Returns the number of zones we have info for."""
        return len(self.zones)

    def get_num_links(self):
        """Returns the number of links we have info for."""
        return len(self.links)

    def to_json(self):
        """Returns the contained information in JSON."""
        json = {}
        zone_list = []
        for zone in self.zones:
            zone_list.append(zone.to_json())
        json['zones'] = zone_list
        # Package link info
        link_list = []
        for link in self.links:
            link_list.append(link.to_json())
        json['links'] = link_list
        return json

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.zones == other.zones) and \
            (self.links == other.links)
        return retval

class ZoneAndLinkInfo(ZoneAndLinkInfoBase):
    """Container for zone and link info."""

    def __init__(
        self, zones=None, links=None,
        json=None, json_schema=None, json_value_keyword='value'
    ):
        super().__init__(zones, links)
        self.json_value_keyword = json_value_keyword
        if json is not None:
            if json_schema is not None:
                validate(json, json_schema)
            self.zones = []
            for zone_item in json['zones']:
                self.zones.append(
                    ZoneItem(
                        json=zone_item,
                        json_value_keyword=self.json_value_keyword
                    )
                )
            self.links = []
            for link_item in json['links']:
                self.links.append(
                    LinkItem(
                        json=link_item,
                        json_value_keyword=self.json_value_keyword
                    )
                )

    def add_resource_value(self, zone_id, res_type, value, pbar=None):
        """
        Adds a resource to a zone. If a request for this type of resource
        is found within the specified zone, it will be replaced.

        Args:
            zone_id : int
                ID of the zone we want to acquire the resource in
            res_type : ResourceType
                Type of resource we want to acquire
            value : int|float
                Value associated with the resource
            pbar : float
                The maximum amount bidders can pay for a unit of this resource.
                If set to None, no value will be sent for pbar
        """
        assert zone_id >= 0, f'Zone ID {zone_id} must be >= 0'
        assert isinstance(res_type, ResourceType), \
            f'Resource {res_type} is invalid'
        if isinstance(value, int) or isinstance(value, float):
            assert value >= 0, f'Value {value} must be >= 0'
        # 1. Check whether zone exists in bundle
        resource_item = ResourceItem(
            res_type, value, pbar, json_value_keyword=self.json_value_keyword
        )
        found_zone = False
        for zone in self.zones:
            if zone.zone_id == zone_id:
                found_zone = True
                # 1.1. Check whether zone of interest already has request for
                # this resource type; if so, then modify it
                found_resource = False
                for resource in zone.resources:
                    if resource.res_type == res_type:
                        found_resource = True
                        resource.set_value(value)
                        resource.set_pbar(pbar)
                        break
                # 1.2. If the previous check failed, then create the resource
                # request
                if not found_resource:
                    zone.resources.append(resource_item)
                break
        # 2. If zone doesn't exist in bundle yet, then create it
        if not found_zone:
            zone_item = ZoneItem(
                zone_id,
                resources=[resource_item],
                json_value_keyword=self.json_value_keyword
            )
            self.zones.append(zone_item)
    
    def add_link_value(self, src_zone_id, dst_zone_id, value, pbar=None):
        """
        Adds a link. If a request for this src-dst pair is found, it will be
        replaced. Links are unidirectional.

        Args:
            src_zone_id : int
                Zone ID of the link source
            dst_zone_id : ResourceType
                Zone ID of the link destination
            value : int|float
                Value associated with the link
            pbar : float
                The maximum amount bidders can pay for a unit of this resource.
                If set to None, no value will be sent for pbar
        """
        assert src_zone_id >= 0, f'src zone ID {src_zone_id} must be >= 0'
        assert dst_zone_id >= 0, f'dst zone ID {dst_zone_id} must be >= 0'
        assert src_zone_id != dst_zone_id, 'src and dst zone ID must be unique'
        if isinstance(value, int) or isinstance(value, float):
            assert value >= 0, f'Value {value} must be >= 0'
        # 1. Check whether request exists for this link; if so, modify it
        found_link = False
        for link in self.links:
            found_link = (link.src_zone_id == src_zone_id) and \
                (link.dst_zone_id == dst_zone_id)
            if found_link:
                link.set_value(value)
                link.set_pbar(pbar)
                break
        # 2. If the previous check failed, then create the link request
        if not found_link:
            link_item = LinkItem(
                src_zone_id,
                dst_zone_id,
                value,
                pbar,
                json_value_keyword=self.json_value_keyword
            )
            self.links.append(link_item)

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.json_value_keyword == other.json_value_keyword) and \
            super().__eq__(other)
        return retval

class ZoneAndLinkSupplyInfo(ZoneAndLinkInfoBase):
    """Container for zone and link supply info."""

    def __init__(self, zones=None, links=None, json=None):
        super().__init__(zones, links)
        if json is not None:
            # TODO: make a schema to validate with before initializing object
            self.zones = []
            for zone_item in json['zones']:
                self.zones.append(ZoneSupplyItem(json=zone_item))
            self.links = []
            for link_item in json['links']:
                self.links.append(LinkSupplyItem(json=link_item))

    def add_resource(self, zone_id, res_type, qty, pbar, cost_func):
        """
        Adds resource supply info to a zone. If info for this type of resource
        is found within the specified zone, it will be replaced.

        Args:
            zone_id : int
                ID of the zone the resource resides in
            res_type : ResourceType
                Type of resource we are supplying
            qty : int
                Number of resource units available to be supplied
            pbar : float
                Maximum amount customers are willing to pay for a unit of this
                resource
            cost_func : CostFunc
                Cost function associated with supplying this resource
        """
        assert zone_id >= 0, f'Zone ID {zone_id} must be >= 0'
        assert isinstance(res_type, ResourceType), \
            f'Resource {res_type} is an invalid type'
        assert qty >= 0, f'qty {qty} must be >= 0'
        assert isinstance(cost_func, CostFunc), \
            f'Cost function {cost_func} is an invalid type'
        # 1. Check whether zone exists yet
        resource_item = ResourceSupplyItem(
            res_type, qty, pbar, cost_func
        )
        found_zone = False
        for zone in self.zones:
            if zone.zone_id == zone_id:
                found_zone = True
                # 1.1. Check whether zone of interest already has info for
                # this resource type; if so, then modify it
                found_resource = False
                for resource in zone.resources:
                    if resource.res_type == res_type:
                        found_resource = True
                        resource.set_value(qty)
                        resource.cost_func = cost_func,
                        resource.pbar = pbar
                        break
                # 1.2. If the previous check failed, then create the resource
                # info
                if not found_resource:
                    zone.resources.append(resource_item)
                break
        # 2. If zone doesn't exist in yet, then create it
        if not found_zone:
            zone_item = ZoneSupplyItem(zone_id, resources=[resource_item])
            self.zones.append(zone_item)
    
    def add_link(self, src_zone_id, dst_zone_id, qty, pbar, cost_func):
        """
        Adds link supply info to a zone. If info for this src-dst pair is
        found, it will be replaced. Links are unidirectional.

        Args:
            src_zone_id : int
                Zone ID of the link source
            dst_zone_id : ResourceType
                Zone ID of the link destination
            qty : int
                Number of resource units available to be supplied
            pbar : float
                Maximum amount customers are willing to pay for a unit of this
                resource
            cost_func : CostFunc
                Cost function associated with supplying this resource
        """
        assert src_zone_id >= 0, f'src zone ID {src_zone_id} must be >= 0'
        assert dst_zone_id >= 0, f'dst zone ID {dst_zone_id} must be >= 0'
        assert src_zone_id != dst_zone_id, 'src and dst zone ID must be unique'
        assert qty >= 0, 'qty {qty} must be >= 0'
        assert isinstance(cost_func, CostFunc), \
            f'Cost function {cost_func} is an invalid type'
        # 1. Check whether info exists for this link; if so, modify it
        found_link = False
        for link in self.links:
            found_link = (link.src_zone_id == src_zone_id) and \
                (link.dst_zone_id == dst_zone_id)
            if found_link:
                link.set_value(value)
                link.cost_func = cost_func
                link.pbar = pbar
                break
        # 2. If the previous check failed, then create the link info
        if not found_link:
            link_item = LinkSupplyItem(
                src_zone_id, dst_zone_id, qty, pbar, cost_func
            )
            self.links.append(link_item)

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            super().__eq__(other)
        return retval

class PriceResponse(ZoneAndLinkInfo):
    """Describes a response for a price request."""

    def __init__(self, zones=None, links=None, json=None):
        super().__init__(
            zones=zones,
            links=links,
            json=json,
            json_schema=PRICE_RESPONSE_SCHEMA,
            json_value_keyword='price'
        )
        if json == None:
            self.price_token = None
        else:
            try:
                self.price_token = json['price_token']
            except KeyError:
                self.price_token = None

    def add_resource_price(self, zone_id, res_type, price, pbar):
        """
        Adds a resource to the price response. If the price response already
        contains a record for this type of resource in the specified zone, the
        previous record will be replaced.

        Args:
            zone_id : int
                ID of the zone the resource resides in
            res_type : ResourceType
                Type of resource we want to add pricing information for
            price : float
                The per-unit price of the resource
            pbar : float
                The maximum amount bidders can pay for a unit of this resource.
                If set to None, no value will be sent for pbar
        """
        assert price >= 0, f'Resource price {price} must be >= 0'
        self.add_resource_value(zone_id, res_type, price, pbar)
    
    def add_link_price(self, src_zone_id, dst_zone_id, price, pbar):
        """
        Adds a link to the price response. If a record for this src-dst pair is
        found, it will be replaced. Links are unidirectional.

        Args:
            src_zone_id : int
                Zone ID of the link source
            dst_zone_id : ResourceType
                Zone ID of the link destination
            price : int
                The per-unit price of the link
            pbar : float
                The maximum amount bidders can pay for a unit of this link. If
                set to None, no value will be sent for pbar
        """
        assert price >= 0, f'Resource price {price} must be >= 0'
        self.add_link_value(src_zone_id, dst_zone_id, price, pbar)

    def set_price_token(self, price_token):
        self.price_token = price_token

    def has_price_token(self):
        return self.price_token is not None

    def get_price_token(self):
        return self.price_token

    def to_json(self):
        """Returns the contained information in JSON."""
        json = super().to_json()
        if self.price_token is not None:
            json['price_token'] = self.price_token
        validate(json, PRICE_RESPONSE_SCHEMA)
        return json

    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.get_price_token() == other.get_price_token()) and \
            super().__eq__(other)
        return retval

    def __str__(self):
        sb = ''
        if self.price_token is not None:
            sb += f'Price token: {self.price_token}\n'
        num_zones = len(self.zones)
        sb += f'{num_zones} zone(s)\n'
        for zone in self.zones:
            sb += str(zone)
        num_links = len(self.links)
        sb += f'{num_links} link(s)\n'
        for link in self.links:
            sb += '\t' + str(link) + '\n'
        return sb

class Bundle(ZoneAndLinkInfo):
    """Describes a bundle of resources used for bidding."""

    def __init__(
        self, payment=0, duration=0, zones=None, links=None, json=None
    ):
        super().__init__(
            zones=zones, links=links, json=json, json_value_keyword='qty'
        )
        if json == None:
            self.payment = payment
            self.duration = duration
            self.price_token = None
            
        else:
            self.payment = json['payment']
            self.duration = json['duration']
            try:
                self.price_token = json['price_token']
            except KeyError:
                self.price_token = None

    def set_price_token(self, price_token):
        self.price_token = price_token

    def has_price_token(self):
        return self.price_token is not None

    def get_price_token(self):
        return self.price_token

    def set_payment(self, payment):
        """
        Sets the bundle payment amount.

        Args:
            payment : float
                Amount to pay for this bundle
        """
        self.payment = payment

    def set_duration(self, duration):
        """
        Sets the bundle duration amount.

        Args:
            duration : float
                Duration we want the bundle for
        """
        self.duration = duration

    def prune(self):
        """
        Checks for any unnecessary information in the bundle, and removes it if
        found. Examples include:
            - Resource requests with quantity zero
            - Link requests with quantity zero
            - Zones with no resource requests
        This method should be called after modifying bundle zones or links.
        """
        # 1. Search for unnecessary information
        resources_to_remove = []
        zones_to_remove = []
        links_to_remove = []
        for zone in self.zones:
            for resource in zone.resources:
                if resource.get_value() == 0:
                    zone_id = zone.zone_id
                    resources_to_remove.append((zone, resource))
        for link in self.links:
            if link.get_value() == 0:
                links_to_remove.append(link)
        # 2. Remove unnecessary resources, if we found any
        for zone, resource in resources_to_remove:
            zone.resources.remove(resource)
        # 3. Remove unnecessary zones. Some may have been created as a result
        # of step 2, while others may have existed before as well
        for zone in self.zones:
            if len(zone.resources) == 0:
                zones_to_remove.append(zone)
        for zone in zones_to_remove:
            self.zones.remove(zone)
        # 4. remove unnecessary links
        for link in links_to_remove:
            self.links.remove(link)

    def add_resource(self, zone_id, res_type, qty):
        """
        Adds a resource to the bundle. If a request for this type of resource
        is found within the specified zone, it will be replaced.

        If qty is zero, any existing request for this resource within the
        specified zone will be deleted.

        Args:
            zone_id : int
                ID of the zone we want to acquire the resource in
            res_type : ResourceType
                Type of resource we want to acquire
            qty : int
                Total number of units of the resource we want to acquire within
                the specified zone
        """
        assert qty >= 0, f'Resource quantity {qty} must be >= 0'
        self.add_resource_value(zone_id, res_type, qty)
        self.prune()
    
    def add_link(self, src_zone_id, dst_zone_id, qty):
        """
        Adds a link to the bundle. If a request for this src-dst pair is found,
        it will be replaced. Links are unidirectional.

        If qty is zero, any existing request for this src-dst pair will be
        deleted.

        Args:
            src_zone_id : int
                Zone ID of the link source
            dst_zone_id : ResourceType
                Zone ID of the link destination
            qty : int
                Total number of units of the resource we want to acquire
        """
        assert qty >= 0, f'Resource quantity {qty} must be >= 0'
        self.add_link_value(src_zone_id, dst_zone_id, qty)
        self.prune()

    def to_json(self):
        """Returns the contained information in JSON."""
        json = super().to_json()
        json['payment'] = self.payment
        json['duration'] = self.duration
        if self.price_token is not None:
            json['price_token'] = self.price_token
        return json
    
    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.payment == other.payment) and \
            (self.duration == other.duration) and \
            (self.get_price_token() == other.get_price_token()) and \
            super().__eq__(other)
        return retval

    def __str__(self):
        sb = ''
        sb += f'\tPayment: {self.payment}\n'
        sb += f'\tDuration: {self.duration}\n'
        sb += f'\t# Zones: {len(self.zones)}\n'
        sb += f'\t# Links: {len(self.links)}\n'
        if self.price_token is not None:
            sb += f'\tPrice token: {self.price_token}\n'
        return sb

class BundleRequest:
    """Describes a request for a bundle of resources."""

    def __init__(self, customer_id=-1, bundle=None, json=None):
        """
        Initializes the bundle request.

        Args:
            customer_id : int
                ID of the customr making the request
            bundle : Bundle
                Bundle describing the resource request
            json : dict
                JSON-formatted dict of bundle information. Can be used to
                construct a BundleRequest from its JSON counterpart
        """
        if json == None:
            self.customer_id = customer_id
            if bundle is not None:
                self.bundle = bundle
            else:
                self.bundle = Bundle()
        else:
            validate(json, BUNDLE_REQUEST_SCHEMA)
            self.customer_id = json['customer_id']
            self.bundle = Bundle(json=json['bundle'])

    def to_json(self):
        """Returns the request information in JSON."""
        json = {}
        json['customer_id'] = self.customer_id
        json['bundle'] = self.bundle.to_json()
        validate(json, BUNDLE_REQUEST_SCHEMA)
        return json
    
    def __eq__(self, other):
        retval = (type(self) == type(other)) and \
            (self.customer_id == other.customer_id) and \
            (self.bundle == other.bundle)
        return retval

    def __str__(self):
        sb = ''
        sb += 'Bundle:\n'
        sb += f'\tCustomer ID: {self.customer_id}\n'
        sb += str(self.bundle)
        return sb
