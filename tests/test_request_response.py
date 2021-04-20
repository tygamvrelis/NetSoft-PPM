#!/usr/bin/python
# Author: Tyler Gamvrelis
# Test requests and responses

# Standard library imports
import os
import sys
import unittest

# Third party imports
from jsonschema import validate, ValidationError, SchemaError, Draft7Validator

# Local application imports
import init_tests
from cost_func import CostFunc, PowerFunc
from request_response import *

class TestBidResult(unittest.TestCase):
    def test_bid_result_schema_is_valid(self):
        raised = False
        msg = ''
        try:
            Draft7Validator.check_schema(BID_RESULT_SCHEMA)
        except SchemaError as e:
            raised = True
            msg = e
        self.assertFalse(raised, msg)

    def test_simple_bid_result_is_schema_compliant(self):
        br = BidResult(True, 1, 'Additional info')
        json = br.to_json()
        raised = False
        msg = ''
        try:
            validate(json, BID_RESULT_SCHEMA)
        except ValidationError as e:
            raised = True
            msg = e
        self.assertFalse(raised, msg)

    def test_bid_result_can_be_constructed_from_its_json(self):
        # Setup
        br = BidResult(True, 1, 'Additional info')
        json = br.to_json()
        # Test
        new_br = BidResult(json=json)
        self.assertTrue(br == new_br)

class TestZoneAndLinkInfo(unittest.TestCase):
    def test_default_zones_are_empty(self):
        zli = ZoneAndLinkInfo()
        num_zones = zli.get_num_zones()
        self.assertTrue(num_zones == 0)

    def test_default_links_are_empty(self):
        zli = ZoneAndLinkInfo()
        num_links = zli.get_num_links()
        self.assertTrue(num_links == 0)

    def test_adding_invalid_resource_type_fails(self):
        # Set up
        zli = ZoneAndLinkInfo()
        RES_TYPE = 'invalid'
        VALUE = 10
        ZONE_ID = 1
        # Test
        with self.assertRaises(AssertionError):
            zli.add_resource_value(ZONE_ID, RES_TYPE, VALUE)

    def test_adding_nonzero_qty_resource_changes_json(self):
        # Set up
        zli = ZoneAndLinkInfo()
        INITIAL_JSON = zli.to_json()
        RES_TYPE = ResourceType.CPU
        VALUE = 10
        ZONE_ID = 1
        # Test
        zli.add_resource_value(ZONE_ID, RES_TYPE, VALUE)
        json = zli.to_json()
        self.assertFalse(INITIAL_JSON == json)

    def test_adding_resource_twice_keeps_most_recent_addition(self):
        # Set up
        zli = ZoneAndLinkInfo()
        RES_TYPE = ResourceType.CPU
        VALUE1 = 10
        VALUE2 = 20
        EXPECTED_RES_ITEM = ResourceItem(RES_TYPE, VALUE2)
        ZONE_ID = 1
        # Test
        zli.add_resource_value(ZONE_ID, RES_TYPE, VALUE1)
        zli.add_resource_value(ZONE_ID, RES_TYPE, VALUE2)
        zone_item = zli.copy_zone_item_by_id(ZONE_ID)
        resource_item = zone_item.get_resource_item_by_type(RES_TYPE)
        self.assertTrue(EXPECTED_RES_ITEM == resource_item)

    def test_adding_different_resource_types_keeps_both(self):
        # Set up
        zli = ZoneAndLinkInfo()
        RES_TYPE1 = ResourceType.CPU
        RES_TYPE2 = ResourceType.MEM
        VALUE1 = 10
        VALUE2 = 5
        EXPECTED_RES_ITEM1 = ResourceItem(RES_TYPE1, VALUE1)
        EXPECTED_RES_ITEM2 = ResourceItem(RES_TYPE2, VALUE2)
        ZONE_ID = 1
        # Test
        zli.add_resource_value(ZONE_ID, RES_TYPE1, VALUE1)
        zli.add_resource_value(ZONE_ID, RES_TYPE2, VALUE2)
        zone_item = zli.copy_zone_item_by_id(ZONE_ID)
        success = True
        success &= zone_item.contains(EXPECTED_RES_ITEM1)
        success &= zone_item.contains(EXPECTED_RES_ITEM2)
        self.assertTrue(success)

    def test_adding_1_resource_in_2_zones_results_in_2_zones(self):
        # Set up. We want different resource types here so that they can be
        # distinguished (to make sure they are stored in the correct zone)
        zli = ZoneAndLinkInfo()
        RES_TYPE1 = ResourceType.CPU
        RES_TYPE2 = ResourceType.MEM
        VALUE = 10
        EXPECTED_RES_ITEM1 = ResourceItem(RES_TYPE1, VALUE)
        EXPECTED_RES_ITEM2 = ResourceItem(RES_TYPE2, VALUE)
        ZONE_ID1 = 1
        ZONE_ID2 = 2
        # Test
        zli.add_resource_value(ZONE_ID1, RES_TYPE1, VALUE)
        zli.add_resource_value(ZONE_ID2, RES_TYPE2, VALUE)
        num_zones = zli.get_num_zones()
        self.assertTrue(num_zones == 2)

    def test_adding_1_resource_in_2_zones_results_in_1_resource_each(self):
        # Set up. We want different resource types here so that they can be
        # distinguished (to make sure they are stored in the correct zone)
        zli = ZoneAndLinkInfo()
        RES_TYPE1 = ResourceType.CPU
        RES_TYPE2 = ResourceType.MEM
        VALUE = 10
        EXPECTED_RES_ITEM1 = ResourceItem(RES_TYPE1, VALUE)
        EXPECTED_RES_ITEM2 = ResourceItem(RES_TYPE2, VALUE)
        ZONE_ID1 = 1
        ZONE_ID2 = 2
        # Test
        zli.add_resource_value(ZONE_ID1, RES_TYPE1, VALUE)
        zli.add_resource_value(ZONE_ID2, RES_TYPE2, VALUE)
        zone_item1 = zli.copy_zone_item_by_id(ZONE_ID1)
        zone_item2 = zli.copy_zone_item_by_id(ZONE_ID2)
        success = True
        success &= (len(zone_item1.resources) == 1)
        success &= (len(zone_item2.resources) == 1)
        self.assertTrue(success)

    def test_adding_1_resource_in_2_zones_keeps_both_in_correct_place(self):
        # Set up. We want different resource types here so that they can be
        # distinguished (to make sure they are stored in the correct zone)
        zli = ZoneAndLinkInfo()
        RES_TYPE1 = ResourceType.CPU
        RES_TYPE2 = ResourceType.MEM
        VALUE = 10
        EXPECTED_RES_ITEM1 = ResourceItem(RES_TYPE1, VALUE)
        EXPECTED_RES_ITEM2 = ResourceItem(RES_TYPE2, VALUE)
        ZONE_ID1 = 1
        ZONE_ID2 = 2
        # Test
        zli.add_resource_value(ZONE_ID1, RES_TYPE1, VALUE)
        zli.add_resource_value(ZONE_ID2, RES_TYPE2, VALUE)
        zone_item1 = zli.copy_zone_item_by_id(ZONE_ID1)
        zone_item2 = zli.copy_zone_item_by_id(ZONE_ID2)
        success = True
        success &= zone_item1.contains(EXPECTED_RES_ITEM1)
        success &= zone_item2.contains(EXPECTED_RES_ITEM2)
        success &= not zone_item1.contains(EXPECTED_RES_ITEM2)
        success &= not zone_item2.contains(EXPECTED_RES_ITEM1)
        self.assertTrue(success)

    def test_adding_nonzero_qty_link_changes_json(self):
        # Set up
        zli = ZoneAndLinkInfo()
        EXPECTED_JSON = zli.to_json()
        RES_TYPE = ResourceType.CPU
        VALUE = 10
        SRC_ZONE_ID = 1
        DST_ZONE_ID = 2
        # Test
        zli.add_link_value(SRC_ZONE_ID, DST_ZONE_ID, VALUE)
        json = zli.to_json()
        self.assertFalse(EXPECTED_JSON == json)
    
    def test_adding_2_different_links_results_in_2_links(self):
        # Set up
        zli = ZoneAndLinkInfo()
        VALUE = 10
        ZONE_ID1 = 1
        ZONE_ID2 = 2
        # Test
        zli.add_link_value(ZONE_ID1, ZONE_ID2, VALUE)
        zli.add_link_value(ZONE_ID2, ZONE_ID1, VALUE)
        self.assertTrue(zli.get_num_links() == 2)

class TestZoneAndLinkSupplyInfo(unittest.TestCase):
    def test_adding_resource_changes_json(self):
        # Set up
        zlsi = ZoneAndLinkSupplyInfo()
        INITIAL_JSON = zlsi.to_json()
        RES_TYPE = ResourceType.CPU
        QTY = 10
        COST_FUNC = PowerFunc(2, 2)
        ZONE_ID = 1
        PBAR = 4
        # Test
        zlsi.add_resource(ZONE_ID, RES_TYPE, QTY, PBAR, COST_FUNC)
        json = zlsi.to_json()
        self.assertFalse(INITIAL_JSON == json)
    
    def test_adding_link_changes_json(self):
        # Set up
        zlsi = ZoneAndLinkSupplyInfo()
        INITIAL_JSON = zlsi.to_json()
        RES_TYPE = ResourceType.CPU
        QTY = 10
        COST_FUNC = PowerFunc(2, 2)
        ZONE_ID1 = 1
        ZONE_ID2 = 2
        PBAR = 5
        # Test
        zlsi.add_link(ZONE_ID1, ZONE_ID2, QTY, PBAR, COST_FUNC)
        json = zlsi.to_json()
        self.assertFalse(INITIAL_JSON == json)

    def test_simple_zlsi_can_be_constructed_from_its_json(self):
        # Set up
        zlsi = ZoneAndLinkSupplyInfo()
        EXPECTED_JSON = zlsi.to_json()
        # Test
        new_zlsi = ZoneAndLinkSupplyInfo(json=EXPECTED_JSON)
        self.assertTrue(zlsi == new_zlsi)

    def test_complex_zlsi_can_be_constructed_from_its_json(self):
        # Set up
        zlsi = ZoneAndLinkSupplyInfo()
        # Set up - Add resource info
        zlsi.add_resource(1, ResourceType.CPU, 190, 4, PowerFunc(2, 2))
        zlsi.add_resource(1, ResourceType.MEM, 200, 3.3, PowerFunc(3, 6))
        zlsi.add_resource(4, ResourceType.MEM, 6789, 1.1, PowerFunc(1, 2))
        zlsi.add_resource(5, ResourceType.CPU, 12345, 6.9, PowerFunc(0.3, 3.3))
        # Set up - Add link info
        zlsi.add_link(1, 2, 1000, 9.9, PowerFunc(2, 2))
        zlsi.add_link(2, 1, 120, 10, PowerFunc(0.5, 1.9))
        zlsi.add_link(3, 4, 1560, 1e-3, PowerFunc(1, 6))
        zlsi.add_link(4, 5, 34250, 5e2, PowerFunc(3.4, 4))
        EXPECTED_JSON = zlsi.to_json()
        # Test
        new_zlsi = ZoneAndLinkSupplyInfo(json=EXPECTED_JSON)
        self.assertTrue(zlsi == new_zlsi)

class TestPriceResponse(unittest.TestCase):
    def test_price_response_schema_is_valid(self):
        raised = False
        msg = ''
        try:
            Draft7Validator.check_schema(PRICE_RESPONSE_SCHEMA)
        except SchemaError as e:
            raised = True
            msg = e
        self.assertFalse(raised, msg)

    def test_simple_price_response_is_schema_compliant(self):
        pr = PriceResponse()
        json = pr.to_json()
        raised = False
        msg = ''
        try:
            validate(json, PRICE_RESPONSE_SCHEMA)
        except ValidationError as e:
            raised = True
            msg = e
        self.assertFalse(raised, msg)
    
    def test_price_response_can_be_constructed_from_its_json_when_pbar_is_none(self):
        """
        In the future, we may wish to not send pbar values from the auctioneer
        to each of the bidders. In such a case, we should be able to set
        pbar = None without causing issues.
        """
        # Setup
        pr = PriceResponse()
        # Setup - Add resources
        pr.add_resource_price(1, ResourceType.CPU, 10, None)
        # Set up - Add links
        pr.add_link_price(1, 2, 10.2, None)
        json = pr.to_json()
        # Test
        new_pr = PriceResponse(json=json)
        self.assertTrue(pr == new_pr)

    def test_complex_price_response_can_be_constructed_from_its_json(self):
        # Setup
        pr = PriceResponse()
        # Setup - Add resources
        pr.add_resource_price(1, ResourceType.CPU, 10, 12)
        pr.add_resource_price(1, ResourceType.MEM, 0.22, 0.12)
        pr.add_resource_price(2, ResourceType.CPU, 22.22323, 0.0123)
        pr.add_resource_price(2, ResourceType.MEM, 0, 1.23)
        # Set up - Add links
        pr.add_link_price(1, 2, 10.2, 1.34)
        pr.add_link_price(1, 4, 0.01, 43.2)
        pr.add_link_price(2, 1, 0, 1.23)
        pr.add_link_price(2, 6, 32.1332423434, 1.01)
        json = pr.to_json()
        # Test
        new_pr = PriceResponse(json=json)
        self.assertTrue(pr == new_pr)

class TestBundle(unittest.TestCase):
    def test_adding_qty0_resource_does_not_change_json(self):
        # Set up
        b = Bundle()
        EXPECTED_JSON = b.to_json()
        RES_TYPE = ResourceType.CPU
        QTY = 0
        ZONE_ID = 1
        # Test
        b.add_resource(ZONE_ID, RES_TYPE, QTY)
        json = b.to_json()
        self.assertTrue(EXPECTED_JSON == json)

    def test_adding_qty0_to_nonzero_qty_removes_old_entry(self):
        # Set up
        b = Bundle()
        RES_TYPE = ResourceType.CPU
        QTY1 = 10
        QTY2 = 0
        EXPECTED_RES_ITEM = ResourceItem(RES_TYPE, QTY2)
        ZONE_ID = 1
        # Test
        b.add_resource(ZONE_ID, RES_TYPE, QTY1)
        b.add_resource(ZONE_ID, RES_TYPE, QTY2)
        num_zones = b.get_num_zones()
        self.assertTrue(num_zones == 0)

    def test_removal_across_different_zones(self):
        # Set up
        b = Bundle()
        RES_TYPE1 = ResourceType.CPU
        RES_TYPE2 = ResourceType.MEM
        QTY = 10
        EXPECTED_RES_ITEM1 = ResourceItem(RES_TYPE1, QTY)
        EXPECTED_RES_ITEM2 = ResourceItem(RES_TYPE2, QTY)
        ZONE_ID1 = 1
        ZONE_ID2 = 2
        # Test
        b.add_resource(ZONE_ID1, RES_TYPE1, QTY)
        b.add_resource(ZONE_ID2, RES_TYPE2, QTY)
        # Test - Now remove zone 1
        b.add_resource(ZONE_ID1, RES_TYPE1, 0)
        zone_item1 = b.copy_zone_item_by_id(ZONE_ID1)
        zone_item2 = b.copy_zone_item_by_id(ZONE_ID2)
        success = (zone_item1 == None) and (zone_item2 is not None)
        success &= (b.get_num_zones() == 1)
        # Test - Now remove zone 2
        b.add_resource(ZONE_ID2, RES_TYPE2, 0)
        zone_item1 = b.copy_zone_item_by_id(ZONE_ID1)
        zone_item2 = b.copy_zone_item_by_id(ZONE_ID2)
        success &= (zone_item1 == None) and (zone_item2 == None)
        success &= (b.get_num_zones() == 0)
        self.assertTrue(success)

    def test_adding_qty0_link_does_not_change_json(self):
        # Set up
        b = Bundle()
        EXPECTED_JSON = b.to_json()
        QTY = 0
        SRC_ZONE_ID = 1
        DST_ZONE_ID = 2
        # Test
        b.add_link(SRC_ZONE_ID, DST_ZONE_ID, QTY)
        json = b.to_json()
        self.assertTrue(EXPECTED_JSON == json)

    def test_adding_then_removing_2_links(self):
        # Set up
        b = Bundle()
        QTY = 10
        ZONE_ID1 = 1
        ZONE_ID2 = 2
        # Test
        b.add_link(ZONE_ID1, ZONE_ID2, QTY)
        b.add_link(ZONE_ID2, ZONE_ID1, QTY)
        # Test - Now remove 1st link
        b.add_link(ZONE_ID1, ZONE_ID2, 0)
        link1 = b.copy_links_starting_at(ZONE_ID1)
        link2 = b.copy_links_starting_at(ZONE_ID2)
        success = True
        success &= (len(link1) == 0)
        success &= (len(link2) == 1)
        # Test - Now remove 2nd
        b.add_link(ZONE_ID2, ZONE_ID1, 0)
        link1 = b.copy_links_starting_at(ZONE_ID1)
        link2 = b.copy_links_starting_at(ZONE_ID2)
        success = True
        success &= (len(link1) == 0)
        success &= (len(link2) == 0)
        self.assertTrue(success)

class TestBundleRequest(unittest.TestCase):
    def test_bundle_request_schema_is_valid(self):
        raised = False
        msg = ''
        try:
            Draft7Validator.check_schema(BUNDLE_REQUEST_SCHEMA)
        except SchemaError as e:
            raised = True
            msg = e
        self.assertFalse(raised, msg)

    def test_default_bundle_request_is_not_schema_compliant(self):
        # We desire this behaviour because details such as the customer ID need
        # to be set
        br = BundleRequest()
        with self.assertRaises(ValidationError):
            json = br.to_json()

    def test_simple_bundle_request_is_schema_compliant(self):
        br = BundleRequest(0)
        json = br.to_json()
        raised = False
        msg = ''
        try:
            validate(json, BUNDLE_REQUEST_SCHEMA)
        except ValidationError as e:
            raised = True
            msg = e
        self.assertFalse(raised, msg)

    def test_schema_compliance_fails_with_unknown_property(self):
        br = BundleRequest(0)
        json = br.to_json()
        json['not a key'] = -1
        raised = False
        msg = ''
        with self.assertRaises(ValidationError):
            validate(json, BUNDLE_REQUEST_SCHEMA)

    def test_simple_bundle_request_can_be_constructed_from_its_json(self):
        # Set up
        br = BundleRequest(0)
        json = br.to_json()
        # Test
        new_br = BundleRequest(json=json)
        self.assertTrue(br == new_br)

    def test_complex_bundle_can_be_constructed_from_its_json(self):
        # Set up
        b = Bundle()
        b.set_duration(10)
        b.set_payment(0.123)
        # Set up - Add resources
        b.add_resource(1, ResourceType.CPU, 10)
        b.add_resource(1, ResourceType.MEM, 20)
        b.add_resource(2, ResourceType.CPU, 5)
        b.add_resource(2, ResourceType.MEM, 20)
        # Set up - Add links
        b.add_link(1, 2, 10)
        b.add_link(1, 4, 1)
        b.add_link(2, 1, 2)
        b.add_link(2, 6, 13)
        br = BundleRequest(customer_id=42, bundle=b)
        json = br.to_json()
        # Test
        new_br = BundleRequest(json=json)
        self.assertTrue(br == new_br)


if __name__ == "__main__":
    unittest.main()
