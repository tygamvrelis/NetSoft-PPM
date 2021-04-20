#!/usr/bin/python
# Author: Tyler Gamvrelis
# PPM

# Standard library imports
from collections import defaultdict
import logging
from queue import PriorityQueue, Empty
from threading import Lock
from time import time

# Local application imports
from request_response import BidResult, PriceResponse
from resource_manager_sim import ResourceManagerSim
from resource_type import ResourceType

# Globals
logger = logging.getLogger(__name__)

class PPM:
    """Implements an auctioneer for the PPM algorithm."""

    def __init__(self, use_price_token=False, token_duration=10):
        """
        Initializes the PPM auctioneer.
        
        Args:
            use_price_token : bool
                Indicates whether a price freeze token should be issue upon a
                price request. If a customer comes back within token_duration,
                then they get whatever price they saw before
            token_duration : int
                If price tokens are enables, this parameter controls how long
                they last before they expire
        """
        self._res_man = ResourceManagerSim()
        self._revenue = 0
        self._expenses = 0
        # If freezing prices for some time...
        self._use_price_token = use_price_token
        self._history_lock = Lock()
        self._hist_q = PriorityQueue()
        self._price_history = defaultdict()
        self._num_req = 0
        self._token_duration = token_duration # seconds
        if self._use_price_token:
            logger.info('PPM is using price tokens')

    def get_price_list(self):
        """Returns a list of per-unit prices for each resource."""
        # Fetch the resource info then get a copy of it
        self._res_man.collect_resource_info()
        res_list = self._res_man.get_resource_info()
        # Pack the price response!
        price_response = PriceResponse()
        for res in res_list:
            price = res.post_price()
            res_type = res.get_res_type()
            if res_type == ResourceType.LINK:
                src_zone_id, dst_zone_id = res.get_src_and_dst()
                price_response.add_link_price(
                    src_zone_id,
                    dst_zone_id,
                    price,
                    res.get_max_valuation()
                )
            else:
                price_response.add_resource_price(
                    res.get_zone_id(),
                    res_type,
                    price,
                    res.get_max_valuation()
                )
        # Save prices for later, if needed
        if self._use_price_token:
            # Generate token
            self._num_req += 1
            token = hash(str(self._num_req))
            price_response.set_price_token(token)
            # Get mapping
            price_mapping = self._res_man.get_res_to_price_mapping()
            # Get priority
            pri = time() + self._token_duration
            with self._history_lock:
                self._hist_q.put((pri, token))
                self._price_history[token] = price_mapping
            logger.debug(
                f'Saved prices with token {token} '
                f'for {self._token_duration} seconds'
            )
        return price_response

    def evolve_time(self):
        """Causes time-dependent updates to be performed, if any."""
        if isinstance(self._res_man, ResourceManagerSim):
            # If our resource manager has simulated aspects, then evolve the
            # simulation
            self._res_man.evolve_time()
        # Update price histories, if needed
        if self._use_price_token:
            t = time()
            with self._history_lock:
                while True:
                    try:
                        pri, token = self._hist_q.get_nowait()
                    except Empty:
                        # Nothing in queue, so nothing to do
                        break
                    if pri < t:
                        try:
                            del self._price_history[token]
                            logger.debug(f'Removed saved price with token {token}')
                        except KeyError:
                            # Token already deleted due to usage
                            pass
                    else:
                        # Add back if this item's time has not yet come
                        self._hist_q.put((pri, token))
                        break

    def _compute_bundle_price(self, bundle):
        """
        Returns the current price of the given bundle.

        Args:
            bundle : Bundle
                The resource bundle whose price is to be computed
        """
        details = ''
        # 1. Build a mapping from resource-specific info to price
        res_to_price_map = self._res_man.get_res_to_price_mapping()
        if self._use_price_token and bundle.has_price_token():
            # Use token if enabled & provided. If token has expired, we don't
            # change the res_to_price_map, which causes the most recent prices
            # to be used instead
            token = bundle.get_price_token()
            try:
                with self._history_lock:
                    price_mapping = self._price_history[token]
                    res_to_price_map = price_mapping
                logger.debug(f'Using prices saved with token {token}')
            except KeyError:
                details = f'token {token} has expired!'
                logger.debug(details)
        # 2. Zones
        zone_price = 0
        zones = bundle.copy_zones()
        for zone in zones:
            zone_id = zone.zone_id
            for resource in zone.resources:
                res_type = resource.get_res_type()
                qty = resource.get_value()
                price = res_to_price_map[zone_id][res_type]
                zone_price += qty * price
        # 3. Links
        link_price = 0
        links = bundle.copy_links()
        for link in links:
            src_zone_id, dst_zone_id = link.get_src_and_dst()
            link_price += qty * res_to_price_map[src_zone_id][dst_zone_id]
            price = res_to_price_map[src_zone_id][dst_zone_id]
        # 4. Account for bundle duration
        duration = bundle.duration
        price = zone_price + link_price
        full_price = price * duration
        logger.debug(
            f'Bundle $ = {full_price} for duration {duration}. '
            f'Per time unit: price = {price} '
            f'(zones = {zone_price}, links = {link_price})'
        )
        return full_price, details

    def _bundle_is_feasible(self, bundle):
        """
        Returns True if the bundle is feasible, otherwise False. If False, a
        reason for infeasibility is returned as well. Feasibility pertains to
        whether:
            1. acceptance of the bundle will exceed resource supply
            2. the bundle requests any resources that do not exist
            3. the bundle requests links that begin or terminate in a zone
               where no resource is requested
        Note that the case where the bundle requests resources in different
        zones but there's no link between the zones could be valid.
        (Communication between zones is not necessarily a requirement for a
        valid bundle. We also do not require links to be unidirectional.)
            
        Args:
            bundle : Bundle
                The resource bundle whose feasibility is to be determined
        """
        is_feasible = True
        reason = ''
        # 1. Build a mapping from resource-specific info to resource record
        res_to_record_mapping = self._res_man.get_res_to_record_mapping()
        # 2. Check feasibility of zones
        zones = bundle.copy_zones()
        for zone in zones:
            zone_id = zone.zone_id
            for resource in zone.resources:
                # If there's a key error, it means a zone-resource type pair
                # was requested which does not exist in the infrastructure
                res_type = resource.get_res_type()
                try:
                    record = res_to_record_mapping[zone_id][res_type]
                except KeyError:
                    is_feasible = False
                    reason = f'zone {zone_id} does not have {res_type}'
                    break
                # Check if we can supply the requested quantity of this
                # resource in this zone
                qty = resource.get_value()
                if not record.can_hold_additional_qty(qty):
                    is_feasible = False
                    reason = f'would exceed supply of resource {res_type} in '
                    reason += f'zone {zone_id}'
                    break
        # 3. Links
        if is_feasible:
            links = bundle.copy_links()
            for link in links:
                # If there's a key error, it means a zone-resource type pair
                # was requested which does not exist in the infrastructure
                src_zone_id, dst_zone_id = link.get_src_and_dst()
                try:
                    record = res_to_record_mapping[src_zone_id][dst_zone_id]
                except KeyError:
                    is_feasible = False
                    reason = f'link {src_zone_id} --> {dst_zone_id} '
                    reason += 'does not exist'
                    break
                # Check whether the zones that the link starts and ends in
                # have any resource requests associated with them. Because of
                # how bundles are implemented, we know that if the zone exists
                # then there are resource requests associated with that zone
                src_zone = bundle.copy_zone_item_by_id(src_zone_id)
                has_src = (src_zone is not None)
                dst_zone = bundle.copy_zone_item_by_id(dst_zone_id)
                has_dst = (dst_zone is not None)
                if not (has_src and has_dst):
                    is_feasible = False
                    reason = f'link {src_zone_id} --> {dst_zone_id} '
                    if not has_src:
                        reason += 'starts '
                    if not has_src and not has_dst:
                        reason += 'and '
                    if not has_dst:
                        reason += 'ends '
                    reason += 'at a zone that no resources have been '
                    reason += 'requested for'
                    break
                # Check if we can supply the requested link quantity
                qty = link.get_value()
                if not record.can_hold_additional_qty(qty):
                    is_feasible = False
                    reason = f'would exceed supply of link {src_zone_id} --> '
                    reason += f'{dst_zone_id}'
                    break
        return is_feasible, reason

    def _accept_bundle(self, bundle):
        """
        Updates resource usage levels as a result of accepting the given
        bundle.

        Args:
            bundle : Bundle
                The resource bundle to be accepted
        """
        duration = bundle.duration
        supply_cost = 0
        # 1. Build a mapping from resource-specific info to resource record
        res_to_record_mapping = self._res_man.get_res_to_record_mapping()
        # 2. Add usage for zones
        zones = bundle.copy_zones()
        for zone in zones:
            zone_id = zone.zone_id
            for resource in zone.resources:
                res_type = resource.get_res_type()
                qty = resource.get_value()
                record = res_to_record_mapping[zone_id][res_type]
                self._res_man.update_res_usage(record, qty, duration)
                supply_cost += record.get_supply_cost()
        # 3. Add usage for links
        links = bundle.copy_links()
        for link in links:
            src_zone_id, dst_zone_id = link.get_src_and_dst()
            qty = link.get_value()
            record = res_to_record_mapping[src_zone_id][dst_zone_id]
            self._res_man.update_res_usage(record, qty, duration)
            supply_cost += record.get_supply_cost()
        # 4. Update bookkeeping
        self._revenue += bundle.payment
        self._expenses += supply_cost
        logger.debug(
            'Updating books...\n'
            f'\tTotal revenue: {self._revenue}\n'
            f'\tTotal supply cost: {self._expenses}\n'
        )
        if self._use_price_token and bundle.has_price_token():
            # Need to expire the token if it was used to compute the prices
            # this time
            token = bundle.get_price_token()
            with self._history_lock:
                try:
                    del self._price_history[token]
                    logger.debug(f'Deleted token {token}')
                except KeyError:
                    # Token happened to expire between time prices were computed
                    # and here
                    pass

    def bid(self, br):
        """
        Core bidding logic.

        Args:
            br : BundleRequest
                The object containing the bundle request along with other data

        Returns: a BidResult object containing the result of this bid attempt
        """
        success = True
        failure_reason = None
        bundle = br.bundle

        # 1. Check edge cases. For example, we don't want to accept a
        # customer's payment if they forgot to include resources in their
        # request
        if success:
            has_zones = bundle.get_num_zones() > 0
            has_links = bundle.get_num_links() > 0
            if not (has_zones or has_links):
                success = False
                failure_reason = 'Bid must contain at least one resource.'
                logger.debug(failure_reason)
            if success and bundle.duration == 0:
                success = False
                failure_reason = 'Bid duration must be > 0'
                logger.debug(failure_reason)

        # 2. Check that accepting the bundle will not exceed resource supply
        if success:
            is_feasible, reason = self._bundle_is_feasible(bundle)
            if not is_feasible:
                success = False
                failure_reason = 'Infeasible bundle (' + reason + ')'
                logger.debug(failure_reason)

        # 3. Check that payment is acceptable
        if success:
            price, details = self._compute_bundle_price(bundle)
            if bundle.payment < price:
                success = False
                failure_reason = f'Insufficient payment ({bundle.payment}). ' \
                    'Perhaps prices changed since you last retrieved them.'
                if details:
                    failure_reason += ' Details: ' + details
                logger.debug(failure_reason)

        # 4. If previous checks succeeded, then accept the bundle
        if success:
            self._accept_bundle(bundle)

        # 5. Regardless of whether bid was accepted, store a record of it
        # TODO: store a record of the bid in the DB. This data could be
        # important for analytics purposes
        return BidResult(success, -1, failure_reason)
