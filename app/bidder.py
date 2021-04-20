#!/usr/bin/python
# Author: Tyler Gamvrelis
# Bidder

# Standard library imports
import argparse
import copy
import json
import logging
import os
import random
import requests
import sys
import time

# Local application imports
ppm_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, os.path.abspath(os.path.join(ppm_path, 'ppm')))
from request_response import BidResult, Bundle, BundleRequest, PriceResponse
from resource_type import ResourceType
from utils import get_bid_uri, get_price_uri, get_script_path, setup_logger

# Globals
BASE = 'http://127.0.0.1:5000'
PRICE_URI = BASE + get_price_uri()
BID_URI = BASE + get_bid_uri()
logger = logging.getLogger(__name__)
REPEATED_MODES = ['rand', '10each', '1each', '1cpu', '1gpu', '1link', '1mem']
PBAR_MODES = ['full_range', 'upper_half', 'lower_half', 'variety']

def get_bundles_dir():
    bundles_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'bundles'
    )
    return bundles_dir

def load_bundles():
    """
    Loads the bundles from the bundles directory.
    
    Returns: a dict mapping the bundle's file basename to its corresponding
        Bundle object
    """
    bundles = {}
    bundles_dir = get_bundles_dir()
    for fname in os.listdir(bundles_dir):
        fname_base, ext = os.path.splitext(fname)
        if ext != '.json':
            continue
        with open(os.path.join(bundles_dir, fname), 'r') as f:
            bundle_txt = f.read()
            bundle = Bundle(json=json.loads(bundle_txt))
            bundles[fname_base] = bundle
    return bundles

class BidderParams:
    def __init__(
        self, bundles, sim_time,
        bundle_duration, bundle_mode, pbar_mode,
        puv_cnstr,
        test_mode
    ):
        self.rng = random.Random(2)
        self.bundles = bundles
        self.sim_time = sim_time
        self.bundle_duration = bundle_duration
        self.bundle_mode = bundle_mode
        self.pbar_mode = pbar_mode
        self.puv_cnstr = puv_cnstr
        self.num_bidders = 3 # Hardcoded for sake of time
        self.test_mode = test_mode
    
    def __str__(self):
        sb = ''
        sb += 'Bidding parameters\n'
        sb += f'\tSimulation time: {self.sim_time}\n'
        sb += f'\tBundle duration: {self.bundle_duration}\n'
        sb += f'\tBundle mode: {self.bundle_mode}\n'
        sb += f'\tpbar mode: {self.pbar_mode}\n'
        sb += f'\tPUV cnstr: {self.puv_cnstr}\n'
        sb += f'\tNum bidders: {self.num_bidders}\n'
        return sb

class Bidder:
    id = 1
    def __init__(self, pbar_mode, puv_cnstr, test_mode=False):
        self.test_mode = test_mode
        self.id = Bidder.id
        Bidder.id += 1
        self.rng = random.Random(self.id)
        self.pbar_mode = pbar_mode
        self.puv_cnstr = puv_cnstr
        # Statistics
        self.num_requests = 0
        self.num_successes = 0
        self.total_payment_offered = 0
        self.total_payment_accepted = 0

    def _get_item_val(self, pbar):
        """Returns a valuation chosen from within the correct pbar range."""
        if self.pbar_mode == 'full_range':
            val_range = (0, pbar)
        elif self.pbar_mode == 'upper_half':
            val_range = (pbar / 2, pbar)
        elif self.pbar_mode == 'lower_half':
            val_range = (0, pbar / 2)
        else:
            raise NotImplementedError()
        return self.rng.uniform(*val_range)

    def _compute_valuation(self, bundle, prices):
        val = 0
        all_vals = []
        # Determine valuation of each resource
        zones = bundle.copy_zones()
        for zone in bundle.zones:
            zone_id = zone.zone_id
            zone_price = prices.copy_zone_item_by_id(zone_id)
            for resource in zone.resources:
                res_type = resource.get_res_type()
                res_price = zone_price.get_resource_item_by_type(res_type)
                pbar = res_price.get_pbar()
                qty = resource.get_value()
                res_val = self._get_item_val(pbar) * qty
                all_vals.append(res_val)
        # Determine valuation of each link
        links = bundle.copy_links()
        for link in links:
            link_price = prices.copy_link(*link.get_src_and_dst())
            pbar = link_price.get_pbar()
            qty = link.get_value()
            link_val = self._get_item_val(pbar) * qty
            all_vals.append(link_val)
        # Determine final valuation
        if self.puv_cnstr:
            val = min(all_vals)
        else:
            val = sum(all_vals)
        val *= bundle.duration
        return val

    def bid(self, bundle):
        # Get prices
        response = requests.get(PRICE_URI)
        prices = PriceResponse(json=response.json())
        if self.test_mode:
            logger.debug(prices)
        # Compute valuation
        val = self._compute_valuation(bundle, prices)
        bundle.set_payment(val)
        br = BundleRequest(self.id, bundle)
        # Get result
        response = requests.post(BID_URI, json=br.to_json())
        result = BidResult(json=response.json())
        # Update bookkeeping
        self.num_requests += 1
        self.total_payment_offered += val
        if result.success:
            self.num_successes += 1
            self.total_payment_accepted += val
        else:
            logger.debug(f'Bidder {self.id}: {str(result)}')

    def __str__(self):
        """Present results."""
        sb = ''
        sb += f'Bidder {self.id}\n'
        sb += f'\tpbar mode: {self.pbar_mode}\n'
        sb += f'\tPUV cnstr: {self.puv_cnstr}\n'
        sb += f'\tNum requests: {self.num_requests}\n'
        try:
            succ_perc = 100 * self.num_successes / self.num_requests
        except ZeroDivisionError:
            succ_perc = 0
        sb += f'\tNum successes: {self.num_successes} (' + \
            str(round(succ_perc, 2)) + '%)\n'
        sb += f'\tTotal payment offered: {self.total_payment_offered}\n'
        try:
            payment_perc = 100 * self.total_payment_accepted / self.total_payment_offered
        except ZeroDivisionError:
            payment_perc = 0
        sb += f'\tTotal payment accepted: {self.total_payment_accepted} (' + \
            str(round(payment_perc, 2)) + '%)\n'
        return sb

def run_bidders(params):
    """
    Runs several bidders in a loop for a specified time then reports results.
    """
    # 1. Set up sim based on params
    # 1.1 Create bidders
    feasible_bundles = [
        bundle for key, bundle in params.bundles.items()
        if 'infeasible' not in key
    ]
    bidders = []
    for i in range(params.num_bidders):
        # pbar mode - rotate across bidders if variety is chosen
        if params.pbar_mode == 'variety':
            NUM_MODES = len(PBAR_MODES)
            assert NUM_MODES == 4, 'This part needs generalization'
            idx = i % (NUM_MODES - 1)
            pbar_mode = PBAR_MODES[idx]
        else:
            pbar_mode = params.pbar_mode
        b = Bidder(pbar_mode, params.puv_cnstr, test_mode=params.test_mode)
        bidders.append(b)
    # 2. Run sim & collect stats
    sim_end = time.time() + params.sim_time
    cur_time = time.time()
    while cur_time < sim_end:
        for bidder in bidders:
            # Pick bundle for bidder...
            if params.bundle_mode in REPEATED_MODES[1:]:
                bundle = copy.deepcopy(params.bundles[params.bundle_mode])
            elif params.bundle_mode == 'rand':
                bundle = copy.deepcopy(params.rng.choice(feasible_bundles))
            else:
                raise NotImplementedError()
            bundle.set_duration(params.bundle_duration)
            # Send the request!
            bidder.bid(bundle)
        #time.sleep(1) # to start with...
        cur_time = time.time()
    # 3. Present results
    logger.info(str(params))
    revenue = 0
    for bidder in bidders:
        logger.info(str(bidder))
        revenue += bidder.total_payment_accepted
    logger.info(f'Auctioneer revenue: {revenue}')

def check_pbar_mode(mode):
    if mode not in PBAR_MODES:
        raise argparse.ArgumentTypeError(
            f'{mode} is an invalid  pbar mode type. Must be in {PBAR_MODES}'
        )
    return mode

def check_repeated_mode(mode):
    if mode not in REPEATED_MODES:
        raise argparse.ArgumentTypeError(
            f'{mode} is an invalid mode type. Must be in {REPEATED_MODES}'
        )
    return mode

def check_nonnegative(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            f'{value} is an invalid nonnegative int value'
        )
    return ivalue

def parse_args():
    """Parses command-line arguments."""
    os.chdir(get_script_path())
    parser = argparse.ArgumentParser(description='Bidder')
    parser.add_argument('--log', help='Set log level', default='info')
    parser.add_argument(
        '--test',
        help='Enables test mode',
        dest='test', action='store_true', default=False
    )
    # Standalone mode. These arguments are for sending a specific bundle request
    parser.add_argument(
        '--bundle',
        help='Standalone mode. '
            'Indicates a specific bundle to use for a single bid request',
        default=None,
    )
    parser.add_argument(
        '--id',
        help='Standalone mode. '
            'Set customer ID. Required if using --bundle argument',
        type=check_nonnegative
    )
    parser.add_argument(
        '--use_price_token',
        help='Standalone mode. '
            'Uses price freeze tokens, if available',
        dest='use_price_token', action='store_true', default=False
    )
    parser.add_argument(
        '--delay_sec',
        help='Standalone mode. '
            'Time to wait between getting prices and requesting bid',
        type=int,
        default=None
    )
    # Repeated mode
    parser.add_argument(
        '--sim_time',
        help='How long to send requests for, in seconds',
        type=int,
        default=120
    )
    parser.add_argument(
        '--bundle_duration',
        help='How long each bundle is requested for, in seconds',
        type=int,
        default=10
    )
    parser.add_argument(
        '--bundle_mode',
        help=f'Controls which bundles are sent. Must be in {REPEATED_MODES}',
        type=check_repeated_mode,
        default=REPEATED_MODES[0]
    )
    parser.add_argument(
        '--pbar_mode',
        help=f'Controls which bundles are sent. Must be in {PBAR_MODES}',
        type=check_pbar_mode,
        default=PBAR_MODES[0]
    )
    parser.add_argument(
        '--puv_cnstr',
        help=f'Enables per-unit valuation constraint',
        dest='puv_cnstr',
        action='store_true',
        default=False
    )
    # TODO: add an option that tests the price token by maybe making 2 users slow while keeping one user fast
    return vars(parser.parse_args())

def main():
    """Main program logic."""
    args = parse_args()
    log_level = args['log']
    test_mode = args['test']
    customer_id = args['id']
    bundle = args['bundle']
    use_price_token = args['use_price_token']
    delay_sec = args['delay_sec']
    
    setup_logger(log_level, __file__)
    logger.info('Started app')
    if test_mode:
        logger.info('~~TEST MODE~~')

    # Validate input
    if bundle is not None and customer_id == None:
        logger.error('bundle argument was provided without customer ID')
        sys.exit(1)

    # Run app
    bundles = load_bundles()
    if bundle is not None:
        # We are running a single bid with a specific bundle
        if bundle not in bundles.keys():
            logger.error(
                f'bundle arugment \'{bundle}\' is invalid. Must be in '
                f'{str(list(bundles.keys()))}'
            )
            sys.exit(1)
        # Get prices
        response = requests.get(PRICE_URI)
        prices = PriceResponse(json=response.json())
        logger.debug(prices)
        if delay_sec is not None:
            logger.info(f'Waiting for {delay_sec} seconds...')
            time.sleep(delay_sec)
        # Use price freeze token, if requested
        b = bundles[bundle]
        if use_price_token and prices.has_price_token():
            token = prices.get_price_token()
            logger.info(f'Bidding with price token {token}')
            b.set_price_token(token)
        # Send bid
        br = BundleRequest(customer_id, b)
        response = requests.post(BID_URI, json=br.to_json())
        result = BidResult(json=response.json())
        logger.info(result)
    else:
        params = BidderParams(
            bundles,
            args['sim_time'],
            args['bundle_duration'],
            args['bundle_mode'],
            args['pbar_mode'],
            args['puv_cnstr'],
            test_mode
        )
        run_bidders(params)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('Interrupted: {0}'.format(e))
        logger.info('Exiting...')
    sys.exit(0)
