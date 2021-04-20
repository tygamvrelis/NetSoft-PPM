#!/usr/bin/python
# Author: Tyler Gamvrelis
# Auctioneer

# Standard library imports
import argparse
import atexit
import logging
import os
import sys
import time

# Third party imports
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MAX_INSTANCES
from flask import Flask, request
from flask_restful import abort, Api, reqparse
from flask_restful import Resource as FlaskResource
from jsonschema import ValidationError

# Local application imports
ppm_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, os.path.abspath(os.path.join(ppm_path, 'ppm')))
from ppm import PPM
from request_response import BundleRequest, BidResult
from utils import get_bid_uri, get_price_uri, get_script_path, setup_logger

# Globals
ppm = None
logger = logging.getLogger(__name__)

# Set up background tasks
def time_evolution():
    """Triggers periodic tasks for simulation purposes."""
    logger.debug('Evolving time...')
    ppm.evolve_time()

def job_error_handler(event):
    logger.debug(f'time_evolution error: {event}')
    global sched
    sched.remove_job('time_evolution')

# Useful source:
#    - https://stackoverflow.com/questions/21214270/how-to-schedule-a-function-to-run-every-hour-on-flask
sched = BackgroundScheduler(daemon=True)
sched.add_job(time_evolution, 'interval', seconds=1, id='time_evolution')
sched.add_listener(job_error_handler, EVENT_JOB_ERROR | EVENT_JOB_MAX_INSTANCES)
executor_logger = logging.getLogger('apscheduler')
executor_logger.setLevel(logging.WARNING)
atexit.register(lambda: sched.shutdown(wait=False))
sched.start()

# Set up REST API
app = Flask(__name__)
api = Api(app)

def customer_id_is_valid(customer_id):
    """
    Returns True if the given customer ID is valid, otherwise False.

    Args:
        customer_id : int
            The object whose validity as a customer ID is to be checked
    """
    retval = False
    if customer_id >= 0:
        retval = True
    return retval

class ResourcesPrices(FlaskResource):
    """Request handler for resource prices."""

    def get(self):
        client_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        client_port = request.environ.get('REMOTE_PORT')
        logger.info(
            f'Received price request from client {client_ip}:{client_port}'
        )
        price_response = ppm.get_price_list()
        return price_response.to_json(), 200

class ResourcesBids(FlaskResource):
    """Request handler for bids on resources."""
    bid_id = 0

    def post(self):
        client_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        client_port = request.environ.get('REMOTE_PORT')
        logger.info(f'Received bid from client {client_ip}:{client_port}')
        ResourcesBids.bid_id += 1
        # Try to parse the request
        json_data = request.get_json(force=True)
        success = (json_data is not None)
        code = 400
        result = BidResult(False, ResourcesBids.bid_id)
        if success:
            try:
                br = BundleRequest(json=json_data)
            except ValidationError as e:
                success = False
                result.msg = e.message
        else:
            result.msg = 'Bid request must include JSON data'
        # Validate customer ID
        if success:
            success = customer_id_is_valid(br.customer_id)
            if not success:
                result.msg = f'Customer ID {br.customer_id} is invalid'
        # If everything up until now has been valid, handle the bid request
        if success:
            logger.info(f'Customer offering {br.bundle.payment} for {br}')
            result = ppm.bid(br)
            result.bid_id = ResourcesBids.bid_id
            logger.info(f'Bid ID: {result.bid_id}')
            code = 201
        retval = result.to_json()
        return retval, code

api.add_resource(ResourcesPrices, get_price_uri())
api.add_resource(ResourcesBids, get_bid_uri())

def parse_args():
    """Parses command-line arguments."""
    os.chdir(get_script_path())
    parser = argparse.ArgumentParser(description='PPM Auctioneer')
    parser.add_argument('--log', help='Set log level', default='info')
    parser.add_argument(
        '--test', help='Enables test mode', dest='test',
        action='store_true', default=False
    )
    parser.add_argument(
        '--use_price_token', help='Enables price freeze tokens',
        dest='use_price_token', action='store_true', default=False
    )
    return vars(parser.parse_args())
        
def main():
    """Main program logic."""
    args = parse_args()
    log_level = args['log']
    test_mode = args['test']
    use_price_token = args['use_price_token']

    setup_logger(log_level, __file__)
    logger.info('Started app')
    if test_mode:
        logger.info('~~TEST MODE~~')

    global ppm
    ppm = PPM(use_price_token=use_price_token)

    # Run REST API
    app.run(debug=test_mode, threaded=False)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('Interrupted: {0}'.format(e))
        logger.info('Exiting...')
    sys.exit(0)
