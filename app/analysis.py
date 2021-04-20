#!/usr/bin/python
# Author: Tyler Gamvrelis
# Real-time plotting

# Standard library imports
import argparse
from collections import defaultdict
import json
import logging
import multiprocessing as mp
import numpy as np
import time
import os
import sys
from time import time
import tkinter as tk

# Third party imports
import matplotlib.cm
import matplotlib.pyplot as plt

# Local application imports
ppm_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, os.path.abspath(os.path.join(ppm_path, 'ppm')))
from request_response import ZoneAndLinkInfo
from resource_type import ResourceType
from utils import get_metrics_log_path, get_script_path, setup_logger

# Globals
logger = logging.getLogger(__name__)

# Helpers
event = mp.Event()
def report_callback_exception(self, exc, val, tb):
    # If we close a plot while it is doing something in the backend, there will
    # likely be a bunch of tk exceptions (which are annoying). Instead of these
    # printing to the terminal and making it hard to quit our program, we catch
    # these exceptions here and set a multiprocessing event which will cause
    # the file loop process to return, hence leading to a graceful shutdown
    global event
    event.set()
tk.Tk.report_callback_exception = report_callback_exception
# https://stackoverflow.com/questions/31883097/elegant-way-to-match-a-string-to-a-random-color-matplotlib
def get_cmap_string(domain, palette='viridis'):
    domain_unique = np.unique(domain)
    hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
    mpl_cmap = matplotlib.cm.get_cmap(palette, lut=len(domain_unique))

    def cmap_out(X, **kwargs):
        return mpl_cmap(hash_table[X], **kwargs)

    return cmap_out

def get_link_str(src_zone_id, dst_zone_id):
    if src_zone_id < dst_zone_id:
        arrow = '-->'
    else:
        arrow = '<--'
    return str(src_zone_id) + arrow + str(dst_zone_id)

# Plot class
class Plotter:
    def __init__(self, plot_type='util', max_pts=120, n_rows=2, fig_size=(10,8)):
        # Data stuff
        self.tvals = []
        self.zone_vals = defaultdict(dict)
        self.link_vals = defaultdict(dict)
        self.zli = None
        # Plot stuff
        self._plot_type = plot_type
        self.zone_axs = {}
        self.link_axs = None
        self.fig = None
        self.fig_size = fig_size
        self.max_pts = max_pts
        self.n_rows = n_rows
        # Color stuff
        self.zone_cmap = None
        self.link_cmap = None
        
    def has_closed(self):
        """
        Returns True if the fig was once open but is now closed; otherwise
        False.
        """
        retval = False
        if self.fig is not None:
            retval = not plt.fignum_exists(self.fig.number)
        return retval

    def handle_zli(self, t, zli):
        """Updates plot vars."""
        # 1. Time
        self.tvals.append(t)
        # Wrap around time if needed
        if self.max_pts != None:
            self.tvals = self.tvals[-self.max_pts:]
        # 2. Do zones:
        zones = zli.copy_zones()
        for zone in zones:
            zone_id = zone.zone_id
            for resource in zone.resources:
                res_type = resource.get_res_type()
                price, usage, supply = resource.get_value()
                if self._plot_type == 'util':
                    util = 100 * usage / supply
                    val = util
                elif self._plot_type == 'price':
                    val = price
                else:
                    raise NotImplementedError()
                try:
                    zone_vals_list = self.zone_vals[zone_id][res_type]
                    zone_vals_list.append(val)
                    # Wrap around if needed
                    if self.max_pts != None:
                        zone_vals_list[:] = zone_vals_list[-self.max_pts:]
                except KeyError:
                    self.zone_vals[zone_id][res_type] = [val]
        # 3. Do links
        links = zli.copy_links()
        for link in links:
            src_zone_id, dst_zone_id = link.get_src_and_dst()
            price, usage, supply = link.get_value()
            if self._plot_type == 'util':
                util = 100 * usage / supply
                val = util
            elif self._plot_type == 'price':
                val = price
            else:
                raise NotImplementedError()
            try:
                link_vals_list = self.link_vals[src_zone_id][dst_zone_id]
                link_vals_list.append(val)
                # Wrap around if needed
                if self.max_pts != None:
                    link_vals_list[:] = link_vals_list[-self.max_pts:]
            except KeyError:
                self.link_vals[src_zone_id][dst_zone_id] = [val]
        # 4. Save ZLI so we can use its structure later
        self.zli = zli

    def _make_plots_if_needed(self, zones, links):
        """Sets up the plot if first time."""
        # 1. Check whether we need to do anything
        if self.fig is None:
            # Some ugly setup stuff
            num_zones = len(zones)
            has_links = len(links) > 0
            num_plots = num_zones
            if has_links:
                num_plots += 1
            # Make fig canvas
            self.fig = plt.figure(figsize=self.fig_size)
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=0.4, wspace=0.4)
            num_cols = (num_plots // self.n_rows) + (num_plots % self.n_rows)
            i = 1
            # Zone subplots
            for zone in zones:
                zone_id = zone.zone_id
                ax = self.fig.add_subplot(self.n_rows, num_cols, i)
                self.zone_axs[zone_id] = ax
                i += 1
            self.zone_cmap = get_cmap_string(
                [res.name for res in list(ResourceType)]
            )
            # Link subplots
            if has_links:
                link_cmap_strs = []
                ax = self.fig.add_subplot(self.n_rows, num_cols, i)
                self.link_axs = ax
                for link in links:
                    src_zone_id, dst_zone_id = link.get_src_and_dst()
                    link_str = get_link_str(src_zone_id, dst_zone_id)
                    link_cmap_strs.append(link_str)
                self.link_cmap = get_cmap_string(link_cmap_strs)
            # Show it...
            plt.ion()
            plt.show()

    def update_plots(self):
        """Updates real-time plots."""
        # 1. Setup
        zones = self.zli.copy_zones()
        links = self.zli.copy_links()
        self._make_plots_if_needed(zones, links)
        # 2. Do zones
        for zone in zones:
            zone_id = zone.zone_id
            ax = self.zone_axs[zone_id]
            # Clear then replot
            ax.clear()
            ymax = 0
            for resource in zone.resources:
                res_type = resource.get_res_type()
                val = self.zone_vals[zone_id][res_type]
                ax.plot(
                    self.tvals, val, c=self.zone_cmap(res_type.name),
                    label=res_type.name, marker='o'
                )
                ymax = max(ymax, max(val))
            if self._plot_type == 'util':
                ax.set_ylabel('Utilization (%)')
                default_y_val = 100
            else:
                ax.set_ylabel('Price')
                default_y_val = 100
            ax.set_xlabel('Time')
            ax.set_title(f'Zone {zone_id}')
            ax.legend()
            # Don't let y axis get too crowded with ticks
            ymin = 0
            if ymin != ymax:
                yticks =  np.linspace(ymin, ymax, 5)
            else:
                yticks = np.linspace(ymin, default_y_val, 5)
            ax.set_yticks(yticks)
        # 3. Do links
        # Clear then replot
        self.link_axs.clear()
        ymax = 0
        for link in links:
            src_zone_id, dst_zone_id = link.get_src_and_dst()
            link_str = get_link_str(src_zone_id, dst_zone_id)
            val = self.link_vals[src_zone_id][dst_zone_id]
            self.link_axs.plot(
                self.tvals, val, c=self.link_cmap(link_str),
                label=link_str, marker='o'
            )
            ymax = max(ymax, max(val))
        if self._plot_type == 'util':
            self.link_axs.set_ylabel('Utilization (%)')
            default_y_val = 100
        else:
            self.link_axs.set_ylabel('Price')
            default_y_val = 100
        self.link_axs.set_xlabel('Time')
        self.link_axs.set_title('Links')
        self.link_axs.legend()
        # Don't let y axis get too crowded with ticks
        ymin = 0
        if ymin != ymax:
            yticks =  np.linspace(ymin, ymax, 5)
        else:
            yticks = np.linspace(ymin, default_y_val, 5)
        self.link_axs.set_yticks(yticks)
        # 4. Draw plot
        self.fig.canvas.draw()

# Main logic
def get_zli_from_line(line):
    """Parses line into a queryable object."""
    global tvals, zone_prices, zone_utils, link_prices, link_utils, zli
    data = json.loads(line)
    # 1. Hacky. Use ZLI class to build an easily-queryable object
    zli = ZoneAndLinkInfo()
    for key1, info in data.items():
        if key1 == 'd_t':
            continue
        for key2, val in info.items():
            if val[0] == ResourceType.LINK.name:
                src_zone_id = int(key1)
                dst_zone_id = int(key2)
                zli.add_link_value(src_zone_id, dst_zone_id, val[1:])
            else:
                zone_id = int(key1)
                res_type = ResourceType[val[0]]
                zli.add_resource_value(zone_id, res_type, val[1:])
    t = float(data['d_t'])
    return t, zli

def file_loop(max_pts):
    """
    Source:
    - https://stackoverflow.com/questions/3290292/read-from-a-log-file-as-its-being-written-using-python
    """
    global event
    metrics_log = get_metrics_log_path()
    with open(metrics_log, 'r') as f:
        do_plots = False
        cond = True
        util_plotter = Plotter(plot_type='util', max_pts=max_pts)
        price_plotter = Plotter(plot_type='price', max_pts=max_pts)
        plotters = [util_plotter, price_plotter]
        while cond:
            where = f.tell()
            line = f.readline()
            if not line:
                if do_plots:
                    # If we didn't get a line then it means we reached the end
                    # of the file. In this case, we should update our plots as
                    # long as there is new data to process since the last time
                    # we plotted
                    for plotter in plotters:
                        plotter.update_plots()
                    do_plots = False
                for plotter in plotters:
                    if plotter.fig is not None:
                        plotter.fig.canvas.flush_events()
                f.seek(where)
            else:
                # Handle the line and set the flag indicating that plots need
                # to be updated
                t, zli = get_zli_from_line(line)
                for plotter in plotters:
                    plotter.handle_zli(t, zli)
                do_plots = True
            # Update termination condition
            cond = not event.is_set()
            all_plots_closed = True
            for plotter in plotters:
                all_plots_closed &= util_plotter.has_closed()
            cond &= not all_plots_closed

def parse_args():
    """Parses command-line arguments."""
    os.chdir(get_script_path())
    parser = argparse.ArgumentParser(description='Data analyzer')
    parser.add_argument('--log', help='Set log level', default='info')
    parser.add_argument(
        '--test', help='Enables test mode', dest='test',
        action='store_true', default=False
    )
    parser.add_argument(
        '--max_pts', help='Max points before plots start wrapping',
        type=int, default=120
    )
    return vars(parser.parse_args())

def main():
    """Main program logic."""
    args = parse_args()
    log_level = args['log']
    test_mode = args['test']
    max_pts = args['max_pts']

    setup_logger(log_level, __file__)
    logger.info('Started app')
    if test_mode:
        logger.info('~~TEST MODE~~')
    
    # Start plotter in a daemon process...makes closing gracefully quite easy
    p = mp.Process(target=file_loop, args=(max_pts,))
    p.daemon = True
    p.start()
    while p.is_alive():
        pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('Interrupted: {0}'.format(e))
        logger.info('Exiting...')
    sys.exit(0)
