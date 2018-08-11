#!/usr/env python

import argparse
import os.path

def arg_file(string):
    """Validate the input string as a file."""
    if not os.path.isfile(string):
        msg = "{0} is not a file".format(string)
        raise argparse.ArgumentTypeError(msg)
    return string

def main():
    parser = argparse.ArgumentParser(description="Utility to generate histograms of the task duration")
    parser.add_argument('--input',
                        dest='input_file', metavar='INPUT', nargs=1, type=arg_file,
                        help='The input file containing the experimental data')

    args = parser.parse_args()

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import math
    import numpy as np

    dt = pd.read_json(args.input_file[0], orient='records')
    phases = ['KptEstimation', 'KptRefinement', 'GenerateRRRSets', 'FindMostInfluentialSet' ]
    dt['Other'] = dt['Total'] - dt[phases].sum(axis=1)
    phases.append('Other')

    dt.sort_values(by='NumThreads')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    from matplotlib.pyplot import cm
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for phase, color in zip(phases, colors):
        if phase is  'GenerateRRRSets':
            ax2.plot(dt['NumThreads'], dt[phase], label=phase, color=color)
            ax2.set_ylabel('Time (ms)', color=color)
            ax2.spines['right'].set_color(color)
            ax2.yaxis.label.set_color(color)
            ax2.tick_params(axis='y', color=color, labelcolor=color)
        else:
            ax1.plot(dt['NumThreads'], dt[phase], label=phase, color=color)

    l = ax1.get_ylim()
    l2 = ax2.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax1.get_yticks())
    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))

    ax1.set_xlabel('Number of Threads')
    ax1.set_xticks(dt['NumThreads'])
    ax1.set_ylabel('Time (ms)')
    ax1.grid(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=2)

    plt.savefig(args.input_file[0] + '.1.pdf', bbox_inches='tight')
    plt.close()

    x = dt['NumThreads']
    width = 0.35

    fig, ax = plt.subplots()
    bottom = dt[phases[0]]
    for i, phase in enumerate(phases):
        c = colors[i]
        if i is 0:
            ax.bar(x, dt[phase], width, color=c, label=phase)
        else:
            ax.bar(x, dt[phase], width, bottom=bottom, color=c, label=phase)
            bottom += dt[phase]

    ax.grid(True)
    ax.set_xticks(x)
    ax.set_xlabel(x)
    h1, l1 = ax.get_legend_handles_labels()
    plt.legend(h1, l1)
    plt.ylabel('Time (ms)')
    plt.xlabel('Number of Threads')
    plt.savefig(args.input_file[0] + '.2.pdf', bbox_inches='tight')
    plt.close()
    return


if __name__ == '__main__':
    main()
