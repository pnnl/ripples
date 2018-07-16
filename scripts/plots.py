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

    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Time (ms)')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=2)

    plt.savefig(args.input_file[0] + '.1.pdf', bbox_inches='tight')
    plt.close()

    x = dt['NumThreads']
    width = 0.35

    plots = []
    bottom = dt[phases[0]]
    for i, phase in enumerate(phases):
        c = colors[i]
        if i is 0:
            plots.append(plt.bar(x, dt[phase], width, color=c))
        else:
            plots.append(plt.bar(x, dt[phase], width, bottom=bottom, color=c))
            bottom += dt[phase]

    plt.legend(plots, phases)
    plt.ylabel('Time (ms)')
    plt.xlabel('Number of Threads')
    plt.savefig(args.input_file[0] + '.2.pdf', bbox_inches='tight')
    plt.close()
    return


if __name__ == '__main__':
    main()
