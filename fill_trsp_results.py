#!/usr/bin/env python3
"""Populate results/trsp_results.tex with values from results/trsp_results.csv."""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / 'results'
CSV_PATH = RESULTS_DIR / 'trsp_results.csv'
TEX_PATH = RESULTS_DIR / 'trsp_results.tex'

NET_MAP   = {'N1': 'grid', 'N2': 'rand_grid', 'N3': 'ames'}
TRAF_MAP  = {'T1': 'PSL(K=1)', 'T2': 'PSL(K=2)', 'T3': 'PSL(K=3)', 'T4': 'PSL(K=4)'}
ALGO_MAP  = {'dijkstra': 'SSP', 'sarsa': 'SARSA', 'dqn': 'DQN', 'ppo': 'PPO', 'a2c': 'A2C'}
MODEL_ORDER = ['SSP', 'SARSA', 'DQN', 'PPO', 'A2C']
NETWORKS    = ['grid', 'rand_grid', 'ames']
TRAFFICS    = ['PSL(K=1)', 'PSL(K=2)', 'PSL(K=3)', 'PSL(K=4)']


def load_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            net  = NET_MAP.get(row['network'])
            traf = TRAF_MAP.get(row['traffic'])
            algo = ALGO_MAP.get(row['algo'])
            if net and traf and algo:
                data[(net, traf, algo)] = (float(row['makespan_mean']),
                                           float(row['success_rate']) * 100)
    return data


def fmt(val, bold=False, decimals=2):
    if val is None:
        return ''
    s = f'{val:.{decimals}f}'
    return f'\\textbf{{{s}}}' if bold else s


def best_makespan(data, net, traf, models):
    vals = {m: data[(net, traf, m)][0] for m in models if (net, traf, m) in data}
    return min(vals, key=vals.get) if vals else None

def best_sr(data, net, traf, models):
    vals = {m: data[(net, traf, m)][1] for m in models if (net, traf, m) in data}
    return max(vals, key=vals.get) if vals else None


def models_for(data, net, traf):
    return [m for m in MODEL_ORDER if (net, traf, m) in data]


def build_row(net_cell, traf_cell, model, makespan_str, sr_str, sep):
    return f'{net_cell} & {traf_cell} & {model} & {makespan_str} & {sr_str} \\\\ {sep}'


def generate_tex(data):
    lines = [
        '% Please add the following required packages to your document preamble:',
        '% \\usepackage{multirow}',
        '\\begin{table}[]',
        '\\centering',
        '\\begin{tabular}{|c|c|c|c|c|}',
        '\\hline',
        '\\textbf{Network} & \\textbf{Traffic} & \\textbf{Model} & \\textbf{Makespan} & \\textbf{Success (\\%)} \\\\ \\hline',
    ]

    for net in NETWORKS:
        net_tex        = net.replace('_', '\\_')
        active_traffics = [t for t in TRAFFICS if models_for(data, net, t)]
        n_rows          = sum(len(models_for(data, net, t)) for t in active_traffics)
        first_net       = True

        for ti, traf in enumerate(active_traffics):
            is_last_traf = ti == len(active_traffics) - 1
            models       = models_for(data, net, traf)
            best_ms      = best_makespan(data, net, traf, models)
            best_success = best_sr(data, net, traf, models)

            for mi, model in enumerate(models):
                is_last_model = mi == len(models) - 1
                net_cell  = f'\\multirow{{{n_rows}}}{{*}}{{{net_tex}}}' if first_net else ''
                traf_cell = f'\\multirow{{{len(models)}}}{{*}}{{{traf}}}' if mi == 0 else ''
                entry        = data.get((net, traf, model))
                makespan_str = fmt(entry[0] if entry else None, bold=(model == best_ms))
                sr_str       = fmt(entry[1] if entry else None, bold=(model == best_success), decimals=0)
                sep          = ('\\hline' if is_last_traf else '\\cline{2-5}') if is_last_model else '\\cline{3-5}'

                lines.append(build_row(net_cell, traf_cell, model, makespan_str, sr_str, sep))
                first_net = False

    lines += [
        '\\end{tabular}',
        '\\caption{TRSP Results}',
        '\\label{tab:trsp_results}',
        '\\end{table}',
    ]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    data = load_csv(CSV_PATH)
    tex  = generate_tex(data)
    TEX_PATH.write_text(tex)
    print(f'Written: {TEX_PATH}')
    print('\nGenerated table preview:')
    print(tex)
