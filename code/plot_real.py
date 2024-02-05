#!/usr/bin/env python
import numpy as np
import pandas as pd
from pdb import set_trace

from aaviz import plot

def main():
    df = pd.read_csv('../results/results_realdata.csv')

    df.delta = df.delta/100
    df.loc[df.delta==1, 'num_nodes'] = 1
    df.loc[df.delta==1, 'num_edges'] = 0
    df['density'] = df.num_edges/df.num_nodes/df.num_nodes

    touchet = pd.read_csv('../../../water/real/input/touchet/Touchet_statistics_by_water_capacity.csv')
    yakima = pd.read_csv('../../../water/real/input/yakima/Yakima_statistics_by_water_capacity.csv')

    touchet['network'] = 'Touchet' + touchet['acre-feet_per_unit'].astype('str')
    yakima['network'] = 'Yakima' + yakima['acre-feet_per_unit'].astype('str')
    touchet['Data'] = 'touchet'
    yakima['Data'] = 'yakima'

    dfs = pd.concat([touchet, yakima], ignore_index=True)
    dfs = dfs[dfs['acre-feet_per_unit']!=1]
    dfs['delta'] = dfs.percent_total_capacity/100
    dfs = dfs

    fig, gs = plot.initiate_figure(x=16, y=2.8, gs_nrows=1, gs_ncols=11,
            gs_wspace=2, fs_fontsize=17)

    markers = ['o', '^', 's', 'o', '^', 's']
    #style = df[lambda_beta_str] #[lambda_str,beta_str]
    colors = [plot.COLORS['mathematica'][i] for i in [3,3,3,4,4,4]]
    dashes = [(2,0), (2,0), (2,0), (2,2), (2,2), (2,2)]
    hue_order = ['Touchet5', 'Touchet10', 'Touchet20', 'Yakima5', 'Yakima10', 'Yakima20']

    ax = plot.subplot(fig=fig, grid=gs[0,0:2], func='sns.lineplot',
            data=dfs,
            pf_x='delta', pf_y='seller_count', pf_hue='network',
            pf_hue_order=hue_order,
            pf_style='network',
            pf_markers=markers,
            pf_palette=colors,
            pf_dashes=dashes,
            pf_linewidth=3,
            ag_xmin=0, ag_xmax=1, ag_ymin=0, ag_ymax=100,
            lg_visible=False,
            la_title='(a)',
            fs_title='normalsize',
            la_xlabel=f'$\delta$',
            la_ylabel='Number of sellers')

    ax = plot.subplot(fig=fig, grid=gs[0,2:5], func='sns.lineplot',
            data=df,
            pf_x='delta', pf_y='num_edges', pf_hue='network',
            pf_hue_order=hue_order,
            pf_style='network',
            pf_markers=markers,
            pf_palette=colors,
            pf_dashes=dashes,
            pf_linewidth=3,
            ag_xmin=0, ag_xmax=1, ag_ymin=0,
            lg_visible=False,
            lg_ncol=2,
            lg_columnspacing=0.8,
            lg_handletextpad=.5,
            la_title='(b)',
            fs_title='normalsize',
            la_xlabel=f'$\delta$',
            la_ylabel='Edges in the Res-needs graph')
            ## sp_yscale='log',

    ax.ticklabel_format(style='sci',axis='y',scilimits=(4,4))

    ax = plot.subplot(fig=fig, grid=gs[0,5:8], func='sns.lineplot',
            data=df,
            pf_x='delta', pf_y='sigma_T_by_sigma_0', pf_hue='network',
            pf_hue_order=hue_order,
            pf_style='network',
            pf_markers=markers,
            pf_palette=colors,
            pf_dashes=dashes,
            pf_linewidth=3,
            ag_xmin=0, ag_xmax=1, ag_ymin=0, ag_ymax=1,
            lg_visible=True,
            fs_legend_title='small',
            fs_legend='small',
            la_title='(c)',
            fs_title='normalsize',
            la_xlabel=f'$\delta$',
            la_ylabel='$\sigma(\mathcal{T})/\sigma_0$')

    ## x = np.linspace(0,1,11)
    ## ax.plot(x, x, color='grey')
    ## plot.text(ax=ax, text=[{'x': 0.1, 'y': 0.05, 'text': 'No trade'}], fontsize=14)

    ax = plot.subplot(fig=fig, grid=gs[0,8:11], func='sns.lineplot',
            data=df,
            pf_x='delta', pf_y='welfare', pf_hue='network',
            pf_hue_order=hue_order,
            pf_style='network',
            pf_markers=markers,
            pf_palette=colors,
            pf_dashes=dashes,
            pf_linewidth=3,
            ag_xmin=0, ag_xmax=1, ag_ymin=0,
            lg_visible=False,
            la_title='(d)',
            fs_title='normalsize',
            la_xlabel=f'$\delta$',
            la_ylabel='wel($\mathcal{T}$)')

    plot.savefig('real_datasets.pdf')
    return

    ax = plot.subplot(fig=fig, grid=gs[0,8:11], func='sns.lineplot',
            data=df,
            pf_x='delta', pf_y='welfare', pf_hue=lambda_beta_str,
            pf_palette=colors,
            pf_style=style,
            pf_markers = markers,
            pf_dashes = ['', (2,2), '', (2,2), '', (2,2)],
            pf_linewidth=3,
            ag_xmin=0, ag_xmax=1, ag_ymin=0,
            lg_visible=False,
            fs_legend_title='normalsize',
            fs_legend='normalsize',
            la_xlabel=f'$\delta$',
            la_ylabel='welfare($\mathcal{T}$)')


if __name__ == '__main__':
    main()
