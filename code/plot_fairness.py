#!/usr/bin/env python
import pandas as pd

import plot

N = 10
k = 5
delta_set = [0.2, 0.4, 0.6, 0.8]

def main():
    df = pd.read_csv('../results/results_fairness.csv')
    df.beta_high = df.beta_high.round(1)
    df.delta = df.delta.round(1)
    df = df[(df.beta_high==0.7) & (df['delta'].isin(delta_set))]
    df['tradeoff'] = df.welfare / df.welfare0

    fig, gs = plot.initiate_figure(x=10, y=2.3, gs_nrows=1, gs_ncols=3,
                                   gs_wspace=.1, fs_fontsize=17)

    markers = ['o', '^', 's', '*']
    ## style = df[lambda_beta_str] #[lambda_str,beta_str]
    colors = plot.COLORS['mathematica']

    ax = plot.subplot(fig=fig, grid=gs[0, 0], func='sns.lineplot',
                      data=df[df['lambda']==0],
                      pf_x='rb', pf_y='tradeoff', pf_hue='delta',
                      pf_palette=colors[0:4],
                      pf_linewidth=3,
                      pf_style='delta',
                      pf_markers=markers,
                      pf_dashes=False,
                      ag_xmin=1, ag_xmax=5, ag_ymin=0, ag_ymax=1,
                      lg_visible=False,
                      fs_legend_title='normalsize',
                      fs_xlabel='large',
                      la_title='$\lambda=0$, $\\beta_{high}=0.7$',
                      la_xlabel=f'$r(\{{b\}})$',
                      la_ylabel='$\\textrm{wel}(\mathcal{T}_r^*)/\\textrm{wel}(\mathcal{T}^*)$')

    ax = plot.subplot(fig=fig, grid=gs[0, 1], func='sns.lineplot',
                      sp_sharey=0,
                      data=df[df['lambda']==0.5],
                      pf_x='rb', pf_y='tradeoff', pf_hue='delta',
                      pf_palette=colors[0:4],
                      pf_linewidth=3,
                      pf_style='delta',
                      pf_markers=markers,
                      pf_dashes=False,
                      ag_xmin=1, ag_xmax=5,
                      lg_visible=True,
                      lg_title='$\delta$',
                      fs_legend_title='normalsize',
                      fs_xlabel='large',
                      la_title='$\lambda=0.5$, $\\beta_{high}=0.7$',
                      la_xlabel=f'$r(\{{b\}})$',
                      la_ylabel='')

    ax = plot.subplot(fig=fig, grid=gs[0, 2], func='sns.lineplot',
                      sp_sharey=0,
                      data=df[df['lambda']==1],
                      pf_x='rb', pf_y='tradeoff', pf_hue='delta',
                      pf_palette=colors[0:4],
                      pf_linewidth=3,
                      pf_style='delta',
                      pf_markers=markers,
                      pf_dashes=False,
                      ag_xmin=1, ag_xmax=5,
                      lg_visible=False,
                      lg_title='$\delta$',
                      fs_legend_title='normalsize',
                      fs_xlabel='large',
                      la_title='$\lambda=1$, $\\beta_{high}=0.7$',
                      la_xlabel=f'$r(\{{b\}})$',
                      la_ylabel='')
    plot.savefig('fairness.pdf')

if __name__ == '__main__':
    main()
