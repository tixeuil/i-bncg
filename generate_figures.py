import random
import networkx as nx
import numpy as np
import copy
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import os
print(os.getcwd())

# Reload past data to regenerate graphs
def read_csv_custom(filename):
     res = pd.read_csv(filename,header=0,index_col=0)
     res.columns = res.columns.astype(float)
     return res

df_cost_c_s = read_csv_custom("df_cost_c_s.csv")
df_host_diameters_c_s = read_csv_custom("df_host_diameters_c_s.csv")
df_equilibrium_diameters_c_s = read_csv_custom("df_equilibrium_diameters_c_s.csv")
df_interests_diameters_c_s = read_csv_custom("df_interests_diameters_c_s.csv")
df_st_min_s = read_csv_custom("df_st_min_s.csv")
df_st_min_cost_eq_c_s = read_csv_custom("df_st_min_cost_eq_c_s.csv")
df_st_max_cost_eq_c_s = read_csv_custom("df_st_max_cost_eq_c_s.csv")
df_time_c_s = read_csv_custom("df_time_c_s.csv")

df_cost_d_s = read_csv_custom("df_costs_d_s.csv")
df_host_diameters_d_s = read_csv_custom("df_host_diameters_d_s.csv")
df_equilibrium_diameters_d_s = read_csv_custom("df_equilibrium_diameters_d_s.csv")
df_interests_diameters_d_s = read_csv_custom("df_interests_diameters_d_s.csv")
df_st_min_cost_eq_d_s = read_csv_custom("df_st_min_cost_eq_d_s.csv")
df_st_max_cost_eq_d_s = read_csv_custom("df_st_max_cost_eq_d_s.csv")
df_time_d_s = read_csv_custom("df_time_d_s.csv")

df_cost_c_m = read_csv_custom("df_cost_c_m.csv")
df_host_diameters_c_m = read_csv_custom("df_host_diameters_c_m.csv")
df_equilibrium_diameters_c_m = read_csv_custom("df_equilibrium_diameters_c_m.csv")
df_interests_diameters_c_m = read_csv_custom("df_interests_diameters_c_m.csv")
df_st_min_m = read_csv_custom("df_st_min_m.csv")
df_st_min_cost_eq_c_m = read_csv_custom("df_st_min_cost_eq_c_m.csv")
df_st_max_cost_eq_c_m = read_csv_custom("df_st_max_cost_eq_c_m.csv")
df_time_c_m = read_csv_custom("df_time_c_m.csv")

df_cost_d_m = read_csv_custom("df_cost_d_m.csv")
df_host_diameters_d_m = read_csv_custom("df_host_diameters_d_m.csv")
df_equilibrium_diameters_d_m = read_csv_custom("df_equilibrium_diameters_d_m.csv")
df_interests_diameters_d_m = read_csv_custom("df_interests_diameters_d_m.csv")
df_st_min_cost_eq_d_m = read_csv_custom("df_st_min_cost_eq_d_m.csv")
df_st_max_cost_eq_d_m = read_csv_custom("df_st_max_cost_eq_d_m.csv")
df_time_d_m = read_csv_custom("df_time_d_m.csv")

"""
This method produces a figure where the ratio of infinite equilibrium is displayed, given simulation results
"""
def draw_figure_infinite(df_metric, title, metric, ymax):
    plt.clf()
    fig, ax = plt.subplots()
    df_plot = pd.DataFrame()
    for p in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        df_plot.insert(0,p,1.0-df_metric.groupby(np.isinf(df_metric[p])).count()[1]/len(df_metric[p]))
    df_plot.mean(0).plot(label='selfish')
    plt.legend()
    plt.ylim(0,ymax)
    ax.set_xlabel(r'$p$ in $G(n,p)$', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{title}_{metric}.png")
    
"""
This method produces a figure where several ratio of infinite equilibrium are displayed, given simulation results
"""
def draw_figures_infinite(d_df_metric, title, metric, ymax):
    plt.clf()
    fig, ax = plt.subplots()
    for l in d_df_metric:
        df_plot = pd.DataFrame()
        for p in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            df_plot.insert(0,p,1.0-d_df_metric[l].groupby(np.isinf(d_df_metric[l][p])).count()[1]/len(d_df_metric[l][p]))
        df_plot.mean(0).plot(label=l)
    plt.legend()
    plt.ylim(0,ymax)
    ax.set_xlabel(r'$p$ in $G(n,p)$', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{title}_{metric}.png")

"""
This method produces a figure (cost, time) from multiple datasets, given optimal spanning tree, cost or time simulations
"""
def draw_figures_path(d_df_metric,title,metric, ymax ):
    plt.clf()
    fig, ax = plt.subplots()
    for l in d_df_metric:
        d_df_metric[l].mean(0).plot(label=l)
    plt.legend()
    plt.ylim(0,ymax)
    ax.set_xlabel(r'$p$ in $G(n,p)$', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{title}_{metric}.png")

"""
This method produces a figure where the price of anarchy and stability are displayed, given optimal spanning tree, minimun cost simulations, and maximum cost simulations
"""
def draw_figures_price_path(df_max_equil, df_min_equil, df_cost, title, metric, ymax):
    plt.clf()
    fig, ax = plt.subplots()
    df_price_anarchy = pd.DataFrame(df_max_equil / df_cost)
    df_price_stability = pd.DataFrame(df_min_equil / df_cost)
    df_price_anarchy.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_price_stability.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(df_price_stability.mean(0))
    df_price_anarchy.mean(0).plot(label='Price of Anarchy')
    df_price_stability.mean(0).plot(label="Price of Stability")
    plt.legend()
    plt.ylim(0,ymax)
    ax.set_xlabel(r'$p$ in $G(n,p)$', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{title}_{metric}.png")
    
"""
This method produces a time figure 
"""
def draw_figures_path_time(d_df_metric,title,metric, ymax):
    
    plt.clf()
    fig, ax = plt.subplots()
    for l in d_df_metric:
        d_df_metric[l].mean(0).plot(label=l)
    plt.legend()
    plt.ylim(0,ymax)
    ax.set_xlabel(r'$p$ in $G(n,p)$', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{title}_{metric}.png")
    

#SUM cost figures (SUM cost simulation must have been run before)

#connected
draw_figure_infinite(df_cost_c_s,"Ratio of equilibrium with infinite cost (connectivity) - SUM", "ratio of equilibrium with infinite cost",ymax=0.65)

#disconnected
draw_figure_infinite(df_cost_d_s,"Ratio of equilibrium with infinite cost (disconnection) - SUM", "ratio of equilibrium with infinite cost",ymax=0.65)

#MAX cost figures (MAX cost simulation must have been run before)

#connected
draw_figure_infinite(df_cost_c_m,"Ratio of equilibrium with infinite cost (connectivity) - MAX", "ratio of equilibrium with infinite cost",ymax=0.65)

#disconnected
draw_figure_infinite(df_cost_d_m,"Ratio of equilibrium with infinite cost (disconnection) - MAX", "ratio of equilibrium with infinite cost",ymax=0.65)

#compare SUM and MAX
draw_figures_infinite(dict({'SUM':df_cost_d_s,'MAX':df_cost_d_m}),"Ratio of equilibrium with infinite cost (disconnection)", "ratio of equilibrium with infinite cost",ymax=0.65)

# comparison with best, min, max trees
draw_figures_path(dict({'Best tree':df_st_min_s,'Selfish connected':df_cost_c_s,'Min connected equilibrium tree':df_st_min_cost_eq_c_s,'Max connected equilibrium tree':df_st_max_cost_eq_c_s}),title='Social cost - SUM (connected)',metric='cost',ymax=350)
draw_figures_path(dict({'Best tree':df_st_min_m,'Selfish connected':df_cost_c_m,'Min connected equilibrium tree':df_st_min_cost_eq_c_m,'Max connected equilibrium tree':df_st_max_cost_eq_c_m}),title='Social cost - MAX (connected)',metric='cost',ymax=75)

draw_figures_path(dict({'Best tree':df_st_min_s,'Selfish disconnected':df_cost_d_s,'Min disconnected equilibrium tree':df_st_min_cost_eq_d_s,'Max disconnected equilibrium tree':df_st_max_cost_eq_d_s}),title='Social cost - SUM (disconnected)',metric='cost',ymax=350)
draw_figures_path(dict({'Best tree':df_st_min_m,'Selfish disconnected':df_cost_d_m,'Min disconnected equilibrium tree':df_st_min_cost_eq_d_m,'Max disconnected equilibrium tree':df_st_max_cost_eq_d_m}),title='Social cost - MAX (disconnected)',metric='cost',ymax=75)

#SUM and #MAX Diameter figures
draw_figures_path(dict({'Host Graph Diameter':df_host_diameters_c_s,'Interests Graph Diameter':df_interests_diameters_c_s,'Equilibrium Graph Diameter':df_equilibrium_diameters_c_s}),title='Graph Diameters - Connected - SUM',metric='hops',ymax=10)
draw_figures_path(dict({'Host Graph Diameter':df_host_diameters_d_s,'Interests Graph Diameter':df_interests_diameters_d_s,'Equilibrium Graph Diameter':df_equilibrium_diameters_d_s}),title='Graph Diameters - Disconnected - SUM',metric='hops',ymax=10)
draw_figures_path(dict({'Host Graph Diameter':df_host_diameters_c_m,'Interests Graph Diameter':df_interests_diameters_c_m,'Equilibrium Graph Diameter':df_equilibrium_diameters_c_m}),title='Graph Diameters - Connected - MAX',metric='hops',ymax=10)
draw_figures_path(dict({'Host Graph Diameter':df_host_diameters_d_m,'Interests Graph Diameter':df_interests_diameters_d_m,'Equilibrium Graph Diameter':df_equilibrium_diameters_d_m}),title='Graph Diameters - Disconnected - MAX',metric='hops',ymax=10)
    
draw_figures_path(dict({'Equilibrium Graph Diameter (connected)':df_equilibrium_diameters_c_s,'Equilibrium Graph Diameter (disconnected)':df_equilibrium_diameters_d_s}),title='Graph Diameters - Connected vs Disconnected - SUM',metric='hops',ymax=10)
draw_figures_path(dict({'Equilibrium Graph Diameter (connected)':df_equilibrium_diameters_c_m,'Equilibrium Graph Diameter (disconnected)':df_equilibrium_diameters_d_m}),title='Graph Diameters - Connected vs Disconnected - MAX',metric='hops',ymax=10)
    
#PoA and PoS
draw_figures_price_path(df_st_max_cost_eq_c_s,df_st_min_cost_eq_c_s,df_st_min_s,"PoA and PoS - SUM (connected)", "ratio",ymax=1.8)
draw_figures_price_path(df_st_max_cost_eq_c_m,df_st_min_cost_eq_c_m,df_st_min_m,"PoA and PoS - MAX (connected)", "ratio",ymax=1.8)

draw_figures_price_path(df_st_max_cost_eq_d_s,df_st_min_cost_eq_d_s,df_st_min_s,"PoA and PoS - SUM (disconnected)", "ratio",ymax=1.8)
draw_figures_price_path(df_st_max_cost_eq_d_m,df_st_min_cost_eq_d_m,df_st_min_m,"PoA and PoS - SUM (disconnected)", "ratio",ymax=1.8)

#Convergence time 
draw_figures_path_time(dict({'Selfish (connectivity)':df_time_c_s,'Selfish (disconnection)':df_time_d_s}),title='Convergence time - SUM',metric='steps',ymax=25)
draw_figures_path_time(dict({'Selfish (connectivity)':df_time_c_m,'Selfish (disconnection)':df_time_d_m}),title='Convergence time - MAX',metric='steps',ymax=30)

#Convergence time
draw_figures_path_time(dict({'Selfish (connectivity)':df_time_c_s.replace(-1, np.nan),'Selfish (disconnection)':df_time_d_s.replace(-1, np.nan)}),title='Convergence time - SUM (remove non equilibrium)',metric='steps',ymax=25)
draw_figures_path_time(dict({'Selfish (connectivity)':df_time_c_m.replace(-1, np.nan),'Selfish (disconnection)':df_time_d_m.replace(-1, np.nan)}),title='Convergence time - MAX (remove non equilibrium)',metric='steps',ymax=30)

#Convergence time
draw_figures_path_time(dict({'Selfish (connectivity)':df_time_c_s,'Selfish (connectivity - remove non equilibrium)':df_time_c_s.replace(-1, np.nan)}),title='Convergence time - SUM connected (all vs. remove eq)',metric='steps',ymax=25)
draw_figures_path_time(dict({'Selfish (connectivity)':df_time_c_m,'Selfish (connectivity - remove non equilibrium)':df_time_c_m.replace(-1, np.nan)}),title='Convergence time - MAX connected (all vs. remove eq)',metric='steps',ymax=25)
draw_figures_path_time(dict({'Selfish (disconnected)':df_time_d_s,'Selfish (disconnected - remove non equilibrium)':df_time_d_s.replace(-1, np.nan)}),title='Convergence time - SUM disconnected (all vs. remove eq)',metric='steps',ymax=30)
draw_figures_path_time(dict({'Selfish (disconnected)':df_time_d_m,'Selfish (disconnected - remove non equilibrium)':df_time_d_m.replace(-1, np.nan)}),title='Convergence time - MAX disconnected (all vs. remove eq)',metric='steps',ymax=30)