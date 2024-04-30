#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import ticker
import matplotlib

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  

def load_throughput_data(throughput_data_location):
    vanilla = None
    data = {}
    with open(throughput_data_location, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()
            if parts[0] == 'Baseline:':
                vanilla = float(parts[1])
                continue
            pt = int(parts[0])
            st = int(parts[2]) if parts[2] != 'None' else 'no upper threshold' 
            time = float(parts[4])
            data[(pt,st)] = time
    df = pd.DataFrame([(t, u, ti) 
                       for (t,u),ti in data.items()], 
                      columns=['Threshold', 'Upper Threshold', 'Time'])
    return df, vanilla
            

def load_data(directory):
    data = {}
    u = set()
    for filename in os.listdir(directory):
        if filename.startswith("preempt") and filename.endswith(".txt"):
            parts = filename.replace('.txt', '').split('_')
            threshold = int(parts[0].split('threshold')[1])
            upper_threshold = int(parts[1].split('threshold')[1])
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                times = np.array([float(line.strip()) for line in file if line.strip()])
                avg_time = np.mean(times) * 1000
            print(f"{threshold} - {upper_threshold} avg time: {avg_time}ms")
            data[(threshold, upper_threshold)] = avg_time
            u.add(upper_threshold)
        
    print(u)
    return data

def create_dataframe(data):
    # Flatten the data list into a DataFrame
    df = pd.DataFrame([(t, u, ti) 
                       for (t,u),ti in data.items()], 
                      columns=['Threshold', 'Upper Threshold', 'Time'])
    return df

def plot_data(df):
    pivot_table = df.pivot_table(values='Time', index='Threshold', columns='Upper Threshold', aggfunc='mean')
    plt.figure(figsize=(5, 4))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Average Time by Threshold and Upper Threshold')
    plt.ylabel('Threshold')
    plt.xlabel('Upper Threshold')
    plt.show()

def plot_grouped_bar_chart(df):
    df['Upper Threshold Category'] = df['Upper Threshold'].apply(lambda x: 'no upper threshold' if x >= 30000 else x)
    df['Upper Threshold Category'] = df['Upper Threshold Category'][df['Threshold'] <= 1000]
    df['Threshold Category'] = df['Threshold'].apply(lambda x: 'no threshold' if x > 1000 else x)
    vanilla = df[df['Threshold'] > 1000]['Time'].values
    pivot_df = df.pivot_table(values='Time', index='Threshold Category', columns='Upper Threshold Category', aggfunc='mean')
    # Plotting
    fig = plt.figure(figsize=(10, 9))
    ax = fig.gca()

    # Setting the positions and width for the bars
    positions = list(range(len(pivot_df.index)))
    width = 0.25  # the width of the bars

    

    # Plotting the bars
    # colors = ['red', 'green', 'blue', 'cyan']  # Add more colors if you have more categories
    for i, column in enumerate(pivot_df.columns):
        b = plt.bar([p + width * i for p in positions], pivot_df[column], width, alpha=0.7, label=str(column))
        b.set_label('No Switch Threshold' if str(column) == 'no upper threshold' else f'Switch Threshold {column}')

    vanilla_line = plt.axhline(y = vanilla, color = 'k', linestyle = '--') 
    vanilla_line.set_label("Vanilla vLLM")

    plt.legend(loc='lower right')

    # specifying horizontal line type 


    # Setting the x-axis labels with the threshold values
    ax.set_xticks([p + width for p in positions])
    ax.set_xticklabels(pivot_df.index)

    y = list(ax.get_yticks())
    y.append(vanilla[0])
    print(y)
    ax.set_yticks(y)
    # Setting the graph title and labels
    plt.xlabel('Preemption Token Threshold')
    plt.ylabel('Average Token Generation Time')
    plt.title('Token Generation Time by Thresholds')
    # plt.title('Average Iteration Times')
    fmtr = ticker.StrMethodFormatter('{x:.0f}ms')
    ax.yaxis.set_major_formatter(fmtr)
    # Showing the plot
    plt.show()

def plot_grouped_bar_chart_throughput():
    throughput_data_location = input('throughput data location: ')
    df, vanilla = load_throughput_data(throughput_data_location)
    print(df)
    df['Upper Threshold Category'] = df['Upper Threshold'].apply(lambda x: 'no upper threshold' if x == np.nan else x)
    print(df)
    df['Threshold Category'] = df['Threshold']
    pivot_df = df.pivot_table(values='Time', index='Threshold Category', columns='Upper Threshold Category', aggfunc='mean')
    # Plotting
    fig = plt.figure(figsize=(10, 9))
    ax = fig.gca()

    # Setting the positions and width for the bars
    positions = list(range(len(pivot_df.index)))
    width = 0.25  # the width of the bars

    

    # Plotting the bars
    # colors = ['red', 'green', 'blue', 'cyan']  # Add more colors if you have more categories
    for i, column in enumerate(pivot_df.columns):
        b = plt.bar([p + width * i for p in positions], pivot_df[column], width, alpha=0.7, label=str(column))
        b.set_label('No Switch Threshold' if str(column) == 'no upper threshold' else f'Switch Threshold {column}')

    vanilla_line = plt.axhline(y = vanilla, color = 'k', linestyle = '--') 
    vanilla_line.set_label("Vanilla vLLM")

    plt.legend(loc='lower right')

    # specifying horizontal line type 

    # Setting the x-axis labels with the threshold values
    ax.set_xticks([p + width for p in positions])
    ax.set_xticklabels(pivot_df.index)

    y = list(ax.get_yticks())
    y.append(vanilla)
    print(y)
    ax.set_yticks(y)
    # Setting the graph title and labels
    plt.xlabel('Preemption Token Threshold')
    plt.ylabel('Throughput (req/s)')
    plt.title('Throughput by Thresholds')
    # plt.title('Average Iteration Times')
    fmtr = ticker.StrMethodFormatter('{x:.1f}')
    ax.yaxis.set_major_formatter(fmtr)
    # Showing the plot
    plt.show()



def main(directory):
    data = load_data(directory)
    df = create_dataframe(data)
    plot_grouped_bar_chart(df)
    plot_grouped_bar_chart_throughput()


if __name__ == "__main__":
    try:
        directory = sys.argv[1]
    except:
        print("Usage: generate.py <directory with output data>")
        sys.exit(1)
    main(directory)
