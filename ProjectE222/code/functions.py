import pandas as pd
import numpy as np
import csv
import random
import matplotlib
matplotlib.use('TkAgg')               #I added this line
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from os import listdir
from flask import Flask, request, send_file, make_response
import io

#might have to change code

#import seaborn as sns 

#Used for Normalization
from sklearn import preprocessing

# Used to shuffle data
import random

# Used to perform dimensionality reduction
from sklearn.decomposition import PCA

# Used for spectral clustering
from sklearn.cluster import SpectralClustering

print(__doc__)

def get_url():
    input_path = 'input.txt'
    input_file = open(input_path, "rt")
    contents = input_file.read()
    url = contents.rstrip()
    input_file.close()
    return str(url)

def download_data(url, filename):
    r = requests.get(url, allow_redirects = True)
    open(filename, 'wb').write(r.content)
    return

def new_download(filename):
    get_url()
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)
    return

def download(output):
    output_file = '/data/'+ output
    new_download(filename=output_file)
    return (str(output) + "Downloaded")

def data_seperation(datafile):
    #datafile = "total_game_data2.csv"

    df = pd.read_csv(datafile)

    # Removes every row where a player had zero minutes played
    df = df[df['Min Played'] != 0]

    #Drop unnecessary columns (one line?)
    df = df[df.columns.drop(list(df.filter(regex='Time')))]
    df = df[df.columns.drop(list(df.filter(regex='HR')))]
    df = df[df.columns.drop(list(df.filter(regex='Calories')))]
    df = df[df.columns.drop(list(df.filter(regex='load score')))]
    df = df[df.columns.drop(list(df.filter(regex='Recovery')))]
    df = df[df.columns.drop(list(df.filter(regex='Speed zone 1')))]

    #Reset index for df, store labels, then remove label column
    df = df.reset_index(drop=True)
    labels = df['Class']
    df = df[df.columns.drop(list(df.filter(regex='Class')))]

    # Column names to use in later loop
    distance_columns = ('Distance in Speed zone 2 [yd] (0.10 - 2.59 mph)', 'Distance in Speed zone 3 [yd] (2.60 - 5.13 mph)', 'Distance in Speed zone 4 [yd] (5.14 - 8.38 mph)', 'Distance in Speed zone 5 [yd] (8.39- mph)')
    accel_columns = ('Number of accelerations (-50.00 - -3.00 m/s)', 'Number of accelerations (-2.99 - -2.00 m/s)', 'Number of accelerations (-1.99 - -1.00 m/s)', 'Number of accelerations (-0.99 - -0.50 m/s)', 'Number of accelerations (0.50 - 0.99 m/s)', 'Number of accelerations (1.00 - 1.99 m/s)', 'Number of accelerations (2.00 - 2.99 m/s)', 'Number of accelerations (3.00 - 50.00 m/s)')

    # Create new column for total accelerations
    df['Total accelerations'] = pd.Series(np.random.randn(len(df)), index=df.index)

    # Convert column dtypes to floats for percentage values
    df['Sprints'] = df['Sprints'].apply(np.float64)
    for i in distance_columns:
        df[i] = df[i].apply(np.float64)
    for i in accel_columns:
        df[i] = df[i].apply(np.float64)

    # Adjust sprints for minutes played, adjust speed zones for total distance, and get total accelerations
    for index, row in df.iterrows():
    # Initialize total acceleration value
        total_accels = 0
        # Change # of sprints to sprints per min
        df.at[index, 'Sprints'] = row['Sprints'] / row['Min Played']
        # Divide distance in speed zone by total distance, save back in speed zone column
        for i in distance_columns:
            df.at[index, i] = (row[i]) / int(row['Total distance [yd]'])
        # Calculate total # of accelerations and save in new column
        for i in accel_columns:
            total_accels += row[i]
        df.at[index, 'Total accelerations'] = total_accels

    # Second loop to divide each acceleration column by the total accelerations, save back in acceleration columns
    for index, row in df.iterrows():
        for i in accel_columns:
            df.at[index, i] = (row[i]) / int(row['Total accelerations'])
    
    df = df[df.columns.drop(list(df.filter(regex='Total distance')))]
    df = df[df.columns.drop(list(df.filter(regex='Distance / min')))]
    df = df[df.columns.drop(list(df.filter(regex='Average')))]
    df = df[df.columns.drop(list(df.filter(regex='Min Played')))]
    df = df[df.columns.drop(list(df.filter(regex='Total accelerations')))]
    df = df[df.columns.drop(list(df.filter(regex='Speed zone 4')))]
    df = df[df.columns.drop(list(df.filter(regex='2.00 - 2.99')))]
    df = df[df.columns.drop(list(df.filter(regex='Max')))]
    df = df[df.columns.drop(list(df.filter(regex='-1.99 - -1.00')))]
    df = df[df.columns.drop(list(df.filter(regex='1.00 - 1.99')))]
    df = df[df.columns.drop(list(df.filter(regex='-2.99 - -2.00')))]


    return(df, labels)

def normalize(df):
    # Normalize

    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    # Save changes in new csv to view later
    csv_ret = "altered_total.csv"
    df.to_csv(csv_ret)

    return(df, csv_ret)

def shuffle_dimension(df, labels):
    [m,n] = df.shape
    #for i in range(n):
    #random.shuffle(df[i])

    # Reduce to two dimensions (uses data from every column, are they all really needed / does this overfit???)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    # Add the labels back to the df
    finalDf = pd.concat([principalDf, labels], axis = 1)

    # Store dimensions of dataframe and convert to an array
    [m,n] = finalDf.shape
    df_array = finalDf.values
    return(df_array, m)

def shape_assign(labels):
    shape_labels = []
    for i in labels:
        if(i == 'Forward'):
            shape_labels.append('x')
        elif(i == 'Midfielder'):
            shape_labels.append('^')
        elif(i == 'Defender'):
            shape_labels.append('.')
    return(shape_labels)

def assign_color(labels):
    colors = []
    for i in labels:
        if i == 0:
            colors.append('r')
        elif i == 1:
            colors.append('b')
        elif i == 2:
            colors.append('k')
    return(colors)

def kmeans_cluster(df_array, m, labels):

    markers = shape_assign(labels)

    # Columns to be used for clustering
    ind1 = 0; ind2 = 1

    X = np.zeros((m,2))
    X[:,0] = df_array[:,ind1]
    X[:,1] = df_array[:,ind2]

 #   plt.scatter( X[:,0],X[:,1], alpha=0.25, cmap = 4 )
  #  plt.show()

    # Scatter plot customization
    plt.rcParams['figure.figsize'] = (15,15) 
    plt.rcParams['font.size'] = 25 
    plt.rcParams['lines.markersize'] = 7

    # Spectral clustering code
    #spectral = SpectralClustering( n_clusters=3 ) # instantiate the k-means model, with 3 clusters
   # spectral.fit(X) # build the model 
    # plot the results!

    kmeans = KMeans(n_clusters = 3).fit(X)
    labels = kmeans.predict(X)
 #   plt.scatter( X[:,0],X[:,1], c=kmeans.labels_, alpha=0.75 )
  #  plt.show()

    x = X[:,0]
    y = X[:,1]

    centroids = kmeans.cluster_centers_

    col = assign_color(kmeans.labels_)
    for i in range(len(X)):
        plt.scatter(x[i], y[i], c=col[i], marker=markers[i], alpha =0.5)

    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    #plt.show()
    return( send_file(bytes_image, attachment_filename = 'plot.png', mimetype= 'image/png'))

def user_data(sprints, DSP2, DSP3, DSP5, ACC1, ACC2, ACC3, ACC4):
        
        
        arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,sprints,0,DSP2, DSP3,0, DSP5, ACC1, 0,0,ACC2, ACC3,0,0,ACC4,0,0,0,0]

        #DSP is distance in particular speed zone

        #ACC is number of acceleration in particular interval
        #1 = -50, -3
        #2 = -.99, -.5
        #3 = 0.5, .99
        #3 = 3, 50

        #norm = np.linalg.norm(arr, ord = 1)

        #norm1 = [place, norm[0], norm[1], norm[2], norm[3], norm[4], norm[5], norm[6], norm[7]]
       
        total_csv = open('total_game_data2.csv', 'r')
        output = open('user_total.csv', 'w')

        writer = csv.writer(output)

        for row in csv.reader(total_csv):
            writer.writerow(row)
        
        writer.writerow(arr)
        total_csv.close()
        output.close()

        ret_file = 'user_total.csv'
        return(ret_file)

def user_kmeans(df_array, m, labels):

    markers = shape_assign(labels)

    # Columns to be used for clustering
    ind1 = 0; ind2 = 1

    X = np.zeros((m,2))
    X[:,0] = df_array[:,ind1]
    X[:,1] = df_array[:,ind2]

 #   plt.scatter( X[:,0],X[:,1], alpha=0.25, cmap = 4 )
  #  plt.show()

    # Spectral clustering code
    #spectral = SpectralClustering( n_clusters=3 ) # instantiate the k-means model, with 3 clusters
   # spectral.fit(X) # build the model 
    # plot the results!

    kmeans = KMeans(n_clusters = 3).fit(X)
    labels = kmeans.predict(X[167:])
   # plt.scatter( X[:,0],X[:,1], c=kmeans.labels_, alpha=0.75 )
  #  plt.show()

   
    x1 = X[:, 0]
    y1 = X[:, 1]
    x = labels[:]
    y = labels[:]

    centroids = kmeans.cluster_centers_

    plt.scatter(x1, y1, c = kmeans.labels_)
    plt.scatter(labels, X[165:168,0])
    plt.show()
    return()

def generate_figure(filename):
    file = data_dir + filename
    with open(file,'r') as csvfile:
        my_file = pd.read_csv(csvfile)
        soccer = my_file
        soccer_numeric = nfl.select_dtypes(include=[np.number])
        soccer_numeric.boxplot()
        bytes_image = io.BytesIO()
        bytes_image
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
    return bytes_image
