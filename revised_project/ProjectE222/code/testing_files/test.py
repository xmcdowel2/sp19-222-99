import functions as f

#url = f.get_url()

datafile = "total_game_data2.csv"

#f.download_data(url, datafile)

print(datafile)

df, labels = f.data_seperation(datafile)
print(labels)
norm, csv_ret = f.normalize(df)

arr, m = f.shuffle_dimension(norm, labels)

f.kmeans_cluster(arr, m, labels)

#output = f.user_data(23, 1500, 3760, 2970, 25, 600, 500, 12)

#df2, labels2 = f.data_seperation(output)

#norm2, csv_ret2 = f.normalize(df2)

#arr2, m2 = f.shuffle_dimension(norm2, labels2)

#f.user_kmeans(arr2, m2, labels2)



