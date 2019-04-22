import functions as f

#url = f.get_url()

datafile = "total_game_data2.csv"

#f.download_data(url, datafile)

print(datafile)

csv_ret = f.data_separation(datafile)
print("separate")
csv_altered = f.normalize('altered_total.csv')
print("norm")
labels = "class.csv"

arr, m = f.shuffle_dimension('altered_total.csv', labels)
print("shuffle")
f.kmeans_cluster(arr, m, labels)

#output = f.user_data(23, 1500, 3760, 2970, 25, 600, 500, 12)

#df2, labels2 = f.data_seperation(output)

#norm2, csv_ret2 = f.normalize(df2)

#arr2, m2 = f.shuffle_dimension(norm2, labels2)

#f.user_kmeans(arr2, m2, labels2)



