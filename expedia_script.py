import os
import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
from sklearn.neighbors import KNeighborsClassifier
import operator
from tqdm import tqdm
from collections import defaultdict

# Set up working directory
os.chdir("/Users/Thms/ds/kaggle/kaggle_expedia/")

# Importing data files into pandas dataframes
print("Importing data files (step 1/13)")
destinations = pd.read_csv("data/destinations.csv")
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")


# ------------------------- Data Leak -------------------------
print("Exploiting data leak to match identical elements of the training and testing set (step 2/13)")
# Use the data leak to match elements of the testing set with elements of the
# training set.

# List of the columns that allow us to identify matching elements.
columns_to_match = ['user_location_country', 'user_location_region',
                    'user_location_city', 'hotel_market',
                    'orig_destination_distance']

groups = train.groupby(columns_to_match)


def exploit_leak(row):
    """ Return the list of hotels clusters corresponding to the row according to the training set. """

    index = tuple([row[t] for t in columns_to_match])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    cluster = list(set(group.hotel_cluster))
    return cluster


# Create a list of the matches between training and testing set.
print("Creating list of exact matches (step 3/13)")
matches = []
for i in tqdm(range(test.shape[0])):
    matches.append(exploit_leak(test.iloc[i]))


# ------------------------- 5 most common clusters -------------------------
print("Identifying the 5 most common clusters (step 4/13)")
# Identify the list of the 5 most common hotels clusters in case we have
# nothing better to predict.
best_5_clusters = list(train.hotel_cluster.value_counts().head().index)

# ------------------------ Best clusters / destination ------------------------
# We analyse the rate of booking according to the number of clicks each
# cluster receive per destination so we can get the "best clusters" for each
# destination.
print("Finding best clusters for every destinations (step 5/13)")

# Grouping logs by destination, then by hotel cluster
groups = train.groupby(["srch_destination_id", "hotel_cluster"])

# Calculating booking rate
print("Calculating booking rate (step 6/13)")
nb_clicks = 0
nb_bookings = 0
for name, group in tqdm(groups):
    nb_clicks = nb_clicks + len(group.is_booking[group.is_booking == False])
    nb_bookings = nb_bookings + len(group.is_booking[group.is_booking == True])
booking_rate = nb_bookings / (nb_bookings + nb_clicks)
# booking_rate = 0.079657 which means that 8% of the clicks generate a booking.

# Create clusters_scores, a dictionnary that stores the scores of every
# cluster for a given destination.
clusters_scores = defaultdict(dict)

print("Creating dictionnaries of scores for each destination (step 7/13)")
for name, group in tqdm(groups):
    nb_clicks = len(group.is_booking[group.is_booking == False])
    nb_bookings = len(group.is_booking[group.is_booking == True])
    score = nb_bookings + booking_rate * nb_clicks
    cluster_name = str(name[0])
    clusters_scores[cluster_name][name[-1]] = score

# Create top_5_clusters, a dictionnary that ranks the 5 best scores
# obtained previously, for each destination.
top_5_clusters = defaultdict(list)
for c in tqdm(clusters_scores):
    d = clusters_scores[c]
    top_5 = [l[0] for l in sorted(
        d.items(), key=operator.itemgetter(1), reverse=True)[:5]]
    top_5_clusters[c] = top_5

# ------------------------- KNN classification -------------------------
# In case we don't have any top cluster for a given destination, we are going
# to match this destination with the most similar ones using a KNN classifier.
# For that, we'll reduce the destinations dataframe to 5 principal components
# and merge the 2 less significant ones into a 2D space with a Gaussian
# Mixture Model.

# Applying PCA to get the 5 principal components.
print("Applying PCA on destinations file (step 8/)")
pca = PCA(n_components=5)
reduced = pca.fit_transform(destinations.ix[:, 'd1':])

# Merging PC4 and PC5 with mixture model.
print("Creating mixture model (step 9/13)")
mixture_components = reduced[:, [3, 4]]
reduced = reduced[:, (0, 1, 2)]
gmm = GMM(n_components=5)
gmm.fit(mixture_components)

# By plotting the mixture we can notice that there are 2 main "blobs" so
# we can simplify this new feature into the likelihood of a point
# belonging to one or the other cluster.
probas = gmm.predict_proba(mixture_components)
b = gmm.means_[:, 1] == min(gmm.means_[:, 1])
p = np.log(probas[:, b] / (1 - probas[:, b]))

# reduced is now a dataframe containing the 4 new meta features we created for
# every destination.
reduced = pd.DataFrame(np.concatenate((reduced, p), axis=1))
reduced.rename(columns={0: "comp_1", 1: "comp_2", 2: "comp_3", 3: "gmm"},
               inplace=True)

# Train KNC. We want 5 neighbours to be able to submit a full prediction
# just with this model.
knc = KNeighborsClassifier(n_neighbors=5)
knc_1 = []
knc_2 = []
knc_3 = []
knc_4 = []
knc_5 = []

print("Train KNN classifier (step 10/13)")
for d in tqdm(range(reduced.shape[0])):
    dest = destinations.loc[d, 'srch_destination_id']
    # Fitting model
    knc = knc.fit(reduced[destinations.srch_destination_id != dest],
                  destinations.srch_destination_id[destinations.
                                                   srch_destination_id != dest])

    nearest_neighbors = knc.kneighbors(np.reshape(
        reduced.loc[d, :].as_matrix(), [1, -1]), return_distance=False)
    nearest_neighbors = nearest_neighbors[0]
    # For each destination, list of first, second, etc. nearest neighbours.
    knc_1.append(nearest_neighbors[0])
    knc_2.append(nearest_neighbors[1])
    knc_3.append(nearest_neighbors[2])
    knc_4.append(nearest_neighbors[3])
    knc_5.append(nearest_neighbors[4])

# Now we need to match the destinations in the testing set with their
# corresponding neighbours. For that we need to create temporary dataframes to
# help us merge the results we obtained with the model and the test dataframe.
# There might be a more elegant way to do so!

print("Updating test dataframe (step 11/13)")
temp1 = pd.DataFrame()
temp1['knc_1'] = knc_1
temp1['knc_2'] = knc_2
temp1['knc_3'] = knc_3
temp1['knc_4'] = knc_4
temp1['knc_5'] = knc_5
temp1["srch_destination_id"] = destinations["srch_destination_id"]

temp2 = np.ones((test.shape[0], 5)) * np.nan
for _, row in tqdm(temp1.iterrows()):
    bool_array = test.srch_destination_id == row.srch_destination_id
    temp2[bool_array.values, 0] = row.knc_1
    temp2[bool_array.values, 1] = row.knc_2
    temp2[bool_array.values, 2] = row.knc_3
    temp2[bool_array.values, 3] = row.knc_4
    temp2[bool_array.values, 4] = row.knc_5

test['knc_1'] = temp2[:, 0]
test['knc_2'] = temp2[:, 1]
test['knc_3'] = temp2[:, 2]
test['knc_4'] = temp2[:, 3]
test['knc_5'] = temp2[:, 4]


# ------------------------- Aggregating results -------------------------
# Time to combine all our "models". For that, we'll take first the exact
# matches, then the best hotels per destination, the best hotels for a
# similar destination and finally the most popular hotels clusters
# overall, in that order.
print("Aggreagating results (step 12/13)")
best_clusters = []
best_clusters_similar = []

for index, row in tqdm(test.iterrows()):

    # Direct prediction
    key = str(row['srch_destination_id'])
    best_clusters.append(top_5_clusters[key])

    # Nearby (cluster-wise) prediction
    most_similar_5 = []

    try:
        key = str(int(row['knc_1']))
    except Exception:
        key = "error"
    if key in top_5_clusters:
        most_similar_5.append(top_5_clusters[key][0])

    try:
        key = str(int(row['knc_2']))
    except Exception:
        key = "error"
    if key in top_5_clusters:
        most_similar_5.append(top_5_clusters[key][0])

    try:
        key = str(int(row['knc_3']))
    except Exception:
        key = "error"
    if key in top_5_clusters:
        most_similar_5.append(top_5_clusters[key][0])

    try:
        key = str(int(row['knc_4']))
    except Exception:
        key = "error"
    if key in top_5_clusters:
        most_similar_5.append(top_5_clusters[key][0])

    try:
        key = str(int(row['knc_5']))
    except Exception:
        key = "error"
    if key in top_5_clusters:
        most_similar_5.append(top_5_clusters[key][0])

    best_clusters_similar.append(most_similar_5)


# Now we aggregate and keep the first 5 best hotels clusters.

def keep_5(items):
    """ Take an iterable as an argument and store its items in unique only if it is not in there already. """
    checked = {}
    unique = []
    for item in items:
        if item in checked:
            continue
        checked[item] = 1
        unique.append(item)
    return unique

predictions = [keep_5(matches[i] + best_clusters[i] + best_clusters_similar[i] +
                      best_5_clusters)[:5] for i in tqdm(range(len(best_clusters)))]

# ------------------------- Write submission file -------------------------
print("Editing submission file (step 13/13)")
with open('submission.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
                            'id', 'hotel_cluster'], delimiter=',')
    writer.writeheader()

    for i in tqdm(range(len(predictions))):
        s1 = '{0}'.format(test.id[i])
        fp0 = predictions[i]
        s2 = ''
        for j in range(len(fp0)):
            s2 = s2 + ' {0}'.format(fp0[j])
        writer.writerow({'id': s1, 'hotel_cluster': s2})
