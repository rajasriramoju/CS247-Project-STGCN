import os
import gzip
import pandas as pd
from io import StringIO
import numpy as np
import sys
from tqdm import tqdm
from pprint import pprint
import math
import geopy.distance
np.set_printoptions(threshold=sys.maxsize)


# Set up our metadata dictionary {station_id : (latitude, longitutde) }
def create_station_metadata():

    # Open up the PEMS metadata - we use this to get lat/long values of stations
    df_meta = pd.read_csv('meta2.txt', delimiter = "\t")
    # We are interested in several fields - ID of the station, Freeway, and lat/long
    df_meta = df_meta[["ID","Fwy","Dir","Latitude","Longitude",]]
    # Create a dictionary for our station meta descriptions for latitude and longitude
    station_data = df_meta.values
    station_location_dict = {int(x[0]):[ (x[3],x[4]), x[1],x[2] ] for x in station_data}

    return station_location_dict


# Get n_stations randomly (requires that the station have metadata, and it is in the dataset)
def get_n_stations(n_stations, dataset_stations, metadata_stations):

    # Get intersection of dataset_stations and metadata_stations
    # station_intersections = np.intersect1d(dataset_stations, metadata_stations)
    station_intersections = set(dataset_stations.tolist()).intersection(metadata_stations)
    # Randomly pick n_stations
    chosen_stations = np.random.choice(list(station_intersections), n_stations)
    return chosen_stations

# Map station ids to indexes (maps from ID to a value between 0 and n_stations-1 )
def map_station_to_index(station_id, chosen_stations):
    return np.where(chosen_stations == station_id)
def map_index_to_station(index, chosen_stations):
    return chosen_stations[index]

# Calculate distance in latlong between two nodes
def latlong_distance(latlong1, latlong2):
    return np.linalg.norm( np.array(latlong1) - np.array(latlong2) )
    # return geopy.distance.distance(latlong1, latlong2).m

# Compute the adjaceny matrix for our chosen stations
def compute_weight_matrix(chosen_stations, station_location_dict):

    # Initialize our matrix (n_stations, n_stations)
    weight_matrix = np.zeros((chosen_stations.shape[0], chosen_stations.shape[0]))

    # Iterate through every element of the matrix
    for i in tqdm(range(chosen_stations.shape[0])):
        for j in range(chosen_stations.shape[0]):
            # We only alter the values if i!=j
            if i != j:
                # Compute the weight value (see data preprocessing section:
                #   https://github.com/VeritasYin/STGCN_IJCAI-18)

                station_i = map_index_to_station(i, chosen_stations)
                station_j = map_index_to_station(j, chosen_stations)

                station_i_fwy = station_location_dict[station_i][1]
                station_j_fwy = station_location_dict[station_j][1]
                station_i_dir = station_location_dict[station_i][2]
                station_j_dir = station_location_dict[station_j][2]

                latlong_i = station_location_dict[station_i][0]
                latlong_j = station_location_dict[station_j][0]


                distance = latlong_distance(latlong_i, latlong_j)
                sigma = 0.1 # according to the paper, sigma should be 10
                # weight = math.exp( - (distance ** 2)  / sigma )
                # weight = distance # math.exp( - (distance ) / sigma)

                # According to the paper, sigma^2 = 10 and eps = 0.5
                # if weight > 0.5 and station_i_fwy == station_j_fwy and station_i_dir ==  station_j_dir:
                weight_matrix[i,j] = distance

    # According to the calculation from their github
    # sigma2 = 0.1
    # epsilon = 0.5
    # n = weight_matrix.shape[0]
    # weight_matrix = weight_matrix / 10000.
    # W2, W_mask = weight_matrix * weight_matrix, np.ones([n, n]) - np.identity(n)
    # weight_matrix = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

    # print(weight_matrix)
    return weight_matrix

# Reorder our stations so those from the same freeway cluster together
def cluster_stations(chosen_stations, chosen_indexes, station_location_dict):

    cluster_list = []
    for i,x in enumerate(chosen_stations):
        # Get the freeway
        freeway = station_location_dict[x][1]
        cluster_list.append((x, chosen_indexes[i], freeway))

    # Sort cluster list by freeway
    cluster_list = sorted(cluster_list, key=lambda x:x[2])
    # print([x[2] for x in cluster_list])

    # Get new chosen stations and chosen indexes
    chosen_stations = [x[0] for x in cluster_list]
    chosen_indexes = [x[1] for x in cluster_list]

    return np.array(chosen_stations), np.array(chosen_indexes)


# Create our dataset for the adjancency matrix and historical road data
#  We randomly select n_stations which form the adjanceny matrix.
def create_custom_dataset(n_stations=228):

    # Get station metadata (locations for each station)
    station_location_dict = create_station_metadata()
    metadata_stations = list(station_location_dict.keys())
    # print(station_location_dict.keys())

    # Open up the PEMS dataset
    data_dir = "PEMS_7_May_June_include_weekends"
    data_files = os.listdir(data_dir)

    chosen_stations = [] # Used to select which stations we are interested in.

    # There are a number of errors that arise with the stations
    #  sometimes a text file won't have data for a particular id
    #  or all values will be nan.  In these cases, we actually take more stations
    #  than n_stations to give us more options later
    buffer_n_stations = int(n_stations * 3)


    # Some overall notes:
    #  There are 4589 stations - we form our adjacency matrix based on
    #    these stations, so we should shrink this value down
    #  There are only 21 freeways, but that value doesn't matter in our system.
    #  To get latlong data for each station, you must find the PEMS metadata:
    #   https://pems.dot.ca.gov/?dnode=Clearinghouse&type=meta&district_id=7&submit=Submit

    # 2d list, where each inner list is a column for a station's avg speed over time
    new_data = [[] for x in range(buffer_n_stations)]

    found_stations = []


    # Open up a file (make sure our data is aligned over time, which requires sorting)
    #  One special note - the first file in our list actually is missing some data - skip it.
    # Otherwise, each file has 288 datapoints across each station (meaning that all stations)
    #   have a full day's worth of data.
    for file_item in tqdm(sorted(data_files)[1:]):

        # Get filedir and data
        filedir = os.path.join(data_dir, file_item)
        gzip_file = gzip.open(filedir, "rb")
        # First, decode from bytes, then StringIO for reading into pandas as csv
        data = StringIO(gzip_file.read().decode("utf-8"))
        df = pd.read_csv(data, header=None) # There are no column names here...
        # Drop columns where all columns are NaN
        # df = df.dropna(axis=1, how='all')

        # Convert to NP - pandas DF is too slow for row iteration.
        numpy_data = df.values

        # If we haven't chosen our list of stations for our dataset, do so now
        if not len(chosen_stations):
            # Check what stations are available in the dataset
            unique_stations = np.unique(numpy_data[:,1])
            # Randomly get some stations for the new dataset
            chosen_stations = get_n_stations(buffer_n_stations, unique_stations, metadata_stations)

        # Iterate through each row, getting the station and freeway
        #  There's probably a faster way to do this, but I'm too tired.
        #  For info on each row:
        #   https://drive.google.com/file/d/1muiKe1uAWJwz2uIz5DZHR1GTEYPa2uGw/view?usp=sharing
        for row in numpy_data:  # Assuming this is aligned in time
            station = row[1] # Station
            freeway = row[3] # Freeway number
            avg_occupancy = row[10] # Average occupancy of all lanes over 5 mins [0,1]
            avg_speed = row[11] # Average mph of cars over all lanes in 5 mins

            # If this is a chosen station, keep the data
            if station in chosen_stations:
                # Add to our new data
                idx = map_station_to_index(station, chosen_stations)[0][0]
                new_data[idx].append(avg_speed)

                found_stations.append(station)

            # if freeway in freeway_dict:
            #     freeway_dict[freeway].add(station)
            # else:
            #     freeway_dict[freeway] = set([station])
        # break
    # Now that we are done with the files, get the new data
    # new_data should be (n_stations, num datapoints)
    n_datapoints = max([len(x) for x in new_data])

    # Iterate through new_data, and make sure we get rid of the bad cases
    bad_indexes = []
    for i,x in enumerate(new_data):

        # Check if the number of datapoints is correct
        if len(x) != n_datapoints:
            bad_indexes.append(i)

        # Make sure values are not all nan for this data
        if np.isnan(x).all():
            bad_indexes.append(i)

    new_data = [np.array(x) for i,x in enumerate(new_data) if i not in bad_indexes]
    chosen_stations = [x for i,x in enumerate(chosen_stations) if i not in bad_indexes]


    # Convert new_data to ndarray
    new_data = np.array(new_data)
    chosen_stations = np.array(chosen_stations)
    print(new_data.shape)
    print(chosen_stations.shape)



    # Now we actually pick the correct number of stations from our better list
    chosen_station_indexes = np.random.choice( np.arange(chosen_stations.shape[0]), n_stations)
    chosen_stations = chosen_stations[chosen_station_indexes]

    # One note - for the sake of later visualization, we need to cluster these stations
    #  where those that share a freeway are closer together
    chosen_stations, chosen_station_indexes = cluster_stations(\
        chosen_stations, chosen_station_indexes, station_location_dict)
    new_data = np.transpose(new_data[chosen_station_indexes])

    print(new_data.shape)
    print(chosen_stations.shape)

    # Now calculate our adjacency matrix based on lat/long
    weight_matrix = compute_weight_matrix(chosen_stations, station_location_dict)

    # Now we save our data
    filename = "preprocessed/PEMSD7_" + str(n_stations) + ".npz"
    with open(filename, 'wb') as f:
        np.savez(f, processed_dataset=new_data, adj_matrix=weight_matrix, station_ids=chosen_stations)
    # Save the V and W matrices
    np.savetxt("preprocessed/V_" + str(n_stations) + ".csv", new_data, delimiter=',')
    np.savetxt("preprocessed/W_" + str(n_stations) + ".csv", weight_matrix, delimiter=',')


create_custom_dataset(n_stations=228) # TODO: make sure you iterate through all the files

# TODOS:
#  - The distance might actually be in meters, so you have to convert from latlong
#     double check this with the original W by checking range of values
