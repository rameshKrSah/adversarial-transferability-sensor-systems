#!/usr/bin/env python
# coding: utf-8

import os
import sys
import getopt
import numpy as np
import pandas as pd
import datetime
import zipfile
import pickle


device_position = ["Chest", "Hand", "Head", "Hip", "Forearm", "Shin", "Thigh", "UpperArm", "Waist"]
environment = ["Building", "Home", "Office", "Street", "Transportation"]
posture = ["Climbing (down)", "Climbing (up)", "Jumping", "Laydown", "Running", "Sitting", "Standing", "Walking"]
activity = ["DeskWork", "Eating/Drinking", "Housework", "Mealpreparation", "Movement", "PersonalGrooming", "Relaxing", 
            "Shopping", "Socializing", "Sleeping", "Sport", "Transportation"]
sub_activity = ["Somethingelse", "GotoWork", "WatchingTV", "TidyingUp", "GoHome", "atHome", "Breakfast", "Brunch", "CofeeBreak",
               "Dinner", "Lunch", "Snack", "Cleaning", "GoForAWalk", "Playing", "ListenToMusic", "Bar/Disco", "CinemaAtHome",
               "Party", "Basketball", "Bicycling", "Dancing", "Gym", "Gymnastics", "IceHockey", "Jogging", "Soccer", 
               "Bicycle", "Bus", "Car", "Motorcycle", "Scooter", "Skateboard", "Train", "Tram"]


device_position_lower = [str.lower(p) for p in device_position]
environment_lower = [str.lower(p) for p in environment]
posture_lower = [str.lower(p) for p in posture]
activity_lower = [str.lower(p) for p in activity]
sub_activity_lower = [str.lower(p) for p in sub_activity]


# the date time format for the dataset
date_time_format = "%d.%m.%y %H:%M:%S"
date_time_milli_seconds_format = "%d.%m.%y %H:%M:%S.%f"

acceleration_frequency = 50
orientation_frequency = 50

sensor_time_position = 1
sensor_value_position = [2, 3, 4]
sensor_label_environment_position = 5
sensor_label_posture_position = 6
sensor_label_device_position = 7
sensor_label_activity_position = 8

label_day_position = 1
label_subject_id_position = 2
label_start_time_position = 3
label_end_time_position = 4
label_position = 5

data_folder = "../dailylog2016_dataset/data"


# - Acceleration (50 Hz), Orientation (50 Hz), GPS (every 10 minutes)
# - Labels: Activity, Device Position, Environment, Posture
# 
# 
# - Device Posistion: Chest, Hand, Head, Hip, Forearm, Shin, Thigh, Upper Arm, Waist
# - Environment: Building, Home, Office, Street, Transportation
# - Posture: Climbing, Jumping, Lay, Running, Sitting, Standing, Walking
# 
# - Activity - Subactivity
# - Desk Work - N/A
# - Eating / Drinking - Breakfast, Brunch, Cofee Break, Dinner, Lunch Snack
# - Housework - Cleaning, Tidying Up
# - Meal Preparation - N/A
# - Movement - Go for a Walk, Go Home, Go to Work
# - Personal Grooming - N/A
# - Relaxing - Playing, Listen to Music, Watching TV
# - Shopping - N/A
# - Socializing - Bar / Disco, Cinema at Home
# - Sleeping - N/A
# - Sport - Basketball, Bicycling, Dancing, Gym, Gymnastics, Ice Hockey, Jogging, Soccer
# - Transportation - Bicycle, Bus, Car, Motorcycle, Scooter, Skateboard, Train, Tram
# 
# 

'''
    Extract data from the accelerometer files for the given label type. 
    @param: data_folder, path to the folder with subjects folder
    @param: label_type, which label type to extract from the accelerometer data files. Possible values are 
        sensor_label_environment_position = 5 for environment labels
        sensor_label_posture_position = 6 for posture labels
        sensor_label_device_position = 7 for device position labels
        sensor_label_activity_position = 8 for activity labels
        
    @param: open_zips, whether to open the acc_csv zip files or not before extracting the sensor values for the given label type
    
'''
def combine_subject_files(data_folder, label_type, open_zips = False):
    # container to store the processed data
    # we have 4 values in each row
    processed_data = np.empty((1, 4))
    
    # get the subject folders
    subject_folders = os.listdir(data_folder)
    
    # for each subject folder
    for folder in subject_folders:
        print(len(processed_data))
        # for each subject folder, we have data and image folder
        subject_data_folder = data_folder + "/" + folder + "/" + "data/"
        
        # now at this location we have acc_csv zip file. 
        if open_zips:
            #open acc_csv zip to get the 14 days data for each subject
            zip_file = subject_data_folder + "acc_csv.zip"
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(subject_data_folder + "acc_csv/")
            except:
                # if not able to open the zip file then just continue to next subject
                continue
            
        # now for each extracted acc csv file
        csv_files = os.listdir(subject_data_folder + "acc_csv/")
        for csv_file in csv_files:
            file_path = subject_data_folder + "acc_csv/" + csv_file
            print("Processing file : " + file_path)
            
            # load the CSV file
            data = pd.read_csv(file_path).to_numpy()

            # get the unique labels in the data
            unique_labels = np.unique(data[:, label_type])
#             print(np.unique(data[:, label_type], return_counts=True))

            # for each unique label
            for label in unique_labels:
#               print(label)
                # get the index for this label
                indexes = np.where(data[:, label_type] == label)[0]

                # get the data at the indexes
                data_for_label = data[indexes]
                # get the sensor value 2:5 and the label for the specified type
                data_for_label = data_for_label[:, np.r_[2:5, label_type]] 
#               print(data_for_label.shape)

                # store the data in container
                processed_data = np.concatenate([processed_data, data_for_label])
                
    return processed_data[1:]


"""
    Given data and path, save the data as pickle.
"""
def save_data(data, path):
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()

"""
    Parse the arguments passed with the script
"""
def parse_arguments(argv):
    data_folder = " "
    output_file = " "
    label_type = -1
    
    try:
        opts, args = getopt.getopt(argv, "hi:o:l:", ["ifile=", "ofile=", "ltype="])
    except:
        print("data_processing.py -i <data_folder_with_subject_folder> -o <output_file_as_pickle> -l <label_type> \nPossible value for label type\n5: Environment\n6: Posture\n7: Device Position\n8: Activity")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == "-h":
            print("data_processing.py -i <data_folder_with_subject_folder> -o <output_file_as_pickle> -l <label_type> \nPossible value for label type\n5: Environment\n6: Posture\n7: Device Position\n8: Activity")
            sys.exit()
            
        elif opt in ("-i", "--ifile"):
            data_folder = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt in ("-l", "--ltype"):
            label_type = int(arg)
            
    return data_folder, label_type, output_file


# Perform the main action
if __name__ == "__main__":
    data_path, label_type, save_path = parse_arguments(sys.argv[1:])
    
    if((len(data_path) != 0) & (label_type > 0) & (len(save_path) != 0)):
        data = combine_subject_files(data_path, label_type)
        save_data(data, save_path)
    else:
        print("data_processing.py -i <data_folder_with_subject_folder> -o <output_file_as_pickle> -l <label_type> \nPossible value for label type\n5: Environment\n6: Posture\n7: Device Position\n8: Activity")
        sys.exit(2)




