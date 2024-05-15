#!/usr/bin/env python
import csv
import cv2
import yaml
import sleap
import matplotlib
import h5py

import numpy as np
import pandas as pd

from multiprocessing import Pool
from itertools import combinations

from naps.sleap_utils import load_tracks_from_slp
from naps.utils.interactor import Interactor

class TrackingArray ():
	def __init__ (self, frames_coords_array = '', tracks = [], nodes = [], interactor_model_dict = {}, chunk_size = 5000, threads = 1, **kwargs):

		# Assign the frame coordsinates array
		self._frames_coords_array = frames_coords_array

		# Assign the tracks and nodes
		self._tracks = tracks
		self._nodes = nodes

		# Store the Interactor model
		self._interactor_model_dict = interactor_model_dict

		# Assign the frame chunk size
		self._chunk_size = chunk_size

		# Assign empty variables for processing
		self._process_frame_start = 0
		self._process_frame_end = 0
		self._process_coords_array = None

		# Assign general variables
		self._threads = threads

	def assign (self, out_file, frame_start = 0, frame_end = -1):

		# Update end frame
		frame_end = frame_end if frame_end != -1 else self._frames_coords_array.shape[0]

		# Assign the pair-wise combinations of each track
		def _yield_combinations ():
			for _tA, _tB in combinations(range(len(self._tracks)), 2):
				yield _tA, _tB

		with open(out_file, 'w') as outfile:
			csv_writer = csv.DictWriter(outfile, fieldnames = ['Origin interactor', 'Destination interactor', 'Interaction Name', 'Interaction Frame', 'Origin Area', 'Destination Area'], delimiter = '\t')
			csv_writer.writeheader()

			for chunk_frame_start in range(frame_start, frame_end, self._chunk_size):

				# Store the frame start for the current processing chunk
				self._process_frame_start = chunk_frame_start

				# Store the frame end for the current processing chunk
				chunk_end = chunk_frame_start + self._chunk_size
				self._process_frame_end = frame_end if frame_end < chunk_end else chunk_end

				# Store the processing chunked coordsinates array
				self._process_coords_array = self._frames_coords_array[self._process_frame_start:self._process_frame_end,:,:,:]

				if self._threads > 1:
					with Pool(processes = self._threads) as pool:
						for report_list in pool.starmap(self._checkForInteractions, _yield_combinations()):
							for report in report_list: csv_writer.writerow(report)
				else:
					for tagA, tagB in _yield_combinations():
						for report in self._checkForInteractions(tagA, tagB):
							csv_writer.writerow(report)

	def plot (self, interactions_filename, models_to_plot, video_filename, out_file, frame_start = 0, frame_end = -1, limit_interactions = [], plot_color = None, line_width = 2):

		# Convert to int32 for cv2 coords
		def int_coords (coords): return np.array(coords).round().astype(np.int32)

		# Update end frame
		frame_end = frame_end if frame_end != -1 else self._frames_coords_array.shape[0]
		frame_end = frame_end - 1 if frame_end >= self._frames_coords_array.shape[0] else frame_end

		# Create dataframe of instances to plot per frame
		frame_instances_dataframe = self.processInteractionsFile(interactions_filename, frame_start, frame_end, limit_interactions = limit_interactions)

		# Assign color for plotting the model:
		# 1) By default, plot red
		# 2) If given a list or array, check and assign the color
		# 3) If given a str, assign the color from matplotlib
		if plot_color == None: plot_color = (255, 0, 0)
		elif isinstance (plot_color, (list, tuple)): 
			if len(plot_color) != 3: raise Exception (f'Unable to assign plot color: {plot_color}. Iterables must be of length 3')
			plot_color = tuple(plot_color)
		elif isinstance (plot_color, str):
			plot_color = matplotlib.colors.to_rgb(plot_color)
			plot_color = tuple(_v * 255 for _v in plot_color)

		# Color the RGB to BGR used by CV2
		plot_color = plot_color[::-1]
			
		# Set the current frame of the job
		current_frame = frame_start
		
		# Initialize OpenCV, then set the starting frame (0-based)
		video = cv2.VideoCapture(video_filename)
		video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

		# Store the width, height, and fps of the input video
		video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
		video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		video_fps = int(video.get(cv2.CAP_PROP_FPS))

		# Open the output video using the input video values
		out_video = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

		# Read in the frame, confirm it was successful
		frame_read, frame_img = video.read()
		while frame_read and current_frame <= frame_end:

			# Assign the instances to plot
			instances_to_plot = frame_instances_dataframe[frame_instances_dataframe['Interaction Frame'] == current_frame]['Instances'].values
			if len(instances_to_plot) > 1: raise Exception('Error parsing interactions for instances')
			if len(instances_to_plot) > 0: instances_to_plot = instances_to_plot[0]

			# Loop each instance for this frame			
			for instance_to_plot in instances_to_plot:

				instance_interactor = Interactor.withModelDict(instance_to_plot, current_frame, self._nodes, self._frames_coords_array[current_frame,:,:,self._tracks.index(instance_to_plot)], self._interactor_model_dict)
				# Loop and plot each model for the instance
				for model_to_plot in models_to_plot:
					instance_model_coords = int_coords(instance_interactor._models['Head_w_antennae'].exterior.coords)
					cv2.polylines(frame_img, [instance_model_coords], True, plot_color, line_width)
			
			# Write the frame
			out_video.write(frame_img)

			# Read the next frame and advance the current frame
			frame_read, frame_img = video.read()
			current_frame += 1

		# Release the video
		out_video.release()

	def _checkForInteractions (self, trackA, trackB):

		# Create list to store the interactions
		interaction_list = []

		# Subset the array by tracks
		filtered_coords_array = np.take(self._process_coords_array, [trackA, trackB], 3)

		# Check each row for an interaction
		for frame, row in enumerate(filtered_coords_array):

			# Skip if missing data
			if np.isnan(row).any(): continue

			# Adjust the frame to account for the start frame
			adj_frame = self._process_frame_start + frame

			# Create the interactors
			interactor1 = Interactor.withModelDict(self._tracks[trackA], adj_frame, self._nodes, row[:,:,0], self._interactor_model_dict)
			interactor2 = Interactor.withModelDict(self._tracks[trackB], adj_frame, self._nodes, row[:,:,1], self._interactor_model_dict)

			# Check if there is an interaction
			interaction_list.extend(interactor1.interacts(interactor2))

		return interaction_list

	@classmethod
	def fromSLEAP (cls, sleap_file, model_dict = {}, model_yaml = '', **kwargs):

		# Assign the tracks from the input file
		labels = sleap.Labels.load_file(sleap_file)
		tracks = [_k.name for _k in labels.tracks]
		labels = None

		# Create an array of [frames, nodes, (x,y), tracks] from the input file
		frames_coords_array, nodes = load_tracks_from_slp(sleap_file)

		# Confirm one a single model was given
		if model_dict and model_yaml: raise Exception ('Only a single Interactor model may be specified')

		# Assign the model dict, if specified
		if model_yaml: interactor_model_dict = TrackingArray.loadYAML(model_yaml)
		elif model_dict: interactor_model_dict = model_dict

		return cls(frames_coords_array = frames_coords_array, tracks = tracks, nodes = nodes, interactor_model_dict = interactor_model_dict, **kwargs)

	@classmethod
	def fromH5 (cls, h5_file, model_dict = {}, model_yaml = '', **kwargs):

		# Assign the tracks from the input file
		with h5py.File(h5_file, "r") as f:
			tracks = f["track_names"][:]
			tracks = list(tracks)
			tracks = [s.decode("utf-8") for s in tracks]
			print(tracks)
			# Create an array of [frames, nodes, (x,y), tracks] from the input file
			frames_coords_array, nodes = f["tracks"][:].T, [n.decode() for n in f["node_names"][:]]

		# Confirm one a single model was given
		if model_dict and model_yaml: raise Exception ('Only a single Interactor model may be specified')

		# Assign the model dict, if specified
		if model_yaml: interactor_model_dict = TrackingArray.loadYAML(model_yaml)
		elif model_dict: interactor_model_dict = model_dict

		return cls(frames_coords_array = frames_coords_array, tracks = tracks, nodes = nodes, interactor_model_dict = interactor_model_dict, **kwargs)

	@staticmethod
	def loadYAML (yaml_filename):

		# Read in the YAML file and return as dict
		with open(yaml_filename, 'r') as yaml_file:
			try: yaml_dict = yaml.safe_load(yaml_file)
			except yaml.YAMLError as err: raise Exception(err)
		return yaml_dict

	@staticmethod
	def processInteractionsFile (interactions_file, frame_start, frame_end, limit_interactions = [], expected_cols = ['Origin interactor', 'Destination interactor', 'Interaction Name', 'Interaction Frame', 'Origin Area', 'Destination Area']):

		# Process the interactions file
		interactions_origin_dataframe = pd.read_csv(interactions_file, sep = '\t')[expected_cols]
		if limit_interactions: interactions_origin_dataframe = interactions_origin_dataframe[interactions_origin_dataframe['Interaction Name'].isin(limit_interactions)]
		interactions_dest_dataframe = interactions_origin_dataframe.copy()

		# Assign the origin instances for each frame
		interactions_origin_dataframe = interactions_origin_dataframe.drop(columns = ['Destination interactor'])
		interactions_origin_dataframe = interactions_origin_dataframe.rename(columns = {'Origin interactor':'Instance'})

		# Assign the destination instances for each frame
		interactions_dest_dataframe = interactions_dest_dataframe.drop(columns = ['Origin interactor'])
		interactions_dest_dataframe = interactions_dest_dataframe.rename(columns = {'Destination interactor':'Instance'})

		# Create the instance dataframe
		instance_dataframe = interactions_origin_dataframe.append(interactions_dest_dataframe, ignore_index = True)
		instance_dataframe = instance_dataframe[instance_dataframe['Interaction Frame'].between(frame_start, frame_end)]

		# Check the presence of interactions within the given frames
		if instance_dataframe.shape[0] == 0: raise Exception(f'No interactions found between frames {frame_start} to {frame_end}')

		instance_dataframe = instance_dataframe.groupby('Interaction Frame')['Instance'].apply(list).reset_index(name = 'Instances')

		return instance_dataframe

	@staticmethod
	def interpolate (coords_array):
		import pandas as pd

		for i in range(coords_array.shape[3]):

			coords_df = pd.DataFrame(coords_array[:,:,0,i])
			coords_df = coords_df.interpolate(method='linear', axis = 0)
			diff_df = coords_df.diff().abs().ge(5)
			coords_df[diff_df] = np.nan
			coords_array[:,:,0,i] = coords_df.to_numpy()

			coords_df = pd.DataFrame(coords_array[:,:,1,i])
			coords_df = coords_df.interpolate(method='linear', axis = 0)
			diff_df = coords_df.diff().abs().ge(5)
			coords_df[diff_df] = np.nan
			coords_array[:,:,1,i] = coords_df.to_numpy()


		return coords_array
