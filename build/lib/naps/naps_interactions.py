#!/usr/bin/env python
import os
import time
import logging
import argparse
from naps.utils.tracking import TrackingArray

logger = logging.getLogger("NAPS Logger")

def interactionParser ():
	'''
	Interaction Parser

	Assign the parameters for interaction assignment

	Parameters
	----------
	sys.argv : list
		Parameters from command line

	Raises
	------
	IOError
		If the specified files do not exist
	'''

	def confirmFile ():
		'''Custom action to confirm file exists'''
		class customAction(argparse.Action):
			def __call__(self, parser, args, value, option_string=None):
				if not os.path.isfile(value):
					raise IOError('%s not found' % value)
				setattr(args, self.dest, value)
		return customAction

	interaction_parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

	# Input files
	interaction_parser.add_argument('--h5-file', help = 'Defines the input h5 file', type = str, action = confirmFile(), required = True)
	interaction_parser.add_argument('--yaml-model', help = 'Defines the YAML model file', type = str, action = confirmFile(), required = True)

	# General options
	interaction_parser.add_argument('--frame-start', help = 'Defines the frame to start interaction assignment', type = int, default = 0)
	interaction_parser.add_argument('--frame-end', help = 'Defines the frame to end interaction assignment', type = int, default = -1)
	interaction_parser.add_argument('--threads', help = 'Number of threads to use', type = int, default = 1)

	#interaction_parser.add_argument('--max-velocity', help = 'Defines the maximum velocity allowed in the dataset', type = int)
	#interaction_parser.add_argument('--remove-tags', help = 'Defines the tags to remove', type = str, nargs = '+', default = [])
	#interaction_parser.add_argument('--min-bin', help = 'Min interaction distance', type = int, default = 50)
	#interaction_parser.add_argument('--max-bin', help = 'Max interaction distance', type = int, default = 450)
	#interaction_parser.add_argument('--frame-buffer', help = 'Frame buffer - i.e. gap between frames allowed', type = int, default = 5)
	#interaction_parser.add_argument('--min-frames', help = 'Min frames for interaction', type = int, default = 5)

	# Output options
	interaction_parser.add_argument('--out-file', help = 'Defines the output tsv file. Defaults to: out.tsv', type = str, default = 'out.tsv')

	# Return the arguments
	return interaction_parser.parse_args()

def main():

	# Assign the interaction args
	interaction_args = interactionParser()

	# Load the tracking array
	logger.info("Loading the tracking array...")
	t0 = time.time()

	# Create the interactions object using a SLEAP file and the yaml model
	interactions = TrackingArray.fromH5(interaction_args.h5_file, model_yaml = interaction_args.yaml_model, threads = interaction_args.threads)

	logger.info("Done loading in %s seconds.", time.time() - t0)

	# Assign interactions from the tracking array
	logger.info("Starting interaction assignment...")
	t0 = time.time()

	# Assign the interactions
	interactions.assign(interaction_args.out_file, frame_start = interaction_args.frame_start, frame_end = interaction_args.frame_end)

	logger.info("Done with assignment in %s seconds.", time.time() - t0)

if __name__ == "__main__":
	main()
