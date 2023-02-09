import json
import os
import math
import librosa
import yaml

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_yaml():
    """ Loads configuration settings from dataset_config.yaml
    
    Returns:
        data (dict): a dict containing the configuration settings
    """
    with open("dataset_config.yaml","r") as f:
        data = yaml.load(f,Loader=yaml.Loader)
        return data

class MFFC_Generator:
    """A dataset generator which stores MFCCs and labels of the music

    Attributes:
        data: contains the labels and MFCC data
    """
    def __init__(self):
        # Dictionary to store labels, and MFCCs
        self.data = {
            "labels": [],
            "mfcc": []
        }    

   
    def generate_dataset(self):

        """Generates a dataset consisting of labels and MFCCs for the dataset, stored in JSON
        """
        config = load_yaml()
        samples_per_segment = int((config["sample_rate"] * config["track_duration"]) / config["segments"])
        mfcc_vectors_per_segment = math.ceil(samples_per_segment / config["hop_length"])

        for i, (path, _, file_names) in enumerate(os.walk(config["dataset_path"])):
            if path is not config['dataset_path']:
                for f in file_names:

                    # Load the audio file
                    audio_path = os.path.join(path, f)

                    try:
                        signal, sample_rate = librosa.load(audio_path, sr=config["sample_rate"])
                    except:
                        continue

                    # Process all segments of the audio
                    for j in range(config["segments"]):

                        # Calculate start and finish sample for current segment
                        start = samples_per_segment * j
                        finish = start + samples_per_segment

                        # Extract MFCC
                        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=config["mfcc"], n_fft=config["n_fft"], hop_length=config["hop_length"])
                        mfcc = mfcc.T

                        # Store MFCC with expected number of vectors and labels
                        if len(mfcc) == mfcc_vectors_per_segment:
                            self.data["mfcc"].append(mfcc.tolist())
                            self.data["labels"].append(i-1)
                            print("{}, segment:{}".format(audio_path, j+1))

        # Save MFCCs to a JSON file
        with open(config["data_path"], "w") as fp:
            json.dump(self.data, fp, indent=4)       
        
        
if __name__ == "__main__":
    mfcc_generator = MFFC_Generator()
    mfcc_generator.generate_dataset()