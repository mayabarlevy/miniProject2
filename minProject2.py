import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os

def calc_mean_erp(trial_points, ecog_data):
    try:
        # Extract the ZIP file
        zip_file_path = 'mini_project_2_data.zip'
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            extract_folder = 'extracted_data'
            zip_ref.extractall(extract_folder)

        # Load data
        trial_points_file = os.path.join(extract_folder, 'mini_project_2_data/events_file_ordered.csv')
        ecog_data_file = os.path.join(extract_folder, 'mini_project_2_data/brain_data_channel_one.csv')

        trial_points = pd.read_csv(trial_points_file, header=None)
        trial_points.columns = ['start', 'peak', 'finger']
        trial_points = trial_points.astype({'start': 'int', 'peak': 'int', 'finger': 'int'})

        ecog_data = pd.read_csv(ecog_data_file, header=None).squeeze("columns").to_numpy()

        # Parameters
        pre_movement = 200  # ms before the starting point
        post_movement = 1000  # ms after the starting point
        num_samples = pre_movement + post_movement + 1

        # Initialize the result matrix
        fingers_erp_mean = np.zeros((5, num_samples))

        for finger in range(1, 6):
            # Filter trials for the current finger
            finger_trials = trial_points[trial_points['finger'] == finger]
            
            # Collect ERP segments for the finger
            erp_segments = []
            for _, trial in finger_trials.iterrows():
                start_idx = trial['start'] - pre_movement
                end_idx = trial['start'] + post_movement
                if start_idx >= 0 and end_idx < len(ecog_data):
                    erp_segments.append(ecog_data[start_idx:end_idx + 1])
            
            # Average the segments
            if erp_segments:
                fingers_erp_mean[finger - 1] = np.mean(erp_segments, axis=0)
            else:
                print(f"No valid trials found for finger {finger}.")

        # Plot the ERP for each finger
        time_axis = np.arange(-pre_movement, post_movement + 1)
        plt.figure(figsize=(10, 6))
        for finger in range(5):
            plt.plot(time_axis, fingers_erp_mean[finger], label=f'Finger {finger + 1}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Brain Response (ECoG Signal)')
        plt.title('Event-Related Potential (ERP) for Each Finger')
        plt.legend()
        plt.grid()
        plt.show()

        # Print the resulting matrix
        if fingers_erp_mean is not None:
            print("Resulting ERP matrix:")
            print(fingers_erp_mean)

        return fingers_erp_mean

    except Exception as e:
        print(f"An error occurred: {e}")

# File paths based on ZIP content
trial_points_file = os.path.join('extracted_data', 'mini_project_2_data/events_file_ordered.csv')
ecog_data_file = os.path.join('extracted_data', 'mini_project_2_data/brain_data_channel_one.csv')

# Example usage
fingers_erp_mean = calc_mean_erp(trial_points_file, ecog_data_file)

