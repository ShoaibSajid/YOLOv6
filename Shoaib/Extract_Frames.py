import cv2

# Function to extract frames
def extract_frames(video_path, num_frames=30, output_folder='frames', crop_size=(400,900)):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval between frames to extract
    interval = total_frames // num_frames

    # Ensure the output folder exists
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract frames
    for i in range(num_frames):
        
        if i+1 in [0,1,2,3,4,5,6,7,8,9,10,19,29,30]: continue
        
        frame_number = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Move to the frame position
        ret, frame = cap.read()  # Read the frame

        if ret:
            # Get the dimensions of the frame
            height, width, _ = frame.shape

            # Calculate the coordinates for the central crop
            center_x, center_y = width // 2, height // 2
            crop_width, crop_height = crop_size

            start_x = max(center_x - crop_width // 2, 0)
            start_y = max(center_y - crop_height // 2, 0)
            end_x = min(center_x + crop_width // 2, width)
            end_y = min(center_y + crop_height // 2, height)

            # Crop the central part of the frame
            cropped_frame = frame[start_y:end_y, start_x:end_x]

            # Save the cropped frame as an image
            frame_filename = os.path.join(output_folder, f'frame_{i+1}.png')
            cv2.imwrite(frame_filename, cropped_frame)
            print(f'Saved: {frame_filename}')
        else:
            print(f"Error: Could not read frame {frame_number}")

    # Release the video capture object
    cap.release()
    print("Frame extraction complete.")

# Usage example
extract_frames('connector.avi')
