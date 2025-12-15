import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to calculate the speed of the ball
def calculate_ball_speed(previous_position, current_position, time_interval):
    distance = np.linalg.norm(np.array(previous_position) - np.array(current_position))  # Euclidean distance
    speed = distance / time_interval  # speed = distance / time
    return speed  # Speed in pixels per second (could be converted to real units if the scale is known)

# Function to track the ball in the video
def track_ball(video_path):
    cap = cv2.VideoCapture(video_path)
    previous_position = None
    current_position = None
    previous_direction = None  # To track trajectory change
    ball_radius = 10  # Adjust this according to your video's ball size
    ball_speed = 0
    time_interval = 1 / 30  # Assume 30 FPS for the video (adjust as needed)
    bounce_heights = []  # Store bounce heights

    trajectory_changes = []
    ball_speeds = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for easier processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the ball using a circular Hough transform
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=ball_radius, maxRadius=ball_radius+5
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            current_position = (circles[0][0], circles[0][1])

            # Calculate ball speed if we have a previous position
            if previous_position is not None:
                ball_speed = calculate_ball_speed(previous_position, current_position, time_interval)
                ball_speeds.append(ball_speed)  # Store ball speed

                # Track trajectory variation (simplified as the angle difference between successive directions)
                if previous_direction is not None:
                    direction_change = np.arctan2(current_position[1] - previous_position[1], current_position[0] - previous_position[0]) - previous_direction
                    trajectory_changes.append(abs(direction_change))  # Store the absolute direction change

                # Update previous direction (angle of movement)
                previous_direction = np.arctan2(current_position[1] - previous_position[1], current_position[0] - previous_position[0])

            # Detect bounce height (assuming the ball is hitting the ground)
            if previous_position is not None and current_position[1] > previous_position[1]:
                bounce_heights.append(previous_position[1])  # Record the Y position at the bounce

            # Update previous position for next frame
            previous_position = current_position

    cap.release()

    # Calculate average trajectory variation (simplified)
    trajectory_variation = np.mean(trajectory_changes) if trajectory_changes else 0

    # Estimate the bounce height (average of recorded bounce points)
    bounce_height = np.mean(bounce_heights) if bounce_heights else 0

    return ball_speeds, trajectory_changes, bounce_heights, trajectory_variation, bounce_height

# Function to classify the type of delivery (simplified)
def classify_delivery(speed, trajectory_variation):
    """Classify delivery type based on speed and trajectory variation."""
    if speed > 30 and trajectory_variation < 5:
        return "Fast Ball"
    elif trajectory_variation > 10:
        return "Spin Ball"
    else:
        return "Unknown Delivery"

# Function to analyze pitch conditions based on ball speed and bounce height
def analyze_pitch_conditions(ball_speed, bounce_height):
    if ball_speed > 35 and bounce_height > 1.5:
        return "Hard Pitch"
    elif ball_speed < 20 and bounce_height < 0.5:
        return "Soft Pitch"
    else:
        return "Normal Pitch"

# Function to save the analysis to a CSV file
def save_to_csv(ball_speeds, trajectory_variation, bounce_height, delivery_type, pitch_type):
    data = {
        'Ball Speed': [np.mean(ball_speeds) if ball_speeds else 0],
        'Trajectory Variation': [trajectory_variation],
        'Bounce Height': [bounce_height],
        'Delivery Type': [delivery_type],
        'Pitch Type': [pitch_type],
    }

    df = pd.DataFrame(data)

    # Ask user to select a path to save the CSV file
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if save_path:
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Data saved to {save_path}")

# Tkinter function to load video and display results
def open_file():
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi")])
    if file_path:
        ball_speeds, trajectory_changes, bounce_heights, trajectory_variation, bounce_height = track_ball(file_path)
        
        # Classify the delivery type
        delivery_type = classify_delivery(np.mean(ball_speeds) if ball_speeds else 0, trajectory_variation)

        # Analyze the pitch condition
        pitch_type = analyze_pitch_conditions(np.mean(ball_speeds) if ball_speeds else 0, bounce_height)

        # Show message box with results
        messagebox.showinfo("Analysis Results", f"Ball Speed: {np.mean(ball_speeds):.2f} pixels/second\n"
                                               f"Trajectory Variation: {trajectory_variation:.2f} radians\n"
                                               f"Bounce Height: {bounce_height:.2f} pixels\n"
                                               f"Delivery Type: {delivery_type}\n"
                                               f"Pitch Type: {pitch_type}")

        # Save results to CSV
        save_to_csv(ball_speeds, trajectory_variation, bounce_height, delivery_type, pitch_type)

        # Plot graphs
        plot_graphs(ball_speeds, trajectory_changes, bounce_heights)

# Function to plot the graphs
def plot_graphs(ball_speeds, trajectory_changes, bounce_heights):
    # Create a Tkinter window to display the plot
    fig, axs = plt.subplots(3, 1, figsize=(6, 9))

    # Plot Ball Speed
    axs[0].plot(ball_speeds, label='Ball Speed', color='blue')
    axs[0].set_title("Ball Speed over Time")
    axs[0].set_xlabel("Frame Number")
    axs[0].set_ylabel("Speed (pixels/sec)")
    axs[0].legend()

    # Plot Trajectory Variation
    axs[1].plot(trajectory_changes, label='Trajectory Variation', color='green')
    axs[1].set_title("Trajectory Variation over Time")
    axs[1].set_xlabel("Frame Number")
    axs[1].set_ylabel("Variation (radians)")
    axs[1].legend()

    # Plot Bounce Heights
    axs[2].plot(bounce_heights, label='Bounce Height', color='red')
    axs[2].set_title("Bounce Height over Time")
    axs[2].set_xlabel("Frame Number")
    axs[2].set_ylabel("Height (pixels)")
    axs[2].legend()

    # Embed the plot into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

# Tkinter setup
window = tk.Tk()
window.title("Cricket Ball Tracker and Analyzer")
window.geometry("800x600")

# Label
label = tk.Label(window, text="Select a Video to Analyze", font=("Arial", 14))
label.pack(pady=20)

# Button to load video file
open_button = tk.Button(window, text="Open Video File", command=open_file, font=("Arial", 12), width=20)
open_button.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
