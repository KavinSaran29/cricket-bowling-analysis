import cv2
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ------------------------ Feature Extraction ------------------------ #
def calculate_ball_speed(previous_position, current_position, time_interval):
    if previous_position is None or current_position is None:
        return 0
    distance = np.linalg.norm(np.array(previous_position) - np.array(current_position))
    return distance / time_interval

def process_video(video_path, label=None):
    cap = cv2.VideoCapture(video_path)
    previous_position = None
    previous_direction = None
    ball_radius = 10
    time_interval = 1 / 30
    trajectory_changes = []
    ball_speeds = []
    bounce_heights = []
    lengths = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30,
                                   minRadius=ball_radius, maxRadius=ball_radius+5)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            current_position = (circles[0][0], circles[0][1])
            if previous_position is not None:
                speed = calculate_ball_speed(previous_position, current_position, time_interval)
                ball_speeds.append(speed)
                pixel_distance = np.linalg.norm(np.array(current_position) - np.array(previous_position))
                lengths.append(pixel_distance)
                if previous_direction is not None:
                    direction_change = np.arctan2(current_position[1] - previous_position[1],
                                                  current_position[0] - previous_position[0]) - previous_direction
                    trajectory_changes.append(abs(direction_change))
                previous_direction = np.arctan2(current_position[1] - previous_position[1],
                                                current_position[0] - previous_position[0])
            if previous_position is not None and current_position[1] > previous_position[1]:
                bounce_heights.append(previous_position[1])
            previous_position = current_position
    cap.release()
    avg_speed = np.mean(ball_speeds) if ball_speeds else 0
    avg_trajectory_variation = np.mean(trajectory_changes) if trajectory_changes else 0
    avg_bounce_height = np.mean(bounce_heights) if bounce_heights else 0
    avg_length = np.mean(lengths) if lengths else 0
    return avg_speed, avg_trajectory_variation, avg_bounce_height, avg_length, ball_speeds, trajectory_changes, bounce_heights, label

# ------------------------ Training & Prediction ------------------------ #
def save_to_csv(video_path, label, csv_file="bowling_data.csv"):
    speed, variation, bounce, length, *_ = process_video(video_path, label)
    df = pd.DataFrame([[speed, variation, bounce, length, label]],
                      columns=["Speed", "Trajectory_Variation", "Bounce_Height", "Length", "Label"])
    if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)

def train_model(data_file="bowling_data.csv"):
    if not os.path.exists(data_file):
        print("Data file not found!")
        return
    df = pd.read_csv(data_file)
    X = df[["Speed", "Trajectory_Variation", "Bounce_Height", "Length"]]
    y = df["Label"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "bowling_model.pkl")
    print("Model trained and saved.")
    accuracy = accuracy_score(y, model.predict(X))
    print(f"Training Accuracy: {accuracy:.2f}")

def predict_next_ball(delivery_type, length):
    if delivery_type == "Fast":
        return "Likely Yorker or Short Ball" if length < 50 else "Full Length Fast Ball"
    elif delivery_type == "Spin":
        return "May follow with a Googly or Flipper" if length > 70 else "Flighted Ball"
    return "Uncertain"

def rate_delivery(speed, bounce, variation):
    score = 0
    if speed > 30: score += 1
    if bounce > 60: score += 1
    if variation > 0.1: score += 1
    if 40 < speed < 80: score += 1
    if bounce < 100: score += 1
    return f"{score}/5 ⭐"

def predict_delivery(video_path):
    if not os.path.exists("bowling_model.pkl"):
        return "Model Missing", "Unknown", [], [], [], "0/5 ⭐"
    model = joblib.load("bowling_model.pkl")
    speed, variation, bounce, length, speeds, variations, bounces, _ = process_video(video_path)
    X = np.array([[speed, variation, bounce, length]])
    prediction = model.predict(X)[0]
    next_ball = predict_next_ball(prediction, length)
    rating = rate_delivery(speed, bounce, variation)
    return prediction, next_ball, speeds, variations, bounces, rating

# ------------------------ GUI ------------------------ #
def launch_ui():
    def open_file():
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi")])
        if file_path:
            prediction, next_ball, speeds, variations, bounces, rating = predict_delivery(file_path)
            results_table.insert("", "end", values=(prediction, next_ball, rating))
            messagebox.showinfo("Prediction", f"Delivery: {prediction}\nNext Ball: {next_ball}\nRating: {rating}")
            show_graphs(speeds, variations, bounces)

    def show_graphs(speeds, variations, bounces):
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        axs[0].plot(speeds, color='blue'); axs[0].set_title("Ball Speed")
        axs[1].plot(variations, color='green'); axs[1].set_title("Trajectory Variation")
        axs[2].plot(bounces, color='red'); axs[2].set_title("Bounce Heights")
        for ax in axs: ax.set_xlabel("Frame")
        fig.tight_layout()
        win = tk.Toplevel(); win.title("Ball Stats")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw(); canvas.get_tk_widget().pack()

    window = tk.Tk()
    window.title("Cricket Ball Analysis")
    window.geometry("800x600")
    tk.Label(window, text="Cricket ML Analysis Tool", font=("Arial", 16, "bold")).pack(pady=10)
    tk.Button(window, text="Open Video File", command=open_file, bg="green", fg="white", font=("Arial", 12)).pack(pady=10)

    frame = tk.Frame(window)
    frame.pack(pady=20)

    columns = ("Delivery", "Next Ball", "Rating")
    global results_table
    results_table = ttk.Treeview(frame, columns=columns, show="headings")
    for col in columns:
        results_table.heading(col, text=col)
        results_table.column(col, width=200)
    results_table.pack()

    window.mainloop()

# ------------------------ MAIN ------------------------ #
if __name__ == "__main__":
    launch_ui()
