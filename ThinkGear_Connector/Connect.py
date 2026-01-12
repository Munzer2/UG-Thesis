import socket
import json
import cv2
import os
import time
import csv
import random
import re
import numpy as np
from datetime import datetime


class EEGRecorder:
    def __init__(self, host='127.0.0.1', port=13854):
        self.host = host
        self.port = port
        self.socket = None
        self.buffer = ""
        self.recording = []
        # New metadata fields
        self.current_image = None
        self.current_label = None
        self.subject_id = None
        self.session_id = None
        self.trial_number = 0
        self.repetition_number = 0
        
    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Connecting to ThinkGear on {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            
            # Enable raw EEG output
            config = {"enableRawOutput": True, "format": "Json"}
            self.socket.send(json.dumps(config).encode('utf-8'))
            
            # Set socket to non-blocking for integration with slideshow
            self.socket.setblocking(False)
            print("Connected to ThinkGear!")
            return True
        except ConnectionRefusedError:
            print("Connection failed. Make sure ThinkGear Connector is running.")
            return False
    
    def read_data(self):
        """Read available data without blocking"""
        try:
            data = self.socket.recv(4096).decode('utf-8')
            if data:
                self.buffer += data
                self._process_buffer()
        except BlockingIOError:
            # No data available right now
            pass
        except Exception as e:
            print(f"Error reading data: {e}")
    
    def _process_buffer(self):
        while '\r' in self.buffer:
            packet, self.buffer = self.buffer.split('\r', 1)
            try:
                reading = json.loads(packet)
                self._handle_packet(reading)
            except json.JSONDecodeError:
                pass
    
    def _handle_packet(self, reading):
        timestamp = time.time()
        
        # Base record with all metadata
        base_record = {
            'timestamp': timestamp,
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'trial_number': self.trial_number,
            'repetition': self.repetition_number,
            'image_file': self.current_image,
            'label': self.current_label
        }
        
        # Raw EEG sample (512 per second) - ESSENTIAL for classification
        if 'rawEeg' in reading:
            record = base_record.copy()
            record.update({
                'type': 'raw',
                'rawEeg': reading['rawEeg']
            })
            self.recording.append(record)
        
        # EEG Power bands (1 per second) - IMPORTANT for classification
        if 'eegPower' in reading:
            power = reading['eegPower']
            record = base_record.copy()
            record.update({
                'type': 'eegPower',
                'theta': power.get('theta', 0),
                'lowAlpha': power.get('lowAlpha', 0),
                'highAlpha': power.get('highAlpha', 0),
                'lowBeta': power.get('lowBeta', 0),
                'highBeta': power.get('highBeta', 0),
            })
            self.recording.append(record)
            # Print alpha for visual feedback
            alpha = power.get('lowAlpha', 0) + power.get('highAlpha', 0)
            beta = power.get('lowBeta', 0) + power.get('highBeta', 0)
            print(f"    Alpha: {alpha:8d} | Beta: {beta:8d}")
    
    def save_to_csv(self, filename=None):
        # Save to the same folder as the script
        save_dir = os.path.dirname(os.path.abspath(__file__))
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"eeg_{self.subject_id}_{self.session_id}_{timestamp}.csv"
        
        filepath = os.path.join(save_dir, filename)
        
        if not self.recording:
            print("No data to save!")
            return filepath
        
        # Get all unique keys
        all_keys = set()
        for record in self.recording:
            all_keys.update(record.keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(self.recording)
        
        print(f"\n{'='*50}")
        print(f"FILE SAVED!")
        print(f"{'='*50}")
        print(f"Records: {len(self.recording)}")
        print(f"Location: {filepath}")
        print(f"{'='*50}")
        return filepath
    
    def close(self):
        if self.socket:
            self.socket.close()


def get_label_from_filename(filename):
    """
    Extract fruit label from filename
    Examples: 'apple.jpg' -> 'apple', 'apple_01.jpg' -> 'apple'
    """
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[_-]?\d+$', '', name)
    return name.lower().strip()


def create_blank_screen(text="", width=1280, height=720):
    """Create a blank screen with optional centered text"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = text.split('\n')
        line_height = 50
        total_height = len(lines) * line_height
        start_y = (height - total_height) // 2
        
        for i, line in enumerate(lines):
            font_scale = 2 if len(line) < 5 else 1
            thickness = 3 if len(line) < 5 else 2
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            x = (width - text_size[0]) // 2
            y = start_y + i * line_height + 30
            cv2.putText(img, line, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return img


def run_eeg_experiment(
    image_folder="images",
    subject_id=None,
    session_id=None,
    viewing_time=10,
    repetitions=10,
    rest_time=3,
    randomize=True
):
    """
    Run a proper EEG classification experiment
    
    Parameters:
    -----------
    image_folder : str - Folder containing fruit images
    subject_id : str - Unique identifier for the subject
    session_id : str - Session identifier
    viewing_time : int - Seconds to show each image
    repetitions : int - Times to show each image
    rest_time : int - Rest period between images (seconds)
    randomize : bool - Whether to randomize image order
    """
    
    # Get subject and session IDs
    if not subject_id:
        subject_id = input("Enter Subject ID (e.g., S01): ").strip()
        if not subject_id:
            subject_id = f"S{datetime.now().strftime('%H%M%S')}"
    
    if not session_id:
        session_id = input("Enter Session ID (e.g., session_01): ").strip()
        if not session_id:
            session_id = "session_01"
    
    # Initialize recorder
    recorder = EEGRecorder()
    recorder.subject_id = subject_id
    recorder.session_id = session_id
    
    if not recorder.connect():
        return
    
    # Get list of images
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    
    if not image_files:
        print(f"No images found in '{image_folder}' folder!")
        recorder.close()
        return
    
    # Create trial list with repetitions
    trials = []
    for img_file in image_files:
        label = get_label_from_filename(img_file)
        img_path = os.path.join(image_folder, img_file)
        
        for rep in range(1, repetitions + 1):
            trials.append({
                'image_file': img_file,
                'image_path': img_path,
                'label': label,
                'repetition': rep
            })
    
    if randomize:
        random.shuffle(trials)
    
    total_trials = len(trials)
    unique_labels = sorted(set(t['label'] for t in trials))
    
    # Print experiment info
    print("\n" + "=" * 60)
    print("EEG FRUIT CLASSIFICATION EXPERIMENT")
    print("=" * 60)
    print(f"Subject ID:      {subject_id}")
    print(f"Session ID:      {session_id}")
    print(f"Images:          {len(image_files)}")
    print(f"Labels:          {unique_labels}")
    print(f"Repetitions:     {repetitions} per image")
    print(f"Total trials:    {total_trials}")
    print(f"Viewing time:    {viewing_time} seconds")
    print(f"Rest time:       {rest_time} seconds")
    print(f"Estimated time:  {(viewing_time + rest_time) * total_trials / 60:.1f} minutes")
    print("=" * 60)
    
    # Create window
    cv2.namedWindow("Experiment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Experiment", 1280, 720)
    
    # Show instructions
    instructions = create_blank_screen(
        "INSTRUCTIONS\n\n"
        "1. Focus on each image when it appears\n"
        "2. Try to minimize blinking\n"
        "3. Stay relaxed but attentive\n"
        "4. A + sign will appear between images\n\n"
        "Press SPACE to begin"
    )
    cv2.imshow("Experiment", instructions)
    
    print("\nPress SPACE to start, or 'q' to quit...")
    
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            recorder.close()
            cv2.destroyAllWindows()
            return
    
    # Stabilization period
    print("\nStabilizing EEG signal...")
    stabilize_screen = create_blank_screen("Relax...\n\nStarting in 5 seconds")
    cv2.imshow("Experiment", stabilize_screen)
    
    start = time.time()
    while time.time() - start < 5:
        recorder.read_data()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            recorder.close()
            cv2.destroyAllWindows()
            return
    
    # Main experiment loop
    print("\n" + "-" * 60)
    print("EXPERIMENT STARTED")
    print("-" * 60)
    
    try:
        for trial_num, trial in enumerate(trials, 1):
            img = cv2.imread(trial['image_path'])
            if img is None:
                print(f"Could not load {trial['image_path']}")
                continue
            
            # Resize to fit window
            h, w = img.shape[:2]
            target_w, target_h = 1280, 720
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # Center on black background
            display = create_blank_screen()
            y_offset = (720 - new_h) // 2
            x_offset = (1280 - new_w) // 2
            display[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
            
            # Set current trial info
            recorder.current_image = trial['image_file']
            recorder.current_label = trial['label']
            recorder.trial_number = trial_num
            recorder.repetition_number = trial['repetition']
            
            print(f"\n[{trial_num:3d}/{total_trials}] {trial['label']:15s} (rep {trial['repetition']:2d}/{repetitions})")
            
            cv2.imshow("Experiment", display)
            
            # Collect data
            start_time = time.time()
            while time.time() - start_time < viewing_time:
                recorder.read_data()
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
            
            # Rest period
            recorder.current_image = "REST"
            recorder.current_label = "REST"
            
            rest_screen = create_blank_screen("+")
            cv2.imshow("Experiment", rest_screen)
            
            start_time = time.time()
            while time.time() - start_time < rest_time:
                recorder.read_data()
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE!")
        print("=" * 60)
        
        complete_screen = create_blank_screen("Experiment Complete!\n\nThank you!")
        cv2.imshow("Experiment", complete_screen)
        cv2.waitKey(3000)
        
    except KeyboardInterrupt:
        print("\n\nExperiment stopped by user.")
    
    finally:
        cv2.destroyAllWindows()
        filename = recorder.save_to_csv()
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        
        type_counts = {}
        label_counts = {}
        
        for record in recorder.recording:
            t = record.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
            
            l = record.get('label', 'unknown')
            if l != 'REST':
                label_counts[l] = label_counts.get(l, 0) + 1
        
        print(f"\nData by type:")
        for t, count in sorted(type_counts.items()):
            print(f"  {t:15s}: {count:6d} records")
        
        print(f"\nData by label:")
        for l, count in sorted(label_counts.items()):
            print(f"  {l:15s}: {count:6d} records")
        
        print(f"\nFile saved: {filename}")
        recorder.close()


def run_eeg_slideshow(image_folder="images", delay=5):
    """Run slideshow while recording EEG data"""
    
    # Initialize EEG recorder
    recorder = EEGRecorder()
    if not recorder.connect():
        return
    
    # Wait for initial connection stabilization
    print("Waiting for headset signal to stabilize...")
    time.sleep(3)
    
    # Get list of images
    images = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    
    if not images:
        print(f"No images found in '{image_folder}' folder!")
        recorder.close()
        return
    
    images.sort()
    print(f"Found {len(images)} images. Starting slideshow...")
    print("Press 'q' to quit early.\n")
    
    try:
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load {img_path}")
                continue
            
            # Set current image for data labeling
            recorder.current_image = os.path.basename(img_path)
            print(f"\n>>> Showing: {recorder.current_image}")
            
            cv2.imshow("EEG Slideshow", img)
            
            # Collect data for 'delay' seconds while showing image
            start_time = time.time()
            while time.time() - start_time < delay:
                recorder.read_data()
                
                # Check for quit key (wait 10ms)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
        
        print("\n\nSlideshow complete!")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    
    finally:
        cv2.destroyAllWindows()
        recorder.save_to_csv()
        recorder.close()


if __name__ == "__main__":
    if not os.path.exists("images"):
        os.makedirs("images")
        print("Created 'images' folder. Please add fruit images and run again.")
        print("\nNaming convention: apple.jpg, banana.png, orange.jpg, etc.")
    else:
        images = [f for f in os.listdir("images") 
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        
        if not images:
            print("No images found in 'images' folder!")
            print("Add fruit images and run again.")
        else:
            print(f"Found {len(images)} images: {images}\n")
            
            # Run the proper experiment
            run_eeg_experiment(
                image_folder="images",
                viewing_time=3,
                repetitions=1,       # Show each image 10 times
                rest_time=3,          # 3 seconds rest
                randomize=True        # Randomize order
            )