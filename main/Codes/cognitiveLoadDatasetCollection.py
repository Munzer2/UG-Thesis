import socket
import json
import cv2
import os
import time
import csv
import random
import numpy as np
from datetime import datetime

# Import target configurations
from simple_design_targets import SIMPLE_DESIGN_TARGETS
from complex_design_targets import COMPLEX_DESIGN_TARGETS
from moderate_design_targets import MODERATE_DESIGN_TARGETS

# Screen dimensions
FULLSCREEN = True
SCREEN_W = 1920
SCREEN_H = 1080

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Combine all targets from the three complexity levels
ALL_TRIALS = SIMPLE_DESIGN_TARGETS + MODERATE_DESIGN_TARGETS + COMPLEX_DESIGN_TARGETS

print(f"Loaded {len(SIMPLE_DESIGN_TARGETS)} simple targets")
print(f"Loaded {len(MODERATE_DESIGN_TARGETS)} moderate targets")
print(f"Loaded {len(COMPLEX_DESIGN_TARGETS)} complex targets")
print(f"Total targets: {len(ALL_TRIALS)}")

# ==========================================
# 2. EEG RECORDER CLASS
# ==========================================
class EEGRecorder:
    def __init__(self, host='127.0.0.1', port=13854):
        self.host = host
        self.port = port
        self.socket = None
        self.buffer = ""
        self.recording = []
        
        # Metadata needed for analysis
        self.subject_id = ""
        self.current_image = ""
        self.current_label = ""   # 'simple', 'moderate', or 'complex'
        self.current_target = ""  # The instruction text
        self.trial_phase = ""     # 'INSTRUCTION', 'FIXATION', or 'TASK'
        self.reaction_time = 0
        
    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Connecting to ThinkGear on {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            # Enable raw output (JSON format)
            self.socket.send(json.dumps({"enableRawOutput": True, "format": "Json"}).encode('utf-8'))
            self.socket.setblocking(False)
            print("Connected successfully!")
            return True
        except ConnectionRefusedError:
            print("Connection failed. Is the ThinkGear Connector app running?")
            return False
    
    def read_data(self):
        """Reads data from the headset buffer without blocking the UI"""
        try:
            data = self.socket.recv(4096).decode('utf-8')
            if data:
                self.buffer += data
                while '\r' in self.buffer:
                    pkt, self.buffer = self.buffer.split('\r', 1)
                    try: self._log_packet(json.loads(pkt))
                    except: pass
        except BlockingIOError:
            pass

    def _log_packet(self, data):
        """Saves a single data packet with current experiment metadata"""
        row = {
            'timestamp': time.time(),
            'subject_id': self.subject_id,
            'image': self.current_image,
            'label': self.current_label,
            'target_instruction': self.current_target,
            'phase': self.trial_phase,
            'reaction_time': self.reaction_time
        }
        
        # 1. Save Raw EEG (512 Hz) - High Fidelity
        if 'rawEeg' in data:
            r = row.copy()
            r.update({'type': 'raw', 'value': data['rawEeg']})
            self.recording.append(r)
            
        # 2. Save Power Bands (1 Hz) - Good for Quick Analysis
        if 'eegPower' in data:
            p = data['eegPower']
            r = row.copy()
            r.update({
                'type': 'power',
                'theta': p.get('theta'),
                'alpha': (p.get('lowAlpha') + p.get('highAlpha')) / 2,
                'beta': (p.get('lowBeta') + p.get('highBeta')) / 2,
                'gamma': (p.get('lowGamma') + p.get('highGamma')) / 2,
                'attention': data.get('eSense', {}).get('attention', 0)
            })
            self.recording.append(r)

    def save(self):
        if not self.recording: 
            print("No data recorded. Nothing to save.")
            return
        filename = f"UI_Exp_{self.subject_id}_{datetime.now().strftime('%H%M%S')}.csv"
        
        # Get all unique columns
        keys = set().union(*(d.keys() for d in self.recording))
        
        with open(filename, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted(keys))
            w.writeheader()
            w.writerows(self.recording)
        print(f"\n[SUCCESS] Data saved to: {filename}")

# ==========================================
# 3. HELPER: UI DRAWER
# ==========================================
def create_screen(text, subtext="", w=1280, h=720, bg=(20, 20, 20)):
    """Creates an image with centered text"""
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Word Wrap Logic
    words = text.split(' ')
    lines = []
    current_line = words[0]
    for word in words[1:]:
        if len(current_line + " " + word) < 25: # Character limit
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    
    # Draw Lines
    y = h // 2 - (len(lines) * 30)
    for line in lines:
        ts = cv2.getTextSize(line, font, 1.5, 2)[0]
        cv2.putText(img, line, ((w - ts[0])//2, y), font, 1.5, (255, 255, 255), 2)
        y += 60
        
    if subtext:
        ts = cv2.getTextSize(subtext, font, 0.8, 1)[0]
        cv2.putText(img, subtext, ((w - ts[0])//2, h - 100), font, 0.8, (150, 150, 150), 1)
        
    return img

# ==========================================
# 4. MAIN EXPERIMENT LOGIC
# ==========================================
def run_experiment():
    # A. CONNECT TO HEADSET
    recorder = EEGRecorder()
    if not recorder.connect(): return
    
    recorder.subject_id = input("Enter Participant Name/ID: ").strip()
    
    # B. SMART SAMPLING - 5 from each category
    # ---------------------------------------------------------
    print("\nPreparing session...")
    
    # Get parent directory (design folders are one level up from Codes/)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Check which files actually exist and group by category
    simple_trials = []
    moderate_trials = []
    complex_trials = []
    
    for t in ALL_TRIALS:
        path = os.path.join(base_path, t['folder'], t['file'])
        if os.path.exists(path):
            t['full_path'] = path
            if 'A_simple' in t['folder']:
                simple_trials.append(t)
            elif 'C_moderate' in t['folder']:
                moderate_trials.append(t)
            elif 'B_complex' in t['folder']:
                complex_trials.append(t)
        else:
            print(f"WARNING: File not found: {path}")

    # 2. Group each category by filename
    def group_by_file(trials):
        grouped = {}
        for t in trials:
            fname = t['file']
            if fname not in grouped:
                grouped[fname] = []
            grouped[fname].append(t)
        return grouped
    
    simple_grouped = group_by_file(simple_trials)
    moderate_grouped = group_by_file(moderate_trials)
    complex_grouped = group_by_file(complex_trials)
    
    # 3. Randomly select 5 unique images from each category
    session_trials = []
    
    # Simple: Pick 5 random files
    simple_files = list(simple_grouped.keys())
    random.shuffle(simple_files)
    for fname in simple_files[:5]:
        chosen_target = random.choice(simple_grouped[fname])
        session_trials.append(chosen_target)
    
    # Moderate: Pick 5 random files
    moderate_files = list(moderate_grouped.keys())
    random.shuffle(moderate_files)
    for fname in moderate_files[:5]:
        chosen_target = random.choice(moderate_grouped[fname])
        session_trials.append(chosen_target)
    
    # Complex: Pick 5 random files
    complex_files = list(complex_grouped.keys())
    random.shuffle(complex_files)
    for fname in complex_files[:5]:
        chosen_target = random.choice(complex_grouped[fname])
        session_trials.append(chosen_target)
    
    # 4. Shuffle the final order (mix all categories)
    random.shuffle(session_trials)
    
    print(f"\nSession Ready: {len(session_trials)} trials prepared")
    print(f"  - Simple: 5 random images")
    print(f"  - Moderate: 5 random images")
    print(f"  - Complex: 5 random images")
    print("(Each image shown once with a random target)")

    # C. SETUP WINDOW
    cv2.namedWindow("Experiment", cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty("Experiment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow("Experiment", SCREEN_W, SCREEN_H)

    cv2.imshow("Experiment", create_screen("Press SPACE to Start"))
    cv2.waitKey(0)

    try:
        for i, trial in enumerate(session_trials):
            
            # ----------------------------------------
            # PHASE 1: INSTRUCTION (5 Seconds)
            # ----------------------------------------
            recorder.trial_phase = "INSTRUCTION"
            recorder.current_target = trial['target']
            recorder.current_image = "instruction_screen"
            recorder.current_label = trial['folder']
            
            img = create_screen(f"TARGET: {trial['target']}", "Memorize this target...")
            cv2.imshow("Experiment", img)
            
            start_instr = time.time()
            while time.time() - start_instr < 5.0:
                recorder.read_data()
                if cv2.waitKey(10) == ord('q'): raise KeyboardInterrupt

            # ----------------------------------------
            # PHASE 2: FIXATION CROSS (1 Second)
            # ----------------------------------------
            recorder.trial_phase = "FIXATION"
            recorder.current_image = "fixation_cross"
            
            cv2.imshow("Experiment", create_screen("+"))
            
            start_fix = time.time()
            while time.time() - start_fix < 1.0:
                recorder.read_data()
                cv2.waitKey(10)

            # ----------------------------------------
            # PHASE 3: VISUAL SEARCH TASK (The Data!)
            # ----------------------------------------
            recorder.trial_phase = "TASK"
            recorder.current_image = trial['file']
            
            # Load and Resize Image
            img = cv2.imread(trial['full_path'])
            h, w = img.shape[:2]
            
            # Maintain Aspect Ratio
            scale = min(SCREEN_W/w, SCREEN_H/h)
            new_w, new_h = int(w*scale), int(h*scale)
            img = cv2.resize(img, (new_w, new_h))
            
            # Place on Black Background (Padding)
            display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            y_off = (SCREEN_H - new_h) // 2
            x_off = (SCREEN_W - new_w) // 2
            display[y_off:y_off+new_h, x_off:x_off+new_w] = img
            
            cv2.imshow("Experiment", display)
            
            # Wait for Response
            start_task = time.time()
            responded = False
            
            while not responded:
                recorder.read_data()
                key = cv2.waitKey(5) & 0xFF
                
                # SPACE BAR = Found
                if key == 32: 
                    rt = time.time() - start_task
                    recorder.reaction_time = rt
                    
                    # Determine complexity level
                    if 'A_simple' in trial['folder']:
                        complexity = "SIMPLE"
                    elif 'C_moderate' in trial['folder']:
                        complexity = "MODERATE"
                    else:
                        complexity = "COMPLEX"
                    
                    print(f"[{i+1}/{len(session_trials)}] {complexity} | RT: {rt:.2f}s | {trial['target']}")
                    responded = True
                
                # 'q' = Quit
                elif key == ord('q'):
                    raise KeyboardInterrupt
                
                # Timeout (40s limit)
                if time.time() - start_task > 40.0:
                    print(f"[{i+1}] TIMEOUT")
                    recorder.reaction_time = 40.0
                    responded = True

    except KeyboardInterrupt:
        print("\nExperiment Aborted by User.")
    finally:
        recorder.save()
        recorder.socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_experiment()