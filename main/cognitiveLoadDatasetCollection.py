import socket
import json
import cv2
import os
import time
import csv
import random
import numpy as np
from datetime import datetime

# Screen dimensions
FULLSCREEN = True
SCREEN_W = 1920
SCREEN_H = 1080

# ==========================================
# 1. CONFIGURATION (Map Instructions Here)
# ==========================================
# This is where you bind the specific target to the specific image.
# Even when we shuffle later, these pairs will stay together.

TRIALS_CONFIG = [
    # --- COMPLEX DESIGNS (High Load) ---
    {
        "folder": "design_A_complex", 
        "file": "DrudgeReport.png", 
        "target": "Find the \"Submit\" button"
    },
    {
        "folder": "design_A_complex", 
        "file": "pnwx.png", 
        "target": "Find the link named \"Small Animal Immobilizers\""
    },
    {
        "folder": "design_A_complex", 
        "file": "seiryu-kan.png", 
        "target": "Find the link named \"Shuriken\""
    },
    {
        "folder": "design_A_complex", 
        "file": "yahooJP.png", 
        "target": "Find the weather icons"
    },
    {
        "folder": "design_A_complex", 
        "file": "yaleSchoolOfArt.png", 
        "target": "Find the \"Log in\" button" 
    },

    # --- SIMPLE DESIGNS (Low Load) ---
    {
        "folder": "design_B_simple", 
        "file": "dropbox.png", 
        "target": "Find the \"login\" button"
    },
    {
        "folder": "design_B_simple", 
        "file": "google.png", 
        "target": "Find the \"I'm feeling lucky\" button"
    },
    {
        "folder": "design_B_simple", 
        "file": "notion.png", 
        "target": "Find the Trash icon"
    },
    {
        "folder": "design_B_simple", 
        "file": "stripe.png", 
        "target": "Find the \"Start now\" button"
    },
    {
        "folder": "design_B_simple", 
        "file": "uber.png", 
        "target": "Find the \"About\" button"
    }
]

TRIALS_CONFIG2 = [
    # --- COMPLEX WEBSITES (High Load) ---
    
    # 1. Yahoo! Japan
    {"folder": "design_A_complex", "file": "yahooJP.png", "target": "Find the search button"},
    {"folder": "design_A_complex", "file": "yahooJP.png", "target": "Find the Travel icon"},
    {"folder": "design_A_complex", "file": "yahooJP.png", "target": "Find the shopping icon"},
    
    # 2. Drudge Report
    {"folder": "design_A_complex", "file": "DrudgeReport.png", "target": "Find the Submit button"},
    {"folder": "design_A_complex", "file": "DrudgeReport.png", "target": "Find the Search box"},
    {"folder": "design_A_complex", "file": "DrudgeReport.png", "target": "Find the Drudge Report logo"},

    # 3. PNWX (Medical Supplies)
    {"folder": "design_A_complex", "file": "pnwx.png", "target": "Find Small Animal Immobilizers link"},
    {"folder": "design_A_complex", "file": "pnwx.png", "target": "Find the GO button"},
    {"folder": "design_A_complex", "file": "pnwx.png", "target": "Find the Lead Curtains link"},

    # 4. Seiryu-Kan (Martial Arts)
    {"folder": "design_A_complex", "file": "seiryu-kan.png", "target": "Find the Shuriken link"},
    {"folder": "design_A_complex", "file": "seiryu-kan.png", "target": "Find the Click here link"},
    {"folder": "design_A_complex", "file": "seiryu-kan.png", "target": "Find the One-time trial seminar section"},

    # 5. Yale School of Art
    {"folder": "design_A_complex", "file": "yaleSchoolOfArt.png", "target": "Find the Visitor Log in button"},
    {"folder": "design_A_complex", "file": "yaleSchoolOfArt.png", "target": "Find the QUICK LINKS section"},
    {"folder": "design_A_complex", "file": "yaleSchoolOfArt.png", "target": "Find the News link"},


    # --- SIMPLE WEBSITES (Low Load) ---

    # 6. Google
    {"folder": "design_B_simple", "file": "google.png", "target": "Find the \"I'm Feeling Lucky\" button"},
    {"folder": "design_B_simple", "file": "google.png", "target": "Find the Gmail link"},
    {"folder": "design_B_simple", "file": "google.png", "target": "Find the Microphone icon"},

    # 7. Dropbox
    {"folder": "design_B_simple", "file": "dropbox.png", "target": "Find the \"Get started\" button"},
    {"folder": "design_B_simple", "file": "dropbox.png", "target": "Find the \"Sign up\" button"},

    # 8. Notion
    {"folder": "design_B_simple", "file": "notion.png", "target": "Find the Trash icon"},
    {"folder": "design_B_simple", "file": "notion.png", "target": "Find the Add new option"},
    
    # 9. Stripe
    {"folder": "design_B_simple", "file": "stripe.png", "target": "Find the \"Start now\" button"},
    {"folder": "design_B_simple", "file": "stripe.png", "target": "Find the \"Contact sales\" button"},
    # 10. Uber
    {"folder": "design_B_simple", "file": "uber.png", "target": "Find the Company link"},
    {"folder": "design_B_simple", "file": "uber.png", "target": "Find the \"Sign up\" button"},
    {"folder": "design_B_simple", "file": "uber.png", "target": "Find the \"Log in\" button"}
]

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
        self.current_label = ""   # 'complex' or 'simple'
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
    
    # B. SMART SAMPLING (The "No Memory" Fix)
    # ---------------------------------------------------------
    print("\nPreparing session...")
    
    # 1. Check which files actually exist
    available_trials = []
    for t in TRIALS_CONFIG2:
        path = os.path.join(t['folder'], t['file'])
        if os.path.exists(path):
            t['full_path'] = path
            available_trials.append(t)
        else:
            # Silent skip or warn once
            pass

    if not available_trials:
        print("ERROR: No images found! Check your 'design_A' and 'design_B' folders.")
        return

    # 2. Group by Filename
    grouped_trials = {}
    for t in available_trials:
        fname = t['file']
        if fname not in grouped_trials:
            grouped_trials[fname] = []
        grouped_trials[fname].append(t)
    
    # 3. Pick ONE random target per website
    session_trials = []
    unique_websites = list(grouped_trials.keys())
    
    # Ensure we show every unique website once
    for fname in unique_websites:
        chosen_target = random.choice(grouped_trials[fname])
        session_trials.append(chosen_target)
    
    # 4. Shuffle the final order
    random.shuffle(session_trials)
    
    print(f"Session Ready: {len(session_trials)} unique websites selected.")
    print("(You will see each website once, with a random target).")

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
                    print(f"[{i+1}/{len(session_trials)}] RT: {rt:.2f}s | {trial['target']}")
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