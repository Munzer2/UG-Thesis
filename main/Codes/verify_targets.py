"""Verify all images in the 3 target files are in the correct class per tertile splits."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_design_targets import SIMPLE_DESIGN_TARGETS
from moderate_design_targets import MODERATE_DESIGN_TARGETS
from complex_design_targets import COMPLEX_DESIGN_TARGETS

# Complexity scores from the previous analysis (complexitymodel values)
SCORES = {
    '0.png': 4.23, '1.png': 5.81, '3.png': 2.94, '7.png': 7.43,
    '8.png': 5.91, '9.png': 5.62, '10.png': 5.61, '11.png': 2.85,
    '12.png': 6.43, '13.png': 5.80, '14.png': 3.94, '15.png': 5.12,
    '16.png': 6.77, '17.png': 4.62, '29.png': 5.80, '38.png': 5.72,
    '45.png': 4.24, '56.png': 7.97, '68.png': 4.91, '75.png': 6.10,
    '79.png': 3.24, '87.png': 5.02, '105.png': 7.42, '116.png': 5.14,
    '120.png': 3.27, '124.png': 3.81, '138.png': 7.57, '142.png': 4.14,
    '222.png': 8.35, '223.png': 4.68, '230.png': 4.18, '236.png': 9.85,
    '242.png': 4.75, '246.png': 1.38, '250.png': 4.87, '269.png': 4.64,
    '272.png': 3.46, '286.png': 5.40, '298.png': 5.41, '299.png': 3.81,
    '302.png': 3.05, '308.png': 4.58, '309.png': 6.18, '323.png': 6.49,
    '334.png': 2.54, '339.png': 3.20, '346.png': 4.05,
}

# Tertile boundaries
LOW = 4.25   # <= 4.25 = Simple
HIGH = 5.24  # <= 5.24 = Moderate, > 5.24 = Complex

def correct_class(score):
    if score <= LOW: return 'Simple'
    elif score <= HIGH: return 'Moderate'
    else: return 'Complex'

def check_list(targets, assigned_class, label):
    unique_files = sorted(set(t['file'] for t in targets))
    wrong = []
    print(f"\n{'='*70}")
    print(f"  {label} — {len(unique_files)} unique images")
    print(f"{'='*70}")
    print(f"  {'Image':<12} {'Score':>8} {'Correct Class':>15} {'Status':>10}")
    print(f"  {'-'*50}")
    for f in unique_files:
        score = SCORES.get(f, None)
        if score is None:
            print(f"  {f:<12} {'N/A':>8} {'???':>15} {'UNKNOWN':>10}")
            continue
        cc = correct_class(score)
        ok = 'OK' if cc == assigned_class else 'WRONG'
        print(f"  {f:<12} {score:>8.2f} {cc:>15} {ok:>10}")
        if cc != assigned_class:
            wrong.append((f, score, cc))
    
    # Also check folder field consistency
    expected_folder = {
        'Simple': 'design_A_simple',
        'Moderate': 'design_C_moderate', 
        'Complex': 'design_B_complex'
    }[assigned_class]
    wrong_folder = [t for t in targets if t['folder'] != expected_folder]
    if wrong_folder:
        print(f"\n  !! FOLDER FIELD MISMATCH: {len(wrong_folder)} entries have wrong 'folder' value")
        for t in wrong_folder:
            print(f"     {t['file']}: folder='{t['folder']}' (expected '{expected_folder}')")
    
    return wrong

print("VERIFICATION: Are all images in the correct complexity class?")
print(f"Tertile boundaries: Simple <= {LOW} | {LOW} < Moderate <= {HIGH} | Complex > {HIGH}")

w1 = check_list(SIMPLE_DESIGN_TARGETS, 'Simple', 'simple_design_targets.py -> Should be <= 4.25')
w2 = check_list(MODERATE_DESIGN_TARGETS, 'Moderate', 'moderate_design_targets.py -> Should be 4.25 < x <= 5.24')
w3 = check_list(COMPLEX_DESIGN_TARGETS, 'Complex', 'complex_design_targets.py -> Should be > 5.24')

total_wrong = w1 + w2 + w3
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Simple targets:   {len(set(t['file'] for t in SIMPLE_DESIGN_TARGETS))} images")
print(f"  Moderate targets: {len(set(t['file'] for t in MODERATE_DESIGN_TARGETS))} images")
print(f"  Complex targets:  {len(set(t['file'] for t in COMPLEX_DESIGN_TARGETS))} images")
print(f"\n  Total misplaced: {len(total_wrong)}")
if total_wrong:
    print(f"\n  MISPLACED IMAGES:")
    for f, score, cc in total_wrong:
        print(f"    {f}: score={score:.2f} -> should be in {cc}")
else:
    print(f"\n  ALL IMAGES ARE IN THE CORRECT CLASS!")
