import cv2
import os

# Define all images to check with their targets
images_to_check = [
    {
        "path": r"design_A_complex\yahooJP.png",
        "targets": [
            "RED ENVELOPE (Mail) icon",
            "BLUE SEARCH button",
            "SUN/CLOUD (Weather) icon"
        ]
    },
    {
        "path": r"design_A_complex\DrudgeReport.png",
        "targets": [
            "\"Submit\" button",
            "\"SEARCH\" box",
            "Top Left Logo"
        ]
    },
    {
        "path": r"design_A_complex\pnwx.png",
        "targets": [
            "\"Small Animal Immobilizers\"",
            "\"Products\" Menu Button",
            "blue \"Pb\" (Lead) icon"
        ]
    },
    {
        "path": r"design_A_complex\seiryu-kan.png",
        "targets": [
            "\"Shuriken\"",
            "\"Map\" or \"Access\" link",
            "red \"English\" text"
        ]
    },
    {
        "path": r"design_A_complex\yaleSchoolOfArt.png",
        "targets": [
            "\"Log in\" button",
            "\"Wiki\" link",
            "\"Quick Links\" sidebar"
        ]
    },
    {
        "path": r"design_B_simple\google.png",
        "targets": [
            "\"I'm Feeling Lucky\" button",
            "\"Gmail\" link",
            "Microphone Icon"
        ]
    },
    {
        "path": r"design_B_simple\dropbox.png",
        "targets": [
            "\"Login\" button",
            "\"Get Started\" button"
        ]
    },
    {
        "path": r"design_B_simple\notion.png",
        "targets": [
            "Trash icon",
            "\"New Page\" button"
        ]
    },
    {
        "path": r"design_B_simple\stripe.png",
        "targets": [
            "\"Start now\" button",
            "\"Contact Sales\" button"
        ]
    },
    {
        "path": r"design_B_simple\uber.png",
        "targets": [
            "\"About\" link",
            "\"Ride\" icon",
            "\"Log in\" button"
        ]
    }
]

print("=" * 80)
print("IMAGE TARGET VERIFICATION TOOL")
print("=" * 80)
print("\nInstructions:")
print("- Each image will be displayed with its targets listed")
print("- Press any key to move to the next image")
print("- Press 'q' to quit")
print("=" * 80)

for item in images_to_check:
    img = cv2.imread(item["path"])
    
    if img is None:
        print(f"\n[ERROR] Could not load: {item['path']}")
        continue
    
    print(f"\n\nCurrent Image: {item['path']}")
    print("Targets to verify:")
    for i, target in enumerate(item["targets"], 1):
        print(f"  {i}. {target}")
    
    # Resize image to fit screen if too large
    h, w = img.shape[:2]
    max_h, max_w = 900, 1600
    if h > max_h or w > max_w:
        scale = min(max_w/w, max_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        img = cv2.resize(img, (new_w, new_h))
    
    cv2.imshow("Target Verification", img)
    key = cv2.waitKey(0)
    
    if key == ord('q'):
        print("\nExiting verification...")
        break

cv2.destroyAllWindows()
print("\n" + "=" * 80)
print("Verification complete!")
print("=" * 80)
