"""
Configuration file for Moderate Design (Medium Cognitive Load) Targets
Each image has 2-3 carefully selected targets for visual search tasks.

These are moderately complex website designs from the design_C_moderate folder.
"""

MODERATE_DESIGN_TARGETS = [
    # 3B Software - Blue gradient software company
    {
        "folder": "design_C_moderate",
        "file": "3.png",
        "target": "Find the PRODUCTS menu"
    },
    {
        "folder": "design_C_moderate",
        "file": "3.png",
        "target": "Find the CUSTOMER CARE link"
    },
    {
        "folder": "design_C_moderate",
        "file": "3.png",
        "target": "Find the Digital Entertainment section"
    },

    # Short Stories - Purple header story site
    {
        "folder": "design_C_moderate",
        "file": "7.png",
        "target": "Find the fiction category"
    },
    {
        "folder": "design_C_moderate",
        "file": "7.png",
        "target": "Find the story of the day section"
    },
    {
        "folder": "design_C_moderate",
        "file": "7.png",
        "target": "Find the search box"
    },

    # Ace Project - Project management software
    {
        "folder": "design_C_moderate",
        "file": "8.png",
        "target": "Find the SIGN UP button"
    },
    {
        "folder": "design_C_moderate",
        "file": "8.png",
        "target": "Find the Features link"
    },
    {
        "folder": "design_C_moderate",
        "file": "8.png",
        "target": "Find the Time Tracking section"
    },

    # Acez.com - Free screensavers with sidebar
    {
        "folder": "design_C_moderate",
        "file": "9.png",
        "target": "Find the Free Screensavers menu"
    },
    {
        "folder": "design_C_moderate",
        "file": "9.png",
        "target": "Find the Download Free Screensaver link"
    },
    {
        "folder": "design_C_moderate",
        "file": "9.png",
        "target": "Find the 3D Screensavers category"
    },

    # Corante - Blog/weblog site
    {
        "folder": "design_C_moderate",
        "file": "10.png",
        "target": "Find the In the Pipeline link"
    },
    {
        "folder": "design_C_moderate",
        "file": "10.png",
        "target": "Find the CATEGORIES section"
    },
    {
        "folder": "design_C_moderate",
        "file": "10.png",
        "target": "Find the Comments link"
    },

    # Addresses.com - People lookup directory
    {
        "folder": "design_C_moderate",
        "file": "11.png",
        "target": "Find the People Lookup button"
    },
    {
        "folder": "design_C_moderate",
        "file": "11.png",
        "target": "Find the Business Lookup button"
    },
    {
        "folder": "design_C_moderate",
        "file": "11.png",
        "target": "Find the Reverse Phone Lookup section"
    },

    # Allok Software - Light blue multimedia converter site
    {
        "folder": "design_C_moderate",
        "file": "13.png",
        "target": "Find the Products menu"
    },
    {
        "folder": "design_C_moderate",
        "file": "13.png",
        "target": "Find the 3GP PSP MP4 iPod Tools section"
    },
    {
        "folder": "design_C_moderate",
        "file": "13.png",
        "target": "Find the Allok Video Converter link"
    },

    # AllPoetry - Poetry community with login
    {
        "folder": "design_C_moderate",
        "file": "14.png",
        "target": "Find the Poetry link"
    },
    {
        "folder": "design_C_moderate",
        "file": "14.png",
        "target": "Find the Log In button"
    },
    {
        "folder": "design_C_moderate",
        "file": "14.png",
        "target": "Find the Free contests link"
    },

    # allproducts.com - Product directory
    {
        "folder": "design_C_moderate",
        "file": "15.png",
        "target": "Find the Search Products button"
    },
    {
        "folder": "design_C_moderate",
        "file": "15.png",
        "target": "Find the Products Categories section"
    },
    {
        "folder": "design_C_moderate",
        "file": "15.png",
        "target": "Find the Hot Products section"
    },

    # aNobii - Red header book sharing site
    {
        "folder": "design_C_moderate",
        "file": "17.png",
        "target": "Find the Shelve icon"
    },
    {
        "folder": "design_C_moderate",
        "file": "17.png",
        "target": "Find the Search button"
    },
    {
        "folder": "design_C_moderate",
        "file": "17.png",
        "target": "Find the Recent Activity section"
    }
]

# Optional: Group by unique website for easier selection
def get_grouped_trials():
    """
    Returns trials grouped by filename for random selection.
    Use this to ensure each website is shown only once per session.
    """
    grouped = {}
    for trial in MODERATE_DESIGN_TARGETS:
        fname = trial['file']
        if fname not in grouped:
            grouped[fname] = []
        grouped[fname].append(trial)
    return grouped

# Optional: Get unique website count
def get_unique_website_count():
    """Returns the number of unique websites (images) in the configuration"""
    return len(set(trial['file'] for trial in MODERATE_DESIGN_TARGETS))

if __name__ == "__main__":
    print(f"Total targets configured: {len(MODERATE_DESIGN_TARGETS)}")
    print(f"Unique websites: {get_unique_website_count()}")
    print(f"Average targets per website: {len(MODERATE_DESIGN_TARGETS) / get_unique_website_count():.1f}")
    
    grouped = get_grouped_trials()
    print("\nWebsites and their target counts:")
    for fname, targets in sorted(grouped.items()):
        print(f"  {fname}: {len(targets)} targets")
