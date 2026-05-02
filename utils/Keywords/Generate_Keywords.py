import json, os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_OUT  = os.path.join(SCRIPT_DIR, "keywords_seed.sql")
JSON_OUT = os.path.join(SCRIPT_DIR, "keywords_seed.json")

# HIGH_PRIORITY flags categories where even a single keyword hit should
# strongly suggest escalating to a higher tier of analysis immediately.
HIGH_PRIORITY = {
    "Shooting", "Explosion", # Instantaneous emergencies
    "Arson", "Robbery", "Burglary", # High-threat process crimes
}

# NEW MODEL: Keywords can exist in multiple categories with different weights.
# The weight reflects the keyword's relevance to THAT SPECIFIC category.
# Format: (keyword_text, weight)  3=primary  2=action  1=context
RAW_KEYWORDS = {

    "Abuse": [
        # Primary Terms (Weight 3)
        ("abuse", 3), ("animal abuse", 3), ("child abuse", 3), ("domestic violence", 3), ("cruelty", 3),
        ("mistreatment", 3), ("maltreatment", 3), ("physical abuse", 3), ("battery", 3),
        # Actions (Weight 2)
        ("hit", 2), ("hitting", 2), ("strike", 2), ("striking", 2), ("punch", 2), ("punching", 2),
        ("kick", 2), ("kicking", 2), ("slap", 2), ("slapping", 2), ("beat", 2), ("beating", 2),
        ("shove", 2), ("shoving", 2), ("push", 2), ("pushing", 2), ("drag", 2), ("dragging", 2),
        ("throw", 2), ("throwing", 2), ("slam", 2), ("slamming", 2), ("choke", 2), ("choking", 2),
        ("stomp", 2), ("stomping", 2), ("yell at", 2), ("scream at", 2), ("berating", 2),
        ("cornering", 2), ("intimidating", 2), ("menacing", 2), ("raise hand", 2), ("lunge at", 2),
        # Context & Descriptors (Weight 1)
        ("animal", 1), ("pet", 1), ("dog", 1), ("cat", 1), ("child", 1), ("toddler", 1),
        ("woman", 1), ("man", 1), ("elderly", 1), ("vulnerable person", 1), ("aggressor", 1),
        ("cowering", 1), ("flinching", 1), ("crying", 1), ("shouting", 1), ("whimpering", 1),
        ("yelping", 1), ("helpless", 1), ("defenseless", 1), ("on the ground", 1), ("cornered", 1),
        ("struggling", 1), ("trying to escape", 1), ("does not fight back", 1), ("restrained", 1),
        ("leash", 1), ("cage", 1), ("bruise", 1), ("injury", 1), ("fearful", 1), ("in distress", 1),
        ("argument", 1), ("victim", 1), # Contextual
    ],

    "Arrest": [
        # Primary (3)
        ("arrest", 3), ("arrested", 3), ("detention", 3), ("apprehension", 3), ("taken into custody", 3),
        # Actions (2)
        ("handcuff", 2), ("handcuffs", 2), ("handcuffing", 2), ("restrain", 2), ("restraining", 2),
        ("detain", 2), ("pin to ground", 2), ("tackle", 2), ("subdue", 2), ("pat down", 2),
        ("frisking", 2), ("escort", 2), ("placed in car", 2), ("hands behind back", 2),
        ("surrender", 2), ("hands up", 2), ("on the ground", 2), ("resisting arrest", 2),
        # Context (1)
        ("police", 1), ("officer", 1), ("cop", 1), ("sheriff", 1), ("law enforcement", 1),
        ("uniform", 1), ("badge", 1), ("patrol car", 1), ("squad car", 1), ("sirens", 1),
        ("flashing lights", 1), ("suspect", 1), ("perp", 1), ("custody", 1), ("weapon drawn", 1),
        ("struggling with officer", 1), ("perpetrator", 1),
    ],

    "Arson": [
        # Primary (3)
        ("arson", 3), ("arsonist", 3), ("deliberately set fire", 3), ("incendiary", 3), ("firebombing", 3),
        # Actions (2)
        ("set fire", 2), ("setting fire", 2), ("ignite", 2), ("igniting", 2), ("torch", 2), ("torching", 2),
        ("light fire", 2), ("start fire", 2), ("pour accelerant", 2), ("dousing", 2),
        ("throwing molotov", 2), ("spread fire", 2), ("walks away from fire", 2),
        # Context (1)
        ("fire", 1), ("flames", 1), ("blaze", 1), ("burning", 1), ("smoke", 1), ("billowing smoke", 1),
        ("gasoline", 1), ("petrol", 1), ("lighter", 1), ("match", 1), ("accelerant", 1),
        ("fuel can", 1), ("gas can", 1), ("molotov cocktail", 1), ("building on fire", 1),
        ("car on fire", 1), ("engulfed in flames", 1), ("charred", 1), ("scorched", 1),
        ("soot", 1), ("firefighter", 1), ("fire truck", 1), ("fire alarm", 1),
    ],

    "Burglary": [
        # Primary (3)
        ("burglary", 3), ("burglar", 3), ("break-in", 3), ("breaking and entering", 3), ("forced entry", 3),
        ("home invasion", 3),
        # Actions (2)
        ("break in", 2), ("force door", 2), ("kick down door", 2), ("pry open", 2), ("jimmy lock", 2),
        ("picking lock", 2), ("break window", 2), ("smash window", 2), ("climb through window", 2),
        ("sneak in", 2), ("trespassing", 2), ("ransacking", 2), ("casing the joint", 2),
        ("disabling alarm", 2), ("covering camera", 2),
        # Context (1)
        ("crowbar", 1), ("lockpick", 1), ("hammer", 1), ("bolt cutters", 1), ("ladder", 1),
        ("masked", 1), ("hooded", 1), ("gloves", 1), ("backpack", 1), ("flashlight", 1),
        ("dark", 1), ("night", 1), ("after hours", 1), ("house", 1), ("residence", 1),
        ("store", 1), ("shop", 1), ("office", 1), ("warehouse", 1), ("property", 1),
        ("broken glass", 1), ("shattered window", 1), ("door ajar", 1), ("forced open", 1),
        ("alarm sounding", 1), ("security camera", 1),
    ],

    "Explosion": [
        # Primary (3)
        ("explosion", 3), ("detonation", 3), ("bombing", 3), ("bomb blast", 3),
        # Actions (2)
        ("explode", 2), ("explodes", 2), ("detonate", 2), ("detonates", 2), ("erupts", 2), ("goes off", 2),
        # Context (1)
        ("bomb", 1), ("ied", 1), ("car bomb", 1), ("pipe bomb", 1), ("explosives", 1),
        ("shockwave", 1), ("blast", 1), ("pressure wave", 1), ("fireball", 1), ("debris flying", 1),
        ("shrapnel", 1), ("building collapses", 1), ("vehicle explodes", 1), ("mushroom cloud", 1),
        ("plume of smoke", 1), ("loud bang", 1), ("wreckage", 1), ("rubble", 1), ("crater", 1),
        ("aftermath", 1), ("shattered windows", 1), ("charred", 1), ("scorched", 1), ("bomb squad", 1),
    ],

    "Fighting": [
        # Primary (3)
        ("fighting", 3), ("brawl", 3), ("mutual combat", 3), ("physical altercation", 3), ("melee", 3),
        ("scuffle", 3), ("riot", 3),
        # Actions (2)
        ("exchange blows", 2), ("trading punches", 2), ("swinging at each other", 2),
        ("punching each other", 2), ("kicking each other", 2), ("wrestling", 2), ("grappling", 2),
        ("shoving match", 2), ("charging at each other", 2),
        # Context (1)
        ("assault", 1), ("attack", 1), ("two people", 1), ("group fight", 1), ("mob", 1),
        ("crowd", 1), ("aggressive", 1), ("hostile", 1), ("confrontation", 1), ("pushing", 1),
        ("shoving", 1), ("yelling", 1), ("shouting", 1), ("argument", 1), ("bar fight", 1),
        ("street fight", 1), ("onlookers", 1), ("security intervening", 1), ("knocked over", 1),
    ],

    "RoadAccidents": [
        # Primary (3)
        ("road accident", 3), ("car accident", 3), ("vehicle collision", 3), ("car crash", 3),
        ("pile-up", 3), ("hit and run", 3),
        # Actions (2)
        ("collide", 2), ("crash", 2), ("hit by car", 2), ("struck by vehicle", 2), ("rear-end", 2),
        ("t-bone", 2), ("sideswipe", 2), ("run over", 2), ("swerve", 2), ("skid", 2), ("loses control", 2),
        ("rollover", 2), ("overturns", 2), ("runs red light", 2),
        # Context (1)
        ("car", 1), ("truck", 1), ("bus", 1), ("motorcycle", 1), ("bicycle", 1), ("vehicle", 1),
        ("pedestrian", 1), ("intersection", 1), ("highway", 1), ("road", 1), ("street", 1),
        ("crosswalk", 1), ("traffic", 1), ("wreck", 1), ("wreckage", 1), ("debris on road", 1),
        ("dented", 1), ("smashed", 1), ("overturned", 1), ("airbags deployed", 1), ("skid marks", 1),
        ("shattered glass", 1), ("ambulance", 1), ("paramedics", 1), ("emergency services", 1),
        ("fire truck", 1), ("tow truck", 1), ("knocked over", 1),
    ],

    "Robbery": [
        # Primary (3)
        ("robbery", 3), ("armed robbery", 3), ("mugging", 3), ("holdup", 3), ("carjacking", 3),
        # Actions (2)
        ("rob", 2), ("robs", 2), ("mug", 2), ("mugs", 2), ("threaten", 2), ("demands money", 2),
        ("hand over valuables", 2), ("at gunpoint", 2), ("at knifepoint", 2), ("taking by force", 2),
        ("point weapon at", 2), ("brandishing weapon", 2), ("grab and run", 2), ("snatch", 2),
        ("purse snatching", 2), ("empty the register", 2),
        # Context (1)
        ("gun", 1), ("knife", 1), ("weapon", 1), ("masked", 1), ("hooded", 1), ("disguised", 1),
        ("assailant", 1), ("robber", 1), ("thief", 1), ("accomplice", 1), ("getaway car", 1),
        ("victim", 1), ("victim present", 1), ("cash register", 1), ("wallet", 1), ("purse", 1),
        ("phone", 1), ("valuables", 1), ("store", 1), ("shop", 1), ("bank", 1), ("atm", 1),
        ("clerk", 1), ("pedestrian", 1), ("flee", 1), ("runs away", 1), ("hands in the air", 1),
    ],

    "Shooting": [
        # Primary (3)
        ("shooting", 3), ("gunfire", 3), ("shots fired", 3), ("gun shot", 3), ("active shooter", 3),
        ("shootout", 3),
        # Actions (2)
        ("shoots", 2), ("fires gun", 2), ("open fire", 2), ("discharges weapon", 2), ("aims gun", 2),
        ("pulls trigger", 2),
        # Context (1)
        ("gun", 1), ("handgun", 1), ("pistol", 1), ("rifle", 1), ("shotgun", 1), ("firearm", 1),
        ("muzzle flash", 1), ("shell casings", 1), ("bullet", 1), ("sound of gunfire", 1),
        ("recoil", 1), ("victim falls", 1), ("person shot", 1), ("person collapses", 1),
        ("wounded", 1), ("shooter", 1), ("gunman", 1), ("armed person", 1), ("taking cover", 1),
        ("ducking", 1), ("crowd scattering", 1), ("people running in panic", 1), ("street", 1),
        ("vehicle", 1), ("public place", 1),
    ],

    "Shoplifting": [
        # Primary (3)
        ("shoplifting", 3), ("shoplifter", 3), ("retail theft", 3), ("concealment of goods", 3),
        # Actions (2)
        ("conceal item", 2), ("hides merchandise", 2), ("pocket item", 2), ("tucks into bag", 2),
        ("slips into coat", 2), ("stuffs into clothing", 2), ("leaves without paying", 2),
        ("exits without purchase", 2), ("walks past checkout", 2), ("bypasses cashier", 2),
        ("removes security tag", 2), ("grab and dash", 2),
        # Context (1)
        ("store", 1), ("shop", 1), ("retail", 1), ("supermarket", 1), ("aisle", 1), ("shelf", 1),
        ("fitting room", 1), ("checkout", 1), ("merchandise", 1), ("product", 1), ("item", 1),
        ("price tag", 1), ("shopping bag", 1), ("personal bag", 1), ("backpack", 1),
        ("shopping cart", 1), ("acting suspicious", 1), ("glancing at cameras", 1),
        ("loss prevention", 1), ("store security", 1), ("security tag", 1), ("security camera", 1),
    ],

    "Stealing": [
        # Primary (3)
        ("stealing", 3), ("theft", 3), ("larceny", 3), ("pickpocket", 3), ("package theft", 3),
        ("porch pirate", 3), ("bicycle theft", 3),
        # Actions (2)
        ("steal", 2), ("steals", 2), ("take item", 2), ("snatch", 2), ("grabs item", 2),
        ("swipe", 2), ("lifts item", 2), ("takes without permission", 2), ("walks off with", 2),
        ("pick pocketing", 2), ("reaches into bag", 2), ("cuts bike lock", 2), ("grabbing package", 2),
        # Context (1)
        ("unattended", 1), ("distracted victim", 1), ("victim unaware", 1), ("no confrontation", 1),
        ("stealthy", 1), ("thief", 1), ("phone", 1), ("laptop", 1), ("wallet", 1), ("purse", 1),
        ("handbag", 1), ("bag", 1), ("backpack", 1), ("bicycle", 1), ("bike", 1), ("package", 1),
        ("mail", 1), ("porch", 1), ("public place", 1), ("crowd", 1), ("bench", 1), ("table", 1),
        ("unlocked car", 1), ("bus stop", 1), ("cafe", 1), ("park", 1),
    ],

    "Vandalism": [
        # Primary (3)
        ("vandalism", 3), ("vandalize", 3), ("property damage", 3), ("destruction of property", 3),
        ("defacement", 3), ("graffiti", 3),
        # Actions (2)
        ("smash", 2), ("smashing", 2), ("break", 2), ("breaking", 2), ("shatter", 2), ("destroy", 2),
        ("damage", 2), ("deface", 2), ("spray paint", 2), ("tagging", 2), ("scratch", 2),
        ("keying car", 2), ("kicks over", 2), ("slashing tires", 2), ("throwing rock", 2),
        # Context (1)
        ("vandal", 1), ("delinquent", 1), ("perpetrator", 1), ("spray paint can", 1),
        ("paint", 1), ("marker", 1), ("rock", 1), ("brick", 1), ("baseball bat", 1), ("crowbar", 1),
        ("knife", 1), ("car window", 1), ("windshield", 1), ("storefront", 1), ("wall", 1),
        ("sign", 1), ("bench", 1), ("mailbox", 1), ("public property", 1), ("broken", 1),
        ("shattered", 1), ("cracked", 1), ("dented", 1), ("tagged", 1), ("vehicle", 1), ("burning", 1),
    ],

    "Normal_Videos_event": [
        # Primary (3)
        ("normal activity", 3), ("no anomaly", 3), ("routine", 3), ("uneventful", 3), ("benign", 3),
        # Actions (2)
        ("walking", 2), ("strolling", 2), ("driving", 2), ("standing", 2), ("waiting", 2),
        ("sitting", 2), ("talking", 2), ("chatting", 2), ("shopping", 2), ("browsing", 2),
        ("eating", 2), ("jogging", 2), ("playing", 2), ("working", 2), ("commuting", 2),
        # Context (1)
        ("pedestrians", 1), ("people", 1), ("person", 1), ("crowd", 1), ("customers", 1),
        ("traffic", 1), ("vehicles", 1), ("daytime", 1), ("peaceful", 1), ("calm", 1),
        ("ordinary", 1), ("everyday", 1), ("no incident", 1), ("no crime", 1), ("public space", 1),
        ("park", 1), ("sidewalk", 1), ("store", 1), ("shop", 1), ("office", 1), ("cafe", 1),
        ("bus stop", 1),
    ],
}


def build_records():
    """Builds a flat list of records from the nested dictionary. Duplicates are now a feature."""
    records = []
    for category, entries in RAW_KEYWORDS.items():
        is_hp = category in HIGH_PRIORITY
        for keyword, weight in entries:
            kw = keyword.strip().lower()
            records.append({
                "category": category,
                "keyword": kw,
                "weight": weight,
                "is_high_priority": is_hp,
            })
    return records


def write_sql(records, path):
    """
    Generates a robust SQL seed file.
    REMOVED the UNIQUE constraint on 'keyword' to allow multi-category assignment.
    UPDATED ON CONFLICT to handle the composite key.
    """
    lines = [
        "-- AgentVigil keyword seed (Weighted, Multi-Category Version)",
        "-- Generated by scripts/generate_keywords.py",
        "-- Run via:  python scripts/upload_keywords.py",
        "",
        "CREATE TABLE IF NOT EXISTS keywords (",
        "    id              SERIAL PRIMARY KEY,",
        "    category        TEXT NOT NULL,",
        "    keyword         TEXT NOT NULL,",
        "    weight          INTEGER NOT NULL DEFAULT 1,",
        "    is_high_priority BOOLEAN NOT NULL DEFAULT FALSE,",
        "    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,",
        "    UNIQUE (category, keyword)",
        ");",
        "",
        "CREATE INDEX IF NOT EXISTS idx_kw_category ON keywords(category);",
        "CREATE INDEX IF NOT EXISTS idx_kw_keyword  ON keywords(keyword);",
        "",
        "INSERT INTO keywords (category, keyword, weight, is_high_priority) VALUES",
    ]
    value_lines = []
    for r in records:
        kw_esc = r["keyword"].replace("'", "''")
        hp = "TRUE" if r["is_high_priority"] else "FALSE"
        value_lines.append(
    f"  ('{r['category']}', '{kw_esc}', {r['weight']}, {hp})"
)
    lines.append(",\n".join(value_lines))
    lines += [
        "ON CONFLICT (category, keyword) DO UPDATE SET",
        "  weight           = EXCLUDED.weight,",
        "  is_high_priority = EXCLUDED.is_high_priority;",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_json(records, path):
    """Generates a JSON representation of the keyword data."""
    grouped = {}
    for r in records:
        grouped.setdefault(r["category"], []).append(
            {"keyword": r["keyword"], "weight": r["weight"]}
        )
    output_data = {
        "high_priority_categories": list(HIGH_PRIORITY),
        "keywords_by_category": grouped,
    }
    with open(path, "w") as f:
        json.dump(output_data, f, indent=2)


def print_summary(records):
    """Prints a summary of the generated keyword data."""
    cat_counts = Counter(r["category"] for r in records)
    unique_keywords = {r["keyword"] for r in records}
    hp_cats = {r["category"] for r in records if r["is_high_priority"]}

    print(f"\n{'─'*65}")
    print(f"  Keyword Generation Summary (Multi-Category Model)")
    print(f"{'─'*65}")
    for cat in sorted(cat_counts):
        count = cat_counts[cat]
        hp_marker = "⚡ HIGH_PRIORITY" if cat in hp_cats else ""
        print(f"  {cat:<25} {count:>4} keyword entries {hp_marker}")
    print(f"{'─'*65}")
    total_records = len(records)
    total_unique_keywords = len(unique_keywords)
    total_categories = len(cat_counts)
    print(f"  Total Records: {total_records} (across {total_categories} categories)")
    print(f"  Unique Keywords: {total_unique_keywords}")
    print(f"  SQL output → {os.path.basename(SQL_OUT)}")
    print(f"  JSON output → {os.path.basename(JSON_OUT)}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    records = build_records()
    write_sql(records, SQL_OUT)
    write_json(records, JSON_OUT)
    print_summary(records)