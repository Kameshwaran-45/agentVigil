import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_OUT  = os.path.join(SCRIPT_DIR, "clip_prompts_seed.sql")
JSON_OUT = os.path.join(SCRIPT_DIR, "clip_prompts_seed.json")

# ══════════════════════════════════════════════════════════════════════
# ANOMALY PROMPTS — Massively expanded based on UCF-Crime & XD-Violence datasets
# ══════════════════════════════════════════════════════════════════════

ANOMALY_PROMPTS: list[tuple[str | None, str]] = [

    # ── Abuse ──────────────────────────────────────────────────────
    ("Abuse", "a person repeatedly kicking an animal"),
    ("Abuse", "a man striking a dog with a large stick"),
    ("Abuse", "a person dragging a helpless animal by a rope"),
    ("Abuse", "a person hitting a restrained dog"),
    ("Abuse", "an individual choking a small animal"),
    ("Abuse", "a person throwing an animal against a wall"),
    ("Abuse", "an aggressive person cornering and yelling at a cowering person"),
    ("Abuse", "a person punching another person who is on the ground"),
    ("Abuse", "an individual stomping on a defenseless person"),
    ("Abuse", "a man grabbing a woman by the hair"),
    ("Abuse", "a person slapping a child across the face"),
    ("Abuse", "an adult shaking a small child violently"),
    ("Abuse", "a person pushing an elderly person to the ground"),
    ("Abuse", "a figure looming over a scared person in a threatening way"),
    ("Abuse", "a person raising their hand to strike someone"),
    ("Abuse", "an animal yelping in pain while being hit"),
    ("Abuse", "a person flinching away from an aggressive individual"),
    ("Abuse", "an individual trapping a person in a corner"),
    ("Abuse", "a close-up of a person being physically mistreated"),
    ("Abuse", "a person being held down and beaten"),
    ("Abuse", "a person being tormented and pushed around"),

    # ── Arrest ─────────────────────────────────────────────────────
    ("Arrest", "a police officer pinning a suspect to the ground"),
    ("Arrest", "multiple police officers restraining a struggling person"),
    ("Arrest", "a person in handcuffs being escorted by law enforcement"),
    ("Arrest", "an officer placing a suspect into the back of a patrol car"),
    ("Arrest", "a suspect with their hands up, surrounded by police"),
    ("Arrest", "a police officer frisking a person against a vehicle"),
    ("Arrest", "a person being tackled to the ground by a cop"),
    ("Arrest", "an officer with a drawn weapon making an arrest"),
    ("Arrest", "a felony traffic stop with suspects exiting a car"),
    ("Arrest", "a SWAT team detaining multiple individuals"),
    ("Arrest", "a person on their knees with hands behind their head"),
    ("Arrest", "an undercover officer revealing a badge and making an arrest"),
    ("Arrest", "a police officer putting a person in a headlock"),
    ("Arrest", "a chaotic scene with police arresting someone in a crowd"),
    ("Arrest", "a suspect being read their rights while being cuffed"),
    ("Arrest", "a person resisting arrest and struggling with officers"),

    # ── Arson ──────────────────────────────────────────────────────
    ("Arson", "a person pouring gasoline on a car and lighting it"),
    ("Arson", "a hooded figure igniting a fire at the base of a building"),
    ("Arson", "someone throwing a Molotov cocktail at a storefront"),
    ("Arson", "a person lighting a rag in a bottle and throwing it"),
    ("Arson", "a fire spreading rapidly across the front of a house"),
    ("Arson", "a person using a lighter to set a pile of trash on fire"),
    ("Arson", "an individual walking away calmly from a growing blaze"),
    ("Arson", "a fuel can and matches next to a smoldering fire"),
    ("Arson", "a vehicle fully engulfed in intentional flames"),
    ("Arson", "a person dousing furniture with a flammable liquid"),
    ("Arson", "an arsonist watching a fire they started"),
    ("Arson", "a black-clad person setting a dumpster on fire in an alley"),
    ("Arson", "a building with flames visible in every window"),
    ("Arrest", "a fire accelerant being poured on a wooden porch"),
    ("Arrest", "a fire spreading up the side of a wooden structure"),
    ("Arrest", "an incendiary device being placed near a door"),

    # ── Assault ────────────────────────────────────────────────────
    ("Assault", "a person sucker-punching someone from the side"),
    ("Assault", "an individual repeatedly punching a defenseless victim"),
    ("Assault", "a man striking another person who does not retaliate"),
    ("Assault", "a person being knocked to the ground by a sudden blow"),
    ("Assault", "an unprovoked physical attack on a single person"),
    ("Assault", "a person kicking someone who is curled up on the ground"),
    ("Assault", "an aggressor hitting a victim with a blunt object"),
    ("Assault", "a person being ambushed and attacked by a group"),
    ("Assault", "a headbutt during a heated confrontation"),
    ("Assault", "a person being shoved violently into a wall"),
    ("Assault", "an attack where the victim is clearly overpowered"),
    ("Assault", "a person falling backward after being hit hard"),
    ("Assault", "an aggressive individual grabbing and striking someone"),
    ("Assault", "a person bleeding from the face after being punched"),
    ("Assault", "a one-sided beating in a parking lot"),
    ("Assault", "a person being hit with a bottle or glass"),

    # ── Burglary ───────────────────────────────────────────────────
    ("Burglary", "a person smashing a glass door to enter a shop"),
    ("Burglary", "someone using a crowbar to pry open a window"),
    ("Burglary", "a masked figure climbing through a kitchen window at night"),
    ("Burglary", "a hooded person picking the lock of a front door"),
    ("Burglary", "a burglar kicking a door open with force"),
    ("Burglary", "a person in dark clothing sneaking around inside a house"),
    ("Burglary", "a thief carrying a television out of a broken window"),
    ("Burglary", "a person using bolt cutters on a gate lock"),
    ("Burglary", "a figure with a flashlight searching through a dark office"),
    ("Burglary", "a person disabling a security camera with a stick"),
    ("Burglary", "a car ramming into the front of a retail store"),
    ("Burglary", "a person climbing a ladder to a second-story window"),
    ("Burglary", "a burglar ransacking drawers and cupboards"),
    ("Burglary", "a person stuffing valuables into a duffel bag"),
    ("Burglary", "an empty home with a door wide open and forced lock"),
    ("Burglary", "a person crawling through a smashed storefront"),

    # ── Explosion ──────────────────────────────────────────────────
    ("Explosion", "a massive fireball erupting from a vehicle"),
    ("Explosion", "a sudden, violent blast with a visible shockwave"),
    ("Explosion", "a building facade blowing outwards with smoke and debris"),
    ("Explosion", "a car exploding in a public street"),
    ("Explosion", "a pressure wave shattering windows of nearby buildings"),
    ("Explosion", "a plume of black smoke rising from an explosion site"),
    ("Explosion", "a secondary explosion at a fire scene"),
    ("Explosion", "a bomb detonating in a trash can"),
    ("Explosion", "an improvised explosive device (IED) going off"),
    ("Explosion", "a gas station exploding in a fiery blast"),
    ("Explosion", "a building collapsing after a large explosion"),
    ("Explosion", "a person being thrown by the force of a blast"),
    ("Explosion", "the aftermath of an explosion with rubble and fire"),

    # ── Fighting ───────────────────────────────────────────────────
    ("Fighting", "two men trading punches in a parking lot"),
    ("Fighting", "a large group of people brawling in the street"),
    ("Fighting", "two individuals wrestling aggressively on the floor"),
    ("Fighting", "a chaotic street fight with people being knocked down"),
    ("Fighting", "two people grappling and swinging wildly at each other"),
    ("Fighting", "a violent confrontation between rival groups"),
    ("Fighting", "a bar fight with chairs and bottles being thrown"),
    ("Fighting", "a person in a headlock during a fight"),
    ("Fighting", "a shoving match escalating into a full-blown fight"),
    ("Fighting", "a circle of people watching two individuals fight"),
    ("Fighting", "a person being pulled into a large melee"),
    ("Fighting", "multiple people kicking someone on the ground"),
    ("Fighting", "a disorganized brawl with no clear sides"),
    ("Fighting", "security guards trying to break up a large fight"),

    # ── RoadAccidents ──────────────────────────────────────────────
    ("RoadAccidents", "a car running a red light and t-boning another vehicle"),
    ("RoadAccidents", "a multi-car pile-up on a highway"),
    ("RoadAccidents", "a vehicle flipping over multiple times after a crash"),
    ("RoadAccidents", "a pedestrian being struck by a car in a crosswalk"),
    ("RoadAccidents", "a motorcycle crashing into the side of a truck"),
    ("RoadAccidents", "a car swerving to avoid something and hitting a pole"),
    ("RoadAccidents", "a head-on collision between two cars"),
    ("RoadAccidents", "a vehicle losing control and spinning out on a wet road"),
    ("RoadAccidents", "the scene of an accident with deployed airbags and smoke"),
    ("RoadAccidents", "a cyclist being hit by a turning car"),
    ("RoadAccidents", "a bus crashing into a storefront"),
    ("RoadAccidents", "a car driving the wrong way down a street, causing a crash"),
    ("RoadAccidents", "a vehicle rear-ended with significant damage"),
    ("RoadAccidents", "paramedics attending to victims at a crash site"),
    ("RoadAccidents", "a car lying on its roof after an accident"),

    # ── Robbery ────────────────────────────────────────────────────
    ("Robbery", "a person snatching a purse from a woman and running"),
    ("Robbery", "a masked person pointing a gun at a store clerk"),
    ("Robbery", "a mugger holding a knife to a victim in an alley"),
    ("Robbery", "a robber forcing a cashier to empty the cash register"),
    ("Robbery", "someone forcefully taking a phone from a person's hand"),
    ("Robbery", "a carjacking with the driver being pulled from the vehicle"),
    ("Robbery", "an ATM robbery with a crowbar"),
    ("Robbery", "a group of people surrounding and robbing a person"),
    ("Robbery", "a robber shoving a person to the ground and taking their wallet"),
    ("Robbery", "a holdup in a bank with employees' hands in the air"),
    ("Robbery", "a thief grabbing jewelry from a display case and fleeing"),
    ("Robbery", "a person being forced to hand over their backpack"),
    ("Robbery", "a getaway car speeding away from a robbery scene"),

    # ── Shooting ───────────────────────────────────────────────────
    ("Shooting", "a person firing a handgun into a crowd"),
    ("Shooting", "a clear muzzle flash from a pistol at night"),
    ("Shooting", "an individual with a rifle aiming at a target"),
    ("Shooting", "a shootout between two armed people"),
    ("Shooting", "a person suddenly collapsing after a loud bang"),
    ("Shooting", "shell casings being ejected from a semi-automatic weapon"),
    ("Shooting", "a drive-by shooting with shots fired from a car window"),
    ("Shooting", "an active shooter moving through a building"),
    ("Shooting", "people running and taking cover from gunfire"),
    ("Shooting", "a person holding a shotgun in a threatening manner"),
    ("Shooting", "a victim on the ground with a visible wound"),
    ("Shooting", "a person being shot at close range"),
    ("Shooting", "a person returning fire in a gun battle"),

    # ── Shoplifting ────────────────────────────────────────────────
    ("Shoplifting", "a person furtively hiding a bottle in their jacket"),
    ("Shoplifting", "someone putting small, expensive items into their pocket"),
    ("Shoplifting", "a person stuffing unpaid merchandise into a backpack"),
    ("Shoplifting", "a shoplifter briskly walking past the registers to the exit"),
    ("Shoplifting", "a person looking around nervously while concealing an item"),
    ("Shoplifting", "someone in a fitting room putting store clothes on under their own"),
    ("Shoplifting", "a person removing a security tag from a product"),
    ("Shoplifting", "a grab-and-dash theft from a retail store"),
    ("Shoplifting", "a person placing items from a shopping cart into a personal bag"),
    ("Shoplifting", "a group of people distracting a clerk while another steals"),
    ("Shoplifting", "a person swapping price tags on merchandise"),

    # ── Stealing ───────────────────────────────────────────────────
    ("Stealing", "a thief grabbing an unattended laptop from a cafe table"),
    ("Stealing", "a pickpocket stealthily removing a wallet from a back pocket"),
    ("Stealing", "a person snatching a cellphone and running off"),
    ("Stealing", "a porch pirate taking a package from a doorstep"),
    ("Stealing", "a person cutting a lock and stealing a bicycle"),
    ("Stealing", "a catalytic converter being sawed off from under a car"),
    ("Stealing", "a person siphoning gasoline from a vehicle's tank"),
    ("Stealing", "a thief taking a bag from an open car window"),
    ("Stealing", "a person reaching over a counter and grabbing cash"),
    ("Stealing", "a person taking mail from a community mailbox"),

    # ── Vandalism ──────────────────────────────────────────────────
    ("Vandalism", "a person spray painting a large tag on a clean wall"),
    ("Vandalism", "someone smashing a car's windshield with a baseball bat"),
    ("Vandalism", "a group of people kicking over trash cans"),
    ("Vandalism", "an individual shattering a bus stop glass panel"),
    ("Vandalism", "a person keying the side of a parked car"),
    ("Vandalism", "someone using a brick to break a shop window"),
    ("Vandalism", "a person slashing the tires of a vehicle"),
    ("Vandalism", "a public statue being defaced with paint"),
    ("Vandalism", "a person ripping signs off a building"),
    ("Vandalism", "a mailbox being destroyed with a sledgehammer"),
    ("Vandalism", "a person deliberately breaking a parking meter"),

    # ── General anomaly catch-alls ─────────────────────────────────
    (None, "a person wearing a ski mask in a public place"),
    (None, "a crowd of people suddenly running in panic"),
    (None, "a person brandishing a weapon threateningly"),
    (None, "a violent act being committed on camera"),
    (None, "a criminal activity in progress"),
    (None, "an emergency situation unfolding on the street"),
    (None, "a person clearly in distress or danger"),
    (None, "a suspicious individual lurking in the shadows"),
]

# ══════════════════════════════════════════════════════════════════════
# NORMAL PROMPTS — Massively expanded for diversity
# ══════════════════════════════════════════════════════════════════════

NORMAL_PROMPTS: list[tuple[None, str]] = [
    # Street / Outdoor / Public Spaces
    (None, "a group of people talking on a street corner"),
    (None, "a person walking their dog in a park"),
    (None, "a quiet residential street at midday"),
    (None, "pedestrians waiting to cross at a crosswalk"),
    (None, "a city square with people strolling and sitting"),
    (None, "a person jogging on a path in the morning"),
    (None, "a cyclist riding in a designated bike lane"),
    (None, "a family having a picnic on the grass"),
    (None, "a street performer playing music for a small crowd"),
    (None, "a person reading a book on a park bench"),
    (None, "a child riding a scooter on the sidewalk"),
    (None, "a quiet alleyway with no people"),
    (None, "a landscape view of a city park"),
    (None, "a person waiting for a bus at a bus stop"),
    (None, "an empty playground at dusk"),
    (None, "a public fountain with water flowing"),
    (None, "a person looking at their phone while walking"),
    (None, "a couple holding hands and walking"),
    (None, "a wide shot of a beach with people relaxing"),
    (None, "a person pushing a baby in a stroller"),

    # Vehicles / Traffic / Transport
    (None, "a normal flow of traffic on a multi-lane highway"),
    (None, "cars stopped at a red traffic light"),
    (None, "a vehicle signaling and making a normal turn"),
    (None, "a parking lot with cars neatly parked in rows"),
    (None, "a delivery truck unloading goods at a store"),
    (None, "a public bus driving along its route"),
    (None, "a person getting into their parked car"),
    (None, "a train arriving at a station platform"),
    (None, "an empty multi-story car park at night"),
    (None, "a person filling their car with gas at a gas station"),
    (None, "a taxi waiting for a passenger"),
    (None, "a view from a dashboard camera of a normal drive"),
    (None, "a sanitation truck collecting trash"),
    (None, "a mail carrier delivering mail to houses"),

    # Indoor / Commercial / Retail
    (None, "a customer paying for groceries at a checkout counter"),
    (None, "shoppers browsing clothes in a department store"),
    (None, "a person examining products on a supermarket shelf"),
    (None, "an office lobby with employees walking through"),
    (None, "a person working at a computer in an office cubicle"),
    (None, "a quiet library with people reading"),
    (None, "a restaurant with patrons eating their meals"),
    (None, "a waiter taking an order from a customer"),
    (None, "a person trying on shoes in a shoe store"),
    (None, "a security guard standing watch near an entrance"),
    (None, "a cleaning crew mopping the floor of a building"),
    (None, "an empty shopping mall after closing hours"),
    (None, "a person using an ATM inside a bank"),
    (None, "a barista making coffee at a cafe"),
    (None, "a hotel reception desk with no one around"),
    (None, "people exercising in a gym"),

    # Generic / Miscellaneous
    (None, "a surveillance camera view of an empty hallway"),
    (None, "a cat sleeping on a windowsill"),
    (None, "rain falling on a city street"),
    (None, "a flag waving in the wind on a flagpole"),
    (None, "a shot of trees swaying in the breeze"),
    (None, "a person simply standing and looking around"),
    (None, "a time-lapse of clouds moving across the sky"),
    (None, "a view from a high-rise building of the city below"),
    (None, "a person opening a door and entering a room"),
    (None, "a sprinkler system watering a lawn"),
]


# ══════════════════════════════════════════════════════════════════════
# BUILD / WRITE (No changes needed below this line)
# ══════════════════════════════════════════════════════════════════════

def build_records() -> list[dict]:
    records = []
    seen = set()

    for category, text in ANOMALY_PROMPTS:
        t = text.strip()
        key = ("anomaly", t)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "prompt_type": "anomaly",
            "category": category,
            "prompt_text": t,
            "enabled": True,
            "weight": 1.0,
        })

    for _, text in NORMAL_PROMPTS:
        t = text.strip()
        key = ("normal", t)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "prompt_type": "normal",
            "category": None,
            "prompt_text": t,
            "enabled": True,
            "weight": 1.0,
        })

    return records


def write_sql(records: list[dict], path: str) -> None:
    lines = [
        "-- AgentVigil CLIP prompt seed (Comprehensive Version)",
        "-- Generated by scripts/generate_clip_prompts.py",
        "-- Run via:  python scripts/upload_clip_prompts.py",
        "",
        "CREATE TABLE IF NOT EXISTS clip_prompts (",
        "    id              SERIAL PRIMARY KEY,",
        "    prompt_type     TEXT    NOT NULL,",
        "    category        TEXT,",
        "    prompt_text     TEXT    NOT NULL,",
        "    enabled         BOOLEAN NOT NULL DEFAULT TRUE,",
        "    weight          REAL    NOT NULL DEFAULT 1.0,",
        "    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,",
        "    UNIQUE (prompt_type, prompt_text)",
        ");",
        "",
        "CREATE INDEX IF NOT EXISTS idx_cp_type     ON clip_prompts(prompt_type);",
        "CREATE INDEX IF NOT EXISTS idx_cp_category ON clip_prompts(category);",
        "CREATE INDEX IF NOT EXISTS idx_cp_enabled  ON clip_prompts(enabled);",
        "",
        "INSERT INTO clip_prompts (prompt_type, category, prompt_text, enabled, weight) VALUES",
    ]

    value_lines = []
    for r in records:
        cat  = f"'{r['category']}'" if r["category"] else "NULL"
        text = r["prompt_text"].replace("'", "''")
        enab = "TRUE" if r["enabled"] else "FALSE"
        value_lines.append(
            f"  ('{r['prompt_type']}', {cat}, '{text}', {enab}, {r['weight']})"
        )

    lines.append(",\n".join(value_lines))
    lines += [
        "ON CONFLICT (prompt_type, prompt_text) DO UPDATE SET",
        "  category = EXCLUDED.category,",
        "  enabled  = EXCLUDED.enabled,",
        "  weight   = EXCLUDED.weight;",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_json(records: list[dict], path: str) -> None:
    grouped: dict[str, list] = {"anomaly": {}, "normal": []}
    for r in records:
        if r["prompt_type"] == "anomaly":
            cat = r["category"] or "_general"
            grouped["anomaly"].setdefault(cat, []).append(r["prompt_text"])
        else:
            grouped["normal"].append(r["prompt_text"])
    with open(path, "w") as f:
        json.dump(grouped, f, indent=2)


def print_summary(records: list[dict]) -> None:
    anomaly = [r for r in records if r["prompt_type"] == "anomaly"]
    normal  = [r for r in records if r["prompt_type"] == "normal"]
    cat_counts = Counter(r["category"] or "_general" for r in anomaly)

    print(f"\n{'─'*58}")
    print(f"  CLIP Prompt Generation Summary (Comprehensive)")
    print(f"{'─'*58}")
    print(f"  ANOMALY prompts:")
    for cat in sorted(cat_counts):
        print(f"    {cat:<28s} {cat_counts[cat]:>3} prompts")
    print(f"  {'─'*40}")
    print(f"    {'TOTAL anomaly':<28s} {len(anomaly):>3} prompts")
    print(f"  {'─'*40}")
    print(f"  NORMAL prompts:")
    print(f"    {'TOTAL normal':<28s} {len(normal):>3} prompts")
    print(f"{'─'*58}")
    print(f"  Grand total: {len(records)} prompts")
    print(f"  SQL output  → {os.path.basename(SQL_OUT)}")
    print(f"  JSON output → {os.path.basename(JSON_OUT)}")
    print(f"{'─'*58}")
    print("  Next step: python scripts/upload_clip_prompts.py\n")


if __name__ == "__main__":
    records = build_records()
    write_sql(records, SQL_OUT)
    write_json(records, JSON_OUT)
    print_summary(records)