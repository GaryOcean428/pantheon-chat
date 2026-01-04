"""
QIG Coordizer - Backward-Compatible Wrapper for FisherCoordizer

Drop-in replacement for qig_tokenizer.py. Provides same API surface
while using pure geometric coordization under the hood.

MIGRATION PATH:
- Old code: from qig_tokenizer import get_tokenizer
- New code: from qig_coordizer import get_coordizer as get_tokenizer

This module will be deleted after full migration to coordizers.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import coordizers
from coordizers.base import FisherCoordizer
from coordizers.vocab_builder import GeometricVocabBuilder

# Try to import PostgresCoordizer for database-backed vocabulary
POSTGRES_COORDIZER_AVAILABLE = False
PostgresCoordizer = None
create_coordizer_from_pg = None

try:
    from coordizers.pg_loader import PostgresCoordizer, create_coordizer_from_pg
    POSTGRES_COORDIZER_AVAILABLE = True
except ImportError as e:
    print(f"[QIGCoordizer] PostgresCoordizer import failed: {e}")

# Import fallback vocabulary (2048 BIP39 words + common words)
try:
    from coordizers.fallback_vocabulary import BIP39_WORDS, COMMON_WORDS, compute_basin_embedding
    FALLBACK_VOCABULARY_AVAILABLE = True
except ImportError:
    FALLBACK_VOCABULARY_AVAILABLE = False
    BIP39_WORDS = []
    COMMON_WORDS = []

# Singleton instance
_coordizer_instance: Optional[FisherCoordizer] = None

# Legacy persistence path (for migration)
LEGACY_TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "data", "qig_tokenizer_state.json"
)

# New coordizer persistence path
COORDIZER_PERSIST_PATH = os.path.join(
    os.path.dirname(__file__), "data", "qig_coordizer_state"
)


class QIGCoordizer(FisherCoordizer):
    """
    QIGTokenizer-compatible coordizer with geometric operations.
    
    Extends FisherCoordizer with QIGTokenizer's API for backward compatibility:
    - Three-tier vocabulary (mnemonic/passphrase/conversation)
    - BIP39 word support
    - Vocabulary observations integration
    - Mode switching (mnemonic/passphrase/conversation)
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        phi_threshold: float = 0.7,
        special_tokens: Optional[List[str]] = None,
    ):
        """Initialize with QIGTokenizer-compatible parameters."""
        super().__init__(
            vocab_size=vocab_size,
            coordinate_dim=64,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )
        
        self.phi_threshold = phi_threshold
        self.mode = "conversation"
        
        # Token metadata (QIGTokenizer compatibility)
        self.token_weights: Dict[str, float] = {}
        self.merge_rules: List[Tuple[str, str]] = []
        self.merge_scores: Dict[Tuple[str, str], float] = {}
        
        # Three-tier vocabulary tracking
        self.mnemonic_vocab_ids: set = set()
        self.passphrase_vocab_ids: set = set()
        self.conversation_vocab_ids: set = set()
        self._conversation_words: set = set()
        
        # Word tokens list for decode compatibility with QIGGenerativeService
        self.word_tokens: list = []
        
        # Word tokens list for decode() compatibility with PostgresCoordizer
        self.word_tokens: List[str] = []
        self.subword_tokens: List[str] = []
        
        # Initialize vocabularies
        self._load_bip39_base()
        self.mnemonic_vocab_ids = set(self.vocab.values())
        
        self._load_passphrase_base()
        self.passphrase_vocab_ids = set(self.vocab.values())
        
        self._load_conversation_base()
        self.conversation_vocab_ids = set(self.vocab.values()) - self.mnemonic_vocab_ids
        
        # Build word_tokens list from vocabulary
        self._build_word_tokens_list()
        
        # Update UNK to centroid
        self._update_unk_to_vocabulary_centroid()
        
        # Initialize vocab builder for geometric discovery
        self.vocab_builder = GeometricVocabBuilder(
            phi_threshold=phi_threshold,
            min_cluster_size=2,
        )
    
    def _load_bip39_base(self):
        """Load BIP39 wordlist as base vocabulary."""
        bip39_path = os.path.join(os.path.dirname(__file__), "bip39_wordlist.txt")
        
        # Try to load from coordizers fallback vocabulary first (guaranteed 2048 words)
        try:
            from coordizers.fallback_vocabulary import BIP39_WORDS
            words = BIP39_WORDS
        except ImportError:
            words = None
        
        # If not available, try file
        if not words and os.path.exists(bip39_path):
            with open(bip39_path, 'r') as f:
                words = [line.strip() for line in f if line.strip()]
        
        # Final fallback - complete 2048 BIP39 words inline
        if not words or len(words) < 100:
            words = self._get_full_bip39_wordlist()
        
        start_id = len(self.special_tokens)
        for i, word in enumerate(words):
            if word not in self.vocab:
                token_id = start_id + i
                self.vocab[word] = token_id
                self.id_to_token[token_id] = word
                self.token_weights[word] = 1.0
                self.token_phi[word] = 0.0
                self.token_frequency[word] = 0
                self.basin_coords[word] = self._initialize_token_coordinate(word, token_id)
    
    def _load_passphrase_base(self):
        """Load passphrase vocabulary."""
        fallback_words = [
            "one", "two", "three", "four", "five",
            "red", "blue", "green", "yellow", "orange",
            "dog", "cat", "bird", "fish", "horse",
            "big", "small", "fast", "slow", "happy",
        ]
        
        vocab_path = os.path.join(os.path.dirname(__file__), "data", "passphrase_vocab.txt")
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r") as f:
                    words = [line.strip() for line in f if line.strip()]
            except Exception:
                words = fallback_words
        else:
            words = fallback_words
        
        start_id = len(self.vocab)
        added = 0
        for word in words:
            if word in self.vocab:
                continue
            token_id = start_id + added
            self.vocab[word] = token_id
            self.id_to_token[token_id] = word
            self.token_weights[word] = 1.1
            self.token_phi[word] = 0.5
            self.token_frequency[word] = 0
            self.basin_coords[word] = self._initialize_token_coordinate(word, token_id)
            added += 1
    
    def _load_conversation_base(self):
        """Load conversation vocabulary.
        
        Uses COMMON_WORDS from fallback_vocabulary.py for better coverage.
        """
        if FALLBACK_VOCABULARY_AVAILABLE and COMMON_WORDS:
            fallback_words = COMMON_WORDS
        else:
            fallback_words = [
                "i", "you", "we", "they", "it", "the", "and", "or", "but",
                "is", "are", "was", "were", "have", "has", "had",
                "question", "answer", "consciousness", "geometry",
            ]
        
        vocab_path = os.path.join(os.path.dirname(__file__), "data", "conversation_vocab.txt")
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r") as f:
                    words = [line.strip() for line in f if line.strip()]
            except Exception:
                words = fallback_words
        else:
            words = fallback_words
        
        start_id = len(self.vocab)
        added = 0
        for word in words:
            self._conversation_words.add(word)
            
            if word in self.vocab:
                self.token_weights[word] = max(self.token_weights.get(word, 1.0), 1.3)
                continue
            
            token_id = start_id + added
            self.vocab[word] = token_id
            self.id_to_token[token_id] = word
            self.token_weights[word] = 1.5
            self.token_phi[word] = 0.65
            self.token_frequency[word] = 0
            self.basin_coords[word] = self._initialize_token_coordinate(word, token_id)
            added += 1
    
    def _get_full_bip39_wordlist(self) -> List[str]:
        """Return the complete 2048 BIP39 wordlist as inline fallback."""
        return [
            "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
            "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
            "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
            "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
            "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
            "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album",
            "alcohol", "alert", "alien", "all", "alley", "allow", "almost", "alone",
            "alpha", "already", "also", "alter", "always", "amateur", "amazing", "among",
            "amount", "amused", "analyst", "anchor", "ancient", "anger", "angle", "angry",
            "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
            "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april",
            "arch", "arctic", "area", "arena", "argue", "arm", "armed", "armor",
            "army", "around", "arrange", "arrest", "arrive", "arrow", "art", "artefact",
            "artist", "artwork", "ask", "aspect", "assault", "asset", "assist", "assume",
            "asthma", "athlete", "atom", "attack", "attend", "attitude", "attract", "auction",
            "audit", "august", "aunt", "author", "auto", "autumn", "average", "avocado",
            "avoid", "awake", "aware", "away", "awesome", "awful", "awkward", "axis",
            "baby", "bachelor", "bacon", "badge", "bag", "balance", "balcony", "ball",
            "bamboo", "banana", "banner", "bar", "barely", "bargain", "barrel", "base",
            "basic", "basket", "battle", "beach", "bean", "beauty", "because", "become",
            "beef", "before", "begin", "behave", "behind", "believe", "below", "belt",
            "bench", "benefit", "best", "betray", "better", "between", "beyond", "bicycle",
            "bid", "bike", "bind", "biology", "bird", "birth", "bitter", "black",
            "blade", "blame", "blanket", "blast", "bleak", "bless", "blind", "blood",
            "blossom", "blouse", "blue", "blur", "blush", "board", "boat", "body",
            "boil", "bomb", "bone", "bonus", "book", "boost", "border", "boring",
            "borrow", "boss", "bottom", "bounce", "box", "boy", "bracket", "brain",
            "brand", "brass", "brave", "bread", "breeze", "brick", "bridge", "brief",
            "bright", "bring", "brisk", "broccoli", "broken", "bronze", "broom", "brother",
            "brown", "brush", "bubble", "buddy", "budget", "buffalo", "build", "bulb",
            "bulk", "bullet", "bundle", "bunker", "burden", "burger", "burst", "bus",
            "business", "busy", "butter", "buyer", "buzz", "cabbage", "cabin", "cable",
            "cactus", "cage", "cake", "call", "calm", "camera", "camp", "can",
            "canal", "cancel", "candy", "cannon", "canoe", "canvas", "canyon", "capable",
            "capital", "captain", "car", "carbon", "card", "cargo", "carpet", "carry",
            "cart", "case", "cash", "casino", "castle", "casual", "cat", "catalog",
            "catch", "category", "cattle", "caught", "cause", "caution", "cave", "ceiling",
            "celery", "cement", "census", "century", "cereal", "certain", "chair", "chalk",
            "champion", "change", "chaos", "chapter", "charge", "chase", "chat", "cheap",
            "check", "cheese", "chef", "cherry", "chest", "chicken", "chief", "child",
            "chimney", "choice", "choose", "chronic", "chuckle", "chunk", "churn", "cigar",
            "cinnamon", "circle", "citizen", "city", "civil", "claim", "clap", "clarify",
            "claw", "clay", "clean", "clerk", "clever", "click", "client", "cliff",
            "climb", "clinic", "clip", "clock", "clog", "close", "cloth", "cloud",
            "clown", "club", "clump", "cluster", "clutch", "coach", "coast", "coconut",
            "code", "coffee", "coil", "coin", "collect", "color", "column", "combine",
            "come", "comfort", "comic", "common", "company", "concert", "conduct", "confirm",
            "congress", "connect", "consider", "control", "convince", "cook", "cool", "copper",
            "copy", "coral", "core", "corn", "correct", "cost", "cotton", "couch",
            "country", "couple", "course", "cousin", "cover", "coyote", "crack", "cradle",
            "craft", "cram", "crane", "crash", "crater", "crawl", "crazy", "cream",
            "credit", "creek", "crew", "cricket", "crime", "crisp", "critic", "crop",
            "cross", "crouch", "crowd", "crucial", "cruel", "cruise", "crumble", "crunch",
            "crush", "cry", "crystal", "cube", "culture", "cup", "cupboard", "curious",
            "current", "curtain", "curve", "cushion", "custom", "cute", "cycle", "dad",
            "damage", "damp", "dance", "danger", "daring", "dash", "daughter", "dawn",
            "day", "deal", "debate", "debris", "decade", "december", "decide", "decline",
            "decorate", "decrease", "deer", "defense", "define", "defy", "degree", "delay",
            "deliver", "demand", "demise", "denial", "dentist", "deny", "depart", "depend",
            "deposit", "depth", "deputy", "derive", "describe", "desert", "design", "desk",
            "despair", "destroy", "detail", "detect", "develop", "device", "devote", "diagram",
            "dial", "diamond", "diary", "dice", "diesel", "diet", "differ", "digital",
            "dignity", "dilemma", "dinner", "dinosaur", "direct", "dirt", "disagree", "discover",
            "disease", "dish", "dismiss", "disorder", "display", "distance", "divert", "divide",
            "divorce", "dizzy", "doctor", "document", "dog", "doll", "dolphin", "domain",
            "donate", "donkey", "donor", "door", "dose", "double", "dove", "draft",
            "dragon", "drama", "drastic", "draw", "dream", "dress", "drift", "drill",
            "drink", "drip", "drive", "drop", "drum", "dry", "duck", "dumb",
            "dune", "during", "dust", "dutch", "duty", "dwarf", "dynamic", "eager",
            "eagle", "early", "earn", "earth", "easily", "east", "easy", "echo",
            "ecology", "economy", "edge", "edit", "educate", "effort", "egg", "eight",
            "either", "elbow", "elder", "electric", "elegant", "element", "elephant", "elevator",
            "elite", "else", "embark", "embody", "embrace", "emerge", "emotion", "employ",
            "empower", "empty", "enable", "enact", "end", "endless", "endorse", "enemy",
            "energy", "enforce", "engage", "engine", "enhance", "enjoy", "enlist", "enough",
            "enrich", "enroll", "ensure", "enter", "entire", "entry", "envelope", "episode",
            "equal", "equip", "era", "erase", "erode", "erosion", "error", "erupt",
            "escape", "essay", "essence", "estate", "eternal", "ethics", "evidence", "evil",
            "evoke", "evolve", "exact", "example", "excess", "exchange", "excite", "exclude",
            "excuse", "execute", "exercise", "exhaust", "exhibit", "exile", "exist", "exit",
            "exotic", "expand", "expect", "expire", "explain", "expose", "express", "extend",
            "extra", "eye", "eyebrow", "fabric", "face", "faculty", "fade", "faint",
            "faith", "fall", "false", "fame", "family", "famous", "fan", "fancy",
            "fantasy", "farm", "fashion", "fat", "fatal", "father", "fatigue", "fault",
            "favorite", "feature", "february", "federal", "fee", "feed", "feel", "female",
            "fence", "festival", "fetch", "fever", "few", "fiber", "fiction", "field",
            "figure", "file", "film", "filter", "final", "find", "fine", "finger",
            "finish", "fire", "firm", "first", "fiscal", "fish", "fit", "fitness",
            "fix", "flag", "flame", "flash", "flat", "flavor", "flee", "flight",
            "flip", "float", "flock", "floor", "flower", "fluid", "flush", "fly",
            "foam", "focus", "fog", "foil", "fold", "follow", "food", "foot",
            "force", "forest", "forget", "fork", "fortune", "forum", "forward", "fossil",
            "foster", "found", "fox", "fragile", "frame", "frequent", "fresh", "friend",
            "fringe", "frog", "front", "frost", "frown", "frozen", "fruit", "fuel",
            "fun", "funny", "furnace", "fury", "future", "gadget", "gain", "galaxy",
            "gallery", "game", "gap", "garage", "garbage", "garden", "garlic", "garment",
            "gas", "gasp", "gate", "gather", "gauge", "gaze", "general", "genius",
            "genre", "gentle", "genuine", "gesture", "ghost", "giant", "gift", "giggle",
            "ginger", "giraffe", "girl", "give", "glad", "glance", "glare", "glass",
            "glide", "glimpse", "globe", "gloom", "glory", "glove", "glow", "glue",
            "goat", "goddess", "gold", "good", "goose", "gorilla", "gospel", "gossip",
            "govern", "gown", "grab", "grace", "grain", "grant", "grape", "grass",
            "gravity", "great", "green", "grid", "grief", "grit", "grocery", "group",
            "grow", "grunt", "guard", "guess", "guide", "guilt", "guitar", "gun",
            "gym", "habit", "hair", "half", "hammer", "hamster", "hand", "happy",
            "harbor", "hard", "harsh", "harvest", "hat", "have", "hawk", "hazard",
            "head", "health", "heart", "heavy", "hedgehog", "height", "hello", "helmet",
            "help", "hen", "hero", "hidden", "high", "hill", "hint", "hip",
            "hire", "history", "hobby", "hockey", "hold", "hole", "holiday", "hollow",
            "home", "honey", "hood", "hope", "horn", "horror", "horse", "hospital",
            "host", "hotel", "hour", "hover", "hub", "huge", "human", "humble",
            "humor", "hundred", "hungry", "hunt", "hurdle", "hurry", "hurt", "husband",
            "hybrid", "ice", "icon", "idea", "identify", "idle", "ignore", "ill",
            "illegal", "illness", "image", "imitate", "immense", "immune", "impact", "impose",
            "improve", "impulse", "inch", "include", "income", "increase", "index", "indicate",
            "indoor", "industry", "infant", "inflict", "inform", "inhale", "inherit", "initial",
            "inject", "injury", "inmate", "inner", "innocent", "input", "inquiry", "insane",
            "insect", "inside", "inspire", "install", "intact", "interest", "into", "invest",
            "invite", "involve", "iron", "island", "isolate", "issue", "item", "ivory",
            "jacket", "jaguar", "jar", "jazz", "jealous", "jeans", "jelly", "jewel",
            "job", "join", "joke", "journey", "joy", "judge", "juice", "jump",
            "jungle", "junior", "junk", "just", "kangaroo", "keen", "keep", "ketchup",
            "key", "kick", "kid", "kidney", "kind", "kingdom", "kiss", "kit",
            "kitchen", "kite", "kitten", "kiwi", "knee", "knife", "knock", "know",
            "lab", "label", "labor", "ladder", "lady", "lake", "lamp", "language",
            "laptop", "large", "later", "latin", "laugh", "laundry", "lava", "law",
            "lawn", "lawsuit", "layer", "lazy", "leader", "leaf", "learn", "leave",
            "lecture", "left", "leg", "legal", "legend", "leisure", "lemon", "lend",
            "length", "lens", "leopard", "lesson", "letter", "level", "liar", "liberty",
            "library", "license", "life", "lift", "light", "like", "limb", "limit",
            "link", "lion", "liquid", "list", "little", "live", "lizard", "load",
            "loan", "lobster", "local", "lock", "logic", "lonely", "long", "loop",
            "lottery", "loud", "lounge", "love", "loyal", "lucky", "luggage", "lumber",
            "lunar", "lunch", "luxury", "lyrics", "machine", "mad", "magic", "magnet",
            "maid", "mail", "main", "major", "make", "mammal", "man", "manage",
            "mandate", "mango", "mansion", "manual", "maple", "marble", "march", "margin",
            "marine", "market", "marriage", "mask", "mass", "master", "match", "material",
            "math", "matrix", "matter", "maximum", "maze", "meadow", "mean", "measure",
            "meat", "mechanic", "medal", "media", "melody", "melt", "member", "memory",
            "mention", "menu", "mercy", "merge", "merit", "merry", "mesh", "message",
            "metal", "method", "middle", "midnight", "milk", "million", "mimic", "mind",
            "minimum", "minor", "minute", "miracle", "mirror", "misery", "miss", "mistake",
            "mix", "mixed", "mixture", "mobile", "model", "modify", "mom", "moment",
            "monitor", "monkey", "monster", "month", "moon", "moral", "more", "morning",
            "mosquito", "mother", "motion", "motor", "mountain", "mouse", "move", "movie",
            "much", "muffin", "mule", "multiply", "muscle", "museum", "mushroom", "music",
            "must", "mutual", "myself", "mystery", "myth", "naive", "name", "napkin",
            "narrow", "nasty", "nation", "nature", "near", "neck", "need", "negative",
            "neglect", "neither", "nephew", "nerve", "nest", "net", "network", "neutral",
            "never", "news", "next", "nice", "night", "noble", "noise", "nominee",
            "noodle", "normal", "north", "nose", "notable", "note", "nothing", "notice",
            "novel", "now", "nuclear", "number", "nurse", "nut", "oak", "obey",
            "object", "oblige", "obscure", "observe", "obtain", "obvious", "occur", "ocean",
            "october", "odor", "off", "offer", "office", "often", "oil", "okay",
            "old", "olive", "olympic", "omit", "once", "one", "onion", "online",
            "only", "open", "opera", "opinion", "oppose", "option", "orange", "orbit",
            "orchard", "order", "ordinary", "organ", "orient", "original", "orphan", "ostrich",
            "other", "outdoor", "outer", "output", "outside", "oval", "oven", "over",
            "own", "owner", "oxygen", "oyster", "ozone", "pact", "paddle", "page",
            "pair", "palace", "palm", "panda", "panel", "panic", "panther", "paper",
            "parade", "parent", "park", "parrot", "party", "pass", "patch", "path",
            "patient", "patrol", "pattern", "pause", "pave", "payment", "peace", "peanut",
            "pear", "peasant", "pelican", "pen", "penalty", "pencil", "people", "pepper",
            "perfect", "permit", "person", "pet", "phone", "photo", "phrase", "physical",
            "piano", "picnic", "picture", "piece", "pig", "pigeon", "pill", "pilot",
            "pink", "pioneer", "pipe", "pistol", "pitch", "pizza", "place", "planet",
            "plastic", "plate", "play", "please", "pledge", "pluck", "plug", "plunge",
            "poem", "poet", "point", "polar", "pole", "police", "pond", "pony",
            "pool", "popular", "portion", "position", "possible", "post", "potato", "pottery",
            "poverty", "powder", "power", "practice", "praise", "predict", "prefer", "prepare",
            "present", "pretty", "prevent", "price", "pride", "primary", "print", "priority",
            "prison", "private", "prize", "problem", "process", "produce", "profit", "program",
            "project", "promote", "proof", "property", "prosper", "protect", "proud", "provide",
            "public", "pudding", "pull", "pulp", "pulse", "pumpkin", "punch", "pupil",
            "puppy", "purchase", "purity", "purpose", "purse", "push", "put", "puzzle",
            "pyramid", "quality", "quantum", "quarter", "question", "quick", "quit", "quiz",
            "quote", "rabbit", "raccoon", "race", "rack", "radar", "radio", "rail",
            "rain", "raise", "rally", "ramp", "ranch", "random", "range", "rapid",
            "rare", "rate", "rather", "raven", "raw", "razor", "ready", "real",
            "reason", "rebel", "rebuild", "recall", "receive", "recipe", "record", "recycle",
            "reduce", "reflect", "reform", "refuse", "region", "regret", "regular", "reject",
            "relax", "release", "relief", "rely", "remain", "remember", "remind", "remove",
            "render", "renew", "rent", "reopen", "repair", "repeat", "replace", "report",
            "require", "rescue", "resemble", "resist", "resource", "response", "result", "retire",
            "retreat", "return", "reunion", "reveal", "review", "reward", "rhythm", "rib",
            "ribbon", "rice", "rich", "ride", "ridge", "rifle", "right", "rigid",
            "ring", "riot", "ripple", "risk", "ritual", "rival", "river", "road",
            "roast", "robot", "robust", "rocket", "romance", "roof", "rookie", "room",
            "rose", "rotate", "rough", "round", "route", "royal", "rubber", "rude",
            "rug", "rule", "run", "runway", "rural", "sad", "saddle", "sadness",
            "safe", "sail", "salad", "salmon", "salon", "salt", "salute", "same",
            "sample", "sand", "satisfy", "satoshi", "sauce", "sausage", "save", "say",
            "scale", "scan", "scare", "scatter", "scene", "scheme", "school", "science",
            "scissors", "scorpion", "scout", "scrap", "screen", "script", "scrub", "sea",
            "search", "season", "seat", "second", "secret", "section", "security", "seed",
            "seek", "segment", "select", "sell", "seminar", "senior", "sense", "sentence",
            "series", "service", "session", "settle", "setup", "seven", "shadow", "shaft",
            "shallow", "share", "shed", "shell", "sheriff", "shield", "shift", "shine",
            "ship", "shiver", "shock", "shoe", "shoot", "shop", "short", "shoulder",
            "shove", "shrimp", "shrug", "shuffle", "shy", "sibling", "sick", "side",
            "siege", "sight", "sign", "silent", "silk", "silly", "silver", "similar",
            "simple", "since", "sing", "siren", "sister", "situate", "six", "size",
            "skate", "sketch", "ski", "skill", "skin", "skirt", "skull", "slab",
            "slam", "sleep", "slender", "slice", "slide", "slight", "slim", "slogan",
            "slot", "slow", "slush", "small", "smart", "smile", "smoke", "smooth",
            "snack", "snake", "snap", "sniff", "snow", "soap", "soccer", "social",
            "sock", "soda", "soft", "solar", "soldier", "solid", "solution", "solve",
            "someone", "song", "soon", "sorry", "sort", "soul", "sound", "soup",
            "source", "south", "space", "spare", "spatial", "spawn", "speak", "special",
            "speed", "spell", "spend", "sphere", "spice", "spider", "spike", "spin",
            "spirit", "split", "spoil", "sponsor", "spoon", "sport", "spot", "spray",
            "spread", "spring", "spy", "square", "squeeze", "squirrel", "stable", "stadium",
            "staff", "stage", "stairs", "stamp", "stand", "start", "state", "stay",
            "steak", "steel", "stem", "step", "stereo", "stick", "still", "sting",
            "stock", "stomach", "stone", "stool", "story", "stove", "strategy", "street",
            "strike", "strong", "struggle", "student", "stuff", "stumble", "style", "subject",
            "submit", "subway", "success", "such", "sudden", "suffer", "sugar", "suggest",
            "suit", "summer", "sun", "sunny", "sunset", "super", "supply", "supreme",
            "sure", "surface", "surge", "surprise", "surround", "survey", "suspect", "sustain",
            "swallow", "swamp", "swap", "swarm", "swear", "sweet", "swift", "swim",
            "swing", "switch", "sword", "symbol", "symptom", "syrup", "system", "table",
            "tackle", "tag", "tail", "talent", "talk", "tank", "tape", "target",
            "task", "taste", "tattoo", "taxi", "teach", "team", "tell", "ten",
            "tenant", "tennis", "tent", "term", "test", "text", "thank", "that",
            "theme", "then", "theory", "there", "they", "thing", "this", "thought",
            "three", "thrive", "throw", "thumb", "thunder", "ticket", "tide", "tiger",
            "tilt", "timber", "time", "tiny", "tip", "tired", "tissue", "title",
            "toast", "tobacco", "today", "toddler", "toe", "together", "toilet", "token",
            "tomato", "tomorrow", "tone", "tongue", "tonight", "tool", "tooth", "top",
            "topic", "topple", "torch", "tornado", "tortoise", "toss", "total", "tourist",
            "toward", "tower", "town", "toy", "track", "trade", "traffic", "tragic",
            "train", "transfer", "trap", "trash", "travel", "tray", "treat", "tree",
            "trend", "trial", "tribe", "trick", "trigger", "trim", "trip", "trophy",
            "trouble", "truck", "true", "truly", "trumpet", "trust", "truth", "try",
            "tube", "tuition", "tumble", "tuna", "tunnel", "turkey", "turn", "turtle",
            "twelve", "twenty", "twice", "twin", "twist", "two", "type", "typical",
            "ugly", "umbrella", "unable", "unaware", "uncle", "uncover", "under", "undo",
            "unfair", "unfold", "unhappy", "uniform", "unique", "unit", "universe", "unknown",
            "unlock", "until", "unusual", "unveil", "update", "upgrade", "uphold", "upon",
            "upper", "upset", "urban", "urge", "usage", "use", "used", "useful",
            "useless", "usual", "utility", "vacant", "vacuum", "vague", "valid", "valley",
            "valve", "van", "vanish", "vapor", "various", "vast", "vault", "vehicle",
            "velvet", "vendor", "venture", "venue", "verb", "verify", "version", "very",
            "vessel", "veteran", "viable", "vibrant", "vicious", "victory", "video", "view",
            "village", "vintage", "violin", "virtual", "virus", "visa", "visit", "visual",
            "vital", "vivid", "vocal", "voice", "void", "volcano", "volume", "vote",
            "voyage", "wage", "wagon", "wait", "walk", "wall", "walnut", "want",
            "warfare", "warm", "warrior", "wash", "wasp", "waste", "water", "wave",
            "way", "wealth", "weapon", "wear", "weasel", "weather", "web", "wedding",
            "weekend", "weird", "welcome", "west", "wet", "whale", "what", "wheat",
            "wheel", "when", "where", "whip", "whisper", "wide", "width", "wife",
            "wild", "will", "win", "window", "wine", "wing", "wink", "winner",
            "winter", "wire", "wisdom", "wise", "wish", "witness", "wolf", "woman",
            "wonder", "wood", "wool", "word", "work", "world", "worry", "worth",
            "wrap", "wreck", "wrestle", "wrist", "write", "wrong", "yard", "year",
            "yellow", "you", "young", "youth", "zebra", "zero", "zone", "zoo",
        ]

    def _build_word_tokens_list(self):
        """Build list of word tokens for decode() compatibility."""
        self.word_tokens = []
        self.subword_tokens = []
        
        for token in self.vocab.keys():
            # Skip special tokens
            if token.startswith('<') and token.endswith('>'):
                continue
            # Real words are alphabetic with 2+ characters
            if token.isalpha() and len(token) >= 2:
                self.word_tokens.append(token)
            else:
                self.subword_tokens.append(token)
    
    def _build_word_tokens(self):
        """Build word_tokens list from vocabulary for decode compatibility."""
        self.word_tokens = []
        for token in self.vocab.keys():
            # Include only real words (alphabetic, 3+ chars, not special tokens)
            if token.startswith('<') and token.endswith('>'):
                continue
            if len(token) >= 3 and token.replace('-', '').replace("'", '').isalpha():
                self.word_tokens.append(token)
    
    def decode(self, basin: np.ndarray, top_k: int = 5) -> list:
        """Decode basin coordinates to most likely tokens.
        
        Uses Fisher-Rao distance for similarity.
        Returns list of (token, similarity) tuples.
        """
        # Normalize input basin
        norm = np.linalg.norm(basin)
        if norm > 1e-10:
            basin = basin / norm
        
        if not self.word_tokens:
            return []
        
        # Compute Fisher-Rao distance to word tokens
        distances = []
        for token in self.word_tokens:
            if token not in self.basin_coords:
                continue
            coords = self.basin_coords[token]
            # Fisher-Rao distance: arccos(dot product) for unit vectors
            dot = np.clip(np.dot(basin, coords), -1.0, 1.0)
            dist = np.arccos(dot)
            distances.append((token, dist))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Return top-k with similarity scores
        results = []
        for token, dist in distances[:top_k]:
            similarity = 1.0 - (dist / np.pi)  # Normalize to [0, 1]
            results.append((token, similarity))
        
        return results
    
    def _update_unk_to_vocabulary_centroid(self):
        """Update UNK token to centroid of vocabulary space."""
        if "<UNK>" not in self.vocab:
            return
        
        vocab_basins = []
        for token, coord in self.basin_coords.items():
            if token not in self.special_tokens:
                vocab_basins.append(coord)
        
        if len(vocab_basins) >= 10:
            from qig_geometry import sphere_project
            centroid = np.mean(vocab_basins, axis=0)
            self.basin_coords["<UNK>"] = sphere_project(centroid)
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to basin coordinates.
        
        Tokenizes text and averages basin coordinates of known tokens.
        
        Args:
            text: Input text to encode
            
        Returns:
            64D basin coordinates on unit sphere
        """
        # Simple whitespace tokenization
        tokens = text.lower().split()
        
        if not tokens:
            return np.zeros(64)
        
        # Collect basin coordinates for known tokens
        coords_list = []
        for token in tokens:
            # Try exact match
            if token in self.basin_coords:
                coords_list.append(self.basin_coords[token])
                continue
            # Try without punctuation
            clean = token.strip('.,!?;:()[]{}"\'-')
            if clean in self.basin_coords:
                coords_list.append(self.basin_coords[clean])
        
        if not coords_list:
            # Return deterministic basin for unknown text
            return self._initialize_token_coordinate(text, hash(text) % 10000)
        
        # Average and normalize
        basin = np.mean(coords_list, axis=0)
        norm = np.linalg.norm(basin)
        if norm > 1e-10:
            basin = basin / norm
        
        return basin
    
    def decode(self, basin: np.ndarray, top_k: int = 5, prefer_words: bool = True) -> List[Tuple[str, float]]:
        """Decode basin coordinates to most likely tokens.
        
        Uses Fisher-Rao distance for similarity matching.
        
        Args:
            basin: Query basin coordinates (64D)
            top_k: Number of top tokens to return
            prefer_words: If True, prefer real words over subword tokens
        
        Returns:
            List of (token, similarity_score) tuples
        """
        # Normalize input basin
        norm = np.linalg.norm(basin)
        if norm > 1e-10:
            basin = basin / norm
        
        # Choose token set based on preference
        if prefer_words and self.word_tokens:
            search_tokens = self.word_tokens
        elif prefer_words:
            # Filter on the fly
            search_tokens = [t for t in self.vocab.keys() 
                           if t.isalpha() and len(t) >= 2 and not t.startswith('<')]
        else:
            search_tokens = [t for t in self.vocab.keys() if not t.startswith('<')]
        
        if not search_tokens:
            search_tokens = list(self.vocab.keys())
        
        # Compute distances
        distances = []
        for token in search_tokens:
            if token not in self.basin_coords:
                continue
            coords = self.basin_coords[token]
            
            # Normalize coords
            coord_norm = np.linalg.norm(coords)
            if coord_norm > 1e-10:
                coords = coords / coord_norm
            
            # Fisher-Rao distance approximation using arccos of dot product
            dot = np.clip(np.dot(basin, coords), -1.0, 1.0)
            dist = np.arccos(dot)
            distances.append((token, dist))
        
        # Sort by distance (ascending = more similar)
        distances.sort(key=lambda x: x[1])
        
        # Return top-k with similarity scores
        results = []
        for token, dist in distances[:top_k]:
            similarity = 1.0 - (dist / np.pi)  # Normalize to [0, 1]
            results.append((token, similarity))
        
        return results
    
    def add_vocabulary_observations(
        self,
        observations: List[Dict],
    ) -> Tuple[int, bool]:
        """
        Add vocabulary observations (QIGTokenizer compatibility).
        
        Args:
            observations: List of {word, frequency, avgPhi, maxPhi, type}
        
        Returns:
            Tuple of (new_tokens_count, weights_updated)
        """
        new_tokens = 0
        weights_updated = False
        sequences_processed = []
        
        skipped_low_freq = 0
        skipped_low_phi = 0
        skipped_not_english = 0
        skipped_vocab_full = 0
        
        for obs in observations:
            word = obs.get('word', '')
            frequency = obs.get('frequency', 0)
            # Support both 'avgPhi' and 'phi' field names for compatibility
            avg_phi = obs.get('avgPhi', obs.get('phi', 0.0))
            max_phi = obs.get('maxPhi', obs.get('phi', 0.0))
            obs_type = obs.get('type', 'word')
            
            # For vocabulary building, accept frequency >= 1 since phi threshold provides quality control
            # (min_frequency=2 is still used for merge rules)
            if not word or frequency < 1:
                skipped_low_freq += 1
                continue
            
            # Collect sequences for merge learning
            if obs_type == 'sequence' and avg_phi >= self.phi_threshold:
                sequences_processed.append((word, avg_phi, frequency))
                continue
            
            # QIG PRINCIPLE: Only learn words with Φ >= threshold
            # Use a more permissive threshold for vocabulary building (0.4 vs 0.7 for merges)
            vocab_phi_threshold = 0.4  # Lower threshold for adding new words
            if avg_phi < vocab_phi_threshold:
                skipped_low_phi += 1
                continue
            
            # Add to vocabulary if not exists
            if word not in self.vocab:
                if not self._is_english_word(word):
                    skipped_not_english += 1
                    continue
                
                new_id = len(self.vocab)
                if new_id < self.vocab_size:
                    self.vocab[word] = new_id
                    self.id_to_token[new_id] = word
                    self.token_frequency[word] = frequency
                    self.basin_coords[word] = self._initialize_token_coordinate(word, new_id)
                    new_tokens += 1
                    print(f"[VocabLearning] Added new word: '{word}' (phi={avg_phi:.2f}, freq={frequency})")
                else:
                    skipped_vocab_full += 1
            
            # Update weights based on Φ
            old_weight = self.token_weights.get(word, 0.0)
            old_phi = self.token_phi.get(word, 0.0)
            
            phi_weight = 1.0 + avg_phi * 2.0
            
            if abs(phi_weight - old_weight) > 0.01 or abs(avg_phi - old_phi) > 0.01:
                weights_updated = True
            
            self.token_weights[word] = phi_weight
            self.token_phi[word] = avg_phi
            self.token_frequency[word] = frequency
        
        # Process sequences for merge learning
        if sequences_processed:
            self._learn_merges_from_sequences(sequences_processed)
            weights_updated = True
        
        # Refresh vocabulary caches
        self.conversation_vocab_ids = set(self.vocab.values())
        
        # Rebuild word_tokens after adding new vocabulary
        self._build_word_tokens()
        
        # Verbose summary logging
        if len(observations) > 0:
            print(f"[VocabLearning] Observation summary:")
            print(f"  Total observations: {len(observations)}")
            print(f"  New words added: {new_tokens}")
            print(f"  Skipped (low frequency < {self.min_frequency}): {skipped_low_freq}")
            print(f"  Skipped (low phi < 0.4): {skipped_low_phi}")
            print(f"  Skipped (not English): {skipped_not_english}")
            print(f"  Skipped (vocab full): {skipped_vocab_full}")
            print(f"  Current vocab size: {len(self.vocab)}")
        
        return new_tokens, weights_updated
    
    def _learn_merges_from_sequences(self, sequences: List[Tuple[str, float, int]]) -> None:
        """Learn merge rules from high-Φ sequences."""
        for sequence, phi, frequency in sequences:
            parts = sequence.split()
            if len(parts) >= 2:
                for i in range(len(parts) - 1):
                    pair = (parts[i], parts[i + 1])
                    if pair not in self.merge_rules:
                        self.merge_rules.append(pair)
                        self.merge_scores[pair] = phi
    
    def _is_english_word(self, word: str) -> bool:
        """Validate English word."""
        try:
            from word_validation import is_valid_english_word
            return is_valid_english_word(word, include_stop_words=True)
        except ImportError:
            if not word or len(word) < 2:
                return False
            word_lower = word.lower().strip()
            if len(word_lower) == 1:
                return word_lower in {'a', 'i'}
            if any(char.isdigit() for char in word_lower):
                return False
            return word_lower[0].isalpha()
    
    def set_mode(self, mode: str) -> None:
        """Set vocabulary mode (mnemonic/passphrase/conversation)."""
        self.mode = mode
    
    def generate_response(
        self,
        context: str,
        agent_role: str = 'zeus',
        allow_silence: bool = False,
        goals: Optional[List] = None
    ) -> dict:
        """Generate a response using geometric retrieval.
        
        Args:
            context: Input context/prompt
            agent_role: Role of the agent generating
            allow_silence: If True, may return empty response
            goals: Optional generation goals
        
        Returns:
            Dict with 'text', 'phi', 'tokens_generated', 'completion_reason', 'qig_pure'
        """
        # Encode context to basin
        context_basin = self.encode(context)
        
        # Get similar words from vocabulary
        similar_words = self.decode(context_basin, top_k=15, prefer_words=True)
        relevant_words = [w for w, s in similar_words if s > 0.3][:10]
        
        if relevant_words:
            response_text = f"Geometric analysis:\n\n"
            response_text += f"Key concepts: {', '.join(relevant_words[:5])}\n"
            if len(relevant_words) > 5:
                response_text += f"Related: {', '.join(relevant_words[5:10])}\n"
            response_text += f"\n[QIG-Pure | {agent_role}]"
            response_phi = 0.5
            completion_reason = 'vocabulary_synthesis'
        elif not allow_silence:
            response_text = f"[No geometric patterns matched | {agent_role}]"
            response_phi = 0.1
            completion_reason = 'no_match'
        else:
            response_text = ''
            response_phi = 0.0
            completion_reason = 'silence'
        
        return {
            'text': response_text,
            'phi': response_phi,
            'tokens_generated': len(response_text.split()),
            'completion_reason': completion_reason,
            'qig_pure': True
        }
    
    def generate_response(
        self,
        context: str,
        agent_role: str = "ocean",
        allow_silence: bool = False,
        **kwargs
    ) -> dict:
        """Generate a response using QIG-pure geometric methods.
        
        This method produces readable English text by traversing the
        vocabulary basin space geometrically. Uses word_tokens (real words)
        instead of BPE fragments for readable output.
        
        Args:
            context: Input prompt/context text
            agent_role: Role hint for generation style
            allow_silence: Whether empty response is allowed
            **kwargs: Additional parameters (max_tokens ignored - geometry determines completion)
        
        Returns:
            Dict with 'text', 'phi', 'tokens_generated', etc.
        """
        # Encode context to basin coordinates
        context_basin = self.encode(context)
        
        if np.linalg.norm(context_basin) < 1e-10:
            if allow_silence:
                return {'text': '', 'phi': 0, 'tokens_generated': 0, 'qig_pure': True}
            # Generate from random high-phi words
            context_basin = self._generate_seed_basin()
        
        # Generate response through geometric traversal
        generated_words = []
        current_basin = context_basin.copy()
        phi_values = []
        
        # Geometric completion loop - stops when basin stabilizes
        max_iterations = 50  # Safety limit (NOT a generation target)
        prev_basin = None
        stable_count = 0
        
        for i in range(max_iterations):
            # Decode current basin to get candidate words (prefer real words)
            candidates = self.decode(current_basin, top_k=20, prefer_words=True)
            
            if not candidates:
                break
            
            # Find best word not recently used (avoid repetition)
            best_word = None
            best_similarity = 0.0
            recent_words = set(generated_words[-5:]) if len(generated_words) >= 5 else set(generated_words)
            
            for word, similarity in candidates:
                if word not in recent_words:
                    best_word = word
                    best_similarity = similarity
                    break
            
            # If all candidates were recently used, take the best one anyway
            if best_word is None and candidates:
                best_word, best_similarity = candidates[0]
            
            if best_word is None:
                break
            
            generated_words.append(best_word)
            phi = self.token_phi.get(best_word, 0.5)
            phi_values.append(phi)
            
            # Update basin by moving toward the selected word's basin
            if best_word in self.basin_coords:
                word_basin = self.basin_coords[best_word]
                # Geodesic interpolation - move toward word basin
                t = 0.3 + np.random.uniform(-0.1, 0.1)
                current_basin = self._geodesic_interpolate(current_basin, word_basin, t)
            
            # Check for geometric completion (basin stabilization)
            if prev_basin is not None:
                dot = np.clip(np.dot(current_basin, prev_basin), -1.0, 1.0)
                movement = np.arccos(dot)  # Fisher-Rao distance
                if movement < 0.1:  # Basin has stabilized
                    stable_count += 1
                    if stable_count >= 3:  # Stable for 3 iterations
                        break
                else:
                    stable_count = 0
            
            prev_basin = current_basin.copy()
            
            # Also stop if phi drops significantly (thought complete)
            if len(phi_values) >= 5:
                recent_phi = np.mean(phi_values[-5:])
                if recent_phi < 0.3:
                    break
        
        # Build response text
        response_text = ' '.join(generated_words)
        avg_phi = np.mean(phi_values) if phi_values else 0.0
        
        return {
            'text': response_text,
            'phi': float(avg_phi),
            'tokens_generated': len(generated_words),
            'completion_reason': 'geometric_stable' if stable_count >= 3 else 'iteration_limit',
            'qig_pure': True
        }
    
    def _generate_seed_basin(self) -> np.ndarray:
        """Generate a seed basin from high-phi words."""
        high_phi_words = [w for w, p in self.token_phi.items() if p > 0.6]
        if not high_phi_words:
            high_phi_words = list(self.vocab.keys())[:100]
        
        if not high_phi_words:
            basin = np.random.randn(64)
            return sphere_project(basin)
        
        sample_size = min(5, len(high_phi_words))
        sample_words = np.random.choice(high_phi_words, sample_size, replace=False)
        basins = [self.basin_coords[w] for w in sample_words if w in self.basin_coords]
        
        if basins:
            combined = np.mean(basins, axis=0)
            norm = np.linalg.norm(combined)
            return combined / norm if norm > 1e-10 else np.random.randn(64) / 10
        
        basin = np.random.randn(64)
        return sphere_project(basin)
    
    def _geodesic_interpolate(self, start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        """Interpolate along geodesic on unit sphere."""
        # Normalize inputs
        start_norm = np.linalg.norm(start)
        end_norm = np.linalg.norm(end)
        
        if start_norm > 1e-10:
            start = start / start_norm
        if end_norm > 1e-10:
            end = end / end_norm
        
        # Linear interpolation (approximation of geodesic)
        result = (1 - t) * start + t * end
        
        # Normalize result
        result_norm = np.linalg.norm(result)
        if result_norm > 1e-10:
            result = result / result_norm
        
        return result
    
    def generate_response(
        self,
        context: str,
        agent_role: str = "ocean",
        allow_silence: bool = False,
        **kwargs
    ) -> dict:
        """Generate a response using QIG-pure geometric methods.
        
        Uses vocabulary-based generation with diversity sampling.
        
        Args:
            context: Input prompt/context text
            agent_role: Role hint for generation style
            allow_silence: Whether empty response is allowed
        
        Returns:
            Dict with 'text', 'phi', 'tokens_generated', etc.
        """
        # Check vocabulary availability
        if not self.word_tokens:
            self._build_word_tokens_list()
        
        if not self.word_tokens:
            return {
                'text': '[No vocabulary available]',
                'phi': 0.0,
                'tokens_generated': 0,
                'completion_reason': 'no_vocabulary',
                'qig_pure': True
            }
        
        # Encode context to basin
        context_basin = self.encode(context)
        
        if np.linalg.norm(context_basin) < 1e-10:
            if allow_silence:
                return {'text': '', 'phi': 0, 'tokens_generated': 0, 'completion_reason': 'empty', 'qig_pure': True}
            # Use random high-phi words as seed
            high_phi = [w for w, p in self.token_phi.items() if p > 0.5 and w in self.basin_coords]
            if high_phi:
                sample = np.random.choice(high_phi, min(5, len(high_phi)), replace=False)
                basins = [self.basin_coords[w] for w in sample]
                context_basin = np.mean(basins, axis=0)
                context_basin = sphere_project(context_basin)
            else:
                context_basin = np.random.randn(64)
                context_basin = sphere_project(context_basin)
        
        # Generation parameters
        temperature = 0.7
        max_words = 30
        
        generated = []
        used_words = set()
        current_basin = context_basin.copy()
        phi_values = []
        
        for i in range(max_words * 2):
            if len(generated) >= max_words:
                break
            
            # Get candidates
            candidates = self.decode(current_basin, top_k=25, prefer_words=True)
            if not candidates:
                break
            
            # Filter used words
            available = [(w, s) for w, s in candidates if w not in used_words]
            if not available:
                used_words = set(generated[-3:]) if len(generated) > 3 else set()
                available = [(w, s) for w, s in candidates if w not in used_words]
            if not available:
                break
            
            # Temperature sampling
            if len(available) > 1:
                sims = np.array([s for w, s in available])
                logits = sims / temperature
                logits = logits - np.max(logits)
                probs = np.exp(logits) / np.sum(np.exp(logits))
                idx = np.random.choice(len(available), p=probs)
                word, sim = available[idx]
            else:
                word, sim = available[0]
            
            generated.append(word)
            used_words.add(word)
            phi_values.append(self.token_phi.get(word, 0.5))
            
            # Update basin
            if word in self.basin_coords:
                word_basin = self.basin_coords[word]
                t = 0.3
                current_basin = (1 - t) * current_basin + t * word_basin
                current_basin += np.random.randn(64) * 0.05
                current_basin = sphere_project(current_basin)
        
        # Remove consecutive duplicates
        final = [generated[0]] if generated else []
        for w in generated[1:]:
            if w != final[-1]:
                final.append(w)
        
        return {
            'text': ' '.join(final),
            'phi': float(np.mean(phi_values)) if phi_values else 0.0,
            'tokens_generated': len(final),
            'completion_reason': 'geometric_stable',
            'qig_pure': True
        }


def get_coordizer():
    """Get or create singleton coordizer instance.
    
    Priority (64D QIG Pure):
    1. PostgresCoordizer (64D QIG-pure geometry from tokenizer_vocabulary)
    2. SimpleWordCoordizer (fallback with database-backed words)
    3. QIGCoordizer with file-based vocabulary (last resort)
    
    PostgresCoordizer is preferred as it maintains 64D geometric purity
    and enables proper vocabulary persistence.
    """
    global _coordizer_instance
    if _coordizer_instance is None:
        # Priority 1: PostgresCoordizer (64D QIG-pure) with retry
        if POSTGRES_COORDIZER_AVAILABLE and create_coordizer_from_pg:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    pg_coordizer = create_coordizer_from_pg()
                    if pg_coordizer and len(pg_coordizer.vocab) >= 50:
                        _coordizer_instance = pg_coordizer
                        print(f"[QIGCoordizer] Using PostgresCoordizer (64D QIG-pure): {len(pg_coordizer.vocab)} tokens from database")
                        return _coordizer_instance
                    else:
                        print(f"[QIGCoordizer] PostgresCoordizer has insufficient vocabulary ({len(pg_coordizer.vocab) if pg_coordizer else 0} tokens)")
                        break  # Don't retry if vocab is empty - it's not transient
                except Exception as e:
                    if attempt < max_retries - 1:
                        import time
                        print(f"[QIGCoordizer] PostgresCoordizer attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    else:
                        print(f"[QIGCoordizer] PostgresCoordizer unavailable after {max_retries} attempts: {e}")
        
        # Priority 2: SimpleWordCoordizer (degraded mode)
        try:
            from simple_word_coordizer import SimpleWordCoordizer
            simple_coordizer = SimpleWordCoordizer()
            
            if simple_coordizer.vocab_size >= 50:
                _coordizer_instance = simple_coordizer
                fallback_note = " (fallback)" if simple_coordizer.is_using_fallback() else ""
                print(f"[QIGCoordizer] DEGRADED: Using SimpleWordCoordizer{fallback_note}: {simple_coordizer.vocab_size} words - 64D purity compromised")
                return _coordizer_instance
            else:
                print(f"[QIGCoordizer] SimpleWordCoordizer has only {simple_coordizer.vocab_size} words")
        except Exception as e:
            print(f"[QIGCoordizer] SimpleWordCoordizer unavailable: {e}")
        
        # Priority 3: File-based QIGCoordizer (last resort)
        _coordizer_instance = QIGCoordizer()
        _load_coordizer_state(_coordizer_instance)
        _migrate_from_legacy_tokenizer(_coordizer_instance)
        print(f"[QIGCoordizer] DEGRADED: Using file-based QIGCoordizer with {len(_coordizer_instance.vocab)} tokens - 64D purity compromised")
    return _coordizer_instance


# Coordizer instance ID for Redis persistence
COORDIZER_INSTANCE_ID = "main"


def _load_coordizer_state(coordizer: QIGCoordizer) -> None:
    """Load persisted coordizer state from Redis."""
    try:
        from redis_cache import CoordizerBuffer
        
        state = CoordizerBuffer.load_state(COORDIZER_INSTANCE_ID)
        if state:
            coordizer.vocab = state['vocab']
            coordizer.id_to_token = state['id_to_token']
            coordizer.token_frequency = state['token_frequency']
            coordizer.token_phi = state['token_phi']
            
            coordizer.basin_coords = {
                token: np.array(coords) 
                for token, coords in state['basin_coords'].items()
            }
            
            extra = state.get('extra', {})
            coordizer.token_weights = extra.get('token_weights', {})
            coordizer.merge_rules = [tuple(r) for r in extra.get('merge_rules', [])]
            coordizer.merge_scores = {tuple(k.split('|')): v for k, v in extra.get('merge_scores', {}).items()}
            
            print(f"[QIGCoordizer] Loaded state from Redis ({len(coordizer.vocab)} tokens)")
            return
    except ImportError:
        print("[QIGCoordizer] Redis cache not available")
    except Exception as e:
        print(f"[QIGCoordizer] Redis load failed: {e}")


def _migrate_from_legacy_tokenizer(coordizer: QIGCoordizer) -> None:
    """Migrate from old QIGTokenizer state if it exists."""
    if not os.path.exists(LEGACY_TOKENIZER_PATH):
        return
    
    try:
        with open(LEGACY_TOKENIZER_PATH, 'r') as f:
            data = json.load(f)
        
        # Restore learned tokens
        for token, weight in data.get('token_weights', {}).items():
            if token in coordizer.vocab:
                coordizer.token_weights[token] = weight
        
        for token, phi in data.get('token_phi', {}).items():
            if token in coordizer.vocab:
                coordizer.token_phi[token] = phi
        
        for token, freq in data.get('token_frequency', {}).items():
            if token in coordizer.vocab:
                coordizer.token_frequency[token] = freq
        
        # Restore merge rules
        for rule in data.get('merge_rules', []):
            if isinstance(rule, list) and len(rule) >= 2:
                pair = (rule[0], rule[1])
                if pair not in coordizer.merge_rules:
                    coordizer.merge_rules.append(pair)
        
        print(f"[QIGCoordizer] Migrated from legacy tokenizer: {len(data.get('learned_vocab', {}))} tokens")
        
        # Save migrated state in new format
        _save_coordizer_state(coordizer)
        
    except Exception as e:
        print(f"[QIGCoordizer] Failed to migrate from legacy: {e}")


def _save_coordizer_state(coordizer: QIGCoordizer) -> None:
    """Save coordizer state to Redis."""
    try:
        from redis_cache import CoordizerBuffer
        
        basin_coords_serializable = {
            token: coords.tolist() if hasattr(coords, 'tolist') else list(coords)
            for token, coords in coordizer.basin_coords.items()
        }
        
        extra_data = {
            'token_weights': coordizer.token_weights,
            'merge_rules': [list(r) for r in coordizer.merge_rules],
            'merge_scores': {'|'.join(k): v for k, v in coordizer.merge_scores.items()},
            'mode': coordizer.mode,
        }
        
        success = CoordizerBuffer.save_state(
            COORDIZER_INSTANCE_ID,
            coordizer.vocab,
            coordizer.id_to_token,
            coordizer.token_frequency,
            coordizer.token_phi,
            basin_coords_serializable,
            extra_data
        )
        
        if success:
            print(f"[QIGCoordizer] Saved state to Redis ({len(coordizer.vocab)} tokens)")
            
    except ImportError:
        print("[QIGCoordizer] Redis cache not available")
    except Exception as e:
        print(f"[QIGCoordizer] Redis save failed: {e}")


def update_tokenizer_from_observations(observations: List[Dict]) -> Tuple[int, bool]:
    """Update coordizer with vocabulary observations (QIGTokenizer compatibility)."""
    coordizer = get_coordizer()
    new_tokens, weights_updated = coordizer.add_vocabulary_observations(observations)
    
    if new_tokens > 0 or weights_updated:
        _save_coordizer_state(coordizer)
    
    return new_tokens, weights_updated


def reset_coordizer() -> None:
    """Reset the coordizer singleton to reload from database.
    
    Call this after populating tokenizer_vocabulary to use the new words.
    Forces a complete reload of the vocabulary.
    """
    global _coordizer_instance
    
    old_type = type(_coordizer_instance).__name__ if _coordizer_instance else 'None'
    old_words = len(getattr(_coordizer_instance, 'word_tokens', [])) if _coordizer_instance else 0
    
    if _coordizer_instance is not None:
        # Close database connection if PostgresCoordizer
        if hasattr(_coordizer_instance, 'close'):
            try:
                _coordizer_instance.close()
            except:
                pass
        _coordizer_instance = None
    
    print(f"[QIGCoordizer] Reset coordizer: was {old_type} with {old_words} words")
    print("[QIGCoordizer] Will reload from database on next get_coordizer() call")
    
    # Force immediate reload to verify it works
    new_instance = get_coordizer()
    new_type = type(new_instance).__name__
    new_words = len(getattr(new_instance, 'word_tokens', []))
    new_bip39 = len(getattr(new_instance, 'bip39_words', []))
    
    print(f"[QIGCoordizer] Reloaded: {new_type} with {new_words} words ({new_bip39} BIP39)")
    
    if new_words < 100:
        print(f"[QIGCoordizer] WARNING: Only {new_words} real words loaded! Check database.")


def get_coordizer_stats() -> dict:
    """Get detailed statistics about the current coordizer."""
    coordizer = get_coordizer()
    
    word_tokens = getattr(coordizer, 'word_tokens', [])
    bip39_words = getattr(coordizer, 'bip39_words', [])
    base_tokens = getattr(coordizer, 'base_tokens', [])
    subword_tokens = getattr(coordizer, 'subword_tokens', [])
    
    return {
        'type': type(coordizer).__name__,
        'vocab_size': len(coordizer.vocab) if hasattr(coordizer, 'vocab') else 0,
        'is_postgres_backed': isinstance(coordizer, PostgresCoordizer) if PostgresCoordizer else False,
        'has_basin_coords': len(coordizer.basin_coords) if hasattr(coordizer, 'basin_coords') else 0,
        'word_tokens_count': len(word_tokens),
        'bip39_words_count': len(bip39_words),
        'base_tokens_count': len(base_tokens),
        'subword_tokens_count': len(subword_tokens),
        'sample_words': word_tokens[:10] if word_tokens else [],
        'sample_bip39': bip39_words[:5] if bip39_words else [],
        'has_real_vocabulary': len(word_tokens) >= 100,
    }


# Backward compatibility alias
get_tokenizer = get_coordizer
