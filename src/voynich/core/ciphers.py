"""
Cipher Families & Encoding Simulators
=======================================
Historical cipher implementations + novel encoding simulators for
information-theoretic fingerprinting reference library.

Carried over from voynich/modules/null_framework.py:
- SimpleSubstitutionCipher, VigenereCipher, HomophonicCipher, NomenclatorCipher

New:
- SyllabicEncoder, AbbreviationEncoder, NullInsertionEncoder
- Reference vocabulary for 7 languages
- generate_reference_text() for synthetic plaintext generation
"""

import random
import re
import string
from collections import Counter
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Carried-Over Cipher Families
# ---------------------------------------------------------------------------

class SimpleSubstitutionCipher:
    """
    Monoalphabetic substitution cipher.
    Each plaintext letter maps to exactly one ciphertext letter.
    Preserves word boundaries, word lengths, and letter frequencies.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        plain_alpha = list('abcdefghijklmnopqrstuvwxyz')
        cipher_alpha = plain_alpha.copy()
        self.rng.shuffle(cipher_alpha)
        self.table = dict(zip(plain_alpha, cipher_alpha))
        self.name = 'simple_substitution'

    def encrypt(self, plaintext: str) -> str:
        result = []
        for ch in plaintext.lower():
            if ch == ' ':
                result.append(' ')
            elif ch in self.table:
                result.append(self.table[ch])
            elif ch.isalpha():
                result.append(ch)
        return ''.join(result)


class VigenereCipher:
    """
    Polyalphabetic substitution cipher (Vigenere family).
    Uses a repeating key to shift each plaintext letter.
    Destroys single-letter frequency patterns but preserves word boundaries.
    """

    def __init__(self, key_length: int = 5, seed: int = 42):
        self.rng = random.Random(seed)
        self.key = [self.rng.randint(1, 25) for _ in range(key_length)]
        self.name = 'polyalphabetic'

    def encrypt(self, plaintext: str) -> str:
        result = []
        key_idx = 0
        for ch in plaintext.lower():
            if ch == ' ':
                result.append(' ')
            elif ch.isalpha():
                shifted = (ord(ch) - ord('a') + self.key[key_idx % len(self.key)]) % 26
                result.append(chr(shifted + ord('a')))
                key_idx += 1
        return ''.join(result)


class HomophonicCipher:
    """
    Homophonic substitution cipher.
    High-frequency letters get multiple ciphertext symbols, flattening
    the frequency distribution. Uses a multi-symbol output alphabet.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.name = 'homophonic'

        freq_tiers = {
            'high': list('aeioust'),
            'medium': list('cnrldhm'),
            'low': list('bfgkpqvwxyz'),
        }

        symbols = [f'{chr(a)}{chr(b)}' for a in range(ord('a'), ord('z') + 1)
                   for b in range(ord('a'), ord('z') + 1)]
        self.rng.shuffle(symbols)
        sym_idx = 0

        self.table: Dict[str, List[str]] = {}
        for ch in freq_tiers['high']:
            n = self.rng.randint(3, 5)
            self.table[ch] = symbols[sym_idx:sym_idx + n]
            sym_idx += n
        for ch in freq_tiers['medium']:
            n = self.rng.randint(2, 3)
            self.table[ch] = symbols[sym_idx:sym_idx + n]
            sym_idx += n
        for ch in freq_tiers['low']:
            self.table[ch] = [symbols[sym_idx]]
            sym_idx += 1
        for ch in string.ascii_lowercase:
            if ch not in self.table:
                self.table[ch] = [symbols[sym_idx]]
                sym_idx += 1

    def encrypt(self, plaintext: str) -> str:
        result = []
        for ch in plaintext.lower():
            if ch == ' ':
                result.append(' ')
            elif ch in self.table:
                result.append(self.rng.choice(self.table[ch]))
        return ''.join(result)


class NomenclatorCipher:
    """
    Nomenclator: code + cipher hybrid.
    Common words get unique code symbols; remaining text uses simple substitution.
    This was the most common diplomatic cipher of the 15th century.
    """

    def __init__(self, n_code_words: int = 50, seed: int = 42):
        self.rng = random.Random(seed)
        self.name = 'nomenclator'

        code_symbols = [f'#{i:03d}' for i in range(n_code_words)]
        self.rng.shuffle(code_symbols)

        common_words = [
            'et', 'in', 'de', 'ad', 'cum', 'per', 'est', 'non', 'sed',
            'qui', 'que', 'ut', 'sic', 'hoc', 'aut', 'vel', 'pro', 'quod',
            'recipe', 'accipe', 'herba', 'aqua', 'contra', 'super', 'ante',
            'post', 'inter', 'sub', 'oleum', 'pulvis', 'misce', 'contere',
            'balneum', 'matrix', 'dolorem', 'febrem', 'sanguinem', 'ventrem',
            'caput', 'corpus', 'manum', 'oculum', 'aurem', 'pedem',
            'calida', 'frigida', 'sicca', 'humida', 'bona', 'mala',
        ]
        self.code_table = dict(zip(common_words[:n_code_words],
                                   code_symbols[:n_code_words]))

        plain_alpha = list('abcdefghijklmnopqrstuvwxyz')
        cipher_alpha = plain_alpha.copy()
        self.rng.shuffle(cipher_alpha)
        self.sub_table = dict(zip(plain_alpha, cipher_alpha))

    def encrypt(self, plaintext: str) -> str:
        words = plaintext.lower().split()
        result = []
        for word in words:
            clean = ''.join(c for c in word if c.isalpha())
            if clean in self.code_table:
                result.append(self.code_table[clean])
            else:
                encrypted = ''.join(self.sub_table.get(c, c) for c in clean)
                result.append(encrypted)
        return ' '.join(result)


# ---------------------------------------------------------------------------
# New Encoding Simulators
# ---------------------------------------------------------------------------

# Vowels and consonants for syllabification
_VOWELS = set('aeiouy')
_CONSONANTS = set('bcdfghjklmnpqrstvwxz')


class SyllabicEncoder:
    """
    Converts plaintext to syllable-unit tokens.
    Simulates a syllabary-based writing system (like Linear B, hiragana).

    Each CV or CVC syllable is mapped to a unique symbol. This dramatically
    reduces conditional character entropy H2 because within-syllable transitions
    become deterministic.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.syllable_map: Dict[str, str] = {}
        self._next_id = 0
        self.name = 'syllabic'

    def _get_symbol(self, syllable: str) -> str:
        """Get or create a unique symbol for a syllable."""
        if syllable not in self.syllable_map:
            # Generate 2-3 character symbols from a reduced alphabet
            alpha = 'abcdefghijklmnopqrst'
            idx = self._next_id
            s1 = alpha[idx % len(alpha)]
            s2 = alpha[(idx // len(alpha)) % len(alpha)]
            self.syllable_map[syllable] = s1 + s2
            self._next_id += 1
        return self.syllable_map[syllable]

    def _syllabify(self, word: str) -> List[str]:
        """Split a word into CV/CVC syllables (simple greedy algorithm)."""
        word = word.lower()
        syllables = []
        current = ''

        i = 0
        while i < len(word):
            ch = word[i]
            if ch not in _VOWELS and ch not in _CONSONANTS:
                i += 1
                continue

            current += ch

            # If we just added a vowel, check for coda consonant
            if ch in _VOWELS:
                # Look ahead: if next is consonant followed by vowel, break here
                # If next is consonant followed by consonant/end, take it as coda
                if i + 1 < len(word) and word[i + 1] in _CONSONANTS:
                    if i + 2 < len(word) and word[i + 2] in _VOWELS:
                        # C follows V follows — break before C (it's next onset)
                        syllables.append(current)
                        current = ''
                    else:
                        # Coda consonant
                        current += word[i + 1]
                        i += 1
                        syllables.append(current)
                        current = ''
                else:
                    syllables.append(current)
                    current = ''
            i += 1

        if current:
            syllables.append(current)

        return syllables if syllables else [word]

    def encode(self, plaintext: str) -> str:
        """Encode plaintext using syllabary mapping."""
        words = plaintext.lower().split()
        encoded_words = []
        for word in words:
            syllables = self._syllabify(word)
            symbols = [self._get_symbol(s) for s in syllables]
            encoded_words.append(''.join(symbols))
        return ' '.join(encoded_words)


class AbbreviationEncoder:
    """
    Simulates medieval abbreviation practices.

    Light mode: common words abbreviated (et -> &, -tion -> -n, etc.)
    Heavy mode: vowels frequently dropped, prefixes abbreviated,
                Tironian-style shorthand.
    """

    _LIGHT_ABBREVS = {
        'et': '7', 'cum': 'c', 'per': 'p', 'pre': 'p', 'pro': 'p',
        'que': 'q', 'quod': 'qd', 'quia': 'qa', 'quam': 'qm',
        'non': 'n', 'sed': 's', 'est': 'e', 'sunt': 'st',
        'enim': 'em', 'autem': 'at', 'tamen': 'tn', 'igitur': 'ig',
    }

    _HEAVY_SUFFIXES = [
        ('tion', 'n'), ('ment', 'mt'), ('orum', 'or'), ('arum', 'ar'),
        ('ibus', 'ib'), ('ione', 'in'), ('alis', 'al'), ('atus', 'at'),
        ('endo', 'nd'), ('ando', 'nd'), ('ante', 'an'), ('ente', 'en'),
    ]

    def __init__(self, mode: str = 'light', seed: int = 42):
        self.mode = mode
        self.rng = random.Random(seed)
        self.name = f'abbreviation_{mode}'

    def encode(self, plaintext: str) -> str:
        words = plaintext.lower().split()
        result = []
        for word in words:
            clean = ''.join(c for c in word if c.isalpha())
            if not clean:
                continue

            # Light: replace known abbreviations
            if clean in self._LIGHT_ABBREVS:
                result.append(self._LIGHT_ABBREVS[clean])
                continue

            if self.mode == 'heavy':
                # Apply suffix abbreviation
                for suffix, abbrev in self._HEAVY_SUFFIXES:
                    if clean.endswith(suffix):
                        clean = clean[:-len(suffix)] + abbrev
                        break
                # Drop some vowels (50% chance for medial vowels)
                chars = list(clean)
                abbreviated = [chars[0]]  # keep first
                for c in chars[1:-1]:
                    if c in _VOWELS and self.rng.random() < 0.5:
                        continue  # drop vowel
                    abbreviated.append(c)
                if len(chars) > 1:
                    abbreviated.append(chars[-1])  # keep last
                clean = ''.join(abbreviated)

            result.append(clean)
        return ' '.join(result)


class NullInsertionEncoder:
    """
    Inserts null (meaningless) characters at random positions.
    Simulates a null-cipher obfuscation layer.

    null_rate: fraction of characters that are nulls (0.0-0.5)
    """

    def __init__(self, null_rate: float = 0.2, seed: int = 42):
        self.null_rate = null_rate
        self.rng = random.Random(seed)
        self.null_chars = list('jvwxz')  # rare letters used as nulls
        self.name = 'null_insertion'

    def encode(self, plaintext: str) -> str:
        result = []
        for ch in plaintext.lower():
            result.append(ch)
            if ch != ' ' and self.rng.random() < self.null_rate:
                result.append(self.rng.choice(self.null_chars))
        return ''.join(result)


# ---------------------------------------------------------------------------
# Encoding Scheme Registry
# ---------------------------------------------------------------------------

ENCODING_SCHEMES = {
    'raw': {
        'factory': lambda seed: None,  # No encoding
        'description': 'Unencoded plaintext',
    },
    'simple_substitution': {
        'factory': lambda seed: SimpleSubstitutionCipher(seed=seed),
        'description': 'Monoalphabetic 1-to-1 letter substitution',
    },
    'polyalphabetic': {
        'factory': lambda seed: VigenereCipher(
            key_length=random.Random(seed).randint(3, 12), seed=seed
        ),
        'description': 'Polyalphabetic cipher with random key length 3-12',
    },
    'homophonic': {
        'factory': lambda seed: HomophonicCipher(seed=seed),
        'description': 'Homophonic substitution flattening frequency distribution',
    },
    'nomenclator': {
        'factory': lambda seed: NomenclatorCipher(seed=seed),
        'description': 'Code+cipher hybrid, top-50 words get code symbols',
    },
    'syllabic': {
        'factory': lambda seed: SyllabicEncoder(seed=seed),
        'description': 'Syllabary encoding (CV/CVC syllables -> unique symbols)',
    },
    'abbreviation_light': {
        'factory': lambda seed: AbbreviationEncoder(mode='light', seed=seed),
        'description': 'Light medieval abbreviation (common words shortened)',
    },
    'abbreviation_heavy': {
        'factory': lambda seed: AbbreviationEncoder(mode='heavy', seed=seed),
        'description': 'Heavy abbreviation + vowel dropping (Tironian-style)',
    },
    'null_insertion': {
        'factory': lambda seed: NullInsertionEncoder(null_rate=0.2, seed=seed),
        'description': 'Random null characters inserted between real ones',
    },
}


def apply_encoding(plaintext: str, scheme_name: str, seed: int = 42) -> str:
    """Apply an encoding scheme to plaintext. Returns encoded text."""
    if scheme_name == 'raw':
        return plaintext

    scheme = ENCODING_SCHEMES[scheme_name]
    encoder = scheme['factory'](seed)

    # Ciphers use .encrypt(), encoders use .encode()
    if hasattr(encoder, 'encrypt'):
        return encoder.encrypt(plaintext)
    elif hasattr(encoder, 'encode'):
        return encoder.encode(plaintext)
    else:
        return plaintext


# ---------------------------------------------------------------------------
# Reference Vocabularies (embedded — no JSON dependency)
# ---------------------------------------------------------------------------

REFERENCE_VOCABS = {
    'latin': {
        'high_freq': [
            'et', 'in', 'de', 'ad', 'cum', 'per', 'est', 'non', 'sed',
            'qui', 'que', 'ut', 'sic', 'hoc', 'aut', 'vel', 'pro', 'quod',
            'recipe', 'accipe', 'herba', 'aqua', 'contra', 'super', 'ante',
            'post', 'inter', 'sub', 'oleum', 'pulvis', 'misce', 'contere',
            'calida', 'frigida', 'sicca', 'humida', 'bona', 'mala',
        ],
        'medium_freq': [
            'artemisia', 'malva', 'ruta', 'salvia', 'mentha', 'camomilla',
            'rosmarinus', 'lavandula', 'absinthium', 'borrago', 'plantago',
            'vulnus', 'febris', 'tussis', 'venenum', 'pestis', 'apostema',
            'unguentum', 'syrupus', 'decoctum', 'emplastrum', 'suffumigium',
            'calidus', 'frigidus', 'siccus', 'humidus', 'temperatus',
            'primus', 'secundus', 'tertius', 'quartus', 'gradus',
            'radix', 'folium', 'flos', 'semen', 'cortex', 'succus',
            'caput', 'venter', 'pectus', 'stomachus', 'hepar', 'renes',
            'sanguis', 'matrix', 'dolor', 'morbus', 'cura', 'remedium',
        ],
    },
    'italian': {
        'high_freq': [
            'herba', 'radice', 'foglia', 'fiore', 'seme', 'acqua', 'olio',
            'polvere', 'calda', 'fredda', 'secca', 'umida', 'corpo', 'sangue',
            'medicina', 'rimedio', 'virtude', 'natura', 'male', 'dolore',
            'testa', 'ventre', 'petto', 'stomaco', 'fegato', 'rene',
            'prendi', 'mescola', 'cuoci', 'bevi', 'metti', 'lava',
            'mattina', 'sera', 'notte', 'giorno', 'luna', 'sole',
        ],
        'medium_freq': [
            'artemisia', 'malva', 'ruta', 'salvia', 'menta', 'camomilla',
            'rosmarino', 'lavanda', 'assenzio', 'borragine', 'piantaggine',
            'ferita', 'febbre', 'tosse', 'veleno', 'peste', 'apostema',
            'unguento', 'sciroppo', 'decotto', 'impiastro', 'suffumigio',
            'caldo', 'freddo', 'secco', 'umido', 'temperato',
            'primo', 'secondo', 'terzo', 'quarto', 'grado',
        ],
    },
    'german': {
        'high_freq': [
            'krut', 'wurzel', 'blat', 'blume', 'sam', 'wasser', 'oel',
            'pulver', 'warm', 'kalt', 'trucken', 'fucht', 'lip', 'blut',
            'arzenie', 'helfe', 'kraft', 'natur', 'siechtum', 'smerze',
            'houbet', 'buch', 'brust', 'mage', 'leber', 'niere',
            'nim', 'mische', 'siude', 'trinke', 'lege', 'wasche',
            'morgen', 'abent', 'naht', 'tac', 'mane', 'sunne',
        ],
        'medium_freq': [
            'byfuoz', 'pappeln', 'raute', 'salbei', 'minze', 'kamillen',
            'rosmarin', 'lavendel', 'wermut', 'borretsch', 'wegerich',
            'wunde', 'fieber', 'huoste', 'gift', 'pestilenz', 'geswulst',
            'salbe', 'latwerge', 'getranc', 'pflaster', 'rouch',
            'heiz', 'kalt', 'durre', 'feuht', 'getempert',
            'erste', 'ander', 'dritte', 'vierde', 'grat',
        ],
    },
    'spanish': {
        'high_freq': [
            'hierba', 'raiz', 'hoja', 'flor', 'semilla', 'agua', 'aceite',
            'polvo', 'caliente', 'frio', 'seco', 'humedo', 'cuerpo', 'sangre',
            'medicina', 'remedio', 'virtud', 'naturaleza', 'mal', 'dolor',
            'cabeza', 'vientre', 'pecho', 'estomago', 'higado', 'rinon',
            'toma', 'mezcla', 'cuece', 'bebe', 'pon', 'lava',
            'manana', 'noche', 'dia', 'luna', 'sol', 'vez',
        ],
        'medium_freq': [
            'artemisia', 'malva', 'ruda', 'salvia', 'menta', 'manzanilla',
            'romero', 'espliego', 'ajenjo', 'borraja', 'llanten',
            'herida', 'fiebre', 'tos', 'veneno', 'peste', 'apostema',
            'ungento', 'jarabe', 'cocimiento', 'emplasto', 'sahumerio',
            'calido', 'frigido', 'seco', 'humido', 'templado',
            'primero', 'segundo', 'tercero', 'cuarto', 'grado',
        ],
    },
    'hebrew': {
        'high_freq': [
            'esev', 'shoresh', 'aleh', 'perach', 'zera', 'mayim', 'shemen',
            'avak', 'cham', 'kar', 'yavesh', 'lach', 'guf', 'dam',
            'refuah', 'tikkun', 'segulah', 'teva', 'choleh', 'keev',
            'rosh', 'beten', 'chazeh', 'kevah', 'kaved', 'kilyah',
            'kach', 'arev', 'bashal', 'shteh', 'sim', 'rechatz',
            'boker', 'erev', 'lailah', 'yom', 'yareach', 'shemesh',
        ],
        'medium_freq': [
            'laanah', 'chalamit', 'ruta', 'marvah', 'naana', 'babuneg',
            'rozmarin', 'ezov', 'artemisia', 'luach', 'leshon',
            'petza', 'kadachat', 'shiul', 'sam', 'dever', 'shechin',
            'mirchah', 'sharbit', 'tavshil', 'retiyah', 'ketoret',
            'chamim', 'karim', 'yeveshim', 'lechim', 'memuzag',
            'rishon', 'sheni', 'shlishi', 'revii', 'madregah',
        ],
    },
    'arabic': {
        'high_freq': [
            'ushb', 'jidr', 'waraq', 'zahr', 'bazr', 'maa', 'zayt',
            'mashuq', 'haar', 'barid', 'yabis', 'ratib', 'jism', 'dam',
            'dawaa', 'ilaaj', 'khassa', 'tabiah', 'marad', 'alam',
            'raas', 'batn', 'sadr', 'maidah', 'kabid', 'kulyah',
            'khudh', 'ikhlat', 'utbukh', 'ishrab', 'daa', 'ighsil',
            'sabah', 'masaa', 'layl', 'yawm', 'qamar', 'shams',
        ],
        'medium_freq': [
            'shiih', 'khubbazi', 'sadhab', 'maramiyah', 'naana', 'babunaj',
            'iklil', 'khuzama', 'afsantin', 'lisaan', 'ribas',
            'jurh', 'hummaa', 'suaal', 'samm', 'taauun', 'khuraj',
            'marham', 'sharba', 'tabikh', 'libaakh', 'bakhur',
            'saakhin', 'baarid', 'yaabis', 'raatib', 'mutadil',
            'awwal', 'thani', 'thalith', 'rabi', 'darajah',
        ],
    },
    'occitan': {
        'high_freq': [
            'erba', 'raiç', 'fuelha', 'flor', 'semença', 'aiga', 'oli',
            'polva', 'cauda', 'freda', 'seca', 'umida', 'cors', 'sang',
            'medicina', 'remedi', 'vertut', 'natura', 'mal', 'dolor',
            'testa', 'ventre', 'peit', 'estomac', 'fetge', 'ren',
            'pren', 'mescla', 'cos', 'beu', 'met', 'lava',
            'matin', 'ser', 'nueit', 'jorn', 'luna', 'solelh',
        ],
        'medium_freq': [
            'artemisia', 'malva', 'ruda', 'salvia', 'menta', 'camamila',
            'romanin', 'lavanda', 'absinti', 'borrage', 'plantage',
            'plaga', 'febre', 'tos', 'veren', 'peste', 'apostema',
            'ongent', 'sirop', 'decoccion', 'emplastra', 'sufumigacion',
            'caut', 'freg', 'sec', 'umit', 'temperat',
            'primier', 'segon', 'terç', 'quart', 'gra',
        ],
    },
}

REFERENCE_LANGUAGES = list(REFERENCE_VOCABS.keys())


# ---------------------------------------------------------------------------
# Plaintext Generation
# ---------------------------------------------------------------------------

def generate_reference_text(language: str, n_words: int = 500, seed: int = 42) -> str:
    """
    Generate synthetic reference text for a given language.
    Uses weighted sampling from high/medium frequency vocabularies.
    """
    if language not in REFERENCE_VOCABS:
        raise ValueError(f"Unknown language: {language}. "
                        f"Available: {REFERENCE_LANGUAGES}")

    vocab = REFERENCE_VOCABS[language]
    high = vocab['high_freq']
    med = vocab['medium_freq']

    rng = random.Random(seed)
    words = []
    for _ in range(n_words):
        if rng.random() < 0.6:
            words.append(rng.choice(high))
        else:
            words.append(rng.choice(med))

    return ' '.join(words)
