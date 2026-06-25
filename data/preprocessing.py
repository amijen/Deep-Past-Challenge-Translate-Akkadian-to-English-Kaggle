import pandas as pd
import re
from fractions import Fraction
from typing import List

MONTH_ROMAN_MAP = {
    "I": "1","II": "2","III": "3","IV": "4",
    "V": "5","VI": "6","VII": "7","VIII": "8",
    "IX": "9","X": "10","XI": "11","XII": "12"
}
TRANSLATION_REPLACEMENTS = {
    "PN": "<gap>",
    "-gold": "pašallum gold",
    "-tax": "šadduātum tax",
    "-textiles": "kutānum textiles",
}
REMOVE_TRANSLATION = [
    r'\bfem\.', r'\bsing\.', r'\bpl\.', r'\bplural\b'
]

# ========== PreprocessorAkkadian  ==========
class PreprocessorAkkadian:
    FRACTIONS_UNICODE = {
        (1,2):"½",(1,3):"⅓",(2,3):"⅔",(1,4):"¼",(3,4):"¾",
        (1,6):"⅙",(5,6):"⅚",(1,8):"⅛",(3,8):"⅜",(5,8):"⅝",(7,8):"⅞"
    }
    ALLOWED_DENOMS = {2,3,4,6,8}
    REPLACEMENTS = {
        "a2":"á","a3":"à","e2":"é","e3":"è","i2":"í","i3":"ì",
        "u2":"ú","u3":"ù","A2":"Á","A3":"À","E2":"É","E3":"È",
        "I2":"Í","I3":"Ì","U2":"Ú","U3":"Ù",
        "sz":"š","SZ":"Š","s,":"ṣ","S,":"Ṣ",
        "t,":"ṭ","T,":"Ṭ","s'":"ś","S'":"Ś",
        "j":"ŋ","J":"Ŋ","Xx":"xₓ",
        "Ḫ":"H","ḫ":"h",
        "KÙ.B.":"KÙ.BABBAR"
    }

    def __init__(self, aggressive=True):
        self.aggressive = aggressive
        self.patterns = {
            "annotations": re.compile(r'\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)', re.I),
            "repeated_words": re.compile(r'\b(\w+)(?:\s+\1\b)+'),
            "whitespace": re.compile(r'\s+'),
            "punct_space": re.compile(r'\s+([.,:])'),
            "repeated_punct": re.compile(r'([.,])\1+')
        }
        self.subscript_trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉","0123456789")
        self.special_chars_trans = str.maketrans("ḫḪ","hH")
        self.forbidden_chars = "——⌈⌋⌊[]+ʾ/;˹˺"
        self.paren_trans = str.maketrans({'(':'{',')':'}'})
        self.forbidden_trans = str.maketrans('','',self.forbidden_chars)

    @staticmethod
    def replace_gaps(text):
        """Your exact static method, unchanged."""
        if not isinstance(text,str):
            return ""
        patterns = [
            r'\[x\]', r'\bx+\b', r'…+', r'\(break\)', r'\(large break\)',
            r'\(\d+\s+broken lines?\)'
        ]
        for p in patterns:
            text = re.sub(p," <gap> ",text,flags=re.I)
        text = re.sub(r'(<gap>\s*){2,}',"<gap> ",text)
        return re.sub(r'\s+'," ",text).strip()

    @staticmethod
    def collapse_gaps(text):
        return re.sub(r'(<gap>\s*){2,}',"<gap> ",text).strip()

    @staticmethod
    def normalize_determinatives(text):
        text = re.sub(r'\(d\)','{d}',text)
        text = re.sub(r'\(ki\)','{ki}',text)
        text = re.sub(r'\(TÚG\)','TÚG',text)
        return text

    @staticmethod
    def shorten_floats(text):
        def repl(m):
            return f"{float(m.group()):.4f}".rstrip('0').rstrip('.')
        return re.sub(r'\d+\.\d{5,}',repl,text)

    def convert_fraction(self,num_str):
        # your exact method (unchanged)
        try:
            value = float(num_str)
        except:
            return num_str
        if value.is_integer():
            return str(int(value))
        whole = int(value)
        frac = abs(value-whole)
        frac_approx = Fraction(frac).limit_denominator(8)
        if frac_approx.denominator not in self.ALLOWED_DENOMS:
            return num_str
        key = (frac_approx.numerator,frac_approx.denominator)
        symbol = self.FRACTIONS_UNICODE.get(key)
        if symbol is None:
            return num_str
        if whole == 0:
            return symbol
        return f"{whole}{symbol}"

    def replace_general_fractions_in_text(self,text):
        if not isinstance(text,str):
            return text
        def repl(match):
            return self.convert_fraction(match.group(0))
        return re.sub(r'\b\d+\.\d+\b',repl,text)

    def apply_multi_replacements(self,text):
        for old,new in self.REPLACEMENTS.items():
            text = text.replace(old,new)
        return text

    def preprocess_batch(self, translations: List[str]) -> List[str]:
        """
        Optimized version: direct loop over translations, no pandas.
        Every transformation step is identical to the original.
        """
        result = []
        for t in translations:
            # 1. Validate – original replaced invalid with ""
            if not isinstance(t, str) or not t.strip():
                result.append("")
                continue

            # 2. Apply all steps in exact original order
            text = t
            text = self.replace_gaps(text)
            text = self.collapse_gaps(text)
            text = self.normalize_determinatives(text)
            text = self.shorten_floats(text)
            text = text.translate(self.special_chars_trans)
            text = text.translate(self.subscript_trans)
            text = re.sub(self.patterns["whitespace"], " ", text)
            text = text.strip()

            if self.aggressive:
                text = re.sub(self.patterns["annotations"], "", text)
                text = text.replace("<gap>", "GAP")
                text = text.translate(self.paren_trans)
                text = text.translate(self.forbidden_trans)
                text = text.replace("GAP", "<gap>")
                text = self.replace_general_fractions_in_text(text)
                text = self.apply_multi_replacements(text)
                for n in range(4, 1, -1):
                    pattern = r'\b((?:\w+\s+){' + str(n-1) + r'}\w+)(?:\s+\1\b)+'
                    text = re.sub(pattern, r'\1', text)
                text = re.sub(self.patterns["punct_space"], r'\1', text)
                text = re.sub(self.patterns["repeated_punct"], r'\1', text)
                text = re.sub(self.patterns["whitespace"], " ", text)

            result.append(text)
        return result


# ========== PreprocessorEnglish  ==========
class PreprocessorEnglish:
    FRACTIONS_UNICODE = {
        (1,2):"½",(1,3):"⅓",(2,3):"⅔",(1,4):"¼",(3,4):"¾",
        (1,6):"⅙",(5,6):"⅚",(1,8):"⅛",(3,8):"⅜",(5,8):"⅝",(7,8):"⅞"
    }
    ALLOWED_DENOMS = {2,3,4,6,8}

    def __init__(self, aggressive=True):
        self.aggressive = aggressive
        self.patterns = {
            "annotations": re.compile(r'\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)', re.I),
            "repeated_words": re.compile(r'\b(\w+)(?:\s+\1\b)+'),
            "whitespace": re.compile(r'\s+'),
            "punct_space": re.compile(r'\s+([.,:])'),
            "repeated_punct": re.compile(r'([.,])\1+')
        }
        self.forbidden_chars = "()——⌈⌋⌊[]+ʾ/;˹˺"
        self.forbidden_trans = str.maketrans('', '', self.forbidden_chars)

    @staticmethod
    def replace_gaps(text):
        if not isinstance(text,str):
            return ""
        patterns = [
            r'\[x\]', r'\bx+\b', r'…+', r'\(break\)', r'\(large break\)'
        ]
        for p in patterns:
            text = re.sub(p," <gap> ",text,flags=re.I)
        text = re.sub(r'(<gap>\s*){2,}',"<gap> ",text)
        return re.sub(r'\s+'," ",text).strip()

    @staticmethod
    def collapse_gaps(text):
        return re.sub(r'(<gap>\s*){2,}',"<gap> ",text).strip()

    def convert_fraction(self,num_str):
        try:
            value = float(num_str)
        except:
            return num_str
        if value.is_integer():
            return str(int(value))
        whole = int(value)
        frac = abs(value-whole)
        frac_approx = Fraction(frac).limit_denominator(8)
        if frac_approx.denominator not in self.ALLOWED_DENOMS:
            return num_str
        key = (frac_approx.numerator,frac_approx.denominator)
        symbol = self.FRACTIONS_UNICODE.get(key)
        if symbol is None:
            return num_str
        if whole == 0:
            return symbol
        return f"{whole}{symbol}"

    def replace_general_fractions_in_text(self,text):
        if not isinstance(text,str):
            return text
        def repl(match):
            return self.convert_fraction(match.group(0))
        return re.sub(r'\b\d+\.\d+\b',repl,text)

    def normalize_months(self,text):
        if not isinstance(text,str):
            return text
        for roman,num in MONTH_ROMAN_MAP.items():
            text = re.sub(rf'\bmonth\s+{roman}\b',f"month {num}",text,flags=re.I)
        return text

    def clean_translation(self,text):
        if not isinstance(text,str):
            return text
        for p in REMOVE_TRANSLATION:
            text = re.sub(p,"",text,flags=re.I)
        for old,new in TRANSLATION_REPLACEMENTS.items():
            text = text.replace(old,new)
        return text

    def preprocess_batch(self, translations: List[str]) -> List[str]:
        """
        Optimized version: direct loop, no pandas.
        Exactly mirrors the original step sequence.
        """
        result = []
        for t in translations:
            # 1. Validate – original replaced invalid with ""
            if not isinstance(t, str) or not t.strip():
                result.append("")
                continue

            # 2. Apply all steps in exact original order
            text = t
            text = self.replace_gaps(text)
            text = self.collapse_gaps(text)
            text = self.normalize_months(text)
            text = self.clean_translation(text)
            text = text.replace("ד"," ")
            text = re.sub(self.patterns["whitespace"], " ", text)
            text = text.strip()

            if self.aggressive:
                text = re.sub(self.patterns["annotations"], "", text)
                text = text.replace("<gap>", "GAP")
                text = text.translate(self.forbidden_trans)
                text = text.replace("GAP", "<gap>")
                text = self.replace_general_fractions_in_text(text)
                for n in range(4, 1, -1):
                    pattern = r'\b((?:\w+\s+){' + str(n-1) + r'}\w+)(?:\s+\1\b)+'
                    text = re.sub(pattern, r'\1', text)
                text = re.sub(self.patterns["punct_space"], r'\1', text)
                text = re.sub(self.patterns["repeated_punct"], r'\1', text)
                text = re.sub(self.patterns["whitespace"], " ", text)

            result.append(text)
        return result