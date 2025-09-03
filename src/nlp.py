from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any, Optional, Set
import pandas as pd
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span


class NLPProcessor:
    def __init__(self, nlp_model, csv_path: Optional[str] = None):
        self.nlp = nlp_model

        self.symptoms: List[str] = []
        self.diseases: List[str] = []
        self.exams: List[str] = ["electrocardiogram", "ecg", "blood pressure", "heart rate"] 

        self.valid_pairs: Set[Tuple[str, str]] = set()

        if csv_path:
            df = pd.read_csv(csv_path)
            if "symptom" in df.columns and "disease" in df.columns:
                self.symptoms = sorted({str(s).strip() for s in df["symptom"].dropna().astype(str)})
                self.diseases = sorted({str(d).strip() for d in df["disease"].dropna().astype(str)})
                self.valid_pairs = {
                    (str(s).strip().lower(), str(d).strip().lower())
                    for s, d in zip(df["symptom"].astype(str), df["disease"].astype(str))
                    if str(s).strip() and str(d).strip()
                }
            else:
                raise ValueError("CSV precisa conter as colunas 'symptom' e 'disease'.")
        else:
            self.symptoms = [
                "chest pain","chest tightness","angina","shortness of breath","dyspnea",
                "orthopnea","paroxysmal nocturnal dyspnea","palpitations","tachycardia","arrhythmia",
                "fatigue","weakness","dizziness","syncope","lightheadedness","sweating",
                "edema","ankle swelling","nausea","vomiting"
            ]
            self.diseases = [
                "coronary artery disease", "myocardial infarction","angina pectoris","heart failure",
                "hypertension","arrhythmia", "atrial fibrillation","cardiomyopathy","congenital heart disease",
                "peripheral artery disease","stroke","deep vein thrombosis","pulmonary embolism","pericarditis",
                "endocarditis", "aortic aneurysm","aortic dissection",
                "cardiac arrest", "sudden cardiac death","ischemic heart disease"
            ]

        self.abbrev_map: Dict[str, Tuple[str, str]] = {
            "mi": ("myocardial infarction", "DISEASE"),
            "hf": ("heart failure", "DISEASE"),
            "ecg": ("electrocardiogram", "EXAM"),
        }

        self.matcher_sym = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher_dis = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher_exa = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        if self.symptoms:
            self.matcher_sym.add("SYMPTOM", [self.nlp.make_doc(t) for t in self.symptoms])
        if self.diseases:
            self.matcher_dis.add("DISEASE", [self.nlp.make_doc(t) for t in self.diseases])
        if self.exams:
            self.matcher_exa.add("EXAM", [self.nlp.make_doc(t) for t in self.exams])

        self.temporal_patterns = [
            re.compile(r"\b\d+\s+(day|days|hour|hours|week|weeks|month|months)\s+ago\b", re.I),
            re.compile(r"\bsince\s+(yesterday|today|day before yesterday)\b", re.I),
        ]

        self.measurement_patterns = [
            re.compile(r"(blood pressure)\s*[:\-]?\s*(\d{2,3}/\d{2,3})\s*(mmhg)\b", re.I),
            re.compile(r"(heart rate)\s*[:\-]?\s*(\d{2,3})\s*(bpm)\b", re.I),
        ]

        self.negation_cues = {"no", "not", "denies", "denied", "without", "negative"}

    def _match_entities(self, doc: Doc) -> List[Tuple[str, str, Tuple[int, int]]]:
        spans: List[Tuple[int, int, str]] = []
        for _m, s, e in self.matcher_sym(doc):
            spans.append((s, e, "SYMPTOM"))
        for _m, s, e in self.matcher_dis(doc):
            spans.append((s, e, "DISEASE"))
        for _m, s, e in self.matcher_exa(doc):
            spans.append((s, e, "EXAM"))

        span_objs: List[Span] = [Span(doc, s, e, label=label) for (s, e, label) in spans]
        span_objs.sort(key=lambda sp: (-(sp.end - sp.start), sp.start))

        final_spans: List[Span] = []
        occupied = set()
        for sp in span_objs:
            if any(i in occupied for i in range(sp.start, sp.end)):
                continue
            for i in range(sp.start, sp.end):
                occupied.add(i)
            final_spans.append(sp)

        return [(sp.text, sp.label_, (sp.start, sp.end)) for sp in final_spans]

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []

        doc: Doc = self.nlp(text)

        matched = self._match_entities(doc)

        converted_from_matcher: List[Tuple[str, str]] = []
        for (t, l, _pos) in matched:
            low = t.lower()
            if low in self.abbrev_map:
                full_term, mapped_label = self.abbrev_map[low]
                converted_from_matcher.append((full_term, mapped_label))
            else:
                converted_from_matcher.append((t, l))

        expanded_from_tokens: List[Tuple[str, str]] = []
        for tok in doc:
            low = tok.text.lower()
            if low in self.abbrev_map:
                full_term, label = self.abbrev_map[low]
                expanded_from_tokens.append((full_term, label))

        all_ents: List[Tuple[str, str]] = converted_from_matcher + expanded_from_tokens
        out: List[Tuple[str, str]] = []
        seen = set()
        for t, l in all_ents:
            key = (t.lower(), l)
            if key not in seen:
                seen.add(key)
                out.append((t, l))
        return out

    def extract_entities_with_negation(self, text: str) -> List[Tuple[str, str, bool]]:
        ents = self.extract_entities(text)
        if not ents:
            return []

        doc = self.nlp(text)
        lowered = text.lower()
        results: List[Tuple[str, str, bool]] = []

        for ent_text, ent_label in ents:
            is_neg = False
            try:
                start_idx = lowered.index(ent_text.lower())
            except ValueError:
                start_idx = -1

            if start_idx >= 0:
                tok_start = None
                for t in doc:
                    if t.idx == start_idx:
                        tok_start = t.i
                        break
                if tok_start is not None:
                    window_start = max(0, tok_start - 4)
                    for t in doc[window_start:tok_start]:
                        if t.text.lower() in self.negation_cues:
                            is_neg = True
                            break
            else:
                for sent in doc.sents:
                    if ent_text.lower() in sent.text.lower():
                        if any(t.text.lower() in self.negation_cues for t in sent):
                            is_neg = True
                        break

            results.append((ent_text, ent_label, is_neg))
        return results

    def extract_temporal_information(self, text: str) -> List[str]:
        if not text:
            return []
        found = []
        for pat in self.temporal_patterns:
            for m in pat.finditer(text):
                found.append(m.group(0))
        return found

    def extract_measurements(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        res: List[Dict[str, Any]] = []
        for pat in self.measurement_patterns:
            for m in pat.finditer(text):
                res.append({
                    "name": m.group(1),
                    "value": m.group(2),
                    "unit": m.group(3)
                })
        return res

    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        if not text:
            return []

        doc = self.nlp(text)
        rels: List[Tuple[str, str, str]] = []

        for sent in doc.sents:
            sent_text = sent.text
            ents_in_sent = self.extract_entities(sent_text)
            symptoms_in_sent = {t.lower(): t for (t, lbl) in ents_in_sent if lbl == "SYMPTOM"}
            diseases_in_sent = {t.lower(): t for (t, lbl) in ents_in_sent if lbl == "DISEASE"}

            for s_low, s_text in symptoms_in_sent.items():
                for d_low, d_text in diseases_in_sent.items():
                    if (s_low, d_low) in self.valid_pairs:
                        rels.append((s_text, "RELATED_TO", d_text))

        seen = set()
        out = []
        for a, r, b in rels:
            key = (a.lower(), r, b.lower())
            if key not in seen:
                seen.add(key)
                out.append((a, r, b))
        return out

    def _normalize_abbrev(self, text_piece: str) -> str:
        low = text_piece.strip().lower()
        if low in self.abbrev_map:
            return self.abbrev_map[low][0]
        return text_piece.strip()
