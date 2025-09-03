from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span  


class NLPProcessor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

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

        self.exams = [
            "electrocardiogram", "ecg", "blood pressure", "heart rate",
            "electrocardiogram"
        ]

        self.abbrev_map = {
            "mi": ("myocardial infarction", "DISEASE"),
            "ecg": ("electrocardiogram", "EXAM"),
        }

        self.matcher_sym = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher_dis = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher_exa = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        self.matcher_sym.add("SYMPTOM", [self.nlp.make_doc(t) for t in self.symptoms])
        self.matcher_dis.add("DISEASE", [self.nlp.make_doc(t) for t in self.diseases])
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

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []

        doc: Doc = self.nlp(text)
        spans: List[Tuple[int, int, str]] = []

        for _m, s, e in self.matcher_sym(doc):
            spans.append((s, e, "SYMPTOM"))
        for _m, s, e in self.matcher_dis(doc):
            spans.append((s, e, "DISEASE"))
        for _m, s, e in self.matcher_exa(doc):
            spans.append((s, e, "EXAM"))

        for tok in doc:
            low = tok.text.lower()
            if low in self.abbrev_map:
                full_term, label = self.abbrev_map[low]
                spans.append((tok.i, tok.i + 1, label))

        span_objs: List[Span] = [Span(doc, s, e, label=label) for (s, e, label) in spans]

        span_objs.sort(key=lambda sp: (-(sp.end - sp.start), sp.start))

        final: List[Tuple[str, str]] = []
        occupied = set()
        for sp in span_objs:
            if any(i in occupied for i in range(sp.start, sp.end)):
                continue
            for i in range(sp.start, sp.end):
                occupied.add(i)
            final.append((sp.text, sp.label_))

        seen = set()
        out: List[Tuple[str, str]] = []
        for ent in final:
            key = (ent[0].lower(), ent[1])
            if key not in seen:
                seen.add(key)
                out.append(ent)
                
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
                # Find the token that starts at this position
                tok_start = None
                for t in doc:
                    if t.idx == start_idx:
                        tok_start = t.i
                        break
                        
                if tok_start is not None:
                    # Check previous tokens for negation cues
                    window_start = max(0, tok_start - 4)
                    for t in doc[window_start:tok_start]:
                        if t.text.lower() in self.negation_cues:
                            is_neg = True
                            break
            else:
                # Fallback: check the whole sentence
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
            stext = sent.text.strip()
            last_disease = None

            # Pattern 1: "<S> is (a)? (adjs)* symptom of <D>"
            m = re.search(
                r"\b(?P<S>[^.,;]+?)\s+is\s+(?:a\s+)?(?:\w+\s+)*symptom\s+of\s+(?P<D>[^.,;]+)",
                stext, flags=re.I
            )
            if m:
                S = m.group("S").strip()
                D = m.group("D").strip()
                S = self._normalize_abbrev(S)
                D = self._normalize_abbrev(D)
                # remove artigos
                S = re.sub(r"^(the|a|an)\s+", "", S, flags=re.I)
                D = re.sub(r"^(the|a|an)\s+", "", D, flags=re.I)
                rels.append((S, "IS_SYMPTOM_OF", D))
                last_disease = D

            # Pattern 2: "<C> can cause <E>"
            m = re.search(
                r"\b(?P<C>[^.,;]+?)\s+can\s+cause\s+(?P<E>[^.,;]+)",
                stext, flags=re.I
            )
            if m:
                C = m.group("C").strip()
                E = m.group("E").strip()
                C = self._normalize_abbrev(C)
                E = self._normalize_abbrev(E)
                if C.lower().startswith("which"):
                    C = last_disease or C
                C = re.sub(r"^(the|a|an)\s+", "", C, flags=re.I)
                E = re.sub(r"^(the|a|an)\s+", "", E, flags=re.I)
                rels.append((C, "CAN_CAUSE", E))

        # Pattern 3: "<C> can generate <E>"
        m = re.search(
            r"\b(?P<C>[^.,;]+?)\s+can\s+generate\s+(?P<E>[^.,;]+)",
            stext, flags=re.I
        )
        if m:
            C = m.group("C").strip()
            E = m.group("E").strip()
            C = self._normalize_abbrev(C)
            E = self._normalize_abbrev(E)
            if C.lower().startswith("which"):
                C = last_disease or C
            C = re.sub(r"^(the|a|an)\s+", "", C, flags=re.I)
            E = re.sub(r"^(the|a|an)\s+", "", E, flags=re.I)
            rels.append((C, "CAN_GENERATE", E))

        # Pattern 4: "<X> confirmed (the)? (suspicion of)? <D>"
        m = re.search(
            r"\b(?P<X>[^.,;]+?)\s+confirmed\s+(?:the\s+)?(?:suspicion\s+of\s+)?(?P<D>[^.,;]+)",
            stext, flags=re.I
        )
        if m:
            X = m.group("X").strip()
            D = m.group("D").strip()
            X = self._normalize_abbrev(X)
            D = self._normalize_abbrev(D)
            X = re.sub(r"^(the|a|an)\s+", "", X, flags=re.I)
            D = re.sub(r"^(the|a|an)\s+", "", D, flags=re.I)
            rels.append((X, "CONFIRMS", D))

        # Remove duplicates (case-insensitive)
        seen = set()
        out = []
        for r in rels:
            key = (r[0].strip().lower(), r[1], r[2].strip().lower())
            if key not in seen:
                seen.add(key)
                out.append(r)

        return out

    def _normalize_abbrev(self, text_piece: str) -> str:
        clean = text_piece.strip()
        low = clean.lower()
        if low in self.abbrev_map:
            return self.abbrev_map[low][0]
        return clean  # mantém capitalização original
