from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span  


class NLPProcessor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

        self.symptoms = [
            "dor no peito", "falta de ar", "dispneia", "palpitações",
            "cansaço", "sudorese", "febre", "dor torácica", "dor de cabeça"
        ]
        self.diseases = [
            "infarto agudo do miocárdio", "cardiopatia", "insuficiência cardíaca",
            "infarto", "pneumonia", "virose", "diabetes", "diabetes mellitus",
            "hipertensão", "hipertensão arterial", "hipertensão arterial não controlada",
            "embolia pulmonar", "cardiomiopatia", "avc", "arritmia cardíaca"
        ]
        self.exams = ["eletrocardiograma", "ecg","pressão arterial", "frequência cardíaca"]

        self.abbrev_map = {
            "iam": ("infarto agudo do miocárdio", "DOENÇA"),
            "ecg": ("ecg", "EXAME"),
        }

        self.matcher_sym = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher_dis = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher_exa = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        self.matcher_sym.add("SYM", [self.nlp.make_doc(t) for t in self.symptoms])
        self.matcher_dis.add("DIS", [self.nlp.make_doc(t) for t in self.diseases])
        self.matcher_exa.add("EXA", [self.nlp.make_doc(t) for t in self.exams])

        self.temporal_patterns = [
            re.compile(r"\bhá\s+\d+\s+(dia|dias|hora|horas|semana|semanas|mês|meses)\b", re.I),
            re.compile(r"\bdesde\s+(ontem|hoje|anteontem)\b", re.I),
        ]

        self.measurement_patterns = [
            re.compile(r"(pressão arterial)\s*[:\-]?\s*(\d{2,3}/\d{2,3})\s*(mmhg)\b", re.I),
            re.compile(r"(frequência cardíaca)\s*[:\-]?\s*(\d{2,3})\s*(bpm)\b", re.I),
        ]

        self.negation_cues = {"não", "nao", "nega", "negou", "sem"}

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []

        doc: Doc = self.nlp(text)
        spans: List[Tuple[int, int, str]] = []

        for _m, s, e in self.matcher_sym(doc):
            spans.append((s, e, "SINTOMA"))
        for _m, s, e in self.matcher_dis(doc):
            spans.append((s, e, "DOENÇA"))
        for _m, s, e in self.matcher_exa(doc):
            spans.append((s, e, "EXAME"))

        # Abreviações literais (IAM, ECG)
        for tok in doc:
            low = tok.text.lower()
            if low in self.abbrev_map:
                spans.append((tok.i, tok.i + 1, self.abbrev_map[low][1]))

        # cria spans (spaCy 3.x): usar construtor Span
        span_objs: List[Span] = [Span(doc, s, e, label=label) for (s, e, label) in spans]

        # resolver overlaps: maior comprimento primeiro
        span_objs.sort(key=lambda sp: (-(sp.end - sp.start), sp.start))

        final: List[Tuple[str, str]] = []
        occupied = set()
        for sp in span_objs:
            if any(i in occupied for i in range(sp.start, sp.end)):
                continue
            for i in range(sp.start, sp.end):
                occupied.add(i)
            final.append((sp.text, sp.label_))

        # dedup (case-insensitive no texto, mas preserva a forma original)
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
                # fallback por sentença
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
                res.append({"name": m.group(1), "value": m.group(2), "unit": m.group(3)})
        return res

    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        if not text:
            return []

        doc = self.nlp(text)
        rels: List[Tuple[str, str, str]] = []

        for sent in doc.sents:
            stext = sent.text.strip()

            # Mantém uma "doença recente" para casos de "que por sua vez"
            last_disease = None

            # Padrão 1: "<S> é (um)? sintoma ... de/da/do/das/dos <D>"
            m = re.search(
            r"\b(?:[AaOo]s?\s+)?(?P<S>[^.,;]+?)\s+é\s+(?:um\s+)?sintoma(?:\s+\w+)?\s+(?:de\s+(?:[AaOo]s?\s+)?|da\s+|do\s+|das\s+|dos\s+)(?P<D>[^.,;]+)",stext, flags=re.I)
            if m:
                S = m.group("S").strip()
                D = m.group("D").strip()
                S = self._normalize_abbrev(S)
                D = self._normalize_abbrev(D)
                # remove artigo inicial se ficou
                S = re.sub(r"^(?:[AaOo]s?\s+)", "", S).strip()
                D = re.sub(r"^(?:[AaOo]s?\s+)", "", D).strip()
                rels.append((S, "É_SINTOMA_DE", D))
                last_disease = D

            # Padrão 2: "<C> pode causar <E>"
            m = re.search(
                r"\b(?:[AaOo]s?\s+)?(?P<C>[^.,;]+?)\s+pode\s+causar\s+(?P<E>[^.,;]+)",
                stext, flags=re.I
            )
            if m:
                C = m.group("C").strip()
                E = m.group("E").strip()
                C = self._normalize_abbrev(C)
                E = self._normalize_abbrev(E)
                C = re.sub(r"^(?:[AaOo]s?\s+)", "", C).strip()
                E = re.sub(r"^(?:[AaOo]s?\s+)", "", E).strip()
                # Heurística "que por sua vez"
                if C.lower().startswith("que"):
                    C = last_disease or C
                rels.append((C, "PODE_CAUSAR", E))

            # Padrão 3: "<C> pode gerar <E>"
            m = re.search(
                r"\b(?:[AaOo]s?\s+)?(?P<C2>[^.,;]+?)\s+pode\s+gerar\s+(?P<E2>[^.,;]+)",
                stext, flags=re.I
            )
            if m:
                C2 = m.group("C2").strip()
                E2 = m.group("E2").strip()
                C2 = self._normalize_abbrev(C2)
                E2 = self._normalize_abbrev(E2)
                C2 = re.sub(r"^(?:[AaOo]s?\s+)", "", C2).strip()
                E2 = re.sub(r"^(?:[AaOo]s?\s+)", "", E2).strip()
                if C2.lower().startswith("que"):
                    C2 = last_disease or C2
                rels.append((C2, "PODE_GERAR", E2))

            # Padrão 4: "<X> confirmou (a)? (suspeita de)? <D>"
            m = re.search(
                r"\b(?:[AaOo]s?\s+)?(?P<X>[^.,;]+?)\s+confirmou\s+(?:a\s+)?(?:suspeita\s+de\s+)?(?P<D2>[^.,;]+)",
                stext, flags=re.I
            )
            if m:
                X = m.group("X").strip()
                D2 = m.group("D2").strip()
                X = self._normalize_abbrev(X)
                D2 = self._normalize_abbrev(D2)
                X = re.sub(r"^(?:[AaOo]s?\s+)", "", X).strip()
                D2 = re.sub(r"^(?:[AaOo]s?\s+)", "", D2).strip()
                rels.append((X, "CONFIRMA", D2))

        # dedup
        seen = set()
        out = []
        for r in rels:
            key = (r[0].strip(), r[1], r[2].strip())
            if key not in seen:
                seen.add(key)
                out.append((r[0].strip(), r[1], r[2].strip()))
        return out

    def _normalize_abbrev(self, text_piece: str) -> str:
        low = text_piece.strip().lower()
        if low in self.abbrev_map:
            return self.abbrev_map[low][0]
        return text_piece
