from typing import Any, Dict, List, Union

from thefuzz import fuzz, process


class FuzzyMatcher:

    def __init__(self, names: List[str]):
        self.names = names

    def get_top_k_matches(self, name: str, k: int) -> List[Dict[str, Union[str, float, int]]]:
        matches = process.extract(name, self.names, scorer=fuzz.ratio, limit=k)
        return [{'name': name, 'matched': matched_name, 'score': score} for matched_name, score in matches]
