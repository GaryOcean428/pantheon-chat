from typing import Dict, List, Tuple
import logging
from .base import BaseCoordizer
import sys
import os
sys.path.append(os.path.abspath("../vex/kernel"))
from coordizer_v2.coordizer import CoordizerV2

logger = logging.getLogger(__name__)

class V2CoordizerWrapper(BaseCoordizer):
    def __init__(self, coordizer: CoordizerV2):
        self.coordizer = coordizer
        super().__init__()
        
    def encode(self, text: str) -> List[int]:
        return self.coordizer.encode(text)
        
    def decode(self, tokens: List[int]) -> str:
        return self.coordizer.decode(tokens)
        
    def coordize(self, text: str):
        return self.coordizer.coordize(text)
        
    @property
    def vocab(self):
        return self.coordizer.bank
        
    def add_vocabulary_observations(self, observations):
        return 0, False
        
    def get_stats(self):
        return {"vocab_size": self.coordizer.vocab_size}
