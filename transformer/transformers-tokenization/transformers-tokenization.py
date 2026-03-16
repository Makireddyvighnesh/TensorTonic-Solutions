from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}

        # Fixed special tokens
        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = 4
    

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.
        """

        words = set()

        for text in texts:
            for word in text.split():
                words.add(word)

        for word in words:
            if word not in self.word_to_id:
                idx = self.vocab_size
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
                self.vocab_size += 1


    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Unknown words -> <UNK>
        """

        ids = []

        for word in text.split():
            ids.append(self.word_to_id.get(word, self.word_to_id["<UNK>"]))

        return ids


    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        """

        words = []

        for i in ids:
            words.append(self.id_to_word.get(i, "<UNK>"))

        return " ".join(words)