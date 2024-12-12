from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=5000, min_frequency=2, show_progress=True)

corpus = [
'i like chocolate',
'i like strawberry',
'i do not like apples',
'i really like oranges',
'i somewhat like grapes',
'i hate avocado',
'i really prefer watermelon'
]

tokenizer.train_from_iterator(corpus, trainer=trainer)
tokenizer.save('bpe_tokenizer.json')

encoded = tokenizer.encode('lowering')
print(encoded.tokens)

encoded = tokenizer.encode('booking')
print(encoded.tokens)

encoded = tokenizer.encode('ball')
print(encoded.tokens)

encoded = tokenizer.encode('i like banks')
print(encoded.tokens)