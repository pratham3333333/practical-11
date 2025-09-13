
# 📘 NLP Practical 11 – Advanced NLP

## **11(a) Multiword Expressions in NLP**

### 🎯 Aim

To identify and process Multiword Expressions (MWEs) like "New York" or "machine learning" using `nltk.MWETokenizer`.

---

### 💻 Code

```python
import nltk
from nltk.tokenize import MWETokenizer, word_tokenize

# Define MWE tokenizer
tokenizer = MWETokenizer([('New', 'York'), ('machine', 'learning')])

# Input text
text = "I live in New York and I love machine learning applications."

# Without MWE
print("Without MWE:", word_tokenize(text))

# With MWE
print("With MWE:", tokenizer.tokenize(word_tokenize(text)))
```

---

### 📤 Expected Output

```
Without MWE: ['I', 'live', 'in', 'New', 'York', 'and', 'I', 'love', 'machine', 'learning', 'applications', '.']
With MWE: ['I', 'live', 'in', 'New_York', 'and', 'I', 'love', 'machine_learning', 'applications', '.']
```

---

### ✅ Conclusion

MWEs group meaningful phrases as single tokens, improving semantic understanding in NLP tasks.

---

## **11(b) Normalized Web Distance & Word Similarity**

### 🎯 Aim

To compute semantic similarity between words using **WordNet** and **Wu-Palmer similarity**.

---

### 💻 Code

```python
from nltk.corpus import wordnet

# Get synsets
car = wordnet.synset('car.n.01')
automobile = wordnet.synset('automobile.n.01')
bus = wordnet.synset('bus.n.01')

# Similarities
print("car vs automobile:", car.wup_similarity(automobile))
print("car vs bus:", car.wup_similarity(bus))
```

---

### 📤 Expected Output

```
car vs automobile: 1.0
car vs bus: 0.67
```

---

### ✅ Conclusion

* *car* and *automobile* are identical (similarity = **1.0**).
* *car* and *bus* are somewhat related (similarity = **0.67**).
* WordNet similarity helps in semantic closeness detection.

---

## **11(c) Word Sense Disambiguation (WSD)**

### 🎯 Aim

To disambiguate the correct sense of a word based on its context using the **Lesk algorithm**.

---

### 💻 Code

```python
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

# Sentence with ambiguous word "bank"
sentence1 = "He deposited money in the bank."
sentence2 = "The fisherman sat on the bank of the river."

# Apply WSD
print("Sentence 1:", lesk(word_tokenize(sentence1), 'bank'))
print("Sentence 2:", lesk(word_tokenize(sentence2), 'bank'))
```

---

### 📤 Expected Output

```
Sentence 1: Synset('depository_financial_institution.n.01')
Sentence 2: Synset('bank.n.01')
```

---

### ✅ Conclusion

* In **Sentence 1**, "bank" = *financial institution*.
* In **Sentence 2**, "bank" = *river edge*.
* Lesk algorithm chooses the correct meaning based on context.

---
