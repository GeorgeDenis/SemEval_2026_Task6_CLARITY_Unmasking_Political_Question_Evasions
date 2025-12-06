




# 3 The RNN/LSTM Era: from sequential models to deep contextual understanding

Before the Transformer architecture reshaped modern NLP, recurrent models dominated nearly every task involving ordered text. The intuition was straightforward: language unfolds over time, so a model that processes information sequentially should, at least in theory, possess the right inductive bias. Classical RNNs attempted to encode this evolving structure by updating a hidden state at each timestep. Formally, the recurrence takes the form:

$$h_t = f(W_x x_t + W_h h_{t-1} + b),$$

where $h_t$ represents the hidden state at time $t$, $x_t$ is the input word embedding, and $f$ is a nonlinear activation function.

The main limitation of this architecture, already recognized in the early 2000s, was its vulnerability to the vanishing and exploding gradient problem. This effectively restricted the model's ability to remember long-range dependencies—an especially damaging weakness for political discourse, where evasive strategies can involve topic shifts, subtle detours, or delayed answers occurring many tokens after the original question.

## 3.1 Word Embeddings: Word2Vec and GloVe

The rise of distributed representations provided the first major leap. Instead of relying on sparse, brittle bag-of-words features, models began using word embeddings to capture semantic and syntactic regularities. Word2Vec's skip-gram objective, for example, maximized the probability of observing a context window around a given word:

$$\max_\theta \sum_{t=1}^{T} \sum_{w_c \in C_t} \log p(w_c \mid w_t),$$

where $C_t$ is the context of token $w_t$. These embeddings allowed models to grasp analogical relations and thematic similarities. For evasion detection, this meant being able to recognize when a reply drifted toward generalities ("prosperity", "leadership") rather than addressing the concrete substance of a question.

However, embeddings were static: the vector for *bank* could not adapt to distinguish a financial institution from the side of a river. In political interviews, where ambiguity is strategic, this lack of contextual sensitivity was a serious bottleneck.

## 3.2 LSTM and BiLSTM Models

LSTMs were introduced to mitigate the deficiencies of standard RNNs. Their gating mechanisms—input, forget, and output gates—regulated the flow of information through time. The cell state update can be summarized as:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,$$

where $f_t$ and $i_t$ are the forget and input gates, respectively. The output gate $o_t$ does not contribute to the cell-state update; instead, it determines how much of the internal memory becomes externally visible through the hidden state, computed as:

$$h_t = o_t \odot \tanh(c_t).$$

This separation between cell-state dynamics and output gating allows gradients to propagate more effectively across long sequences while maintaining a stable internal memory representation.

Bi-directional variants (BiLSTMs) further improved performance by processing sequences from both directions. This is particularly relevant in political interviews: the meaning of a candidate's reply often becomes clearer only after reading later clauses that subtly contradict or reframe earlier statements.

Despite these advantages, LSTMs had fundamental limitations. Their sequential nature made them slow to train and evaluate, and their capacity to track discourse-level phenomena remained inherently constrained. As answers grew longer or more rhetorically complex, the models' performance plateaued.

Although these architectures were initially developed for broad NLP tasks, they became directly relevant when Thomas et al. (2024) introduced the “I Never Said That” dataset and the associated taxonomy for response clarity in political interviews. The dataset formalizes the phenomenon of equivocation by labeling each question–answer pair along two axes: a coarse distinction between clear replies, ambivalent replies, and clear non-replies, and a fine-grained set of nine evasion strategies. Early attempts to model this task relied on sequential encoders such as LSTMs or BiLSTMs, which could capture local coherence but often failed to track the deeper rhetorical maneuvers that unfold across several clauses. This gap between linguistic behavior and model capacity ultimately motivated the shift toward architectures capable of richer contextual reasoning.

# 4 Transformers and DeBERTa: The current State of the Art

The clarity‐classification problem introduced by Thomas et al. is inherently relational: an answer cannot be judged in isolation, but must be evaluated with respect to the expectations set by the question. This makes the task particularly well aligned with the strengths of Transformer architectures, which excel at modeling long-range dependencies and subtle mismatches between linguistic segments. In the CLARITY dataset, many instances of evasion are not marked by explicit refusal but by gradual drift, selective omission, or reframing—patterns that require the model to attend jointly to both sides of the interaction. Transformers, through their self-attention mechanism, offer precisely the representational flexibility needed to capture these dynamics.

The introduction of the Transformer architecture marked a decisive shift in NLP. Instead of relying on recurrence, Transformers use attention mechanisms to relate any two positions in a sequence directly. This allows them to capture global interactions, making them ideally suited for analyzing question–answer pairs in political discourse.

## 4.1 The Self-Attention mechanism

Self-attention computes contextualized representations for each token by comparing it to all others in the sequence. Given queries (Q), keys (K), and values (V), the mechanism is defined as:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

This formulation allows the model to identify when an answer aligns with a question, when it avoids the topic altogether, and when it introduces unrelated information as a distraction. Such relational patterns are the hallmark of evasive strategies.

BERT demonstrated that bidirectional attention is critical for tasks involving deep semantic inference. Because the meaning of a political answer depends on both what is said and what is deliberately omitted, the bidirectional nature of Transformers is particularly advantageous.

## 4.2 DeBERTa and Disentangled Attention

DeBERTa introduced an important refinement: disentangling content and positional information. While BERT fuses both types of information in a single embedding, DeBERTa treats them separately, enabling more nuanced attention scoring. The core mechanism can be written as:

$$\alpha_{ij} = (q_i^c + q_i^p)(k_j^c + k_j^p)^\top,$$

where $q_i^c$ and $k_j^c$ encode content, while $q_i^p$ and $k_j^p$ represent relative positional offsets.

This design allows the model to reason more effectively about discourse structure. For example, political answers often shift from direct responses to broad commentary. Being able to separate position from meaning helps the model detect when the speaker is drifting away from the informational target of the question.

## 4.3 Cross-encoder architectures for answer classification

For classification tasks involving question–answer pairs, the most effective Transformer-based design is the cross-encoder. Instead of encoding the question and answer independently, the model ingests them jointly as a single sequence:

```
[CLS] Question [SEP] Answer [SEP]
```

The **[CLS]** token ("classification") is a synthetic token inserted at the beginning of every input sequence; its final hidden representation is used by the classifier as a summary embedding of the entire question–answer pair.  
The **[SEP]** token ("separator") is used to mark boundaries between segments, allowing the model to distinguish where the question ends and the answer begins.  

This allows every token in the question to attend to every token in the answer, and vice versa. The resulting representation captures deviations, inconsistencies, and evasive patterns much more reliably than separate encoders could. Empirically, cross-encoders consistently outperform bi-encoders in tasks requiring relational understanding, which is precisely the core requirement of clarity and evasion classification.

This joint encoding is essential for the CLARITY taxonomy, because the distinction between a Clear Non-Reply and an Ambivalent Reply, or between Deflection and General Evasion, hinges on how the content of the answer diverges from the informational target of the question. A cross-encoder does not merely encode the answer; it implicitly learns the rhetorical relationship between the two segments, mirroring the judgment process followed by annotators in the dataset.  





# 6 Comparative Summary and Conclusions

The baselines reported in the original “I Never Said That” study reinforce this developmental arc. Traditional linear models trained on handcrafted features achieved only modest performance, especially on the nine-category evasion task. LSTM-based models improved the situation slightly but struggled with the nuanced boundary cases that define political communication. The most substantial gains emerged with Transformer-based models, particularly cross-encoders, whose performance provided the first strong baselines for both levels of the taxonomy.  

The evolution of models for response clarity classification reflects broader trends in NLP. Early approaches depended heavily on feature engineering and statistical classifiers such as SVMs and Naive Bayes. Their inability to model semantic nuance or discourse coherence made them ill-suited for detecting political evasion.

RNNs and LSTMs extended capability by enabling sequential modeling and capturing patterns over time. They were a significant improvement but still struggled with context length and global reasoning. Political discourse frequently involves rhetorical devices whose interpretation depends on distant parts of the answer, something LSTMs inherently handle poorly.

Transformers, with their global self-attention operations, finally broke through these limitations. Their ability to model long-range dependencies, combined with deeper contextual understanding, made them the natural choice for tasks requiring fine-grained interpretation of question–answer relationships. Architectures like DeBERTa further enhanced this capability through improved attention mechanisms that separate content and positional information.

Taken together, these observations show that the CLARITY task is not merely a classification problem but a test of a model’s ability to track pragmatic intent across a dialogue turn. The dataset captures a spectrum of evasive behaviors that cannot be detected through surface-level similarity alone, and the steady progression from RNNs to modern Transformer architectures illustrates how representational depth translates directly into performance on such subtle, discourse-driven tasks.  

A synthetic comparison of model families can be summarized as follows:

| Model Family                    | Strengths                                            | Limitations                                | Typical Performance Range |
| ------------------------------- | ---------------------------------------------------- | ------------------------------------------ | ------------------------- |
| Classical ML (SVM, NB)          | Fast, interpretable, easy to train                   | No semantic depth, fragile to paraphrasing | ~40–55% F1                |
| RNN / LSTM                      | Sequential modeling, better semantics                | Limited long-range reasoning, slow         | ~55–65% F1                |
| Transformers (BERT)             | Strong contextual learning                           | High computational cost                    | ~70–78% F1                |
| Advanced Transformers (DeBERTa) | Best-in-class reasoning, fine-grained discrimination | Heavy inference cost                       | 80%+ F1                   |

Overall, the State of the Art clearly favors Transformer-based cross-encoder architectures, especially for tasks where subtle deviations between question and answer reveal the speaker's intent. The CLARITY taxonomy, with its hierarchical structure, aligns closely with the multi-layer representations learned by these models, making modern Transformers not just suitable but essential for high-performance evasion classification.
