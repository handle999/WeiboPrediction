# LLM-based Text Feature Extraction

We use the BERT ([bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)) to extract content features with an initial dimension of 768. To make the features more tractable, we apply PCA to reduce the dimension to 32. This produces bert_f, which is a key component of our content feature set.

At the same time, we try to use QWen3 ([Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)), the results will be discussed in **Results** subsection.

Need to install `transformers`. Note that *bert* and *Qwen* need different versions of `transformers`, which can be get in there huggingface pages. And this can be **really complex**!!! So it is adviced to use bert only.
