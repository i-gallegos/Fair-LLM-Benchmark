================================================================
HONEST: Measuring Hurtful Sentence Completion in Language Models
================================================================


.. image:: https://img.shields.io/pypi/v/honest.svg
        :target: https://pypi.python.org/pypi/honest

.. image:: https://img.shields.io/travis/MilaNLProc/honest.svg
        :target: https://travis-ci.com/MilaNLProc/honest

.. image:: https://readthedocs.org/projects/honest/badge/?version=latest
        :target: https://honest.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://raw.githubusercontent.com/aleen42/badges/master/src/medium.svg
    :target: https://medium.com/towards-data-science/can-too-much-bert-be-bad-for-you-92f0014e099b
    :alt: Medium Blog Post



...


Large language models (LLMs) have revolutionized the field of NLP. However, LLMs capture and proliferate hurtful stereotypes, especially in text generation. We propose **HONEST**, a score to measure hurtful sentence completions in language models. It uses a systematic template- and lexicon-based bias evaluation methodology in six languages (English, Italian, French, Portuguese, Romanian, and Spanish) for binary gender and in English for LGBTQAI+ individuals.

...

See the papers for additional details:

Nozza D., Bianchi F., and Hovy D. "HONEST: Measuring hurtful sentence completion in language models." The 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, 2021. https://aclanthology.org/2021.naacl-main.191

Nozza D., Bianchi F., Lauscher L., and Hovy D. "Measuring Harmful Sentence Completion in Language Models for LGBTQIA+ Individuals." The Second Workshop on Language Technology for Equality, Diversity and Inclusion at the Annual Meeting of the Association for Computational Linguistics 2022. https://aclanthology.org/2022.ltedi-1.4/


Tutorials
---------


.. |colab1_2| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/13iAwHmtdYIAzDt8O5Ldat2vbKz9Ej6PT?usp=sharing
    :alt: Open In Colab
    

.. |colab1_3| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1vVgarK99TVm2QKdaJtJjye1470BD1_Bb?usp=sharing
    :alt: Open In Colab

+--------------------------------------------------------------------------------+------------------+
| Name                                                                           | Link             |
+================================================================================+==================+
| Compute HONEST score with BERT models (+Viz) (stable **v0.2.1**)               | |colab1_2|       |
+--------------------------------------------------------------------------------+------------------+
| Compute HONEST score with GPT models (+Viz) (stable **v0.2.1**)                | |colab1_3|       |
+--------------------------------------------------------------------------------+------------------+


Installing
----------

.. code-block:: bash

    pip install -U honest


Using
-----

.. code-block:: python

    # Load HONEST templates
    evaluator = honest.HonestEvaluator(lang)
    masked_templates = evaluator.templates(data_set="binary") # or "queer_nonqueer" or "all"

    # Load BERT model
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    model = AutoModelForMaskedLM.from_pretrained(name_model)

    # Define nlp_fill pipeline
    nlp_fill = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=k)

    print("FILL EXAMPLE:",nlp_fill('all women like to [M].'.replace('[M]',tokenizer.mask_token)))

    # Fill templates (please check if the filled words contain any special character)
    filled_templates = [[fill['token_str'].strip() for fill in nlp_fill(masked_sentence.replace('[M]',tokenizer.mask_token))] for masked_sentence in masked_templates.keys()]

    honest_score = evaluator.honest(filled_templates)
    print(name_model, k, honest_score)

Citation
--------

Please use the following bibtex entries if you use this score in your project:

::

    @inproceedings{nozza-etal-2021-honest,
        title = {"{HONEST}: Measuring Hurtful Sentence Completion in Language Models"},
        author = "Nozza, Debora and Bianchi, Federico  and Hovy, Dirk",
        booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
        month = jun,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.naacl-main.191",
        doi = "10.18653/v1/2021.naacl-main.191",
        pages = "2398--2406",
    }

    @inproceedings{nozza-etal-2022-measuring,
        title = {Measuring Harmful Sentence Completion in Language Models for LGBTQIA+ Individuals},
        author = "Nozza, Debora and Bianchi, Federico and Lauscher, Anne and Hovy, Dirk",
        booktitle = "Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion",
        publisher = "Association for Computational Linguistics",
        year={2022}
    }

Development Team
----------------

* Federico Bianchi <f.bianchi@unibocconi.it> Bocconi University
* Debora Nozza <debora.nozza@unibocconi.it> Bocconi University
* Dirk Hovy <dirk.hovy@unibocconi.it> Bocconi University

Software Details
----------------

* Free software: MIT license
* Documentation: https://honest.readthedocs.io.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

Note
----

Remember that this is a research tool :)
