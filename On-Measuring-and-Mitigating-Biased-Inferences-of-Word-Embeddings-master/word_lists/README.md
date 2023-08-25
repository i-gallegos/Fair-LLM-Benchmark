Instruction
```
usage: generate_templates.py [-h] [--noun] [--adj] --p PREMISE_TYPE --h HYP_TYPE --output OUT_FILE

Expand templates into a set of premise-hypothesis pairs and write the result into a CSV file.

optional arguments:
  -h, --help         show this help message and exit
  --noun             Use noun templates. With this argument, the premise and hypothesis types should be nouns.
  --adj              Use adjective templates. With this argument, the premise and hypothesis types should be adjectives.
  --p PREMISE_TYPE   Premise word type. If noun templates are used, this argument should be one of [person_hyponyms polarized_nouns occupations
                     gendered_words rulers religion_demonyms country_demonyms]. If adjective templates are used, this argument should be one of [religions
                     countries able_adjectives age_adjectives polarized_adjectives color hair_adjectives physical_adjectives ethnicities class].
  --h HYP_TYPE       Hypothesis word type. If noun templates are used, this argument should be one of [person_hyponyms polarized_nouns occupations
                     gendered_words rulers religion_demonyms country_demonyms]. If adjective templates are used, this argument should be one of [religions
                     countries able_adjectives age_adjectives polarized_adjectives color hair_adjectives physical_adjectives ethnicities class].
  --output OUT_FILE  The name of the output CSV file
  ```

  This will produce a file whose first column is the id and the last two columns are p, h pairs. Can you write a script uses such a file and produces an output csv file with the following columns
  ```
  id,entail_probability,contradict_probability,neutral_probability
  ```