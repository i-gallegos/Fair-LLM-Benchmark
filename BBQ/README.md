# BBQ
Repository for the Bias Benchmark for QA dataset.

Authors: Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel R. Bowman.

## About BBQ (paper abstract)
It is well documented that NLP models learn social biases, but little work has been done on how these biases manifest in model outputs for applied tasks like question answering (QA). We introduce the Bias Benchmark for QA (BBQ), a dataset of question sets constructed by the authors that highlight attested social biases against people belonging to protected classes along nine social dimensions relevant for U.S. English-speaking contexts. Our task evaluates model responses at two levels: (i) given an under-informative context, we test how strongly responses refect social biases, and (ii) given an adequately informative context, we test whether the model's biases override a correct answer choice. We fnd that models often rely on stereotypes when the context is under-informative, meaning the model's outputs consistently reproduce harmful biases in this setting. Though models are more accurate when the context provides an informative answer, they still rely on stereotypes and average up to 3.4 percentage points higher accuracy when the correct answer aligns with a social bias than when it conficts, with this difference widening to over 5 points on examples targeting gender for most models tested.

## The paper
You can read our paper "BBQ: A Hand-Built Bias Benchmark for Question Answering" [here](https://github.com/nyu-mll/BBQ/blob/main/QA_bias_benchmark.pdf). The paper has been published in the Findings of ACL 2022 [here](https://aclanthology.org/2022.findings-acl.165/).

## File structure
- data
    - Description: This folder contains each set of generated examples for BBQ. This is the folder you would use to test BBQ.
    - Contents: 11 jsonl files, each containing all templated examples. Each category is a separate file.
- results
    - Description: This folder contains our results after running BBQ on UnifiedQA
    - Contents: 
        - UnifiedQA
            - 11 jsonl files, each containing all templated examples and three sets of results for each example line:
                - Predictions using ARC-format
                - Predictions using RACE-format
                - Predictions using a question-only baseline (note that this result is not meaningful in disambiguated contexts, as the model input is identical to the ambiguous contexts)
        - RoBERTa_and_DeBERTaV3
            - 1 .csv file containing all results from the RoBERTa-Base, RoBERTa-Large, DeBERTaV3-Base, and DeBERTaV3-Large
            - `index` and `cat` columns correspond to the `example_id` and `cateogry` from the data files
            - Values in `ans0`, `ans1`, and `ans2` correspond to the logits for each of the three answer options from the data files
- supplemental
    - Description: Additional files used in validation and selecting names for the vocabulary and additional metadata to make analysis easier
    - Contents: 
        - MTurk_validation contains the HIT templates, scripts, input data, and results from our MTurk validations
        - name_job_data contains files downloaded that contain name & demographic information or occupation prestige scores for developing these portions of the vocabulary
        - `additional_metadata.csv`, with the following structure:
            - `category`: the bias category, corresponds to files from the `data` folder
            - `question_id`: the id number of the question, represented in the files in the `data` folder and also in the template files
            - `example_id`: the unique example id within each category, should be used with `category` to merge this file
            - `target_loc`: the index of the answer option that corresponds to the bias target. Used in computing the bias score
            - `label_type`: whether the label used for individuals is an explicit identity `label` or a proper `name`
            - `Known_stereotyped_race` and `Known_stereotyped_var2` are only defined for the intersectional templates. Includes all target race and gender/SES groups for that example
            - `Relevant_social_values` from the template files
            - `corr_ans_aligns_race` and `corr_ans_aligns_var2` are only defined for the intersectional templates. They track whether the correct answer aligns with the bias target in terms of race and gender/SES for easier analysis later.
            - `full_cond` is only defined for the intersectional templates. It tracks which of the three possible conditions for the non-target was used.
            - `Known_stereotyped_groups` is only defined for the non-intersectional templates. Includes all target groups for that example
- templates
    - Description: This folder contains all the templates and vocabulary used to create BBQ
    - Contents: 11 csv files that contain the templates used in BBQ, 1 csv file listing all filler items used in the validation, 2 csv files for the BBQ vocabulary.

## Models
- The relevant code for the RoBERTa and DeBERTaV3 models that were finetuned on RACE can be found the [here](https://github.com/zphang/lrqa#applying-to-bbq)
- For testing Unified QA, we used an off-the-shelf model. String formatting for inference was created by concatenating the following fields from the data files:
    - RACE-style-format: `question + \n + '(a)' + ans_0 + '(b)' + ans_1 + '(c)' + ans2 + \n + context`
    - ARC-style-format: `context + question + \n + '(a)' + ans_0 + '(b)' + ans_1 + '(c)' + ans2`

