# RedditBias

This repository contains the code and data for bias evaluation with *RedditBias* (to appear at ACL21).  The code for the debiasing approaches and the conversational downstream evaluation can be found here: https://github.com/umanlp/redditbias_debias_conv_ai.

## Privacy & Ethics
RedditBias is created from real-world conversations. To protect the users whose comments are included in our data set, we have removed all identifying information, e.g., user names, and kept only the text needed for our analysis. However, if you find your text in our data set and you feel misrepresented being included in this data set, please reach out to us with the following information: comment to be removed & reddit username. Thank you!

## How to Use this Code
For bias evaluation with RedditBias, please use Evaluation/measure_bias.py. The rest of the code you can find in this repository documents the data set creation and offers other useful functions.

### Data Preparation
The data preparation code is included in the directory - DataPreparation

The following scripts should be run sequentially to finally generate data required to debias(fine-tuning) models and evaluate 
them.

- DataPreparation/reddit_data.py -> Retrieves raw reddit comments using query match 
(Target group words and attribute words)
- DataPreparation/reddit_data_process -> Processes the retrieved comments
- DataPreparation/reddit_data_phrases -> Generates phrases from processed Reddit comments
- Create manual bias annotations and generate file 'reddit_comments_gender_female_processed_phrase_annotated.csv'
- DataPreparation/reddit_data_phrases_replace_target.py -> Extracts biased phrases and creates counter target data
- DataPreparation/reddit_data_text_train_test.py -> Creates train test split of biased phrases
- evaluation/measure_bias.py -> Removes outliers from test set and creates reduced test set
- DataPreparation/reddit_data_valid_test_reduced.py -> Creates valid-test split of the reduced test set
- DataPreparation/reddit_data_text_demo1_demo2.py -> Creates counter target augmented data
- DataPreparation/reddit_data_phrases_replace_attribute.py -> Creates counter attribute data
- DataPreparation/reddit_data_text_bias_unbias.py -> Creates test files of counter attribute augmented data

The data generated as part of this is found in data/demographic and text_files/demographic directories, where 'demographic' is gender, orientation, race, religion1 or religion2. The txt files in folder text_files/ are used for train, validation and evaluation during fine-tuning the DialoGPT model using Debiasing methods.

A brief description of files in data/religion1 is:

- **religion2_muslims.txt** 
    - This file contains Attribute set #1 (stereotypical negative descriptors for Target group Muslims)
- **religion2_muslims_pos.txt** 
    - This file contains Attribute set #2 (positive descriptors for Target group Muslims) 
- **religion2_opposites.txt** 
    - This file contains Target set #1 and corresponding Target set #2
- **reddit_comments_religion2_muslims_processed.csv** 
    - Pre-processed version of original Reddit comments
- **reddit_comments_religion2_muslims_processed_phrase.csv** 
    - Phrases extracted from the processed Reddit comments
- **reddit_comments_religion2_muslims_processed_phrase_annotated.csv** 
    - Manual bias annotations for Reddit comments and phrases
- **reddit_comments_religion2_christians_biased_test_reduced.csv** and **reddit_comments_religion2_muslims_biased_test_reduced.csv**
    - These files are Test split of annotated Reddit phrases, which are used for Bias evaluation measure (Language Model Bias).
- **reddit_comments_religion2_christians_biased_valid_reduced.csv** and **reddit_comments_religion2_muslims_biased_valid_reduced.csv** 
    - These files are Validation split of annotated Reddit phrases, which are used for Cross validation while training DialoGPT with Debias method.
- **reddit_comments_religion2_muslims_processed_phrase_biased_testset_neg_attr_reduced.csv** and **reddit_comments_religion2_muslims_processed_phrase_biased_testset_pos_attr_reduced.csv**
    - These files are test split of Reddit phrases used for Bias evaluation over contrasting Attributes for marginalised demographic.

**Note:** The unprocessed reddit comment files could not be uploaded to GitHub due to size constraints. Find it on https://drive.google.com/drive/folders/1FC79WZyuVJRGXf4OzGoX4z84wvwhBxgh?usp=sharing

### Language Model Bias (Significance test Bias evaluation)

- Evaluation/measure_bias.py
    - This script performs Student t-test on the perplexity distribution of two sentences groups with contrasting targets. For example works on files: **reddit_comments_religion2_christians_biased_test_reduced.csv** and **reddit_comments_religion2_muslims_biased_test_reduced.csv**. Set variable 'REDUCE_SET' to remove outliers from target set. Unset variable ''REDUCE_SET' if you are already using reduced test set for input

- Evaluation/measure_bias_attribute_swap.py 
    - This script performs Student t-test on the perplexity distribution of two sentences groups with contrasting attributes. For example works on files: **reddit_comments_religion2_muslims_processed_phrase_unbiased_testset_pos_attr_reduced.csv** and **reddit_comments_religion2_muslims_processed_phrase_biased_testset_neg_attr_reduced.csv**. Set variable 'REDUCE_SET' to remove outliers from target set. Unset variable ''REDUCE_SET' if you are already using reduced test set for input


### Generate response from models

- Decoding/generate.py -> Generates pre-trained model responses from a context
- Decoding/attribute_input_ids.py -> Creates token ids of attribute words
- Decoding/target_input_ids.py -> Creates token ids of target words
