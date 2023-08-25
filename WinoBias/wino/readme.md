---
## WinoBias dataset ##
1. The data/ folder contains the WinoDataset we generated.
2. Gender swapping lists are listed in extra_gendered_words.txt and generalized_swaps.txt files. The swapping can be finished by the word_swapper.py file.
3. If you want to try the WinoBias dataset using [allennlp](https://allennlp.org/models), remember to add "pos_tag != '-'" in line 274 [here](https://github.com/allenai/allennlp/blob/5f9fb419273f99c949ccdabab22fdc8e9b895c1c/allennlp/data/dataset_readers/dataset_utils/ontonotes.py#L274).

*How To evaulate on our WinoBias Dataset*:
- Download the codes from [e2e-coref](https://github.com/kentonl/e2e-coref)
- Remember to proprecess WinoBias dataset to the corresponding input format [refer here](https://github.com/kentonl/e2e-coref/blob/9d1ee1972f6e34eb5d1dcbb1fd9b9efdf53fc298/setup_training.sh#L38). 
- Replace the [evaluation file and path](https://github.com/kentonl/e2e-coref/blob/9d1ee1972f6e34eb5d1dcbb1fd9b9efdf53fc298/experiments.conf#L79) to the winobias dataset.


Others:
- You do not need the anonymize.py file for our winobias dataset. The script is for anonymizing the OntoNote coref file. To use this, you need fo first generate the NER tags for the OntoNotes coref files, and then run `python anonymize.py file_with_NERs ontonotes_coref.conll outputfile`.
- For the NER tags we use [tagger](https://github.com/glample/tagger) in our paper.
