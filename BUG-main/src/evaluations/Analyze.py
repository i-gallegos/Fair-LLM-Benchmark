"""
Usage: Analyze.py directory_path
"""

import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

all_entities, m_entities, f_entities, n_entities = [], [], [], []

f_pronouns = ["she", "herself", "her", "She", "Herself", "Her"]
m_pronouns = ["he", "his", "himself", "him", "He", "His", "Himself", "Him"]


def create_lists(path):
    """
    Takes a txt file of possible entities and fills up the python lists of entities.
    """
    df = pd.read_csv(path)

    for index, row in df.iterrows():
        if isinstance(row["entities"], str):
            entities = row["entities"].split(",")
            entities = [e.rstrip().lstrip().lower() for e in entities]
            all_entities.extend(entities)
            f = int(row["F_workers"])
            m = int(row["M_workers"])
            if f > m:
                f_entities.extend(entities)
            elif m > f:
                m_entities.extend(entities)
            else:
                n_entities.extend(entities)


def drop_less_quality_data(data):
    """
    drop less quality data (discovered by the human annotations statistics)
    """
    data = data[data['data_index'] != "6"]
    data = data[data['data_index'] != "7"]
    data = data[data['data_index'] != "8"]
    data = data[data['data_index'] != "12"]
    data = data[data['data_index'] != "13"]
    data = data[data['data_index'] != "16"]
    data = data[data['data_index'] != "19"]
    data = data[data['corpus'] != "perseus"]
    data = data[data['distance'] <= 20]
    data = data[data['num_of_pronouns'] <= 4]
    return data


def create_BUG(directory_path):
    """
    Receives the spike data and a list of possible entities,
    filters the data such that the professions column will be only words from the list
    and creates 2 csv files:
    the filtered data
    the distribution over the professions
    """
    new_data = pd.DataFrame()
    i = 0
    for file_name in os.listdir(directory_path):
        i += 1
        if file_name.endswith("tsv"):
            data = pd.read_csv(os.path.join(directory_path, file_name), sep='\t')
        elif file_name.endswith("csv"):
            data = pd.read_csv(os.path.join(directory_path, file_name))
        else:
            continue
        data = data.rename(columns={"er": "g", "gender": "g"})
        data = data.rename(columns={"er_first_index": "g_first_index", "gender_first_index": "g_first_index"})
        data = data.rename(columns={"er_last_index": "g_last_index", "gender_last_index": "g_last_index"})
        data['profession'] = data['profession'].str.lower()
        data['data_index'] = file_name[-5] if not file_name[-6].isdigit() else file_name[-6:-4]
        data['corpus'] = file_name.split("_")[0]
        # assumption: path is- ...._datax.[t,c]sv, while x is the index of the query
        new_data = new_data.append(data[data['profession'].isin(all_entities)])

    new_data['stereotype'] = new_data[['profession', 'g']].apply(is_stereotype, axis=1)
    new_data['distance'] = new_data[['g_first_index', 'profession_first_index']].apply(find_distance, axis=1)
    new_data['num_of_pronouns'] = new_data[['sentence_text']].apply(num_of_pronouns, axis=1)

    new_data = new_data.drop_duplicates(subset=['sentence_text'], keep="last")
    duplicate_pronouns = ["he or she", "she or he", "her or his", "his or her", "her / his", "his / her",
                          "he / she", "she / he"]
    for p in duplicate_pronouns:
        new_data = new_data[~new_data.sentence_text.str.contains(p)]

    new_data['predicted gender'] = new_data[['g']].apply(predict_gender, axis=1)

    new_data = drop_less_quality_data(new_data)
    new_data = clean_columns(new_data)

    new_data.to_csv("data\\full_BUG.csv")
    professions = new_data['profession']
    distribution = professions.value_counts()
    distribution.to_csv("data/data_distribution.csv")
    create_balanced("data\\full_BUG.csv")

    with open('dropped_en.txt', 'r', encoding="utf8") as f:
        dropped_rows = [line.strip() + "\n" for line in f]

    with open("data/en_pro.txt", "w+", encoding="utf-8") as output_file_pro, \
            open("data/en_anti.txt", "w+", encoding="utf-8") as output_file_anti, \
            open("data/en.txt", "w+", encoding="utf-8") as output_file_all:
        for index, row in tqdm(new_data.iterrows()):
            is_dropped = False
            for line in dropped_rows:
                if row["sentence_text"] in line:
                    is_dropped = True
                    break
            if not is_dropped:
                line = row['predicted gender'] + "\t" + str(row['profession_first_index']) + "\t" + \
                       row['sentence_text'] + "\t" + row['profession'] + "\t" + row['corpus']
                if row["stereotype"] == 1:
                    output_file_pro.write(line + "\n")
                elif row["stereotype"] == -1:
                    output_file_anti.write(line + "\n")
                output_file_all.write(line + "\n")

    corpora = ["wikipedia", "covid19", "pubmed"]
    for corpus in corpora:
        with open("data/en_{}_pro.txt".format(corpus), "w+", encoding="utf-8") as output_file_pro, \
                open("data/en_{}_anti.txt".format(corpus), "w+", encoding="utf-8") as output_file_anti, \
                open("data/en_{}.txt".format(corpus), "w+", encoding="utf-8") as output_file_all:
            for index, row in tqdm(new_data.iterrows()):
                is_dropped = False
                for line in dropped_rows:
                    if row["sentence_text"] in line:
                        is_dropped = True
                        break
                if not is_dropped and row['corpus'] == corpus:
                    line = row['predicted gender'] + "\t" + str(row['profession_first_index']) + "\t" + \
                           row['sentence_text'] + "\t" + row['profession'] + "\t" + row['corpus']
                    if row["stereotype"] == 1:
                        output_file_pro.write(line + "\n")
                    elif row["stereotype"] == -1:
                        output_file_anti.write(line + "\n")
                    output_file_all.write(line + "\n")

    print("Size of data: " + str(new_data.shape[0]))
    return new_data


def clean_columns(df):
    cleaned_df = df[['sentence_text', 'profession', 'g', 'profession_first_index', 'g_first_index',
                     'predicted gender', 'stereotype', 'distance', 'num_of_pronouns', 'corpus', 'data_index']]
    return cleaned_df


def create_balanced(data_path):
    data = pd.read_csv(data_path)
    data_female = data[data["g"] == "her"]
    data_female = data_female.append(data[data["g"] == "she"])
    data_female = data_female.append(data[data["g"] == "herself"])
    data_male = data[data["g"] == "his"]
    data_male = data_male.append(data[data["g"] == "he"])
    data_male = data_male.append(data[data["g"] == "himself"])
    data_f_a = data_female[data_female["stereotype"] == -1]
    data_f_s = data_female[data_female["stereotype"] == 1]
    data_m_a = data_male[data_male["stereotype"] == -1]
    data_m_s = data_male[data_male["stereotype"] == 1]

    n = data_f_s.shape[0]
    balanced = data_f_s
    balanced = balanced.append(data_f_a.sample(n=n))
    balanced = balanced.append(data_m_s.sample(n=n))
    balanced = balanced.append(data_m_a.sample(n=n))
    balanced.to_csv("data/balanced_BUG.csv")


def remove_tags_from_sentence(df):
    """
    Adds ref tag to the gender pronoun word, and ent tag to the entity word.
    """
    df = df.values
    sentence = df[0]
    try:
        sentence_a = sentence.split("<")[0]
        sentence_b = sentence.split(">")[1]
        sentence_c = sentence_b.split("<")[0]
        sentence_d = sentence.split(">")[2]
    except:
        print(sentence)
        return
    return sentence_a + sentence_c + sentence_d


def add_tags_to_sentence(df):
    """
    Adds ref tag to the gender pronoun word, and ent tag to the entity word.
    """
    df = df.values
    sentence = df[0]
    p_idx = int(df[1])
    split_sentence = sentence.split(' ')
    split_sentence[p_idx] = "<ent>" + split_sentence[p_idx] + "</ent>"
    return " ".join(split_sentence)


def find_distance(df):
    """
    Finds the number of words between the gender pronoun word, and the entity word.
    """
    df = df.values
    g_idx = df[0]
    p_idx = df[1]
    return abs(g_idx - p_idx)


def predict_gender(df):
    """
    returns "male" is the pronoun in the sentence is a male pronoun, and "female" otherwise
    """
    df = df.values
    g = df[0]
    if g in m_pronouns:
        return "male"
    return "female"


def is_stereotype(df):
    """
    1 : stereotype, -1 : non-stereotype, 0 neutral entity
    """
    df = df.values
    profession = df[0]
    g = df[1]
    if (profession in f_entities and g in f_pronouns) or (profession in m_entities and g in m_pronouns):
        return 1
    if (profession in f_entities and g in m_pronouns) or (profession in m_entities and g in f_pronouns):
        return -1
    return 0


def num_of_pronouns(df):
    """
    Finds the number of words between the gender pronoun word, and the entity word.
    """
    sentence = df.values[0].lower()
    ans = 0
    pronouns = ["he", "she", "his", "himself", "herself", "her", "him"]
    try:
        for word in pronouns:
            ans += sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), sentence))
        return ans
    except:
        return 1


def samples_for_double_validation(data):
    """
    Sample 200 random example from the annotations for validate we agree
    """
    data['validation'] = 0
    new_data = data.sample(n=200)
    new_data[['predicted gender', 'validation', 'sentence_text', 'profession', 'g', 'g_first_index',
                'profession_first_index', 'stereotype',
                'corpus', 'data_index', 'correct']].to_csv("double_validation.csv")


def samples_to_classify(data):
    """
    Sample 1000 random example from the data for us to classify the sentences as correct/incorrect reference
    Sample 50 random examples from each corpus and for each query
    """
    data['correct?'] = 0
    new_data = data.sample(n=1000)
    new_data['sentence_text'] = \
        new_data[['sentence_text', 'profession_first_index']].apply(add_tags_to_sentence, axis=1)
    for i in range(1, 11):
        df_new = new_data[100 * (i - 1):100 * i]
        df_new[['predicted gender', 'correct?', 'sentence_text', 'profession', 'g', 'g_first_index',
                'profession_first_index', 'stereotype',
                'corpus', 'data_index']].to_csv("human_annotation\\data_samples{}.csv".format(i))

    for i in range(1, 22):
        df_new = data[data.data_index == str(i)].sample(n=50)
        df_new['sentence_text'] = \
            df_new[['sentence_text', 'profession_first_index']].apply(add_tags_to_sentence, axis=1)
        df_new[['predicted gender', 'correct?', 'sentence_text', 'profession', 'g', 'g_first_index',
                'profession_first_index', 'stereotype',
                'corpus', 'data_index']].to_csv("human_annotation\\data_samples_query{}.csv".format(i))

    for i in ["wikipedia", "perseus", "covid19", "pubmed"]:
        df_new = data[data.corpus == i].sample(n=50)
        df_new['sentence_text'] = \
            df_new[['sentence_text', 'profession_first_index']].apply(add_tags_to_sentence, axis=1)
        df_new[['predicted gender', 'correct?', 'sentence_text', 'profession', 'g', 'g_first_index',
                'profession_first_index', 'stereotype',
                'corpus', 'data_index']].to_csv("human_annotation\\data_samples_{}.csv".format(i))


def plot_dict(dict_attributes, color, title):
    """
    receives a dictionary of attributes and sentences count / sentences statistics, ant plot the data as histograms
    """
    histo = plt.bar(dict_attributes.keys(), dict_attributes.values(), color=color)
    for bar in histo:
        height = bar.get_height()
        plt.annotate(height if height > 1 else "{:.2f}".format(height),
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom')
    plt.title(title)
    plt.savefig("data/graphs/{}.png".format(title))
    plt.close()


def statistics_over_samples(data):
    """
    plot statistics over data of human annotations.
    numbers over histograms are:
        (number of "correct" samples with attribute) / (number of all samples with attribute)
    comment out code lines are data with less quality.
    """

    print("accuracy of human annotations" + str(sum(data['correct'] == '1') / data.shape[0]))



    statistics_gender_pronouns = {
        "himself": sum(data[data['g'] == 'himself']['correct'] == '1') / sum(data['g'] == 'himself'),
        "he": sum(data[data['g'] == 'he']['correct'] == '1') / sum(data['g'] == 'he'),
        "she": sum(data[data['g'] == 'she']['correct'] == '1') / sum(data['g'] == 'she'),
        "herself": sum(data[data['g'] == 'herself']['correct'] == '1') / sum(data['g'] == 'herself'),
        "his": sum(data[data['g'] == 'his']['correct'] == '1') / sum(data['g'] == 'his'),
        "her": sum(data[data['g'] == 'her']['correct'] == '1') / sum(data['g'] == 'her'),
        "female": (sum(data[data['g'] == 'her']['correct'] == '1') +
                   sum(data[data['g'] == 'she']['correct'] == '1') +
                  sum(data[data['g'] == 'herself']['correct'] == '1')) / \
                  (sum(data['g'] == 'her') + sum(data['g'] == 'she') + sum(data['g'] == 'herself')),
        "male": (sum(data[data['g'] == 'he']['correct'] == '1') +
                 sum(data[data['g'] == 'his']['correct'] == '1') + sum(data[data['g'] == 'him']['correct'] == '1') +
                 sum(data[data['g'] == 'himself']['correct'] == '1')) / \
                (sum(data['g'] == 'he') + sum(data['g'] == 'his') + sum(data['g'] == 'himself') +
                 sum(data['g'] == 'him'))
    }
    plot_dict(statistics_gender_pronouns, "pink", "Statistics - Gender Pronouns")

    data['distance'] = data[['g_first_index', 'profession_first_index']].apply(find_distance, axis=1)
    # distance_average = sum(data['distance']) / 300
    statistics_distance = {
        "d <= 5": sum(data[data['distance'] <= 5]['correct'] == '1') / sum(data['distance'] <= 5),
        "5 < d <= 10": (sum(data[data['distance'] <= 10]['correct'] == '1') -
                        sum(data[data['distance'] <= 5]['correct'] == '1')) /
                       (sum(data['distance'] <= 10) - sum(data['distance'] <= 5)),
        "10 < d <= 15": (sum(data[data['distance'] <= 15]['correct'] == '1') -
                         sum(data[data['distance'] <= 10]['correct'] == '1')) /
                        (sum(data['distance'] <= 15) - sum(data['distance'] <= 10)),
        "15 < d <=20": (sum(data[data['distance'] <= 20]['correct'] == '1') -
                        sum(data[data['distance'] <= 15]['correct'] == '1')) /
                       (sum(data['distance'] <= 20) - sum(data['distance'] <= 15)),
        # "20 < d": sum(data[data['distance'] > 20]['correct'] == '1') / sum(data['distance'] > 20)
    }
    plot_dict(statistics_distance, "teal", "Statistics - Distance Between Words")

    m_data = data[data['profession'].isin(m_entities)]
    f_data = data[data['profession'].isin(f_entities)]
    n_data = data[data['profession'].isin(n_entities)]

    # data[f_data[f_data['g'] == 'he']['correct'] == '0']
    #
    # "non-stereotypes": (sum(f_data[f_data['g'] == 'he']['correct'] == '1') +
    #                     sum(f_data[f_data['g'] == 'his']['correct'] == '1') +
    #                     sum(m_data[m_data['g'] == 'her']['correct'] == '1') +
    #                     sum(m_data[m_data['g'] == 'she']['correct'] == '1')) / \
    #                    (sum(f_data['g'] == 'he') + sum(f_data['g'] == 'his') +
    #                     sum(m_data['g'] == 'her') + sum(m_data['g'] == 'she')),

    statistics_male_female = {
        "male\n entities": sum(m_data['correct'] == '1') / len(m_data),
        "female\n entities": sum(f_data['correct'] == '1') / len(f_data),
        "neutral\n entities": sum(n_data['correct'] == '1') / len(n_data),
    }
    plot_dict(statistics_male_female, "darkmagenta", "Statistics - Male, Female Entities")

    statistics_stereotype = {
        "stereotypes": (sum(m_data[m_data['g'] == 'he']['correct'] == '1') +
                        sum(m_data[m_data['g'] == 'his']['correct'] == '1') +
                        sum(f_data[f_data['g'] == 'her']['correct'] == '1') +
                        sum(f_data[f_data['g'] == 'she']['correct'] == '1')) / \
                       (sum(m_data['g'] == 'he') + sum(m_data['g'] == 'his') +
                        sum(f_data['g'] == 'her') + sum(f_data['g'] == 'she')),
        "s\n male": (sum(m_data[m_data['g'] == 'he']['correct'] == '1') +
                     sum(m_data[m_data['g'] == 'his']['correct'] == '1')) / \
                    (sum(m_data['g'] == 'he') + sum(m_data['g'] == 'his')),
        "s\n female": (sum(f_data[f_data['g'] == 'her']['correct'] == '1') +
                       sum(f_data[f_data['g'] == 'she']['correct'] == '1')) / \
                      (sum(f_data['g'] == 'her') + sum(f_data['g'] == 'she')),
        "non-stereotypes": (sum(f_data[f_data['g'] == 'he']['correct'] == '1') +
                            sum(f_data[f_data['g'] == 'his']['correct'] == '1') +
                            sum(m_data[m_data['g'] == 'her']['correct'] == '1') +
                            sum(m_data[m_data['g'] == 'she']['correct'] == '1')) / \
                           (sum(f_data['g'] == 'he') + sum(f_data['g'] == 'his') +
                            sum(m_data['g'] == 'her') + sum(m_data['g'] == 'she')),
        "non-s\n male": (sum(f_data[f_data['g'] == 'he']['correct'] == '1') +
                         sum(f_data[f_data['g'] == 'his']['correct'] == '1')) / \
                        (sum(f_data['g'] == 'he') + sum(f_data['g'] == 'his')),
        "non-s\n female": (sum(m_data[m_data['g'] == 'her']['correct'] == '1') +
                           sum(m_data[m_data['g'] == 'she']['correct'] == '1')) / \
                          (sum(m_data['g'] == 'her') + sum(m_data['g'] == 'she'))
    }
    plot_dict(statistics_stereotype, "orange", "Statistics - Stereotypes")

    statistics_num_of_pronouns = {
        "1": (sum(data[data['num_of_pronouns'] == 1]['correct'] == '1')) / \
             (sum(data['num_of_pronouns'] == 1)),  # 146
        "2": (sum(data[data['num_of_pronouns'] == 2]['correct'] == '1')) / \
             (sum(data['num_of_pronouns'] == 2)),  # 81
        "3": (sum(data[data['num_of_pronouns'] == 3]['correct'] == '1')) / \
             (sum(data['num_of_pronouns'] == 3)),  # 47
        "4": (sum(data[data['num_of_pronouns'] == 4]['correct'] == '1')) / \
             (sum(data['num_of_pronouns'] == 4)),  # 20
        # "more then 4": (sum(data[data['num_of_pronouns'] > 4]['correct'] == '1')) / \
        #                 (sum(data['num_of_pronouns'] > 4))  # 6
    }
    plot_dict(statistics_num_of_pronouns, "maroon", "Statistics - number of pronouns")

    count_corpus = {
        "wikipedia": (sum(data[data['corpus'] == 'wikipedia']['correct'] == '1')) / sum(data['corpus'] == 'wikipedia'),
        "covid19": (sum(data[data['corpus'] == 'covid19']['correct'] == '1')) / sum(data['corpus'] == 'covid19'),
        # "perseus": (sum(data[data['corpus'] == 'perseus']['correct'] == '1')) / sum(data['corpus'] == 'perseus'),
        "pubmed": (sum(data[data['corpus'] == 'pubmed']['correct'] == '1')) / sum(data['corpus'] == 'pubmed'),
    }
    plot_dict(count_corpus, "g", "Statistics - statistics_count_corpus")


def statistics_over_corpus(data_path):
    """
    plot statistics over all sentences.
    numbers over histograms are:
        number of all samples with attribute
    comment out code lines are data with less quality.
    """
    data = pd.read_csv(data_path)
    data['g'] = data['g'].apply(lambda x: x.lower())
    count_gender_pronouns = {
        "herself": sum(data['g'] == 'herself'),
        "she": sum(data['g'] == 'she'),
        "himself": sum(data['g'] == 'himself'),
        "he": sum(data['g'] == 'he'),
        "her": sum(data['g'] == 'her'),
        "his": sum(data['g'] == 'his'),
        # "female": sum(data['g'] == 'her') + sum(data['g'] == 'she') + sum(data['g'] == 'herself'),
        # "male": sum(data['g'] == 'he') + sum(data['g'] == 'his') + sum(data['g'] == 'him') + sum(data['g'] == 'himself')
    }
    plot_dict(count_gender_pronouns, "pink", "count - Gender Pronouns")
    print(count_gender_pronouns)

    m_data = data[data['profession'].isin(m_entities)]
    f_data = data[data['profession'].isin(f_entities)]
    n_data = data[data['profession'].isin(n_entities)]

    count_male_female = {
        "male\n entities": m_data.shape[0],
        "female\n entities": f_data.shape[0],
        "neutral\n entities": n_data.shape[0]
    }
    plot_dict(count_male_female, "darkmagenta", "count - Male, Female Entities")

    count_stereotype = {
        "stereotypes": sum(data['stereotype'] == 1),
        "non-stereotypes": sum(data['stereotype'] == -1),
    }
    plot_dict(count_stereotype, "orange", "count - Stereotypes")

    count_data_index = {
        "1": sum(data['data_index'] == '1'),
        "2": sum(data['data_index'] == '2'),
        "3": sum(data['data_index'] == '3'),
        "4": sum(data['data_index'] == '4'),
        "5": sum(data['data_index'] == '5'),
        # "6": sum(data['data_index'] == '6'),
        # "7": sum(data['data_index'] == '7'),
        # "8": sum(data['data_index'] == '8'),
        "9": sum(data['data_index'] == '9'),
        "10": sum(data['data_index'] == '10'),
        "11": sum(data['data_index'] == '11'),
        # "12": sum(data['data_index'] == '12'),
        # "13": sum(data['data_index'] == '13'),
        "14": sum(data['data_index'] == '14'),
        "15": sum(data['data_index'] == '15'),
        # "16": sum(data['data_index'] == '16'),
        "17": sum(data['data_index'] == '17'),
        "18": sum(data['data_index'] == '18'),
        # "19": sum(data['data_index'] == '19'),
        "20": sum(data['data_index'] == '20'),
        "21": sum(data['data_index'] == '21'),
    }
    plot_dict(count_data_index, "gold", "count - statistics_data_index")

    count_corpus = {
        "wikipedia": sum(data['corpus'] == 'wikipedia'),
        "covid19": sum(data['corpus'] == 'covid19'),
        # "perseus": sum(data['corpus'] == 'perseus'),
        "pubmed": sum(data['corpus'] == 'pubmed'),
        # "ungd": sum(data['corpus'] == 'ungd'),
    }
    plot_dict(count_corpus, "g", "count - statistics_count_corpus")

    data['num_of_pronouns'] = data[['sentence_text']].apply(num_of_pronouns, axis=1)
    count_num_of_pronouns = {
        "1": sum(data['num_of_pronouns'] == 1),
        "2": sum(data['num_of_pronouns'] == 2),
        "3": sum(data['num_of_pronouns'] == 3),
        "4": sum(data['num_of_pronouns'] == 4),
        "more then 4": sum(data['num_of_pronouns'] > 4),
    }
    plot_dict(count_num_of_pronouns, "maroon", "count - number of pronouns")

    data['distance'] = data[['g_first_index', 'profession_first_index']].apply(find_distance, axis=1)

    count_distance = {
        "d <= 5": sum(data['distance'] <= 5),
        "5 < d <= 10": (sum(data['distance'] <= 10) - sum(data['distance'] <= 5)),
        "10 < d <= 15": (sum(data['distance'] <= 15) - sum(data['distance'] <= 10)),
        "15 < d <=20": (sum(data['distance'] <= 20) - sum(data['distance'] <= 15)),
        "20 < d": sum(data['distance'] > 20)
    }
    plot_dict(count_distance, "teal", "count - Distance Between Words")


if __name__ == '__main__':
    create_lists("inc_occ_gender.csv")
    filtered_data = create_BUG(sys.argv[1])

    data_filtered = "data\\full_BUG.csv"
    # statistics_over_corpus(data_filtered)

