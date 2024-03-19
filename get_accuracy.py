import os
import eer_util
# 7850-281318-0003.flac --> this is rejected, [0.22598357]
# 7850-281318-0004.flac --> accepted as speaker: 3081, [0.36840414]

test_file = 'devclean_test_result.txt'
total_test_num = 0
true_positive_num = 0
false_negative_num = 0
true_negative_num = 0
false_positive_num = 0

TH_value = 0.22
test_list = []
enrolled_name = []
train_path = 'data/train'
for spks in os.listdir(train_path):
    spks_path = os.path.join(train_path, spks)
    if os.path.isdir(spks_path):
        enrolled_name.append(spks)

pos_scores = []
neg_scores = []
with open(test_file) as f:
    for line in f:
        tmp = line.rstrip().split(' --> ')
        spk_name = tmp[0].split('-')[0]
        test_name = 'Unknown'
        try:
            sc_val = float(tmp[1].split(', ')[-1][1:-1])
            if 'accepted as speaker:' in tmp[1]:
                rm_line = tmp[1].replace('accepted as speaker: ', '')
                test_name = rm_line.split(', ')[0]

            total_test_num += 1
            if spk_name in enrolled_name:
                if spk_name == test_name:
                    true_positive_num += 1
                    pos_scores.append(sc_val)
                else:
                    false_negative_num += 1
                    neg_scores.append(sc_val)
            else:
                if test_name == 'Unknown':
                    true_negative_num += 1
                    pos_scores.append(sc_val)
                else:
                    false_positive_num += 1
                    neg_scores.append(sc_val)

            test_list.append([spk_name, test_name, sc_val])
        except:
            continue

    true_positive_num = 0
    false_negative_num = 0
    true_negative_num = 0
    false_positive_num = 0
    enrolled_case = 0
    # eer, thd = eer_util.eer_test(pos_scores, neg_scores)
    for item in test_list:
        spk_name, test_name, confidence = item
        if spk_name in enrolled_name:
            enrolled_case += 1
            if TH_value > confidence:
                false_negative_num += 1
            else:
                if spk_name == test_name:
                    true_positive_num += 1
                else:
                    false_positive_num += 1
        else:
            if TH_value > confidence:
                true_negative_num += 1
            else:
                false_negative_num += 1

    tp_rate = true_positive_num / (true_positive_num + false_negative_num)
    fp_rate = false_positive_num / (false_positive_num + true_negative_num)

    print("Total person: 40\tenrolled: 30, un-enrolled: 10")
    print("Total test: {}".format(len(test_list)))
    print("true-accept case: {}".format(true_positive_num))
    print("false-accept case: {}".format(false_positive_num))
    print("threshold: {}".format(TH_value))
    print("true-accept rate: {:.3f}%".format(tp_rate * 100.0))
    print("false-accept rate: {:.3f}%".format(fp_rate * 100.0))

