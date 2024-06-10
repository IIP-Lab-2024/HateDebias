import os

def generate_sequence(sequence):
    sequence_list = []
    for item1 in sequence:
        for item2 in sequence:
            for item3 in sequence:
                for item4 in sequence:
                    new_sequence = [item1,item2,item3,item4]
                    if len(set(new_sequence)) == 4:
                        sequence_list.append(new_sequence)
    return sequence_list

define_sequence = ['gender','age','country','ethnicity']
sequence_list = generate_sequence(define_sequence)
for sequence in sequence_list:
    item1, item2, item3, item4 = sequence
    save_path = 'Fine-Tune-FGM-' + item1[0] + item2[0] + item3[0] + item4[0]
    os.system(
        'python3 Fine-Tune-FGM.py --attribute1 %s --attribute2 %s --attribute3 %s --attribute4 %s --save_path %s' % (item1, item2, item3, item4, save_path)
    )