# https://learning.edx.org/course/course-v1:UTAustinX+CSMS.ML.323+1T2022/block-v1:UTAustinX+CSMS.ML.323+1T2022+type@sequential+block@721383d3bf624baaacfa0d56f0dc8407/block-v1:UTAustinX+CSMS.ML.323+1T2022+type@vertical+block@9b65cf246eb44051a92c5b45e48e351a
import sys
from matplotlib import pyplot as plt

filename = sys.argv[1]

with open(filename,'r') as f:
    lines = f.readlines()

    sample_average = {
        'average_rs': [float(n) for n in lines[0].strip().split()],
        'average_best_action_taken': [float(n) for n in lines[1].strip().split()],
    }
    constant = {
        'average_rs': [float(n) for n in lines[2].strip().split()],
        'average_best_action_taken': [float(n) for n in lines[3].strip().split()],
    }

    assert len(sample_average['average_rs']) == len(sample_average['average_best_action_taken']) == \
        len(constant['average_rs']) == len(constant['average_best_action_taken']) == 10000

    fig,axes = plt.subplots(2,1)

    axes[1].set_ylim([0.,1.])

    axes[0].plot(sample_average['average_rs'])
    axes[1].plot(sample_average['average_best_action_taken'])

    axes[0].plot(constant['average_rs'])
    axes[1].plot(constant['average_best_action_taken'])

    fig.show()
    _ = input()

