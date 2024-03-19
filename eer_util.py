import matplotlib.pyplot as plt
import numpy


def draw_eer(pos_scores, neg_scores, inverse):
    neg_scores = numpy.array(neg_scores)
    pos_scores = numpy.array(pos_scores)
    max_thv = max(pos_scores.max(), neg_scores.max())
    min_thv = min(pos_scores.min(), neg_scores.min())

    print("max score: {}".format(max_thv))
    llrAxisis = range(0, int(1000*(max_thv-min_thv)), max(1, int(max_thv-min_thv)))
    llrAxisis = numpy.array(llrAxisis) * 0.001 + min_thv

    totalPositiveNum = len(pos_scores)
    FAR = numpy.zeros(len(llrAxisis))
    for idx, llr in enumerate(llrAxisis):
        if inverse:
            FAR[idx] = len(numpy.nonzero(pos_scores < llr)[0]) * 1.0 / totalPositiveNum
        else:
            FAR[idx] = 1 - len(numpy.nonzero(pos_scores < llr)[0]) * 1.0 / totalPositiveNum

    totalNegativeNum = len(neg_scores)
    print("total pos num: {}, total neg num: {}".format(totalPositiveNum, totalNegativeNum))
    FRR = numpy.zeros(len(llrAxisis))
    for idx, llr in enumerate(llrAxisis):
        if inverse:
            FRR[idx] = 1 - len(numpy.nonzero(neg_scores < llr)[0]) * 1.0 / totalNegativeNum
        else:
            FRR[idx] = len(numpy.nonzero(neg_scores < llr)[0]) * 1.0 / totalNegativeNum

    fig, ax = plt.subplots()
    ax.axis([llrAxisis[0], llrAxisis[-1], 0, 1])
    ax.plot(llrAxisis, FAR, 'm-', label='FAR')
    ax.plot(llrAxisis, FRR, 'b-', label='FRR')
    plt.xlabel("threshold")
    plt.ylabel("Error rate")
    plt.title("Speaker verification: EER curve")
    legend = ax.legend(loc='upper right', shadow=False, fontsize='x-large')

    plt.show()


def eer_test(pos_scores, neg_scores):

    neg_scores = numpy.array(neg_scores)
    pos_scores = numpy.array(pos_scores)

    max_thv = max(pos_scores.max(), neg_scores.max())
    axis = range(0, int(1000*max_thv), 1)
    axis = numpy.array(axis) * 0.001

    total_positive_num = len(pos_scores)
    true_accept_rate = numpy.zeros(len(axis))
    for idx, llr in enumerate(axis):
        true_accept_rate[idx] = 1 - len(numpy.nonzero(pos_scores < llr)[0]) * 1.0 / total_positive_num

    total_negative_num = len(neg_scores)
    false_accept_rate = numpy.zeros(len(axis))
    for idx, llr in enumerate(axis):
        false_accept_rate[idx] = len(numpy.nonzero(neg_scores < llr)[0]) * 1.0 / total_negative_num

    # find index, eer
    far = numpy.copy(true_accept_rate)
    frr = numpy.copy(false_accept_rate)
    diff = numpy.absolute(far - frr)
    eer = numpy.min(diff)
    idx = numpy.argmin(diff)

    print("EER={}\nthreshold={}".format(true_accept_rate[idx], axis[idx]))

    fig, ax = plt.subplots()
    ax.axis([axis[0], axis[-1], 0, 1])
    ax.plot(axis, true_accept_rate, 'm-', label='True Accept Rate')
    ax.plot(axis, false_accept_rate, 'b-', label='False Accept Rate')
    plt.xlabel("threshold")
    plt.ylabel("Error rate")
    plt.title("Speaker verification: EER curve")
    ax.legend(loc='upper right', shadow=False, fontsize='x-large')
    plt.show()

    fig.savefig('EER_mfcc.jpg')  # save the figure to file
    plt.close(fig)  # close the figure

    return eer, axis[idx]
