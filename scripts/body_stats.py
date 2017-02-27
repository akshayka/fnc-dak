import csv
import matplotlib.pyplot as plt

TRAIN_BODIES = 'train_bodies.csv'
with open(TRAIN_BODIES, 'rb') as f:
    reader = csv.reader(f)
    header = reader.next()

    num_words = []
    num_distinct_words = []
    for row in reader:
        words = row[1].split()
        num_words.append(len(words))
        num_distinct_words.append(len(set([w.lower() for w in words])))

    assert len(num_distinct_words) == len(num_words)
    num_bodies = float(len(num_words))
    print 'Total number of articles: %d' % num_bodies

    print 'Average number of words per body: %0.2f' % (
        sum(num_words) / num_bodies)
    print 'Average number of distinct words per body: %0.2f' % (
        sum(num_distinct_words) / num_bodies)

    print 'Max number of words per body: %0.2f' % max(num_words)
    print 'Max number of distinct words per body: %0.2f' % (
        max(num_distinct_words))

    print 'Min number of words per body: %0.2f' % min(num_words)
    print 'Min number of distinct words per body: %0.2f' % (
        min(num_distinct_words))
    
    plt.figure()
    plt.hist(num_words, bins=50)
    plt.title('Number of words per body')
    plt.savefig('body_word_hist.png')
    plt.figure()

    plt.hist(num_distinct_words, bins=50)
    plt.title('Number of distinct words per body')
    plt.savefig('body_distinct_word_hist.png')
    plt.figure()
