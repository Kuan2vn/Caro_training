import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores1, mean_scores1): #scores2, mean_scores2):
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    # plt.subplot(1,2,1)
    plt.title('Training...')
    plt.xlabel('Number of Games')
    # plt.ylabel('Score')
    plt.ylabel('Move Taken')
    plt.plot(scores1, label = 'Move taken/ game')
    plt.plot(mean_scores1, label = 'Average move taken/ game')


    # plt.plot(reward, label = 'reward')


    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores1)-1, scores1[-1], str(scores1[-1]))
    plt.text(len(mean_scores1)-1, mean_scores1[-1], str(mean_scores1[-1]))

    # plt.text(len(reward)-1, reward[-1], str(reward[-1]))

    plt.show(block=False)
    plt.pause(.1)

    
    # plt.subplot(1,2,2)

    # plt.clf()

    # plt.title('Training...')
    # plt.xlabel('Number of Games')
    # # plt.ylabel('Score')
    # plt.ylabel('Move Taken')
    # plt.plot(scores2, label = 'Move taken/ game')
    # plt.plot(mean_scores2, label = 'Average move taken/ game')


    # # plt.plot(reward, label = 'reward')


    # plt.ylim(ymin=0)
    # plt.legend()
    # plt.text(len(scores2)-1, scores2[-1], str(scores2[-1]))
    # plt.text(len(mean_scores2)-1, mean_scores2[-1], str(mean_scores2[-1]))

    # # plt.text(len(reward)-1, reward[-1], str(reward[-1]))

    # plt.show(block=False)
    # plt.pause(.1)