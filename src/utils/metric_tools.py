from sklearn.metrics import confusion_matrix
import matplotlib as plt

def plot_confusion_matrix(confusion_matrix, class_labels,
                  normalize=False,
                  title='Confusion Matrix',
                  cmap=plt.cm.Blues):
    """ Code courtesy of Abinav Sagar: https://towardsdatascience.com/convolutional-neural-network-for-breast-cancer-classification-52f1213dcc9 """

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusion_matrix)

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=55)
    plt.yticks(tick_marks, class_labels)
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
