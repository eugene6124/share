import knn
import rf
import svc
import logistic
import classifier
import pca


def main():
    print("PCA")
    pca.pca()
    print("RandomForest")
    rf.rf(2)
    print("KNN")
    knn.knn(2)
    print("SVC")
    svc.svc()
    print("GRID_SVC")
    svc.gridSearchScore()
    print("Logistic")
    logistic.Logistic().fit()
    print("DNN Classifier")
    classifier_model = classifier.classifier()
    classifier_model.fit()


if __name__ == "__main__":
    main()