from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from src.factories import LearningSetFactory, MultiClassClassifierFactory


def main():
    print("Starting Test..")

    rfc = abc = bc = trfc = tabc = tbc = 0
    for i in range(100):
        print(i)
        train_set, train_labels, test_set, test_labels = LearningSetFactory.get_learning_sets_and_labels(0.8)

        multiclass_random_forest = MultiClassClassifierFactory.make_default_classifier(RandomForestClassifier, train_set,
                                                                                       train_labels)
        multiclass_ada_boost_classifier = MultiClassClassifierFactory.make_ada_boost_classifier(train_set, train_labels)
        multiclass_bagging_classifier = MultiClassClassifierFactory.make_default_classifier(BaggingClassifier,
                                                                                                    train_set, train_labels)

        two_layer_random_forest = MultiClassClassifierFactory.make_default_two_layer_classifier(RandomForestClassifier,
                                                                                                  train_set, train_labels)
        two_layer_ada_boost_classifier = MultiClassClassifierFactory.make_ada_boost_two_layer_classifier(train_set,
                                                                                                         train_labels)
        two_layer_bagging_classifier = MultiClassClassifierFactory.make_default_two_layer_classifier(BaggingClassifier,
                                                                                                    train_set, train_labels)

        rfc += multiclass_random_forest.accuracy(test_set, test_labels)
        abc += multiclass_ada_boost_classifier.accuracy(test_set, test_labels)
        bc += multiclass_bagging_classifier.accuracy(test_set, test_labels)
        trfc += two_layer_random_forest.accuracy(test_set, test_labels)
        tabc += two_layer_ada_boost_classifier.accuracy(test_set, test_labels)
        tbc += two_layer_bagging_classifier.accuracy(test_set, test_labels)

    print("Done.\nBenchmark:")

    print("Multiclass Random Forest Classifier :", rfc/100)
    print("Multiclass Ada Boost Classifier     :", abc/100)
    print("Multiclass Bagging Classifier       :", bc/100)
    print("Two Layer Random Forest Classifier  :", trfc/100)
    print("Two Layer Ada Boost Classifier      :", tabc/100)
    print("Two Layer Bagging Classifier        :", tbc /100)
    print("Done.")

if __name__ == '__main__':
    main()