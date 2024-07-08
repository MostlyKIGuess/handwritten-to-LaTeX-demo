from learner.trainer import HandwrittenSymbolsClassifier
SCORE = 0
TOTAL = 2

classifier = HandwrittenSymbolsClassifier(root_dir='./learner/datasets/extracted_images/', epochs=5)

try:
    classifier.load_model('./learner/models/test.torch')
    SCORE += 1
except Exception as e:
    try:
        classifier.train()
        classifier.save_model("./learner/models/", "test.torch")
        classifier.load_model('./learner/models/test.torch')
        SCORE += 1
    except Exception as e:
        print(e)

try:
    result = classifier.predict('./tests/75bb990518714322ba46ba3743bb7b90.png')
    print(f'Predicted class: {result}')
    SCORE += 1
except Exception as e:
    print(e)

if classifier.epoch_losses and classifier.epoch_accuracies:
    classifier.plot_metrics()
else:
    print("No training metrics available to plot.")

if SCORE < TOTAL:
    raise Exception(f'Test failed: {SCORE}/{TOTAL}')
else:
    print(f'Test passed: {SCORE}/{TOTAL}')