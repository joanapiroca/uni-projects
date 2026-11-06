#!/usr/bin/env python3
# d9034c78-c8b2-4820-8671-0e919beeaaf3
# 10d1c6c4-6216-4721-b23c-d0de3e0fb138
# c8cb695-9dc2-465b-b4ea-1637eb270929
import argparse
import dataclasses

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrap_samples", default=100, type=int, help="Bootstrap samples")
parser.add_argument("--data_size", default=1000, type=int, help="Data set size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.


class ArtificialData:
    @dataclasses.dataclass
    class Sentence:
        """ Information about a single dataset sentence."""
        gold_edits: int  # Number of required edits to be performed.
        predicted_edits: int  # Number of edits predicted by a model.
        predicted_correct: int  # Number of correct edits predicted by a model.
        human_rating: int  # Human rating of the model prediction.

    def __init__(self, args: argparse.Namespace):
        generator = np.random.RandomState(args.seed)

        self.sentences = []
        for _ in range(args.data_size):
            gold = generator.poisson(2)
            correct = generator.randint(gold + 1)
            predicted = correct + generator.poisson(0.5)
            human_rating = max(0, int(100 - generator.uniform(5, 8) * (gold - correct)
                                      - generator.uniform(8, 13) * (predicted - correct)))
            self.sentences.append(self.Sentence(gold, predicted, correct, human_rating))

def cova(x,y):
    
    suma = 0
    x_mean = x.mean()
    y_mean = y.mean()
    for i in range(len(x)):
        suma += (x[i]-x_mean)*(y[i]-y_mean)
    return suma

def var(x):

    suma = 0
    x_mean = x.mean()
    for i in range(len(x)):
        suma += ((x[i]-x_mean)**2)
    return suma

def main(args: argparse.Namespace) -> tuple[float, float]:
    # Create the artificial data.
    data = ArtificialData(args)

    # Create `args.bootstrap_samples` bootstrapped samples of the dataset by
    # sampling sentences of the original dataset, and for each compute
    # - average of human ratings,
    # - TP, FP, FN counts of the predicted edits.
    human_ratings, predictions = [], []
    generator = np.random.RandomState(args.seed)
    contador = 0
    for _ in range(args.bootstrap_samples):
        # Bootstrap sample of the dataset.
        sentences = generator.choice(data.sentences, size=len(data.sentences), replace=True)
        suma = 0
        fp = 0
        tp = 0
        fn = 0
        
        vector = [] # Position 0 to TP, position 1 to FP and position 2 to FN
        for element in sentences:
            suma += element.human_rating
            tp += element.predicted_correct
            fp += (element.predicted_edits - element.predicted_correct)
            fn += (element.gold_edits - element.predicted_correct)
            if(element.predicted_correct > element.gold_edits):
                contador+=1
            if(element.predicted_correct > element.predicted_edits):
                contador+=1
        
        # TODO: Append the average of human ratings of `sentences` to `human_ratings`.
        human_ratings.append((suma/sentences.shape[0]))

        # TODO: Compute TP, FP, FN counts of predicted edits in `sentences`
        # and append them to `predictions`.
        vector.append(tp) 
        vector.append(fp)
        vector.append(fn)
        predictions.append(vector)

    
    # Compute Pearson correlation between F_beta score and human ratings
    # for betas between 0 and 2.

    betas, correlations = [], []
    for beta in np.linspace(0, 2, 201):
        betas.append(beta)

        # TODO: For each bootstrap dataset, compute the F_beta score using
        # the counts in `predictions` and then manually compute the Pearson
        # correlation between the computed scores and `human_ratings`. Append
        # the result to `correlations`.
        f_betas = []
        
        for i in range(args.bootstrap_samples):
            tp = predictions[i][0]
            fp = predictions[i][1]
            fn = predictions[i][2]
            f_b = ((tp + (beta**2)*tp)/(tp + fp + ((beta**2)*(tp+fn))))
            f_betas.append(f_b)
        
        cov = cova(np.array(f_betas),np.array(human_ratings))
        
        var_human = var(np.array(human_ratings))
        var_fbetas = var(np.array(f_betas))
        pearson = (cov)/(np.sqrt(var_fbetas)*np.sqrt(var_human))           
        correlations.append(pearson)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(betas, correlations)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"Pearson correlation of $F_\beta$-score and human ratings")
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    # TODO: Assign the highest correlation to `best_correlation` and
    # store corresponding beta to `best_beta`.
    index = np.argmax(correlations)
    best_beta = betas[index]
    best_correlation = correlations[index]

    return best_beta, best_correlation


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_beta, best_correlation = main(args)

    print("Best correlation of {:.3f} was found for beta {:.2f}".format(
        best_correlation, best_beta))
