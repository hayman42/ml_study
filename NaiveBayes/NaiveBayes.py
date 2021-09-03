import csv
import random
import re
from collections import defaultdict
from math import e, pi

# posterior probability = (prior probability * likelihood) / marginal probability
class NaiveBayes:
    def load_csv(self, data, header=False):
        lines = csv.reader(data)
        dataset = list(lines)
        if header:
            # remove header
            dataset = dataset[1:]
        for i in range(len(dataset)):
            dataset[i] = [float(x) if re.search('\d', x) else x for x in dataset[i]]

        return dataset

    def split_data(self, data, percent):
        train_size = int(len(data) * percent)
        trainset = []
        for i in range(train_size):
            idx = random.randrange(len(data))
            trainset.append(data[idx])
            data.pop(idx)

        return trainset, data #testset

    def group_by_class(self, data, target):
        target_map = defaultdict(list)
        for i in range(len(data)):
            features = data[i]
            if not features:
                continue
            x = features[target]
             # designating the last column as the class column
            target_map[x].append(features[:-1]) 

        return target_map

    def mean(self, numbers):
        return sum(numbers) / len(numbers)

    def stdev(self, numbers):
        mean = self.mean(numbers)
        squared_diff_list = []
        for num in numbers:
            squared_diff = (num-mean)**2
            squared_diff_list.append(squared_diff)
        variance = sum(squared_diff_list) / (len(numbers)-1)

        return variance ** 0.5

    def summarize(self, features):
        for feature in zip(*features):
            yield {
                'stdev': self.stdev(feature),
                'mean': self.mean(feature)
            }

    def prior_prob(self, group, target, data):
        return len(group[target]) / len(data)

    def train(self, trainset, target):
        group = self.group_by_class(trainset, target)
        self.summaries = {}
        for target, features in group.items():
            self.summaries[target] = {
                'prior_prob': self.prior_prob(group, target, trainset),
                'summary': [x for x in self.summarize(features)]
            }

    def normal_pdf(self, x, mean, stdev):
        # N(x; µ, σ) = (1 / 2πσ) * (e ^ (x–µ)^2/-2σ^2
        variance = stdev ** 2
        exp_squared_diff = (x-mean) ** 2
        exp_power = -exp_squared_diff / (2*variance)
        exponent =  e ** exp_power
        denominator = ((2 * pi) ** 0.5) * stdev
        normal_prob = exponent / denominator

        return normal_prob

    def joint_prob(self, test_data):
        joint_probs = {}
        for target, features in self.summaries.items():
            total_features = len(features['summary'])
            likelihood = 1
            for i in range(total_features):
                feature = test_data[i]
                mean = features['summary'][i]['mean']
                stdev = features['summary'][i]['stdev']
                normal_prob = self.normal_pdf(feature, mean, stdev)
                likelihood *= normal_prob
            prior_prob = features['prior_prob']
            joint_probs[target] = prior_prob * likelihood

        return joint_probs

    def marginal_prob(self, joint_prob):
        return sum(joint_prob.values())

    def posterior_prob(self, test_data):
        posterior_probs = {}
        joint_prob = self.joint_prob(test_data)
        marginal_prob = self.marginal_prob(joint_prob)
        for target, joint in joint_prob.items():
            posterior_probs[target] = joint / marginal_prob

        return posterior_probs

    def get_maxprob(self, test_data):
        posterior_probs = self.posterior_prob(test_data)
        return max(posterior_probs, key=posterior_probs.get)

    def predict(self, test_data):
        max_probs = []
        for data in test_data:
            max_prob = self.get_maxprob(data)
            max_probs.append(max_prob)

        return max_probs

    def accuracy(self, testset, predicted):
        correct = 0
        actual = [item[-1] for item in testset]
        for act, pred in zip(actual, predicted):
            if act == pred:
                correct += 1

        return correct / len(testset)

if __name__ == '__main__':
    NB = NaiveBayes()

    data = NB.load_csv(open('iris.csv','r'), header = True)
    trainset, testset = NB.split_data(data, percent=0.7)
    print (f'Using {len(trainset)} data for training set and {len(testset)} data for test set')

    group = NB.group_by_class(data, -1)
    print (f'Grouped into {len(group.keys())} classes: {list(group.keys())}')

    NB.train(trainset, -1)

    predicted = NB.predict(testset)
    accuracy = NB.accuracy(testset, predicted)
    print(f'Accuracy : {accuracy:.0%}')